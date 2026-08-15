"""Run build_pair_timeline_v2 over the same selected pair rows.

The input is the previous validation JSON with a ``rows`` list.  This runner
scans raw Scenario-protobuf TFRecords once, then evaluates every pair belonging
to each selected scene.  It keeps the original pair order and continues after
individual failures.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows", payload)
    if not isinstance(rows, list):
        raise ValueError("Input must contain a rows list")
    return rows


def _json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-script", required=True, help="Path to the timeline script")
    parser.add_argument(
        "--builder-function",
        default="build_pair_timeline_v2",
        help="Builder function exported by the timeline script",
    )
    parser.add_argument("--input", type=Path, required=True, help="Previous 500-row validation JSON")
    parser.add_argument("--tfrecord", required=True, help="Raw Scenario TFRecord directory or glob")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--use-interhub-prior", action="store_true")
    args = parser.parse_args()

    v2_path = Path(args.v2_script).resolve()
    sys.path.insert(0, str(v2_path.parent))
    import importlib.util

    spec = importlib.util.spec_from_file_location("build_pair_timeline_v2", v2_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {v2_path}")
    v2 = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = v2
    spec.loader.exec_module(v2)
    builder = getattr(v2, args.builder_function)

    rows = _load_rows(args.input)
    scene_to_rows: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        scene_to_rows.setdefault(str(row["scene_id"]), []).append(row)
    pending = set(scene_to_rows)
    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    v2._require_runtime_deps()
    paths = v2.resolve_tfrecord_paths(args.tfrecord)
    path_scene_hint: Dict[str, str] = {}
    # trajdata's remote cache has one raw Scenario TFRecord per global scene
    # index.  Prefer those exact files when the supplied spec points at the
    # split directory; this avoids scanning tens of thousands of unrelated
    # records just to recover the selected 500 scenes.
    split_dir = Path(args.tfrecord)
    if split_dir.is_dir() and "training_splitted" in split_dir.name:
        direct_paths = []
        for scene_id in scene_to_rows:
            if not scene_id.startswith("scene_"):
                continue
            try:
                index = int(scene_id[len("scene_"):])
            except ValueError:
                continue
            candidate = split_dir / f"training_splitted_{index}.tfrecords"
            if candidate.is_file():
                direct_paths.append(str(candidate))
                path_scene_hint[str(candidate)] = scene_id
        if direct_paths:
            paths = sorted(set(direct_paths))
    print(f"[START] rows={len(rows)} scenes={len(pending)} shards={len(paths)}", flush=True)

    for shard_number, path in enumerate(paths, start=1):
        if not pending:
            break
        # Each training_splitted_N file contains one raw Scenario record.
        # The v1 iterator avoids constructing a tf.data pipeline for every
        # tiny split file while preserving the same protobuf bytes.
        for record_index, serialized in enumerate(
            v2.tf.compat.v1.io.tf_record_iterator(path)
        ):
            scenario = v2._parse_scenario(serialized)
            scene_id = path_scene_hint.get(path, str(scenario.scenario_id))
            if scene_id not in pending:
                continue

            for row in scene_to_rows[scene_id]:
                try:
                    interhub_start = row.get("interhub_window_audit_only") if args.use_interhub_prior else None
                    interhub_end = row.get("interhub_window_end_audit_only") if args.use_interhub_prior else None
                    result = builder(
                        scenario=scenario,
                        agent_a_id=int(row["agent_A"]),
                        agent_b_id=int(row["agent_B"]),
                        interhub_start=interhub_start,
                        interhub_end=interhub_end,
                    )
                    result["source"] = {
                        "tfrecord": path,
                        "record_index_in_shard": int(record_index),
                    }
                    item = {
                        "index": len(results) + len(errors) + 1,
                        "input": row,
                        "result": result,
                    }
                    results.append(item)
                    _json_write(
                        args.output_dir / (
                            f"pair_{len(results) + len(errors):04d}_"
                            f"{scene_id}_A{row['agent_A']}_B{row['agent_B']}.json"
                        ),
                        item,
                    )
                    print(
                        f"[OK {len(results)+len(errors)}/{len(rows)}] {scene_id} "
                        f"A={row['agent_A']} B={row['agent_B']} "
                        f"status={result['analysis']['status']}",
                        flush=True,
                    )
                except Exception as exc:  # continue after one bad pair
                    error = {
                        "index": len(results) + len(errors) + 1,
                        "input": row,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                    errors.append(error)
                    print(
                        f"[ERROR {len(results)+len(errors)}/{len(rows)}] {scene_id} "
                        f"A={row['agent_A']} B={row['agent_B']}: {exc}",
                        flush=True,
                    )
            pending.remove(scene_id)
            if not pending:
                break

        print(
            f"[SCAN] shard={shard_number}/{len(paths)} found={len(scene_to_rows)-len(pending)} "
            f"remaining={len(pending)} elapsed={time.time()-started:.1f}s",
            flush=True,
        )

    if pending:
        for scene_id in sorted(pending):
            for row in scene_to_rows[scene_id]:
                errors.append({
                    "index": len(results) + len(errors) + 1,
                    "input": row,
                    "error_type": "SceneNotFound",
                    "error": f"Scenario {scene_id} not found in supplied TFRecords",
                })

    status_counts = Counter(
        item["result"]["analysis"]["status"]
        for item in results
    )
    facts = Counter()
    for item in results:
        role = item["result"].get("role_centric_facts", {})
        facts[f"transition_agent={role.get('transition_agent')}"] += 1
        facts[f"reference_agent={role.get('reference_agent')}"] += 1
        for key in (
            "transition_agent_changed_lane",
            "transition_agent_entered_reference_lane",
            "reference_agent_maintained_lane",
            "reference_agent_speed_reduced",
            "transition_agent_behind_to_ahead",
        ):
            value = role.get(key)
            facts[f"{key}={value}"] += 1

    summary = {
        "schema_version": "pair_timeline_v2_batch_v1",
        "input": str(args.input),
        "tfrecord": args.tfrecord,
        "num_selected": len(rows),
        "num_success": len(results),
        "num_errors": len(errors),
        "unresolved_scene_count": len(pending),
        "use_interhub_prior": bool(args.use_interhub_prior),
        "analysis_status_counts": dict(status_counts),
        "role_fact_counts": dict(facts),
        "elapsed_seconds": round(time.time() - started, 3),
        "results": results,
        "errors": errors,
    }
    _json_write(args.output_dir / "batch_results.json", summary)
    print(json.dumps({k: summary[k] for k in summary if k not in {"results", "errors"}}, ensure_ascii=False, indent=2), flush=True)
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
