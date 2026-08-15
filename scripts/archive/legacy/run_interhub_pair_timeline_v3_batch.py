#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build v3 pair timelines for InterHub scene_motion_v3 records.

The InterHub records are kept as the stable input identity.  This runner
resolves scene_<scenario_index> to the corresponding raw Waymo split file,
maps ``ego`` to the raw scenario SDC track id, and deliberately does not pass
the InterHub window into the builder as an event-selection prior.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


def json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def jsonl_write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def scene_index(value: Any, scene_id: Any) -> int | None:
    if value is not None:
        try:
            return int(value)
        except (TypeError, ValueError):
            pass
    match = re.search(r"scene_(\d+)", str(scene_id or ""))
    return int(match.group(1)) if match else None


def compact_input(path: Path, data: dict[str, Any]) -> dict[str, Any]:
    source = data.get("source", {}) if isinstance(data.get("source"), dict) else {}
    interaction = data.get("interaction", {}) if isinstance(data.get("interaction"), dict) else {}
    behavior = data.get("behavior", {}) if isinstance(data.get("behavior"), dict) else {}
    participant_ids = interaction.get("key_agent_ids") or interaction.get("participant_ids") or []
    participant_ids = [str(x) for x in participant_ids]
    window = interaction.get("interhub", {}).get("window", {})
    if not isinstance(window, dict):
        window = {}
    record_id = path.stem
    return {
        "interhub_record_id": record_id,
        "interhub_source_file": str(path),
        "interhub_scene_id": source.get("scene_id"),
        "scenario_index": scene_index(source.get("scenario_index"), source.get("scene_id")),
        "interhub_agent_A": participant_ids[0] if len(participant_ids) > 0 else None,
        "interhub_agent_B": participant_ids[1] if len(participant_ids) > 1 else None,
        "participant_ids": participant_ids,
        "interhub_window_start": window.get("start_frame"),
        "interhub_window_end": window.get("end_frame"),
        "source_behavior": behavior.get("type"),
        "selected_behavior_event": data.get("selected_behavior_event"),
    }


def resolve_agent_id(agent_id: Any, scenario: Any) -> int:
    if str(agent_id).lower() == "ego":
        index = int(scenario.sdc_track_index)
        return int(scenario.tracks[index].id)
    return int(agent_id)


def resolve_tfrecord(split_dir: Path, index: int) -> Path | None:
    candidates = [
        split_dir / f"training_splitted_{index}.tfrecords",
        split_dir / f"training_splitted_{index}.tfrecord",
        split_dir / f"training_splitted_{index}.tfrecords.gz",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--tfrecord-dir", type=Path, required=True)
    parser.add_argument("--builder-script", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    builder_path = args.builder_script.resolve()
    sys.path.insert(0, str(builder_path.parent))
    import importlib.util

    spec = importlib.util.spec_from_file_location("pair_timeline_v3_builder", builder_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import builder: {builder_path}")
    builder_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = builder_module
    spec.loader.exec_module(builder_module)
    builder_module._require_runtime_deps()

    input_paths = sorted(args.input_dir.glob("*_v3_classified.json"))
    if args.limit > 0:
        input_paths = input_paths[: args.limit]

    root = args.output_dir
    manifest_dir = root / "00_manifest"
    mapping_dir = root / "01_scene_mapping"
    timeline_dir = root / "02_pair_timelines" / "pair_timelines"
    timeline_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    started = time.time()
    scenario_cache: dict[int, tuple[Any, str, int, int]] = {}

    for ordinal, input_path in enumerate(input_paths, start=1):
        compact: dict[str, Any] = {
            "interhub_record_id": input_path.stem,
            "interhub_source_file": str(input_path),
        }
        try:
            data = json.loads(input_path.read_text(encoding="utf-8"))
            compact = compact_input(input_path, data)
            manifest_rows.append(compact)
            scene_id = compact["interhub_scene_id"]
            index = compact["scenario_index"]
            a_raw = compact["interhub_agent_A"]
            b_raw = compact["interhub_agent_B"]
            if index is None or a_raw is None or b_raw is None:
                raise ValueError("missing scenario_index or pair participant ids")
            if len(compact["participant_ids"]) != 2:
                raise ValueError("expected exactly two InterHub participants")

            if index not in scenario_cache:
                tfrecord = resolve_tfrecord(args.tfrecord_dir, int(index))
                if tfrecord is None:
                    raise FileNotFoundError(f"raw TFRecord not found for scenario_index={index}")
                iterator = builder_module.tf.compat.v1.io.tf_record_iterator(str(tfrecord))
                serialized = next(iterator)
                scenario = builder_module._parse_scenario(serialized)
                scenario_cache[index] = (scenario, str(tfrecord), 0, int(scenario.tracks[scenario.sdc_track_index].id))

            scenario, tfrecord_path, record_index, ego_id = scenario_cache[index]
            a_id = resolve_agent_id(a_raw, scenario)
            b_id = resolve_agent_id(b_raw, scenario)
            if a_id == b_id:
                raise ValueError(f"resolved pair collapses to one track: {a_raw}, {b_raw} -> {a_id}")

            # InterHub's window is audit metadata only.  Do not pass it into
            # build_pair_timeline_v3 as an event-selection prior.
            result = builder_module.build_pair_timeline_v3(
                scenario=scenario,
                agent_a_id=a_id,
                agent_b_id=b_id,
                interhub_start=None,
                interhub_end=None,
            )
            result["interhub_record_id"] = compact["interhub_record_id"]
            result["interhub_scene_label"] = scene_id
            result["interhub_window_audit_only"] = {
                "start_frame": compact["interhub_window_start"],
                "end_frame": compact["interhub_window_end"],
                "used_as_soft_event_prior": False,
            }
            result["source"] = {
                "tfrecord": tfrecord_path,
                "record_index_in_shard": record_index,
                "raw_waymo_scene_id": str(scenario.scenario_id),
                "interhub_record_id": compact["interhub_record_id"],
                "interhub_source_file": compact["interhub_source_file"],
                "interhub_scene_label": scene_id,
                "scenario_index": index,
                "interhub_agent_A": a_raw,
                "interhub_agent_B": b_raw,
                "waymo_agent_A": a_id,
                "waymo_agent_B": b_id,
                "waymo_ego_track_id": ego_id,
            }
            timeline_path = timeline_dir / f"{compact['interhub_record_id']}_pair_timeline.json"
            json_write(timeline_path, result)
            status = str(result.get("analysis", {}).get("status", "unknown"))
            status_counts[status] += 1
            item = {"index": ordinal, "input": compact, "result": result}
            results.append(item)
            mapping_rows.append({
                **compact,
                "raw_waymo_scene_id": str(scenario.scenario_id),
                "tfrecord_path": tfrecord_path,
                "record_index_in_shard": record_index,
                "waymo_agent_A": a_id,
                "waymo_agent_B": b_id,
                "waymo_ego_track_id": ego_id,
                "timeline_file": str(timeline_path),
                "mapping_status": "timeline_success",
                "timeline_status": status,
                "error_message": None,
            })
        except Exception as exc:
            error = {
                "index": ordinal,
                "input": compact,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            errors.append(error)
            mapping_rows.append({
                **compact,
                "raw_waymo_scene_id": None,
                "tfrecord_path": None,
                "record_index_in_shard": None,
                "waymo_agent_A": None,
                "waymo_agent_B": None,
                "waymo_ego_track_id": None,
                "timeline_file": None,
                "mapping_status": "error",
                "timeline_status": None,
                "error_message": f"{type(exc).__name__}: {exc}",
            })
        if ordinal % 100 == 0 or ordinal == len(input_paths):
            print(
                f"[PROGRESS] {ordinal}/{len(input_paths)} success={len(results)} errors={len(errors)} "
                f"cached_scenes={len(scenario_cache)} elapsed={time.time()-started:.1f}s",
                flush=True,
            )

    jsonl_write(manifest_dir / "interhub_records_manifest.jsonl", manifest_rows)
    jsonl_write(mapping_dir / "interhub_waymo_scene_mapping.jsonl", mapping_rows)
    mapping_fields = sorted({key for row in mapping_rows for key in row})
    with (mapping_dir / "interhub_waymo_scene_mapping.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=mapping_fields)
        writer.writeheader()
        writer.writerows(mapping_rows)

    batch = {
        "schema_version": "pair_motion_facts_v3_multichannel_batch",
        "input_dir": str(args.input_dir),
        "tfrecord_dir": str(args.tfrecord_dir),
        "builder_script": str(builder_path),
        "interhub_window_used_for_event_selection": False,
        "num_selected": len(input_paths),
        "num_success": len(results),
        "num_errors": len(errors),
        "analysis_status_counts": dict(status_counts),
        "elapsed_seconds": round(time.time() - started, 3),
        "results": results,
        "errors": errors,
    }
    json_write(root / "02_pair_timelines" / "batch_results.json", batch)
    json_write(root / "02_pair_timelines" / "batch_summary.json", {
        key: value for key, value in batch.items() if key not in {"results", "errors"}
    })
    print(json.dumps({key: value for key, value in batch.items() if key not in {"results", "errors"}}, ensure_ascii=False, indent=2))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
