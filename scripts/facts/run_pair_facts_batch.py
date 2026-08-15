#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the canonical facts-only pair builder over InterHub candidate records.

This is the only active batch entry point for the Waymo pair-facts layer.
InterHub supplies the candidate scene and the two participant IDs.  The full
raw Waymo Scenario is then loaded and analyzed; InterHub windows are retained
as provenance only and never select, crop, or extend an event.

This command stops before interaction semantics, episode fusion, LLM input
construction, and Qwen generation.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

try:
    from . import build_pair_timeline as builder
except ImportError:  # pragma: no cover - direct script execution
    import build_pair_timeline as builder


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _scene_index(source: Mapping[str, Any]) -> int | None:
    for value in (source.get("scenario_index"), source.get("scene_id")):
        try:
            return int(value)
        except (TypeError, ValueError):
            pass
        match = re.search(r"scene_(\d+)", str(value or ""))
        if match:
            return int(match.group(1))
    return None


def _candidate_pair(data: Mapping[str, Any]) -> tuple[Any, Any]:
    interaction = data.get("interaction", {})
    if not isinstance(interaction, Mapping):
        interaction = {}
    ids = interaction.get("key_agent_ids") or interaction.get("participant_ids") or []
    if not isinstance(ids, list) or len(ids) != 2:
        raise ValueError("expected exactly two InterHub participant ids")
    return ids[0], ids[1]


def _resolve_agent_id(value: Any, scenario: Any) -> int:
    if str(value).lower() == "ego":
        return int(scenario.tracks[scenario.sdc_track_index].id)
    return int(value)


def _shard_path(directory: Path, index: int) -> Path:
    for suffix in (".tfrecords", ".tfrecord", ".tfrecords.gz"):
        candidate = directory / f"training_splitted_{index}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Waymo shard not found for scenario_index={index}: {directory}")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--tfrecord-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--record-index", type=int, default=0)
    parser.add_argument("--compression-type", default="")
    args = parser.parse_args()

    builder._require_runtime_deps()
    input_paths = sorted(args.input_dir.glob("*_v3_classified.json"))
    if args.limit > 0:
        input_paths = input_paths[: args.limit]
    if not input_paths:
        raise FileNotFoundError(f"No *_v3_classified.json files found in {args.input_dir}")

    output_dir = args.output_dir
    facts_dir = output_dir / "pair_facts"
    manifest_rows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    scenario_cache: dict[int, tuple[Any, str, int]] = {}
    started = time.time()

    for ordinal, input_path in enumerate(input_paths, start=1):
        data = _read_json(input_path)
        source = data.get("source", {})
        source = source if isinstance(source, Mapping) else {}
        interhub_scene = source.get("scene_id")
        index = _scene_index(source)
        raw_a, raw_b = _candidate_pair(data)
        manifest = {
            "ordinal": ordinal,
            "interhub_record_id": input_path.stem,
            "interhub_source_file": str(input_path),
            "interhub_scene_id": interhub_scene,
            "scenario_index": index,
            "interhub_agent_A": str(raw_a),
            "interhub_agent_B": str(raw_b),
        }
        manifest_rows.append(manifest)
        try:
            if index is None:
                raise ValueError("missing scenario_index and scene_<index> source scene_id")
            if index not in scenario_cache:
                shard = _shard_path(args.tfrecord_dir, index)
                scenario, source_file, record_index = builder.load_scenario_by_record_index(
                    [str(shard)], args.record_index, args.compression_type
                )
                scenario_cache[index] = (scenario, source_file, record_index)
            scenario, source_file, record_index = scenario_cache[index]
            agent_a = _resolve_agent_id(raw_a, scenario)
            agent_b = _resolve_agent_id(raw_b, scenario)
            if agent_a == agent_b:
                raise ValueError(f"pair collapses to one Waymo track: {raw_a}, {raw_b}")

            result = builder.build_pair_timeline_v3(
                scenario=scenario,
                agent_a_id=agent_a,
                agent_b_id=agent_b,
                interhub_start=None,
                interhub_end=None,
            )
            result["interhub_record_id"] = input_path.stem
            result["interhub_scene_label"] = interhub_scene
            result["source"] = {
                "tfrecord": source_file,
                "record_index_in_shard": record_index,
                "raw_waymo_scene_id": str(scenario.scenario_id),
                "scenario_index": index,
                "interhub_agent_A": str(raw_a),
                "interhub_agent_B": str(raw_b),
                "waymo_agent_A": agent_a,
                "waymo_agent_B": agent_b,
            }
            output_path = facts_dir / f"{input_path.stem}_pair_facts.json"
            _write_json(output_path, result)
            status_counts["facts_extracted"] += 1
            mapping_rows.append({
                **manifest,
                "raw_waymo_scene_id": str(scenario.scenario_id),
                "waymo_agent_A": agent_a,
                "waymo_agent_B": agent_b,
                "timeline_file": str(output_path),
                "status": "facts_extracted",
                "error": None,
            })
        except Exception as exc:  # keep one bad candidate from hiding others
            errors.append({
                **manifest,
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
            mapping_rows.append({
                **manifest,
                "raw_waymo_scene_id": None,
                "waymo_agent_A": None,
                "waymo_agent_B": None,
                "timeline_file": None,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            })
        if ordinal % 100 == 0 or ordinal == len(input_paths):
            print(
                f"[PROGRESS] {ordinal}/{len(input_paths)} success="
                f"{status_counts['facts_extracted']} errors={len(errors)} "
                f"cached_scenes={len(scenario_cache)} elapsed={time.time()-started:.1f}s",
                flush=True,
            )

    _write_jsonl(output_dir / "manifest.jsonl", manifest_rows)
    _write_jsonl(output_dir / "mapping.jsonl", mapping_rows)
    _write_jsonl(output_dir / "errors.jsonl", errors)
    _write_json(output_dir / "coverage_report.json", {
        "interhub_records": len(input_paths),
        "mapped_records": sum(row["status"] == "facts_extracted" for row in mapping_rows),
        "facts_extracted": status_counts["facts_extracted"],
        "errors": len(errors),
        "unique_scenario_indices": len(scenario_cache),
        "semantic_layer_run": False,
        "llm_generation_run": False,
        "builder": "scripts/facts/build_pair_timeline.py",
        "schema_version": "pair_physical_facts_v1",
    })
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
