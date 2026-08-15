#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Attach InterHub/Waymo/visual provenance and finalize episode descriptions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--raw-output-dir", type=Path, required=True)
    parser.add_argument("--final-output-dir", type=Path, required=True)
    parser.add_argument("--episode-map", type=Path, required=True)
    parser.add_argument("--scene-map", type=Path, required=True)
    parser.add_argument("--visual-map", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--timeline-batch", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--coverage-output", type=Path, required=True)
    parser.add_argument("--visual-text-jsonl", type=Path, required=True)
    parser.add_argument("--visual-text-csv", type=Path, required=True)
    args = parser.parse_args()

    episode_rows = load_csv(args.episode_map)
    scene_rows = {str(row.get("interhub_record_id")): row for row in load_csv(args.scene_map)}
    visual_rows = {str(row.get("interhub_record_id")): row for row in load_jsonl(args.visual_map)}
    episode_by_final = {str(row.get("final_input_file")): row for row in episode_rows}
    episode_by_name = {Path(str(row.get("final_input_file"))).name: row for row in episode_rows}

    args.final_output_dir.mkdir(parents=True, exist_ok=True)
    inputs = sorted(args.input_dir.glob("*_llm_input.json"))
    final_records: list[dict[str, Any]] = []
    missing_outputs: list[dict[str, Any]] = []
    invalid_outputs: list[dict[str, Any]] = []

    for input_path in inputs:
        episode = episode_by_final.get(str(input_path)) or episode_by_name.get(input_path.name)
        if episode is None:
            missing_outputs.append({"input_file": str(input_path), "error": "episode_map_missing"})
            continue
        input_stem = input_path.stem
        if input_stem.endswith("_llm_input"):
            input_stem = input_stem[: -len("_llm_input")]
        raw_path = args.raw_output_dir / f"{input_stem}_qwen_llm_description.json"
        if not raw_path.is_file():
            missing_outputs.append({"input_file": str(input_path), "output_file": str(raw_path), "error": "qwen_output_missing"})
            continue
        try:
            description = load_json(raw_path)
            if not description.get("description_short") or not description.get("description_detailed"):
                raise ValueError("description_short/description_detailed missing")
        except Exception as exc:
            invalid_outputs.append({"input_file": str(input_path), "output_file": str(raw_path), "error": str(exc)})
            continue

        record_id = str(episode.get("interhub_record_id"))
        scene = scene_rows.get(record_id, {})
        visual = visual_rows.get(record_id, {})
        description["interhub_record_id"] = record_id
        description["episode_id"] = episode.get("episode_id")
        description["interhub_scene_id"] = scene.get("interhub_scene_id")
        description["raw_waymo_scene_id"] = scene.get("raw_waymo_scene_id")
        description["visual_paths"] = visual.get("visual_paths", [])
        description["provenance_join"] = {
            "interhub_source_file": scene.get("interhub_source_file"),
            "tfrecord_path": scene.get("tfrecord_path"),
            "record_index_in_shard": scene.get("record_index_in_shard"),
            "interhub_agent_A": scene.get("interhub_agent_A"),
            "interhub_agent_B": scene.get("interhub_agent_B"),
            "waymo_agent_A": scene.get("waymo_agent_A"),
            "waymo_agent_B": scene.get("waymo_agent_B"),
            "visual_mapping_status": visual.get("visual_mapping_status"),
        }
        if isinstance(description.get("source"), dict):
            description["source"].update({
                "interhub_record_id": record_id,
                "interhub_scene_id": scene.get("interhub_scene_id"),
                "raw_waymo_scene_id": scene.get("raw_waymo_scene_id"),
                "interhub_source_file": scene.get("interhub_source_file"),
            })
        final_name = input_path.stem.replace("_llm_input", "") + "_description.json"
        final_path = args.final_output_dir / final_name
        final_path.write_text(json.dumps(description, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        final_records.append(description)

    timeline_batch = load_json(args.timeline_batch)
    timeline_results = timeline_batch.get("results", []) if isinstance(timeline_batch, dict) else []
    manifest = load_jsonl(args.manifest)
    scene_map = list(scene_rows.values())
    coverage = {
        "interhub_records": len(manifest),
        "mapped_records": sum(bool(row.get("raw_waymo_scene_id")) for row in scene_map),
        "timeline_success": sum(row.get("mapping_status") == "timeline_success" for row in scene_map),
        "pair_interaction_detected": sum(
            (item.get("result", {}).get("analysis", {}) or {}).get("status") == "pair_event_detected"
            for item in timeline_results
        ),
        "interaction_episodes": len(inputs),
        "llm_inputs": len(inputs),
        "llm_generation_success": len(final_records),
        "final_valid_descriptions": len(final_records),
        "mapping_errors": sum(not bool(row.get("raw_waymo_scene_id")) for row in scene_map),
        "timeline_errors": int(timeline_batch.get("num_errors", 0)) if isinstance(timeline_batch, dict) else None,
        "missing_qwen_outputs": len(missing_outputs),
        "invalid_descriptions": len(invalid_outputs),
        "visual_mapped_records": sum(row.get("visual_mapping_status") == "mapped" for row in visual_rows.values()),
    }
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for record in final_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    visual_rows: list[dict[str, Any]] = []
    for record in final_records:
        join = record.get("provenance_join", {})
        visual_rows.append({
            "interhub_record_id": record.get("interhub_record_id"),
            "episode_id": record.get("episode_id"),
            "interhub_scene_id": record.get("interhub_scene_id"),
            "raw_waymo_scene_id": record.get("raw_waymo_scene_id"),
            "interhub_agent_A": join.get("interhub_agent_A"),
            "interhub_agent_B": join.get("interhub_agent_B"),
            "waymo_agent_A": join.get("waymo_agent_A"),
            "waymo_agent_B": join.get("waymo_agent_B"),
            "event_type": record.get("event", {}).get("type"),
            "event_frame": record.get("event", {}).get("frame"),
            "visual_mapping_status": join.get("visual_mapping_status"),
            "visual_paths": record.get("visual_paths", []),
            "description_short": record.get("description_short"),
            "description_detailed": record.get("description_detailed"),
        })
    args.visual_text_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.visual_text_jsonl.open("w", encoding="utf-8") as handle:
        for row in visual_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    args.visual_text_csv.parent.mkdir(parents=True, exist_ok=True)
    visual_fields = sorted({key for row in visual_rows for key in row})
    with args.visual_text_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=visual_fields)
        writer.writeheader()
        for row in visual_rows:
            writer.writerow({**row, "visual_paths": ";".join(row.get("visual_paths", []))})
    dump_json(args.coverage_output, {
        **coverage,
        "missing_qwen_output_records": missing_outputs,
        "invalid_output_records": invalid_outputs,
    })
    print(json.dumps(coverage, ensure_ascii=False, indent=2))
    return 0 if not missing_outputs and not invalid_outputs else 2


if __name__ == "__main__":
    raise SystemExit(main())
