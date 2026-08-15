#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract raw vehicle trajectories and local map geometry once.

This script is the TFRecord stage of the raw-trajectory experiment.  It does
not call an LLM.  The resulting per-vehicle JSON files can be reused by
``validate_raw_vehicle_local_map.py`` with ``--input-json-dir``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.facts import build_pair_timeline as facts_builder
from scripts.llm import validate_raw_vehicle_local_map as raw_runner


VEHICLE_JSON_DIRNAME = "vehicle_json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tfrecord", required=True)
    parser.add_argument(
        "--scene-id",
        action="append",
        help="Scene ID; repeat the option to process multiple scenes",
    )
    parser.add_argument(
        "--scene-ids-file",
        help="Text file containing one scene ID per line; # comments are ignored",
    )
    parser.add_argument(
        "--record-index",
        type=int,
        help="0-based scene index in one TFRecord shard",
    )
    parser.add_argument("--record-start", type=int)
    parser.add_argument("--record-end", type=int)
    parser.add_argument(
        "--all-scenes",
        action="store_true",
        help="Process from --record-start through TFRecord EOF",
    )
    vehicle_group = parser.add_mutually_exclusive_group(required=True)
    vehicle_group.add_argument("--vehicle-id", type=int)
    vehicle_group.add_argument("--all-vehicles", action="store_true")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--jsonl-output")
    parser.add_argument("--csv-output")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--min-contiguous-valid-frames", type=int, default=20)
    parser.add_argument("--compression-type", default="")
    parser.add_argument("--radius-m", type=float, default=raw_runner.DEFAULT_LOCAL_MAP_RADIUS_M)
    parser.add_argument("--max-map-lanes", type=int, default=raw_runner.DEFAULT_MAX_MAP_LANES)
    parser.add_argument(
        "--max-map-points-per-lane",
        type=int,
        default=raw_runner.DEFAULT_MAX_MAP_POINTS_PER_LANE,
    )
    return parser.parse_args()


def _scene_ids(args: argparse.Namespace) -> list[str]:
    values = [str(value).strip() for value in (args.scene_id or [])]
    if args.scene_ids_file:
        if values:
            raise ValueError("use either --scene-id or --scene-ids-file, not both")
        values = [
            line.strip()
            for line in Path(args.scene_ids_file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    if not values:
        raise ValueError("scene mode requires --scene-id or --scene-ids-file")
    return values


def _tracks(scenario: Any, vehicle_id: int | None, limit: int | None) -> list[Any]:
    tracks = sorted(
        (
            track
            for track in scenario.tracks
            if int(track.object_type) == 1
            and (vehicle_id is None or int(track.id) == int(vehicle_id))
        ),
        key=lambda track: int(track.id),
    )
    if limit is not None:
        tracks = tracks[: int(limit)]
    return tracks


def _blank_description() -> dict[str, Any]:
    return {
        "description_short": "",
        "description_detailed": "",
        "behavior_segments": [],
        "uncertainty_notes": "",
    }


def _extract_scene(
    args: argparse.Namespace,
    scenario: Any,
    source_file: str,
    record_index: int,
    scene_output_dir: Path,
    jsonl_output: str | None = None,
    csv_output: str | None = None,
) -> dict[str, Any]:
    scene_output_dir.mkdir(parents=True, exist_ok=True)
    vehicle_json_dir = scene_output_dir / VEHICLE_JSON_DIRNAME
    vehicle_json_dir.mkdir(parents=True, exist_ok=True)
    lane_map = facts_builder.build_lane_map(scenario)
    tracks = _tracks(scenario, args.vehicle_id, args.limit)
    if not tracks:
        raise ValueError(f"no vehicle tracks found for scene {scenario.scenario_id}")

    jsonl_path = Path(jsonl_output) if jsonl_output else (
        scene_output_dir / "raw_vehicle_local_map_inputs.jsonl"
    )
    csv_path = Path(csv_output) if csv_output else (
        scene_output_dir / "raw_vehicle_local_map_inputs_summary.csv"
    )
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    skipped = 0

    with jsonl_path.open("w", encoding="utf-8") as jsonl_handle:
        for track in tracks:
            vehicle_id = int(track.id)
            normalized_track = facts_builder.extract_agent_track(scenario, vehicle_id)
            max_run = raw_runner._max_contiguous_valid_run(normalized_track)
            payload = raw_runner.build_experiment_input(
                scenario,
                vehicle_id,
                args.radius_m,
                args.max_map_lanes,
                args.max_map_points_per_lane,
                lane_map=lane_map,
                source_file=source_file,
                record_index=record_index,
            )
            is_skipped = max_run < args.min_contiguous_valid_frames
            status = (
                "skipped_insufficient_contiguous_valid_frames"
                if is_skipped
                else "ready_for_llm"
            )
            skip_reason = (
                f"max_contiguous_valid_frames={max_run} < "
                f"min_contiguous_valid_frames={args.min_contiguous_valid_frames}"
                if is_skipped
                else ""
            )
            output_path = vehicle_json_dir / (
                f"{scenario.scenario_id}__vehicle_{vehicle_id}"
                "_raw_local_map_input.json"
            )
            vehicle_output = {
                "status": status,
                "max_contiguous_valid_frames": max_run,
                "skip_reason": skip_reason,
                "description": _blank_description(),
                "input": payload,
            }
            output_path.write_text(
                json.dumps(vehicle_output, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            jsonl_handle.write(
                json.dumps(
                    {
                        "scene_id": str(scenario.scenario_id),
                        "vehicle_id": vehicle_id,
                        "status": status,
                        "max_contiguous_valid_frames": max_run,
                        "skip_reason": skip_reason,
                        "json_path": str(output_path),
                        "input": payload,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            if is_skipped:
                skipped += 1
            rows.append({
                "scene_id": str(scenario.scenario_id),
                "vehicle_id": vehicle_id,
                "status": status,
                "max_contiguous_valid_frames": max_run,
                "valid_frame_count": len(payload["timeline"]["valid_frame_indices"]),
                "local_lane_count": len(payload["local_map_geometry"]["lanes"]),
                "llm_called": False,
                "description_short": "",
                "description_detailed": "",
                "behavior_segments": "[]",
                "uncertainty_notes": "",
                "json_path": str(output_path),
                "error_type": "",
                "error": "",
                "skip_reason": skip_reason,
            })

    with csv_path.open("w", encoding="utf-8-sig", newline="") as csv_handle:
        writer = csv.DictWriter(csv_handle, fieldnames=raw_runner.CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "scene_id": str(scenario.scenario_id),
        "requested_vehicle_count": len(tracks),
        "extracted": len(tracks) - skipped,
        "skipped": skipped,
        "min_contiguous_valid_frames": int(args.min_contiguous_valid_frames),
        "jsonl_output": str(jsonl_path),
        "csv_output": str(csv_path),
        "vehicle_json_dir": str(vehicle_json_dir),
    }
    (scene_output_dir / "extraction_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def _aggregate(
    summaries: list[dict[str, Any]],
    output_dir: Path,
    jsonl_output: str | None,
    csv_output: str | None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = Path(jsonl_output) if jsonl_output else (
        output_dir / "all_scenes_raw_vehicle_local_map_inputs.jsonl"
    )
    csv_path = Path(csv_output) if csv_output else (
        output_dir / "all_scenes_raw_vehicle_local_map_inputs_summary.csv"
    )
    with jsonl_path.open("w", encoding="utf-8") as out_handle:
        for summary in summaries:
            with Path(summary["jsonl_output"]).open(encoding="utf-8") as in_handle:
                for line in in_handle:
                    out_handle.write(line)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as out_handle:
        writer = csv.DictWriter(out_handle, fieldnames=raw_runner.CSV_FIELDS)
        writer.writeheader()
        for summary in summaries:
            with Path(summary["csv_output"]).open(
                encoding="utf-8-sig", newline=""
            ) as in_handle:
                for row in csv.DictReader(in_handle):
                    writer.writerow({field: row.get(field, "") for field in raw_runner.CSV_FIELDS})
    result = {
        "scene_count": len(summaries),
        "extracted": sum(int(item["extracted"]) for item in summaries),
        "skipped": sum(int(item["skipped"]) for item in summaries),
        "jsonl_output": str(jsonl_path),
        "csv_output": str(csv_path),
        "scenes": summaries,
    }
    (output_dir / "extraction_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    args = _parse_args()
    if args.min_contiguous_valid_frames <= 0:
        raise ValueError("--min-contiguous-valid-frames must be positive")
    if args.radius_m <= 0 or args.max_map_lanes <= 0 or args.max_map_points_per_lane <= 1:
        raise ValueError("map locality and sampling limits must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    has_range = args.record_start is not None or args.record_end is not None
    has_record_index = args.record_index is not None
    if has_record_index and has_range:
        raise ValueError("use either --record-index or record range options, not both")
    if has_record_index and (args.scene_id or args.scene_ids_file):
        raise ValueError("--record-index cannot be combined with scene selection")
    if has_record_index and args.record_index < 0:
        raise ValueError("--record-index must be non-negative")
    if has_range:
        if args.record_start is None:
            raise ValueError("record mode requires --record-start")
        if args.record_end is None and not args.all_scenes:
            raise ValueError("record mode requires --record-end or --all-scenes")
        if args.record_end is not None and args.all_scenes:
            raise ValueError("use either --record-end or --all-scenes, not both")
        if args.vehicle_id is not None:
            raise ValueError("record mode processes all vehicles; omit --vehicle-id")
        if args.scene_id or args.scene_ids_file:
            raise ValueError("record mode cannot be combined with scene selection")
        if args.record_start < 0 or (
            args.record_end is not None and args.record_end <= args.record_start
        ):
            raise ValueError("record range must satisfy 0 <= start < end")
    elif args.all_scenes:
        raise ValueError("--all-scenes requires record mode")

    facts_builder._require_runtime_deps()
    paths = facts_builder.resolve_tfrecord_paths(args.tfrecord)
    output_dir = Path(args.output_dir)
    summaries: list[dict[str, Any]] = []

    if has_record_index:
        scenario, source_file, record_index = facts_builder.load_scenario_by_record_index(
            paths,
            int(args.record_index),
            args.compression_type,
        )
        summary = _extract_scene(
            args,
            scenario,
            source_file,
            record_index,
            output_dir,
        )
        summary["record_index_in_shard"] = int(record_index)
        (output_dir / "extraction_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False))
        return 0

    if has_range:
        if len(paths) != 1:
            raise ValueError("record mode requires exactly one TFRecord shard")
        dataset = facts_builder.tf.data.TFRecordDataset(
            paths[0], compression_type=args.compression_type
        )
        scene_output_dir = output_dir / "scenes"
        for record_index, serialized_record in enumerate(dataset):
            if args.record_end is not None and record_index >= args.record_end:
                break
            if record_index < args.record_start:
                continue
            scenario = facts_builder._parse_scenario(bytes(serialized_record.numpy()))
            summary = _extract_scene(
                args,
                scenario,
                paths[0],
                int(record_index),
                scene_output_dir / str(scenario.scenario_id),
            )
            summary["record_index_in_shard"] = int(record_index)
            summaries.append(summary)
            print(json.dumps(summary, ensure_ascii=False), flush=True)
    else:
        scene_ids = _scene_ids(args)
        for scene_id in scene_ids:
            scenario, source_file, record_index = facts_builder.load_scenario_by_id(
                paths, scene_id, args.compression_type
            )
            scene_output_dir = output_dir if len(scene_ids) == 1 else output_dir / scene_id
            summary = _extract_scene(
                args,
                scenario,
                source_file,
                record_index,
                scene_output_dir,
            )
            summaries.append(summary)
            print(json.dumps(summary, ensure_ascii=False), flush=True)

    if len(summaries) == 1 and not has_range:
        return 0
    result = _aggregate(
        summaries,
        output_dir,
        args.jsonl_output,
        args.csv_output,
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
