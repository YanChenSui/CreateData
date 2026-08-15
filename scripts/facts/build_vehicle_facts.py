#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build objective physical facts for every vehicle in one WOMD scenario.

This is the vehicle-level counterpart of ``build_pair_timeline.py``.  It does
not enumerate objects of interest, generate pair candidates, or compute pair
relations / pair geometry.  The only selection is the Waymo track type:

    for track in scenario.tracks:
        if track.object_type == TYPE_VEHICLE:
            ...

The output is JSONL with one ``vehicle_physical_facts_v1`` object per vehicle.
Short or otherwise unusable tracks are retained with ``status`` set to
``insufficient_data``; they are never silently dropped.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:
    from . import build_pair_timeline as facts_builder
except ImportError:  # pragma: no cover - direct script execution
    import build_pair_timeline as facts_builder


TYPE_VEHICLE = 1
SCHEMA_VERSION = "vehicle_physical_facts_v1"
MIN_FACT_FRAMES = 2
BATCH_SCENE_DIRNAME = "scenes"
ALL_VEHICLE_FACTS_NAME = "all_vehicle_facts.jsonl"


def _jsonable(value: Any) -> Any:
    """Use the canonical builder conversion when available."""
    return facts_builder._jsonable(value)


def _insufficient_fact(section: str, error: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "insufficient_data",
        "section": section,
    }
    if error:
        result["audit_error"] = error
    return result


def _vehicle_metadata(track: Any) -> dict[str, Any]:
    valid = getattr(track, "valid")
    valid_indices = [int(i) for i, is_valid in enumerate(valid) if bool(is_valid)]
    return {
        "is_sdc": bool(track.is_sdc),
        "is_object_of_interest": bool(track.is_object_of_interest),
        "track_to_predict": bool(track.track_to_predict),
        "valid_frame_count": len(valid_indices),
        "first_valid_frame": valid_indices[0] if valid_indices else None,
        "last_valid_frame": valid_indices[-1] if valid_indices else None,
    }


def _timeline(scenario: Any, track: Any) -> dict[str, Any]:
    timestamps = [round(float(value), 4) for value in scenario.timestamps_seconds]
    return {
        "num_frames": int(track.T),
        "current_time_index": int(scenario.current_time_index),
        "timestamps_seconds": timestamps,
        "valid_frame_indices": [
            int(i) for i, is_valid in enumerate(track.valid) if bool(is_valid)
        ],
        "full_scenario_used": True,
    }


def _track_motion_summary(track: Any) -> dict[str, Any]:
    """Summarize motion directly from the complete valid WOMD track.

    This is descriptive evidence, not a WOMD-native stationary label.  Invalid
    states are excluded, and path length is accumulated only across adjacent
    valid frames so missing-state gaps do not become artificial movement.
    """
    valid_indices = np.flatnonzero(np.asarray(track.valid, dtype=bool))
    valid_speed = np.asarray(track.speed, dtype=np.float64)[valid_indices]
    valid_xy = np.asarray(track.xy, dtype=np.float64)[valid_indices]

    finite_speed = valid_speed[np.isfinite(valid_speed)]
    finite_xy_mask = np.all(np.isfinite(valid_xy), axis=1)
    finite_xy = valid_xy[finite_xy_mask]

    endpoint_displacement = None
    spatial_extent = None
    if len(finite_xy) >= 1:
        endpoint_displacement = 0.0
        if len(finite_xy) >= 2:
            endpoint_displacement = float(np.linalg.norm(finite_xy[-1] - finite_xy[0]))
        span = np.ptp(finite_xy, axis=0)
        spatial_extent = float(np.linalg.norm(span))

    path_length = 0.0
    for previous_index, current_index in zip(valid_indices[:-1], valid_indices[1:]):
        if int(current_index) != int(previous_index) + 1:
            continue
        previous_xy = np.asarray(track.xy[previous_index], dtype=np.float64)
        current_xy = np.asarray(track.xy[current_index], dtype=np.float64)
        if np.all(np.isfinite(previous_xy)) and np.all(np.isfinite(current_xy)):
            path_length += float(np.linalg.norm(current_xy - previous_xy))

    return {
        "valid_frame_count": int(len(valid_indices)),
        "max_speed_mps": (
            float(np.max(finite_speed)) if len(finite_speed) else None
        ),
        "p95_speed_mps": (
            float(np.percentile(finite_speed, 95)) if len(finite_speed) else None
        ),
        "endpoint_displacement_m": endpoint_displacement,
        "spatial_extent_m": spatial_extent,
        "path_length_m": float(path_length) if finite_xy_mask.any() else None,
    }


def _empty_section_facts(error: str | None = None) -> dict[str, Any]:
    return {
        "lateral_motion_evidence": _insufficient_fact(
            "lateral_motion_evidence", error
        ),
        "speed_change": _insufficient_fact("speed_change", error),
        "heading_motion": _insufficient_fact("heading_motion", error),
        "turn_maneuver": _insufficient_fact("turn_maneuver", error),
        "u_turn_evidence": _insufficient_fact("u_turn_evidence", error),
        "physical_lane_chain": _insufficient_fact("physical_lane_chain", error),
        "route_transition": _insufficient_fact("route_transition", error),
        "map_context": _insufficient_fact("map_context", error),
    }


def build_vehicle_fact(scenario: Any, track_id: int) -> dict[str, Any]:
    """Build one vehicle record from the complete scenario timeline."""
    track = facts_builder.extract_agent_track(scenario, int(track_id))
    scene_id = str(scenario.scenario_id)
    valid_count = int(sum(bool(value) for value in track.valid))
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "vehicle_record_id": f"{scene_id}::vehicle_{int(track.agent_id)}",
        "scene_id": scene_id,
        "vehicle_id": int(track.agent_id),
        "status": "insufficient_data" if valid_count < MIN_FACT_FRAMES else "facts_extracted",
        "metadata": _vehicle_metadata(track),
        "timeline": _timeline(scenario, track),
        "track_motion_summary": _track_motion_summary(track),
        # Quality is an evidence gate only.  It does not repair or remove
        # raw states; caption eligibility remains a downstream decision.
        "trajectory_quality": facts_builder.extract_trajectory_quality(track),
        "physical_facts": _empty_section_facts(),
    }

    # Speed and heading are single-track facts and do not require a map.
    try:
        result["physical_facts"]["speed_change"] = facts_builder.extract_speed_change_facts(track)
    except Exception as exc:  # retain the vehicle even if one fact extractor fails
        result["physical_facts"]["speed_change"] = _insufficient_fact(
            "speed_change", f"{type(exc).__name__}: {exc}"
        )
    try:
        result["physical_facts"]["heading_motion"] = facts_builder.extract_heading_motion_facts(track)
    except Exception as exc:  # retain the vehicle even if one fact extractor fails
        result["physical_facts"]["heading_motion"] = _insufficient_fact(
            "heading_motion", f"{type(exc).__name__}: {exc}"
        )
    try:
        result["physical_facts"]["u_turn_evidence"] = facts_builder.extract_u_turn_evidence(track)
    except Exception as exc:  # retain the vehicle even if one fact extractor fails
        result["physical_facts"]["u_turn_evidence"] = _insufficient_fact(
            "u_turn_evidence", f"{type(exc).__name__}: {exc}"
        )

    try:
        lane_map = facts_builder.build_lane_map(scenario)
        lane_timeline = facts_builder.build_lane_timeline(track, lane_map)
        # Lane-relative lateral motion is retained only as physical evidence.
        # It is not a standalone semantic behavior such as "shift left/right".
        # The evidence may support a lane-change fact when combined with
        # a confirmed physical-lane transition.
        lateral_facts = facts_builder.extract_physical_lateral_facts(
            track, lane_timeline, lane_map
        )
        result["physical_facts"]["lateral_motion_evidence"] = lateral_facts
        result["physical_facts"]["physical_lane_chain"] = (
            facts_builder.build_physical_lane_chain(
                track, lane_timeline, lane_map, lateral_facts
            )
        )
        result["physical_facts"]["turn_maneuver"] = (
            facts_builder.extract_turn_maneuver_facts(
                track,
                lane_timeline,
                lane_map,
                result["physical_facts"]["heading_motion"],
                result["physical_facts"]["physical_lane_chain"],
            )
        )
        result["physical_facts"]["route_transition"] = facts_builder._compute_route_transition(
            track, lane_timeline, lane_map
        )
        result["physical_facts"]["map_context"] = facts_builder._compute_vehicle_map_context(
            scenario, track, lane_map
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        # Map-dependent facts remain explicit unknown/insufficient values, but
        # the vehicle record itself is still emitted.
        for key in (
            "lateral_motion_evidence",
            "physical_lane_chain",
            "turn_maneuver",
            "route_transition",
            "map_context",
        ):
            result["physical_facts"][key] = _insufficient_fact(key, error)
        result["audit_error"] = error

    if valid_count < MIN_FACT_FRAMES:
        result["status_reason"] = (
            f"Only {valid_count} valid frame(s); vehicle retained with insufficient_data."
        )
    return _jsonable(result)


def build_vehicle_facts(scenario: Any) -> list[dict[str, Any]]:
    """Build one record for every vehicle track, without OOI/pair selection."""
    records: list[dict[str, Any]] = []
    for track in scenario.tracks:
        if int(track.object_type) != TYPE_VEHICLE:
            continue
        records.append(build_vehicle_fact(scenario, int(track.id)))
    return records


def _safe_scene_filename(scene_id: str) -> str:
    value = "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in str(scene_id)
    ).strip("._")
    return value or "unknown_scene"


def _write_vehicle_records(
    records: list[dict[str, Any]],
    output: Path,
) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(f".{output.name}.tmp")
    try:
        with temporary_output.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(
                    json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
                )
        temporary_output.replace(output)
    finally:
        if temporary_output.exists():
            temporary_output.unlink()

    return _summarize_vehicle_records(records, output)


def _summarize_vehicle_records(
    records: list[Mapping[str, Any]],
    output: Path,
) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    for record in records:
        status = str(record["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        "vehicle_records": len(records),
        "status_counts": status_counts,
        "output": str(output),
    }


def _read_existing_scene_summary(
    output: Path,
    scene_id: str,
    source_file: str,
    record_index: int,
    expected_vehicle_count: int,
) -> dict[str, Any] | None:
    """Return metadata for a complete existing scene JSONL, or None.

    A malformed or vehicle-count-incomplete file is treated as incomplete and
    will be rebuilt.  An empty file is valid only for a scene with zero
    vehicles.  This keeps --skip-existing safe after an interrupted write.
    """
    if not output.is_file():
        return None
    if output.stat().st_size == 0 and int(expected_vehicle_count) != 0:
        return None

    records: list[Mapping[str, Any]] = []
    try:
        with output.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    return None
                if str(value.get("scene_id")) != str(scene_id):
                    return None
                if "vehicle_id" not in value or "status" not in value:
                    return None
                records.append(value)
    except (OSError, json.JSONDecodeError, UnicodeError):
        return None

    if len(records) != int(expected_vehicle_count):
        return None

    return {
        "scene_id": str(scene_id),
        "source_file": source_file,
        "record_index_in_shard": int(record_index),
        **_summarize_vehicle_records(records, output),
        "skipped_existing": True,
    }


def _merge_scene_vehicle_records(
    scene_summaries: list[dict[str, Any]],
    output: Path,
) -> int:
    """Create one aggregate JSONL from the successfully written scene files."""
    output.parent.mkdir(parents=True, exist_ok=True)
    record_count = 0
    with output.open("w", encoding="utf-8") as destination:
        for scene_summary in scene_summaries:
            scene_output = Path(str(scene_summary["output"]))
            with scene_output.open("r", encoding="utf-8") as source:
                for line in source:
                    if not line.strip():
                        continue
                    destination.write(line if line.endswith("\n") else line + "\n")
                    record_count += 1
    return record_count


def _build_batch(
    paths: list[str],
    record_start: int,
    record_end: int | None,
    compression_type: str,
    output_dir: Path,
    skip_existing: bool,
) -> dict[str, Any]:
    """Build a contiguous record range in one sequential TFRecord pass."""
    if len(paths) != 1:
        raise ValueError("batch record mode requires --tfrecord to resolve to exactly one shard")
    if record_start < 0 or (record_end is not None and record_end <= record_start):
        raise ValueError("batch record range must satisfy 0 <= record-start < record-end")

    output_dir.mkdir(parents=True, exist_ok=True)
    scene_output_dir = output_dir / BATCH_SCENE_DIRNAME
    scene_output_dir.mkdir(parents=True, exist_ok=True)
    scene_summaries: list[dict[str, Any]] = []
    failed_scenes: list[dict[str, Any]] = []
    total_vehicle_records = 0

    dataset = facts_builder.tf.data.TFRecordDataset(
        paths[0], compression_type=compression_type
    )
    for record_index, serialized_record in enumerate(dataset):
        if record_end is not None and record_index >= record_end:
            break
        if record_index < record_start:
            continue
        try:
            scenario = facts_builder._parse_scenario(bytes(serialized_record.numpy()))
            output = scene_output_dir / f"{_safe_scene_filename(scenario.scenario_id)}.jsonl"
            if skip_existing:
                expected_vehicle_count = sum(
                    1 for track in scenario.tracks
                    if int(track.object_type) == TYPE_VEHICLE
                )
                existing_summary = _read_existing_scene_summary(
                    output,
                    str(scenario.scenario_id),
                    paths[0],
                    int(record_index),
                    expected_vehicle_count,
                )
                if existing_summary is not None:
                    scene_summaries.append(existing_summary)
                    total_vehicle_records += int(existing_summary["vehicle_records"])
                    print(json.dumps(existing_summary, ensure_ascii=False), flush=True)
                    continue

            records = build_vehicle_facts(scenario)
            summary = _write_vehicle_records(records, output)
            scene_summary = {
                "scene_id": str(scenario.scenario_id),
                "source_file": paths[0],
                "record_index_in_shard": int(record_index),
                **summary,
                "skipped_existing": False,
            }
            scene_summaries.append(scene_summary)
            total_vehicle_records += int(summary["vehicle_records"])
            print(json.dumps(scene_summary, ensure_ascii=False), flush=True)
        except Exception as exc:
            failure = {
                "record_index_in_shard": int(record_index),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            failed_scenes.append(failure)
            print(json.dumps({"failed_scene": failure}, ensure_ascii=False), flush=True)

    all_vehicle_facts = output_dir / ALL_VEHICLE_FACTS_NAME
    aggregate_vehicle_records = _merge_scene_vehicle_records(
        scene_summaries,
        all_vehicle_facts,
    )

    summary = {
        "mode": "batch",
        "source_file": paths[0],
        "record_start": int(record_start),
        "record_end": int(record_end) if record_end is not None else None,
        "requested_scene_count": (
            int(record_end - record_start) if record_end is not None else "all_remaining"
        ),
        "processed_scene_count": len(scene_summaries),
        "computed_scene_count": sum(
            1 for item in scene_summaries if not item.get("skipped_existing", False)
        ),
        "skipped_scene_count": sum(
            1 for item in scene_summaries if item.get("skipped_existing", False)
        ),
        "failed_scene_count": len(failed_scenes),
        "vehicle_records": total_vehicle_records,
        "all_vehicle_records": aggregate_vehicle_records,
        "all_vehicle_facts": str(all_vehicle_facts),
        "scene_output_dir": str(scene_output_dir),
        "scenes": scene_summaries,
        "failed_scenes": failed_scenes,
        "output_dir": str(output_dir),
    }
    (output_dir / "scene_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "failed_scenes.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["record_index_in_shard", "error_type", "error"],
        )
        writer.writeheader()
        writer.writerows(failed_scenes)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tfrecord", required=True, help="TFRecord file, directory, or glob")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--scene-id", help="Exact WOMD Scenario.scenario_id")
    group.add_argument("--record-index", type=int, help="0-based record index in one shard")
    parser.add_argument(
        "--record-start",
        type=int,
        help="Batch mode: inclusive 0-based record index",
    )
    parser.add_argument(
        "--record-end",
        type=int,
        help="Batch mode: exclusive 0-based record index",
    )
    parser.add_argument(
        "--all-scenes",
        action="store_true",
        help="Batch mode: process from --record-start through TFRecord EOF",
    )
    parser.add_argument("--compression-type", default="")
    parser.add_argument("--output", help="Single-scene output JSONL path")
    parser.add_argument("--output-dir", help="Batch-scene output directory")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip complete existing scene JSONL files in batch or single-scene mode",
    )
    args = parser.parse_args()

    has_range = args.record_start is not None or args.record_end is not None
    has_single = args.scene_id is not None or args.record_index is not None
    if has_range and has_single:
        parser.error("batch record range cannot be combined with --scene-id or --record-index")
    if has_range:
        if args.record_start is None:
            parser.error("batch mode requires --record-start")
        if args.record_end is None and not args.all_scenes:
            parser.error("batch mode requires --record-end or --all-scenes")
        if args.record_end is not None and args.all_scenes:
            parser.error("use either --record-end or --all-scenes, not both")
        if args.output:
            parser.error("batch mode uses --output-dir, not --output")
        if not args.output_dir:
            parser.error("batch mode requires --output-dir")
    else:
        if not has_single:
            parser.error("provide --scene-id, --record-index, or a batch record range")
        if not args.output:
            parser.error("single-scene mode requires --output")
        if args.output_dir:
            parser.error("single-scene mode uses --output, not --output-dir")
        if args.all_scenes:
            parser.error("--all-scenes requires batch mode")
    return args


def main() -> int:
    args = _parse_args()
    facts_builder._require_runtime_deps()
    paths = facts_builder.resolve_tfrecord_paths(args.tfrecord)
    if args.record_start is not None:
        summary = _build_batch(
            paths,
            int(args.record_start),
            int(args.record_end) if args.record_end is not None else None,
            args.compression_type,
            Path(args.output_dir),
            bool(args.skip_existing),
        )
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        return 1 if summary["failed_scene_count"] else 0

    if args.scene_id is not None:
        scenario, source_file, record_index = facts_builder.load_scenario_by_id(
            paths, args.scene_id, args.compression_type
        )
    else:
        scenario, source_file, record_index = facts_builder.load_scenario_by_record_index(
            paths, int(args.record_index), args.compression_type
        )

    if args.skip_existing:
        expected_vehicle_count = sum(
            1 for track in scenario.tracks
            if int(track.object_type) == TYPE_VEHICLE
        )
        existing_summary = _read_existing_scene_summary(
            Path(args.output),
            str(scenario.scenario_id),
            source_file,
            int(record_index),
            expected_vehicle_count,
        )
        if existing_summary is not None:
            print(json.dumps(existing_summary, ensure_ascii=False))
            return 0

    records = build_vehicle_facts(scenario)
    summary = _write_vehicle_records(records, Path(args.output))
    print(json.dumps({
        "scene_id": str(scenario.scenario_id),
        "source_file": source_file,
        "record_index_in_shard": int(record_index),
        **summary,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
