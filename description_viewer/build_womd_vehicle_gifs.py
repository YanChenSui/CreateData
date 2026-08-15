#!/usr/bin/env python3
"""Render WOMD vehicle descriptions with the existing InterHub renderer.

The input path is deliberately independent of InterHub interaction records:
WOMD Scenario protobufs provide all vehicle tracks, ``full_vehicle_facts``
selects target vehicles, and the per-vehicle Qwen JSON provides the overlay
windows.  No candidate pair, agent A/B role, or InterHub scene mapping is read.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import build_pair_timeline as facts_builder
from utils.visualize_utils import draw_womd_pic


STATIONARY_MAX_INTERNAL_GAP_FRAMES = 3
MIN_CONTIGUOUS_VALID_FRAMES = 3


def _read_json_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    paths: list[Path]
    if path.is_file():
        paths = [path]
    elif path.is_dir():
        paths = sorted(
            candidate for candidate in path.rglob("*")
            if candidate.is_file()
            and candidate.suffix.lower() in {".json", ".jsonl"}
        )
    else:
        raise FileNotFoundError(path)

    for candidate in paths:
        try:
            if candidate.suffix.lower() == ".jsonl":
                values = [
                    json.loads(line)
                    for line in candidate.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            else:
                values = [json.loads(candidate.read_text(encoding="utf-8"))]
        except (OSError, json.JSONDecodeError):
            continue
        for value in values:
            if isinstance(value, dict):
                records.append(value)
    return records


def _facts_by_vehicle(path: Path, scene_id: str) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for record in _read_json_records(path):
        if str(record.get("scene_id", scene_id)) != str(scene_id):
            continue
        vehicle_id = record.get("vehicle_id")
        if vehicle_id is None:
            vehicle = record.get("vehicle")
            if isinstance(vehicle, Mapping):
                vehicle_id = vehicle.get("id")
        try:
            result[int(vehicle_id)] = record
        except (TypeError, ValueError):
            continue
    return result


def _description_for(
    path: Path,
    scene_id: str,
    vehicle_id: int,
) -> dict[str, Any]:
    candidates: list[Path]
    if path.is_file():
        candidates = [path]
    elif path.is_dir():
        exact_name = f"{scene_id}__vehicle_{vehicle_id}_vehicle_qwen_description.json"
        candidates = [
            candidate for candidate in path.rglob("*.json")
            if candidate.name == exact_name
            or f"vehicle_{vehicle_id}_vehicle_qwen_description" in candidate.name
        ]
    else:
        return {}
    for candidate in sorted(candidates):
        try:
            value = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            return value
    return {}


def _scenario_tracks(scenario: Any):
    tracks = []
    for track in scenario.tracks:
        # WOMD's Scenario.ObjectType enum uses 1 for VEHICLE.  The
        # pair-timeline loader exposes track extraction, but not this enum
        # constant, so keep the protobuf value local to this adapter.
        if int(track.object_type) == 1:
            tracks.append(facts_builder.extract_agent_track(scenario, int(track.id)))
    return tracks


def _max_contiguous_valid_run(track: Any) -> int:
    valid = np.asarray(track.valid, dtype=bool)
    best = current = 0
    for is_valid in valid:
        if bool(is_valid):
            current += 1
            best = max(best, current)
        else:
            current = 0
    return int(best)


def _fill_stationary_internal_gaps(
    states: np.ndarray,
    row: int,
    valid: np.ndarray,
    max_gap_frames: int = STATIONARY_MAX_INTERNAL_GAP_FRAMES,
) -> None:
    """Fill only short internal gaps for a stationary render track.

    Leading/trailing invalid frames remain NaN.  Gaps longer than the fixed
    render-only tolerance remain NaN as well, so this never invents a long
    unobserved presence interval.
    """
    valid_indices = np.flatnonzero(valid)
    for left, right in zip(valid_indices[:-1], valid_indices[1:]):
        gap = int(right - left - 1)
        if gap <= 0 or gap > max_gap_frames:
            continue
        left_state = states[row, int(left)].copy()
        right_state = states[row, int(right)].copy()
        if not (
            np.all(np.isfinite(left_state[0:2]))
            and np.all(np.isfinite(right_state[0:2]))
        ):
            continue
        heading = left_state[4]
        if not np.isfinite(heading):
            heading = right_state[4]
        for frame in range(int(left) + 1, int(right)):
            states[row, frame, 0:2] = left_state[0:2]
            states[row, frame, 2:4] = 0.0
            states[row, frame, 4] = heading


def _state_array(
    tracks: list[Any],
    stationary_ids: set[int] | None = None,
) -> np.ndarray:
    if not tracks:
        raise ValueError("scenario contains no vehicle tracks")
    total_frames = max(track.T for track in tracks)
    states = np.full((len(tracks), total_frames, 5), np.nan, dtype=np.float64)
    for row, track in enumerate(tracks):
        valid = np.asarray(track.valid, dtype=bool)
        states[row, valid, 0:2] = np.asarray(track.xy[valid], dtype=np.float64)
        states[row, valid, 2:4] = np.asarray(track.velocity[valid], dtype=np.float64)
        states[row, valid, 4] = np.asarray(track.yaw[valid], dtype=np.float64)
        if int(track.agent_id) in (stationary_ids or set()):
            _fill_stationary_internal_gaps(states, row, valid)
    return states


def _lane_polylines(scenario: Any) -> list[np.ndarray]:
    try:
        lane_map = facts_builder.build_lane_map(scenario)
    except Exception:
        return []
    return [
        np.asarray(segment.xy, dtype=np.float64)
        for segment in lane_map.segments.values()
        if len(segment.xy) >= 2
    ]


def _is_stationary_description(description: Mapping[str, Any]) -> bool:
    """Reuse the existing stationary decision recorded with the description."""
    generation = description.get("generation")
    if isinstance(generation, Mapping) and generation.get("stationary_detected") is True:
        return True

    # Keep compatibility with older description JSON files that do not carry
    # generation.stationary_detected.  This is only a deterministic fallback;
    # it does not introduce a second behavior classifier.
    summary = description.get("track_motion_summary")
    if not isinstance(summary, Mapping):
        return False
    try:
        max_speed = float(summary.get("max_speed_mps"))
        path_length = float(summary.get("path_length_m"))
    except (TypeError, ValueError):
        return False
    return np.isfinite(max_speed) and np.isfinite(path_length) and max_speed <= 0.1 and path_length <= 0.1


def _remove_vehicle_outputs(scene_output_dir: Path, stem: str) -> None:
    """Remove only stale outputs for one vehicle before a rerun/skip."""
    for gif_path in (
        scene_output_dir / f"{stem}.gif",
        scene_output_dir / "stationary" / f"{stem}.gif",
    ):
        gif_path.unlink(missing_ok=True)
    for png_dir in (
        scene_output_dir / "png" / stem,
        scene_output_dir / "stationary" / "png" / stem,
    ):
        if png_dir.exists():
            shutil.rmtree(png_dir)


def _render_vehicle(
    scene_id: str,
    vehicle_id: int,
    description: Mapping[str, Any],
    all_agents: list[str],
    agent_states: np.ndarray,
    lane_polylines: list[np.ndarray],
    dt: float,
    num_frames: int,
    output_dir: Path,
    fps: float,
    dpi: int,
    considertime: float,
) -> dict[str, Any]:
    import imageio.v2 as imageio

    stem = f"{scene_id}__vehicle_{vehicle_id}"
    scene_output_dir = output_dir / scene_id
    _remove_vehicle_outputs(scene_output_dir, stem)
    vehicle_output_dir = (
        scene_output_dir / "stationary"
        if _is_stationary_description(description)
        else scene_output_dir
    )
    png_dir = vehicle_output_dir / "png" / stem
    png_dir.mkdir(parents=True, exist_ok=True)
    frame_paths: list[Path] = []
    behavior_segments = description.get("behavior_segments", [])
    supporting_ranges = description.get("supporting_frame_ranges", [])
    for frame in range(num_frames):
        frame_paths.append(Path(draw_womd_pic(
            all_agents=all_agents,
            agent_states=agent_states,
            timestamp_index=frame,
            all_timesteps=range(num_frames),
            dt=dt,
            lane_polylines=lane_polylines,
            target_vehicle_id=str(vehicle_id),
            behavior_segments=behavior_segments,
            supporting_frame_ranges=supporting_ranges,
            save_path=str(png_dir),
            dpi=dpi,
            considertime=considertime,
            output_stem="frame",
            title_prefix=f"WOMD {scene_id} | Vehicle {vehicle_id}",
        )))
    gif_path = vehicle_output_dir / f"{stem}.gif"
    imageio.mimsave(
        gif_path,
        [imageio.imread(path) for path in frame_paths],
        duration=1.0 / float(fps),
        loop=0,
    )
    return {
        "scene_id": scene_id,
        "vehicle_id": vehicle_id,
        "gif": str(gif_path),
        "png_dir": str(png_dir),
        "frames": len(frame_paths),
    }


def _render_scenario(args: argparse.Namespace, scenario: Any) -> dict[str, Any]:
    scene_id = str(scenario.scenario_id)
    facts = _facts_by_vehicle(args.full_vehicle_facts, scene_id)
    tracks = _scenario_tracks(scenario)
    tracks_by_id = {int(track.agent_id): track for track in tracks}
    selected_ids = sorted(facts) if args.vehicle_id is None else sorted(set(args.vehicle_id))
    missing = [vehicle_id for vehicle_id in selected_ids if vehicle_id not in tracks_by_id]
    if missing:
        raise KeyError(f"vehicle IDs not found as WOMD vehicle tracks: {missing}")
    if not selected_ids:
        raise ValueError("no target vehicles found in full_vehicle_facts")

    descriptions = {
        int(track.agent_id): _description_for(
            args.vehicle_qwen_description, scene_id, int(track.agent_id)
        )
        for track in tracks
    }
    stationary_ids = {
        vehicle_id
        for vehicle_id, description in descriptions.items()
        if _is_stationary_description(description)
    }
    all_agents = [str(track.agent_id) for track in tracks]
    agent_states = _state_array(tracks, stationary_ids=stationary_ids)
    dt_values = np.diff(np.asarray(scenario.timestamps_seconds, dtype=float))
    dt_values = dt_values[np.isfinite(dt_values) & (dt_values > 0)]
    dt = float(np.median(dt_values)) if len(dt_values) else 0.1
    lane_polylines = _lane_polylines(scenario)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    scene_output_dir = args.output_dir / scene_id
    for vehicle_id in selected_ids:
        track = tracks_by_id[vehicle_id]
        description = descriptions.get(vehicle_id, {})
        max_valid_run = _max_contiguous_valid_run(track)
        stem = f"{scene_id}__vehicle_{vehicle_id}"
        if max_valid_run < MIN_CONTIGUOUS_VALID_FRAMES:
            _remove_vehicle_outputs(scene_output_dir, stem)
            status = (
                "skipped_insufficient_contiguous_valid_frames"
                if _is_stationary_description(description)
                else "skipped_insufficient_valid_frames"
            )
            print(
                f"[skip] scene={scene_id} vehicle={vehicle_id} status={status} "
                f"max_contiguous_valid_frames={max_valid_run}",
                file=sys.stderr,
            )
            results.append({
                "scene_id": scene_id,
                "vehicle_id": vehicle_id,
                "status": status,
                "max_contiguous_valid_frames": max_valid_run,
            })
            continue
        results.append(_render_vehicle(
            scene_id,
            vehicle_id,
            description,
            all_agents,
            agent_states,
            lane_polylines,
            dt,
            len(scenario.timestamps_seconds),
            args.output_dir,
            args.fps,
            args.dpi,
            args.considertime,
        ))
    return {"scene_id": scene_id, "vehicles": results}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tfrecord", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--scene-id")
    group.add_argument("--record-index", type=int)
    group.add_argument(
        "--record-index-range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Render an inclusive range of records, for example: 0 9",
    )
    parser.add_argument("--full-vehicle-facts", required=True, type=Path)
    parser.add_argument("--vehicle-qwen-description", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--vehicle-id", type=int, action="append")
    parser.add_argument("--compression-type", default="")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--considertime", type=float, default=5.0)
    args = parser.parse_args()
    if args.fps <= 0 or args.dpi <= 0 or args.considertime <= 0:
        raise ValueError("fps, dpi, and considertime must be positive")

    facts_builder._require_runtime_deps()
    paths = facts_builder.resolve_tfrecord_paths(args.tfrecord)
    if args.scene_id is not None:
        scenarios = [facts_builder.load_scenario_by_id(
            paths, args.scene_id, args.compression_type
        )[0]]
    else:
        if args.record_index is not None:
            record_indices = [int(args.record_index)]
        else:
            start, end = (int(value) for value in args.record_index_range)
            if start < 0 or end < start:
                raise ValueError("record-index-range must satisfy 0 <= START <= END")
            record_indices = list(range(start, end + 1))
        scenarios = [
            facts_builder.load_scenario_by_record_index(
                paths, record_index, args.compression_type
            )[0]
            for record_index in record_indices
        ]

    rendered = [_render_scenario(args, scenario) for scenario in scenarios]
    if len(rendered) == 1:
        output = rendered[0]
    else:
        output = {"scenes": rendered}
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
