#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare an LLM description from raw trajectory plus local map geometry.

This is an experiment-only input path.  It deliberately does not pass any
derived behavior facts such as acceleration, turn, lane change, trajectory
quality, or lane assignment to the model.  The output contains the raw
observation payload and, when requested, an OpenAI-compatible LLM response.

Example::

    python scripts/llm/validate_raw_vehicle_local_map.py \
        --tfrecord /path/to/womd.tfrecord \
        --scene-id a662cc105b15a772 \
        --vehicle-id 2225 \
        --output raw_vehicle_2225_local_map.json \
        --call-llm --model qwen-3.6
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

facts_builder = None


def _get_facts_builder() -> Any:
    """Import the TensorFlow-backed facts module only for TFRecord mode."""
    global facts_builder
    if facts_builder is None:
        from scripts.facts import build_pair_timeline

        facts_builder = build_pair_timeline
    return facts_builder


DEFAULT_LOCAL_MAP_RADIUS_M = 60.0
DEFAULT_MAX_MAP_LANES = 40
DEFAULT_MAX_MAP_POINTS_PER_LANE = 32
DEFAULT_MODEL = os.getenv("RAW_MAP_MODEL", "qwen-3.6")
DEFAULT_BASE_URL = os.getenv(
    "RAW_MAP_BASE_URL",
    os.getenv("QWEN_BASE_URL", "http://172.17.0.1:60200/v1"),
)
DEFAULT_API_KEY_ENV = "RAW_MAP_API_KEY"
DEFAULT_WORKERS = 4
DEFAULT_MIN_CONTIGUOUS_VALID_FRAMES = 20
DEFAULT_MAX_TOKENS = 2048
VEHICLE_JSON_DIRNAME = "vehicle_json"
DEFAULT_PROMPT_PATH = (
    PROJECT_ROOT / "prompt" / "raw_vehicle_local_map_validation_prompt.txt"
)
CSV_FIELDS = [
    "scene_id",
    "vehicle_id",
    "status",
    "max_contiguous_valid_frames",
    "valid_frame_count",
    "local_lane_count",
    "llm_called",
    "description_short",
    "description_detailed",
    "behavior_segments",
    "uncertainty_notes",
    "json_path",
    "error_type",
    "error",
    "skip_reason",
]


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _relative_xy(point_xy: np.ndarray, origin_xy: np.ndarray) -> list[float]:
    return [
        round(float(point_xy[0] - origin_xy[0]), 3),
        round(float(point_xy[1] - origin_xy[1]), 3),
    ]


def _sample_polyline(polyline: np.ndarray, max_points: int) -> np.ndarray:
    if len(polyline) <= max_points:
        return polyline
    indices = np.linspace(0, len(polyline) - 1, max_points, dtype=int)
    return polyline[np.unique(indices)]


def _raw_trajectory(track: Any, origin_xy: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for frame in range(track.T):
        valid = bool(track.valid[frame])
        row: dict[str, Any] = {
            "frame": int(frame),
            "time_s": _finite_float(track.timestamps[frame]),
            "valid": valid,
        }
        if valid:
            row.update({
                "x_relative_m": round(float(track.xy[frame, 0] - origin_xy[0]), 3),
                "y_relative_m": round(float(track.xy[frame, 1] - origin_xy[1]), 3),
                "z_m": _finite_float(track.z[frame]),
                "vx_mps": _finite_float(track.velocity[frame, 0]),
                "vy_mps": _finite_float(track.velocity[frame, 1]),
                "speed_mps": _finite_float(track.speed[frame]),
                "heading_rad": _finite_float(track.yaw[frame]),
            })
        else:
            row.update({
                "x_relative_m": None,
                "y_relative_m": None,
                "z_m": None,
                "vx_mps": None,
                "vy_mps": None,
                "speed_mps": None,
                "heading_rad": None,
            })
        rows.append(row)
    return rows


def _max_contiguous_valid_run(track: Any) -> int:
    """Return the longest consecutive run of valid raw trajectory frames."""
    best = current = 0
    for value in np.asarray(track.valid, dtype=bool):
        if bool(value):
            current += 1
            best = max(best, current)
        else:
            current = 0
    return int(best)


def _local_map_geometry(
    lane_map: Any,
    track: Any,
    origin_xy: np.ndarray,
    radius_m: float,
    max_lanes: int,
    max_points_per_lane: int,
) -> list[dict[str, Any]]:
    valid_xy = np.asarray(track.xy[np.asarray(track.valid, dtype=bool)], dtype=float)
    if len(valid_xy) == 0:
        return []

    candidates: list[tuple[float, int, Any]] = []
    for lane_id, lane in lane_map.segments.items():
        polyline_xy = np.asarray(lane.xy, dtype=float)
        if len(polyline_xy) == 0:
            continue
        # This is only a geometric locality filter.  No per-frame lane
        # assignment or behavior label is passed to the model.
        sample = polyline_xy[:: max(1, len(polyline_xy) // 64)]
        delta = sample[:, None, :] - valid_xy[None, :: max(1, len(valid_xy) // 64), :]
        min_distance = float(np.sqrt(np.min(np.sum(delta * delta, axis=2))))
        if min_distance <= radius_m:
            candidates.append((min_distance, int(lane_id), lane))

    candidates.sort(key=lambda item: (item[0], item[1]))
    result: list[dict[str, Any]] = []
    for min_distance, lane_id, lane in candidates[:max_lanes]:
        sampled = _sample_polyline(np.asarray(lane.polyline, dtype=float), max_points_per_lane)
        result.append({
            "lane_id": lane_id,
            "lane_type_raw": int(lane.lane_type),
            "min_distance_to_track_m": round(min_distance, 3),
            "centerline_xy_relative_m": [
                _relative_xy(point[:2], origin_xy) for point in sampled
            ],
            "entry_lane_ids": [int(value) for value in lane.entry_lanes],
            "exit_lane_ids": [int(value) for value in lane.exit_lanes],
            "left_neighbor_lane_ids": [int(value) for value in lane.left_neighbors],
            "right_neighbor_lane_ids": [int(value) for value in lane.right_neighbors],
        })
    return result


def build_experiment_input(
    scenario: Any,
    vehicle_id: int,
    radius_m: float,
    max_lanes: int,
    max_points_per_lane: int,
    lane_map: Any | None = None,
    source_file: str | None = None,
    record_index: int | None = None,
) -> dict[str, Any]:
    track = facts_builder.extract_agent_track(scenario, int(vehicle_id))
    valid_indices = np.flatnonzero(np.asarray(track.valid, dtype=bool))
    if len(valid_indices) == 0:
        raise ValueError(f"vehicle {vehicle_id} has no valid raw frames")
    origin_xy = np.asarray(track.xy[int(valid_indices[0])], dtype=float)
    if lane_map is None:
        lane_map = facts_builder.build_lane_map(scenario)
    return {
        "experiment": "raw_trajectory_plus_local_map_geometry",
        "input_policy": {
            "derived_behavior_facts_included": False,
            "lane_assignment_included": False,
            "trajectory_quality_included": False,
            "coordinate_note": (
                "x/y are relative to the vehicle's first valid-frame center; "
                "heading is the raw Waymo heading in radians."
            ),
        },
        "scene_id": str(scenario.scenario_id),
        "vehicle_id": int(track.agent_id),
        "timeline": {
            "num_frames": int(track.T),
            "timestamps_seconds": [
                round(float(value), 4) for value in scenario.timestamps_seconds
            ],
            "valid_frame_indices": [int(value) for value in valid_indices],
        },
        "raw_trajectory": _raw_trajectory(track, origin_xy),
        "local_map_geometry": {
            "origin_xy_world_m": [round(float(value), 3) for value in origin_xy],
            "radius_m": float(radius_m),
            "lanes": _local_map_geometry(
                lane_map,
                track,
                origin_xy,
                radius_m,
                max_lanes,
                max_points_per_lane,
            ),
        },
        "source": {
            "tfrecord": source_file,
            "record_index": record_index,
        },
    }


def build_llm_prompt(
    payload: Mapping[str, Any],
    prompt_file: str | Path = DEFAULT_PROMPT_PATH,
) -> str:
    template = Path(prompt_file).read_text(encoding="utf-8").strip()
    return (
        f"{template}\n\n"
        "Raw experiment input:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)}"
    )


def call_llm(
    prompt: str,
    model: str,
    base_url: str,
    api_key_env: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str:
    from openai import OpenAI

    client = OpenAI(
        api_key=os.getenv(api_key_env, "EMPTY"),
        base_url=base_url,
    )
    request_kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "Return valid JSON only. Do not include markdown fences.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": int(max_tokens),
        "stream": False,
    }
    # Match the project Qwen runner.  Without this, some Qwen deployments can
    # spend the whole response budget on reasoning and leave message.content
    # empty even though the request itself succeeded.
    if str(model).lower().startswith("qwen"):
        request_kwargs["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": False},
        }
    budgets = [int(max_tokens), max(int(max_tokens) * 2, 4096)]
    last_finish_reason = None
    for attempt, budget in enumerate(budgets):
        request_kwargs["max_tokens"] = budget
        response = client.chat.completions.create(**request_kwargs)
        choice = response.choices[0]
        last_finish_reason = choice.finish_reason
        message = choice.message
        content = getattr(message, "content", None)
        if isinstance(content, list):
            content = "".join(
                str(item.get("text", ""))
                for item in content
                if isinstance(item, Mapping)
            )
        if content and str(content).strip():
            return str(content)
        reasoning = getattr(message, "reasoning_content", None)
        if (
            attempt == 0
            and reasoning
            and str(reasoning).strip()
            and str(choice.finish_reason) == "length"
        ):
            continue
        if reasoning and str(reasoning).strip():
            raise RuntimeError(
                "LLM returned reasoning_content but no final content; "
                f"finish_reason={choice.finish_reason}"
            )
        raise RuntimeError(
            "LLM returned empty content; "
            f"finish_reason={choice.finish_reason}"
        )
    raise RuntimeError(
        "LLM returned reasoning_content but no final content after retry; "
        f"finish_reason={last_finish_reason}"
    )


def _description_from_response(
    response: str,
    num_frames: int,
) -> dict[str, Any]:
    """Extract the user-facing description fields while preserving raw output."""
    try:
        parsed = json.loads(response)
    except (TypeError, ValueError):
        parsed = {}
    if not isinstance(parsed, Mapping):
        parsed = {}
    segments: list[dict[str, Any]] = []
    raw_segments = parsed.get("behavior_segments", [])
    if isinstance(raw_segments, list):
        for item in raw_segments:
            if not isinstance(item, Mapping):
                continue
            try:
                start_frame = int(item["start_frame"])
                end_frame = int(item["end_frame"])
            except (KeyError, TypeError, ValueError):
                continue
            description = item.get("description")
            if (
                start_frame < 0
                or end_frame < start_frame
                or end_frame >= int(num_frames)
                or not isinstance(description, str)
                or not description.strip()
            ):
                continue
            segments.append({
                "start_frame": start_frame,
                "end_frame": end_frame,
                "description": description.strip(),
            })
    return {
        "description_short": str(parsed.get("description_short", "")),
        "description_detailed": str(parsed.get("description_detailed", "")),
        "behavior_segments": segments,
        "uncertainty_notes": str(parsed.get("uncertainty_notes", "")),
    }


def _build_vehicle_result(
    scenario: Any,
    track: Any,
    lane_map: Any,
    radius_m: float,
    max_lanes: int,
    max_points_per_lane: int,
    source_file: str,
    record_index: int,
    prompt_file: str,
    call_model: bool,
    model: str,
    base_url: str,
    api_key_env: str,
    max_tokens: int,
) -> dict[str, Any]:
    """Build one vehicle payload and optionally call the LLM.

    This function is independent per vehicle and is safe to run in a worker
    thread.  File writing remains in the caller so output artifacts stay
    deterministic.
    """
    vehicle_id = int(track.id)
    payload = build_experiment_input(
        scenario,
        vehicle_id,
        radius_m,
        max_lanes,
        max_points_per_lane,
        lane_map=lane_map,
        source_file=source_file,
        record_index=record_index,
    )
    prompt = build_llm_prompt(payload, prompt_file)
    vehicle_output: dict[str, Any] = {
        "description": {
            "description_short": "",
            "description_detailed": "",
            "behavior_segments": [],
            "uncertainty_notes": "",
        },
        "input": payload,
        "llm_prompt": prompt,
    }
    if call_model:
        response = call_llm(prompt, model, base_url, api_key_env, max_tokens)
        vehicle_output["description"] = _description_from_response(
            response,
            int(payload["timeline"]["num_frames"]),
        )
        vehicle_output["llm"] = {
            "model": model,
            "base_url": base_url,
            "response": response,
        }
    return {
        "vehicle_id": vehicle_id,
        "payload": payload,
        "output": vehicle_output,
    }


def _max_contiguous_valid_indices(indices: Any) -> int:
    """Return the longest consecutive run from a serialized frame-index list."""
    values = sorted({int(value) for value in indices})
    if not values:
        return 0
    best = current = 1
    for previous, value in zip(values, values[1:]):
        if value == previous + 1:
            current += 1
            best = max(best, current)
        else:
            current = 1
    return int(best)


def _load_preextracted_input(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load one extraction artifact and return (wrapper, experiment input)."""
    wrapper = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(wrapper, Mapping):
        raise ValueError(f"input JSON must contain an object: {path}")
    payload = wrapper.get("input")
    if not isinstance(payload, Mapping):
        # Also accept a bare experiment input JSON for easier reuse.
        payload = wrapper
    if not payload.get("scene_id") or payload.get("vehicle_id") is None:
        raise ValueError(f"input JSON has no scene_id/vehicle_id: {path}")
    return dict(wrapper), dict(payload)


def _preextracted_input_paths(args: argparse.Namespace) -> list[Path]:
    if args.input_json:
        paths = [Path(value) for value in args.input_json]
    else:
        root = Path(args.input_json_dir)
        if not root.exists():
            raise FileNotFoundError(f"input JSON directory does not exist: {root}")
        paths = sorted(
            path
            for path in root.rglob("*.json")
            if path.name.endswith("_raw_local_map_input.json")
            or path.name.endswith("_raw_local_map.json")
        )
    if not paths:
        raise ValueError("no pre-extracted vehicle JSON files found")
    return paths


def _build_preextracted_result(
    path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    wrapper, payload = _load_preextracted_input(path)
    timeline = payload.get("timeline")
    if not isinstance(timeline, Mapping):
        raise ValueError(f"input JSON has no timeline: {path}")
    valid_indices = timeline.get("valid_frame_indices", [])
    max_run = int(wrapper.get(
        "max_contiguous_valid_frames",
        _max_contiguous_valid_indices(valid_indices),
    ))
    is_skipped = max_run < args.min_contiguous_valid_frames
    skip_reason = (
        f"max_contiguous_valid_frames={max_run} < "
        f"min_contiguous_valid_frames={args.min_contiguous_valid_frames}"
        if is_skipped
        else ""
    )
    status = (
        "skipped_insufficient_contiguous_valid_frames"
        if is_skipped
        else "ready_for_llm"
    )
    prompt = build_llm_prompt(payload, args.prompt_file)
    description = {
        "description_short": "",
        "description_detailed": "",
        "behavior_segments": [],
        "uncertainty_notes": "",
    }
    output: dict[str, Any] = {
        "status": status,
        "max_contiguous_valid_frames": max_run,
        "skip_reason": skip_reason,
        "description": description,
        "input": payload,
        "llm_prompt": prompt,
        "source_input_json": str(path),
    }
    if args.call_llm and not is_skipped:
        response = call_llm(
            prompt,
            args.model,
            args.base_url,
            args.api_key_env,
            args.max_tokens,
        )
        output["description"] = _description_from_response(
            response,
            int(timeline["num_frames"]),
        )
        output["llm"] = {
            "model": args.model,
            "base_url": args.base_url,
            "response": response,
        }
        output["status"] = "success"
    return {
        "scene_id": str(payload["scene_id"]),
        "vehicle_id": int(payload["vehicle_id"]),
        "payload": payload,
        "output": output,
        "source_path": str(path),
        "status": output["status"],
        "max_contiguous_valid_frames": max_run,
        "skip_reason": skip_reason,
        "llm_called": bool(args.call_llm and not is_skipped),
    }


def _run_preextracted_batch(args: argparse.Namespace) -> int:
    input_paths = _preextracted_input_paths(args)
    output_dir = Path(args.output_dir)
    vehicle_json_dir = output_dir / VEHICLE_JSON_DIRNAME
    vehicle_json_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = Path(args.jsonl_output) if args.jsonl_output else (
        output_dir / "raw_vehicle_local_map.jsonl"
    )
    csv_path = Path(args.csv_output) if args.csv_output else (
        output_dir / "raw_vehicle_local_map_summary.csv"
    )
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    futures: dict[Path, Any] = {}
    input_metadata: dict[Path, dict[str, Any]] = {}
    for path in input_paths:
        try:
            wrapper, payload = _load_preextracted_input(path)
            max_run = int(wrapper.get(
                "max_contiguous_valid_frames",
                _max_contiguous_valid_indices(
                    payload["timeline"]["valid_frame_indices"]
                ),
            ))
            input_metadata[path] = {
                "scene_id": str(payload["scene_id"]),
                "vehicle_id": int(payload["vehicle_id"]),
                "max_contiguous_valid_frames": max_run,
                "valid_frame_count": len(payload["timeline"]["valid_frame_indices"]),
                "local_lane_count": len(payload["local_map_geometry"]["lanes"]),
            }
        except Exception:
            input_metadata[path] = {
                "scene_id": "",
                "vehicle_id": "",
                "max_contiguous_valid_frames": "",
                "valid_frame_count": "",
                "local_lane_count": "",
            }
    print(json.dumps({
        "event": "batch_started",
        "total": len(input_paths),
        "workers": int(args.workers),
        "call_llm": bool(args.call_llm),
    }, ensure_ascii=False), flush=True)

    completed_results: dict[Path, dict[str, Any]] = {}
    future_errors: dict[Path, BaseException] = {}
    completed_count = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for path in input_paths:
            futures[path] = executor.submit(_build_preextracted_result, path, args)
        future_paths = {future: path for path, future in futures.items()}
        for future in as_completed(futures.values()):
            path = future_paths[future]
            completed_count += 1
            try:
                result = future.result()
                completed_results[path] = result
                print(json.dumps({
                    "event": "vehicle_completed",
                    "progress": f"{completed_count}/{len(input_paths)}",
                    "scene_id": result["scene_id"],
                    "vehicle_id": result["vehicle_id"],
                    "status": result["status"],
                    "max_contiguous_valid_frames": result[
                        "max_contiguous_valid_frames"
                    ],
                    "llm_called": result["llm_called"],
                }, ensure_ascii=False), flush=True)
            except BaseException as exc:
                future_errors[path] = exc
                metadata = input_metadata[path]
                print(json.dumps({
                    "event": "vehicle_completed",
                    "progress": f"{completed_count}/{len(input_paths)}",
                    "scene_id": metadata["scene_id"],
                    "vehicle_id": metadata["vehicle_id"],
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }, ensure_ascii=False), flush=True)

    csv_rows: list[dict[str, Any]] = []
    success = skipped = failed = 0
    with jsonl_path.open("w", encoding="utf-8") as jsonl_handle:
        for path in input_paths:
            try:
                if path in future_errors:
                    raise future_errors[path]
                result = completed_results[path]
                scene_id = result["scene_id"]
                vehicle_id = result["vehicle_id"]
                output_path = vehicle_json_dir / (
                    f"{scene_id}__vehicle_{vehicle_id}_raw_local_map.json"
                )
                output_path.write_text(
                    json.dumps(result["output"], ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                output = result["output"]
                description = output["description"]
                jsonl_handle.write(
                    json.dumps(
                        {
                            "scene_id": scene_id,
                            "vehicle_id": vehicle_id,
                            "status": result["status"],
                            "max_contiguous_valid_frames": result[
                                "max_contiguous_valid_frames"
                            ],
                            "skip_reason": result["skip_reason"],
                            "input_json": result["source_path"],
                            "json_path": str(output_path),
                            "description": description,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                status = result["status"]
                if status == "success":
                    success += 1
                elif status.startswith("skipped_"):
                    skipped += 1
                csv_rows.append({
                    "scene_id": scene_id,
                    "vehicle_id": vehicle_id,
                    "status": status,
                    "max_contiguous_valid_frames": result[
                        "max_contiguous_valid_frames"
                    ],
                    "valid_frame_count": len(
                        result["payload"]["timeline"]["valid_frame_indices"]
                    ),
                    "local_lane_count": len(
                        result["payload"]["local_map_geometry"]["lanes"]
                    ),
                    "llm_called": result["llm_called"],
                    "description_short": description.get("description_short", ""),
                    "description_detailed": description.get("description_detailed", ""),
                    "behavior_segments": json.dumps(
                        description.get("behavior_segments", []),
                        ensure_ascii=False,
                    ),
                    "uncertainty_notes": description.get("uncertainty_notes", ""),
                    "json_path": str(output_path),
                    "error_type": "",
                    "error": "",
                    "skip_reason": result["skip_reason"],
                })
            except Exception as exc:
                failed += 1
                metadata = input_metadata[path]
                jsonl_handle.write(
                    json.dumps({
                        "status": "failed",
                        "scene_id": metadata["scene_id"],
                        "vehicle_id": metadata["vehicle_id"],
                        "input_json": str(path),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }, ensure_ascii=False)
                    + "\n"
                )
                csv_rows.append({
                    "scene_id": metadata["scene_id"],
                    "vehicle_id": metadata["vehicle_id"],
                    "status": "failed",
                    "max_contiguous_valid_frames": metadata[
                        "max_contiguous_valid_frames"
                    ],
                    "valid_frame_count": metadata["valid_frame_count"],
                    "local_lane_count": metadata["local_lane_count"],
                    "llm_called": False,
                    "description_short": "",
                    "description_detailed": "",
                    "behavior_segments": "",
                    "uncertainty_notes": "",
                    "json_path": str(path),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "skip_reason": "",
                })

    with csv_path.open("w", encoding="utf-8-sig", newline="") as csv_handle:
        writer = csv.DictWriter(csv_handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(csv_rows)
    summary = {
        "mode": "preextracted_json_batch",
        "input_json_count": len(input_paths),
        "success": success,
        "skipped": skipped,
        "failed": failed,
        "llm_called": any(bool(row["llm_called"]) for row in csv_rows),
        "min_contiguous_valid_frames": int(args.min_contiguous_valid_frames),
        "jsonl_output": str(jsonl_path),
        "csv_output": str(csv_path),
        "vehicle_json_dir": str(vehicle_json_dir),
    }
    (output_dir / "batch_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 1 if failed else 0


def _resolve_scene_ids(args: argparse.Namespace) -> list[str]:
    scene_ids = [str(value).strip() for value in (args.scene_id or [])]
    if args.scene_ids_file:
        if scene_ids:
            raise ValueError("use either --scene-id or --scene-ids-file, not both")
        scene_ids = [
            line.strip()
            for line in Path(args.scene_ids_file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    scene_ids = list(dict.fromkeys(scene_ids))
    if not scene_ids:
        raise ValueError("provide at least one --scene-id or --scene-ids-file")
    return scene_ids


def _run_scene_batch(
    args: argparse.Namespace,
    scenario: Any,
    source_file: str,
    record_index: int,
    output_dir: Path,
    jsonl_output: str | None = None,
    csv_output: str | None = None,
) -> dict[str, Any]:
    """Run the existing all-vehicle batch flow for one scene."""
    output_dir.mkdir(parents=True, exist_ok=True)
    vehicle_json_dir = output_dir / VEHICLE_JSON_DIRNAME
    vehicle_json_dir.mkdir(parents=True, exist_ok=True)
    lane_map = facts_builder.build_lane_map(scenario)
    vehicle_tracks = sorted(
        (
            track
            for track in scenario.tracks
            if int(track.object_type) == 1
        ),
        key=lambda track: int(track.id),
    )
    if args.limit is not None:
        vehicle_tracks = vehicle_tracks[: int(args.limit)]

    records: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    jsonl_path = Path(jsonl_output) if jsonl_output else (
        output_dir / "raw_vehicle_local_map.jsonl"
    )
    csv_path = Path(csv_output) if csv_output else (
        output_dir / "raw_vehicle_local_map_summary.csv"
    )
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    futures: dict[int, Any] = {}
    max_valid_runs: dict[int, int] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for track in vehicle_tracks:
            vehicle_id = int(track.id)
            normalized_track = facts_builder.extract_agent_track(scenario, vehicle_id)
            max_contiguous_valid_frames = _max_contiguous_valid_run(normalized_track)
            max_valid_runs[vehicle_id] = max_contiguous_valid_frames
            futures[vehicle_id] = executor.submit(
                _build_vehicle_result,
                scenario,
                track,
                lane_map,
                args.radius_m,
                args.max_map_lanes,
                args.max_map_points_per_lane,
                source_file,
                record_index,
                args.prompt_file,
                bool(
                    args.call_llm
                    and max_contiguous_valid_frames
                    >= args.min_contiguous_valid_frames
                ),
                args.model,
                args.base_url,
                args.api_key_env,
                args.max_tokens,
            )

    with jsonl_path.open("w", encoding="utf-8") as jsonl_handle:
        # Consume futures in vehicle-ID order so JSONL/CSV remain stable even
        # though LLM requests finish at different times.
        for vehicle_id in sorted(futures):
            output_path = vehicle_json_dir / (
                f"{scenario.scenario_id}__vehicle_{vehicle_id}"
                "_raw_local_map.json"
            )
            try:
                result = futures[vehicle_id].result()
                payload = result["payload"]
                vehicle_output = result["output"]
                max_contiguous_valid_frames = max_valid_runs[vehicle_id]
                is_skipped = (
                    max_contiguous_valid_frames < args.min_contiguous_valid_frames
                )
                status = (
                    "skipped_insufficient_contiguous_valid_frames"
                    if is_skipped
                    else "success"
                )
                skip_reason = (
                    f"max_contiguous_valid_frames={max_contiguous_valid_frames} "
                    f"< min_contiguous_valid_frames={args.min_contiguous_valid_frames}"
                    if is_skipped
                    else ""
                )
                output_path.write_text(
                    json.dumps(vehicle_output, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                jsonl_handle.write(
                    json.dumps(
                        {
                            "description": vehicle_output["description"],
                            "status": status,
                            "max_contiguous_valid_frames": max_contiguous_valid_frames,
                            "skip_reason": skip_reason,
                            "input": vehicle_output["input"],
                            "llm_prompt": vehicle_output["llm_prompt"],
                            **({"llm": vehicle_output["llm"]} if "llm" in vehicle_output else {}),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                record = {
                    "vehicle_id": vehicle_id,
                    "max_contiguous_valid_frames": max_contiguous_valid_frames,
                    "valid_frame_count": len(payload["timeline"]["valid_frame_indices"]),
                    "local_lane_count": len(payload["local_map_geometry"]["lanes"]),
                    "output": str(output_path),
                }
                if is_skipped:
                    skipped.append({**record, "skip_reason": skip_reason})
                else:
                    records.append(record)
                description = vehicle_output["description"]
                csv_rows.append({
                    "scene_id": str(scenario.scenario_id),
                    "vehicle_id": vehicle_id,
                    "status": status,
                    "max_contiguous_valid_frames": max_contiguous_valid_frames,
                    "valid_frame_count": len(payload["timeline"]["valid_frame_indices"]),
                    "local_lane_count": len(payload["local_map_geometry"]["lanes"]),
                    "llm_called": bool(args.call_llm and not is_skipped),
                    "description_short": description.get("description_short", ""),
                    "description_detailed": description.get("description_detailed", ""),
                    "behavior_segments": json.dumps(
                        description.get("behavior_segments", []),
                        ensure_ascii=False,
                    ),
                    "uncertainty_notes": description.get("uncertainty_notes", ""),
                    "json_path": str(output_path),
                    "error_type": "",
                    "error": "",
                    "skip_reason": skip_reason,
                })
            except Exception as exc:
                error_record = {
                    "status": "failed",
                    "scene_id": str(scenario.scenario_id),
                    "vehicle_id": vehicle_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                jsonl_handle.write(
                    json.dumps(error_record, ensure_ascii=False) + "\n"
                )
                failed.append({
                    "vehicle_id": vehicle_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                })
                csv_rows.append({
                    "scene_id": str(scenario.scenario_id),
                    "vehicle_id": vehicle_id,
                    "status": "failed",
                    "max_contiguous_valid_frames": "",
                    "valid_frame_count": "",
                    "local_lane_count": "",
                    "llm_called": False,
                    "description_short": "",
                    "description_detailed": "",
                    "behavior_segments": "",
                    "uncertainty_notes": "",
                    "json_path": "",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "skip_reason": "",
                })

    with csv_path.open("w", encoding="utf-8-sig", newline="") as csv_handle:
        writer = csv.DictWriter(csv_handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(csv_rows)

    summary = {
        "scene_id": str(scenario.scenario_id),
        "requested_vehicle_count": len(vehicle_tracks),
        "success": len(records),
        "skipped": len(skipped),
        "failed": len(failed),
        "min_contiguous_valid_frames": int(args.min_contiguous_valid_frames),
        "llm_called": any(bool(row["llm_called"]) for row in csv_rows),
        "jsonl_output": str(jsonl_path),
        "csv_output": str(csv_path),
        "records": records,
        "skipped_records": skipped,
        "failed_records": failed,
    }
    (output_dir / "batch_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def _aggregate_scene_batches(
    scene_summaries: list[dict[str, Any]],
    output_dir: Path,
    jsonl_output: str | None,
    csv_output: str | None,
) -> dict[str, Any]:
    """Combine per-scene JSONL/CSV artifacts into stable aggregate files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = Path(jsonl_output) if jsonl_output else (
        output_dir / "all_scenes_raw_vehicle_local_map.jsonl"
    )
    csv_path = Path(csv_output) if csv_output else (
        output_dir / "all_scenes_raw_vehicle_local_map_summary.csv"
    )
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with jsonl_path.open("w", encoding="utf-8") as aggregate_jsonl:
        for summary in scene_summaries:
            with Path(summary["jsonl_output"]).open(encoding="utf-8") as scene_jsonl:
                for line in scene_jsonl:
                    aggregate_jsonl.write(line)

    with csv_path.open("w", encoding="utf-8-sig", newline="") as aggregate_csv:
        writer = csv.DictWriter(aggregate_csv, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for summary in scene_summaries:
            with Path(summary["csv_output"]).open(
                encoding="utf-8-sig", newline=""
            ) as scene_csv:
                for row in csv.DictReader(scene_csv):
                    writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})

    aggregate = {
        "scene_count": len(scene_summaries),
        "success": sum(int(summary["success"]) for summary in scene_summaries),
        "skipped": sum(int(summary.get("skipped", 0)) for summary in scene_summaries),
        "failed": sum(int(summary["failed"]) for summary in scene_summaries),
        "llm_called": any(bool(summary["llm_called"]) for summary in scene_summaries),
        "jsonl_output": str(jsonl_path),
        "csv_output": str(csv_path),
        "scenes": scene_summaries,
    }
    (output_dir / "multi_scene_summary.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return aggregate


def _build_record_batch(
    paths: list[str],
    record_start: int,
    record_end: int | None,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    """Process a contiguous range of scenes in one TFRecord pass."""
    if len(paths) != 1:
        raise ValueError(
            "record-index batch mode requires --tfrecord to resolve to exactly one shard"
        )
    if record_start < 0 or (record_end is not None and record_end <= record_start):
        raise ValueError("record range must satisfy 0 <= record-start < record-end")

    output_dir.mkdir(parents=True, exist_ok=True)
    scene_output_dir = output_dir / "scenes"
    scene_output_dir.mkdir(parents=True, exist_ok=True)
    scene_summaries: list[dict[str, Any]] = []
    failed_scenes: list[dict[str, Any]] = []
    dataset = facts_builder.tf.data.TFRecordDataset(
        paths[0], compression_type=args.compression_type
    )
    for record_index, serialized_record in enumerate(dataset):
        if record_end is not None and record_index >= record_end:
            break
        if record_index < record_start:
            continue
        try:
            scenario = facts_builder._parse_scenario(bytes(serialized_record.numpy()))
            scene_output = scene_output_dir / str(scenario.scenario_id)
            summary = _run_scene_batch(
                args,
                scenario,
                paths[0],
                int(record_index),
                scene_output,
            )
            summary["record_index_in_shard"] = int(record_index)
            (scene_output / "batch_summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            scene_summaries.append(summary)
            print(json.dumps(summary, ensure_ascii=False), flush=True)
        except Exception as exc:
            failure = {
                "record_index_in_shard": int(record_index),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            failed_scenes.append(failure)
            print(json.dumps({"failed_scene": failure}, ensure_ascii=False), flush=True)

    aggregate = _aggregate_scene_batches(
        scene_summaries,
        output_dir,
        args.jsonl_output,
        args.csv_output,
    )
    summary = {
        "mode": "record_batch",
        "source_file": paths[0],
        "record_start": int(record_start),
        "record_end": int(record_end) if record_end is not None else None,
        "requested_scene_count": (
            int(record_end - record_start)
            if record_end is not None
            else "all_remaining"
        ),
        "processed_scene_count": len(scene_summaries),
        "failed_scene_count": len(failed_scenes),
        "vehicle_records": int(
            aggregate["success"] + aggregate["skipped"] + aggregate["failed"]
        ),
        "successful_vehicle_records": int(aggregate["success"]),
        "skipped_vehicle_records": int(aggregate["skipped"]),
        "failed_vehicle_records": int(aggregate["failed"]),
        "all_scenes_jsonl": aggregate["jsonl_output"],
        "all_scenes_csv": aggregate["csv_output"],
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
    parser.add_argument(
        "--tfrecord",
        help="Source TFRecord; not needed when using --input-json-dir/--input-json",
    )
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--input-json-dir",
        help=(
            "Read previously extracted vehicle JSON files from this directory; "
            "this mode does not read TFRecord"
        ),
    )
    input_group.add_argument(
        "--input-json",
        action="append",
        help="Previously extracted vehicle JSON path; repeat for multiple files",
    )
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
        help="0-based record index in one TFRecord shard",
    )
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
    vehicle_group = parser.add_mutually_exclusive_group()
    vehicle_group.add_argument("--vehicle-id", type=int)
    vehicle_group.add_argument(
        "--all-vehicles",
        action="store_true",
        help="Process every vehicle track in the scene",
    )
    parser.add_argument("--output", help="Single-vehicle experiment JSON path")
    parser.add_argument(
        "--output-dir",
        help="Batch output directory; one JSON file is written per vehicle",
    )
    parser.add_argument(
        "--jsonl-output",
        help=(
            "Batch JSONL aggregation path; defaults to "
            "<output-dir>/raw_vehicle_local_map.jsonl"
        ),
    )
    parser.add_argument(
        "--csv-output",
        help=(
            "Batch CSV summary path; defaults to "
            "<output-dir>/raw_vehicle_local_map_summary.csv"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional maximum number of vehicles in batch mode",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Concurrent workers for batch mode (default: 4)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=(
            "Maximum output tokens per LLM request; if the first request is "
            "truncated, one retry uses up to twice this budget (default: 2048)"
        ),
    )
    parser.add_argument(
        "--min-contiguous-valid-frames",
        type=int,
        default=DEFAULT_MIN_CONTIGUOUS_VALID_FRAMES,
        help=(
            "Do not generate a first-round description when the longest "
            "contiguous valid run is shorter than this (default: 20)"
        ),
    )
    parser.add_argument("--compression-type", default="")
    parser.add_argument("--radius-m", type=float, default=DEFAULT_LOCAL_MAP_RADIUS_M)
    parser.add_argument("--max-map-lanes", type=int, default=DEFAULT_MAX_MAP_LANES)
    parser.add_argument(
        "--max-map-points-per-lane",
        type=int,
        default=DEFAULT_MAX_MAP_POINTS_PER_LANE,
    )
    parser.add_argument("--call-llm", action="store_true")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    parser.add_argument(
        "--prompt-file",
        default=str(DEFAULT_PROMPT_PATH),
        help="Prompt template used for the raw-trajectory experiment",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.input_json_dir or args.input_json:
        if args.tfrecord or args.scene_id or args.scene_ids_file:
            raise ValueError(
                "pre-extracted JSON mode cannot be combined with --tfrecord or scene selection"
            )
        if args.record_index is not None or args.record_start is not None or args.record_end is not None:
            raise ValueError("pre-extracted JSON mode cannot be combined with record selection")
        if args.all_scenes or args.all_vehicles or args.vehicle_id is not None:
            raise ValueError(
                "pre-extracted JSON mode selects vehicles from the input JSON files"
            )
        if not args.output_dir or args.output:
            raise ValueError("pre-extracted JSON mode requires --output-dir and not --output")
        if args.limit is not None:
            raise ValueError("--limit is only available for TFRecord batch mode")
        if args.min_contiguous_valid_frames <= 0:
            raise ValueError("--min-contiguous-valid-frames must be positive")
        if args.workers <= 0:
            raise ValueError("--workers must be positive")
        if args.max_tokens <= 0:
            raise ValueError("--max-tokens must be positive")
        if args.radius_m <= 0 or args.max_map_lanes <= 0 or args.max_map_points_per_lane <= 1:
            raise ValueError("map locality and sampling limits must be positive")
        return _run_preextracted_batch(args)
    if not args.tfrecord:
        raise ValueError("TFRecord mode requires --tfrecord")
    global facts_builder
    facts_builder = _get_facts_builder()
    scene_ids = _resolve_scene_ids(args) if args.scene_id or args.scene_ids_file else []
    has_range = args.record_start is not None or args.record_end is not None
    has_single_record = args.record_index is not None
    has_single_scene = bool(scene_ids)
    if has_range and (has_single_record or has_single_scene):
        raise ValueError(
            "record range cannot be combined with --scene-id, --scene-ids-file, or --record-index"
        )
    if has_range:
        if args.record_start is None:
            raise ValueError("batch mode requires --record-start")
        if args.record_end is None and not args.all_scenes:
            raise ValueError("batch mode requires --record-end or --all-scenes")
        if args.record_end is not None and args.all_scenes:
            raise ValueError("use either --record-end or --all-scenes, not both")
        if args.vehicle_id is not None:
            raise ValueError("record batch mode processes all vehicles; omit --vehicle-id")
        args.all_vehicles = True
    elif not has_single_record and not has_single_scene:
        raise ValueError(
            "provide --scene-id, --scene-ids-file, --record-index, or a batch record range"
        )
    elif has_single_record and has_single_scene:
        raise ValueError("use either --record-index or --scene-id/--scene-ids-file")
    if args.all_scenes and not has_range:
        raise ValueError("--all-scenes requires batch record mode")
    if args.radius_m <= 0 or args.max_map_lanes <= 0 or args.max_map_points_per_lane <= 1:
        raise ValueError("map locality and sampling limits must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive")
    if args.min_contiguous_valid_frames <= 0:
        raise ValueError("--min-contiguous-valid-frames must be positive")
    if has_range:
        if args.output or not args.output_dir:
            raise ValueError("record batch mode requires --output-dir and not --output")
    elif args.all_vehicles:
        if not args.output_dir or args.output:
            raise ValueError("batch mode requires --output-dir and not --output")
    elif len(scene_ids) > 1:
        if not args.all_vehicles:
            raise ValueError("multiple scenes require --all-vehicles")
    elif args.vehicle_id is None and not args.all_vehicles:
        raise ValueError("vehicle mode requires --vehicle-id or --all-vehicles")
    elif not args.output or args.output_dir or args.limit is not None:
        raise ValueError("single-vehicle mode requires --output and does not use --output-dir/--limit")
    if not has_range and args.all_vehicles and not args.output_dir:
        raise ValueError("all-vehicles mode requires --output-dir")
    if not has_range and not args.all_vehicles and len(scene_ids) != 1:
        raise ValueError("single-vehicle mode accepts exactly one scene")
    if not has_range and not args.all_vehicles and args.jsonl_output:
        raise ValueError("--jsonl-output is only available in batch mode")
    if not has_range and not args.all_vehicles and args.csv_output:
        raise ValueError("--csv-output is only available in batch mode")

    facts_builder._require_runtime_deps()
    paths = facts_builder.resolve_tfrecord_paths(args.tfrecord)

    if has_range:
        summary = _build_record_batch(
            paths,
            int(args.record_start),
            int(args.record_end) if args.record_end is not None else None,
            args,
            Path(args.output_dir),
        )
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        return 1 if summary["failed_scene_count"] else 0

    if args.all_vehicles:
        output_dir = Path(args.output_dir)
        scene_summaries: list[dict[str, Any]] = []
        for scene_id in scene_ids:
            scenario, source_file, record_index = facts_builder.load_scenario_by_id(
                paths,
                scene_id,
                args.compression_type,
            )
            # Keep the one-scene layout backward compatible.  With multiple
            # scenes, isolate each scene's JSON files and per-scene manifests.
            scene_output_dir = output_dir if len(scene_ids) == 1 else output_dir / scene_id
            summary = _run_scene_batch(
                args,
                scenario,
                source_file,
                record_index,
                scene_output_dir,
                args.jsonl_output if len(scene_ids) == 1 else None,
                args.csv_output if len(scene_ids) == 1 else None,
            )
            scene_summaries.append(summary)
            print(json.dumps(summary, ensure_ascii=False))

        if len(scene_summaries) == 1:
            return 1 if scene_summaries[0]["failed"] else 0
        aggregate = _aggregate_scene_batches(
            scene_summaries,
            output_dir,
            args.jsonl_output,
            args.csv_output,
        )
        print(json.dumps(aggregate, ensure_ascii=False))
        return 1 if aggregate["failed"] else 0

    if has_single_record:
        scenario, source_file, record_index = facts_builder.load_scenario_by_record_index(
            paths,
            int(args.record_index),
            args.compression_type,
        )
    else:
        scenario, source_file, record_index = facts_builder.load_scenario_by_id(
            paths,
            scene_ids[0],
            args.compression_type,
        )
    lane_map = facts_builder.build_lane_map(scenario)
    single_track = facts_builder.extract_agent_track(scenario, int(args.vehicle_id))
    max_contiguous_valid_frames = _max_contiguous_valid_run(single_track)
    insufficient_contiguous_validity = (
        max_contiguous_valid_frames < args.min_contiguous_valid_frames
    )

    payload = build_experiment_input(
        scenario,
        int(args.vehicle_id),
        args.radius_m,
        args.max_map_lanes,
        args.max_map_points_per_lane,
        lane_map=lane_map,
        source_file=source_file,
        record_index=record_index,
    )
    output: dict[str, Any] = {
        "status": (
            "skipped_insufficient_contiguous_valid_frames"
            if insufficient_contiguous_validity
            else "success"
        ),
        "max_contiguous_valid_frames": max_contiguous_valid_frames,
        "description": {
            "description_short": "",
            "description_detailed": "",
            "behavior_segments": [],
            "uncertainty_notes": "",
        },
        "input": payload,
        "llm_prompt": build_llm_prompt(payload, args.prompt_file),
    }
    if args.call_llm and not insufficient_contiguous_validity:
        response = call_llm(
            output["llm_prompt"],
            args.model,
            args.base_url,
            args.api_key_env,
            args.max_tokens,
        )
        output["description"] = _description_from_response(
            response,
            int(payload["timeline"]["num_frames"]),
        )
        output["llm"] = {
            "model": args.model,
            "base_url": args.base_url,
            "response": response,
        }
        output["status"] = "success"
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "scene_id": payload["scene_id"],
        "vehicle_id": payload["vehicle_id"],
        "valid_frame_count": len(payload["timeline"]["valid_frame_indices"]),
        "max_contiguous_valid_frames": max_contiguous_valid_frames,
        "local_lane_count": len(payload["local_map_geometry"]["lanes"]),
        "llm_called": bool(args.call_llm and not insufficient_contiguous_validity),
        "status": output["status"],
        "output": str(output_path),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
