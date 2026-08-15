#!/usr/bin/env python3
"""Render one target-highlighted GIF per vehicle review record.

This is an audit-only renderer.  It consumes the target/context tracks and
lane polylines already exported in ``vehicle_review_manifest.json``; it does
not modify the facts or source timeline.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np


def read_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("items"), list):
        raise ValueError("vehicle review manifest must contain an items list")
    return value


def valid_states(track: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    states = track.get("states", [])
    return [state for state in states if isinstance(state, Mapping) and state.get("valid")]


def state_at(track: Mapping[str, Any], frame: int) -> Mapping[str, Any] | None:
    states = track.get("states", [])
    if frame < 0 or frame >= len(states):
        return None
    state = states[frame]
    return state if isinstance(state, Mapping) and state.get("valid") else None


def vehicle_polygon(state: Mapping[str, Any], length: float = 4.5, width: float = 1.8) -> np.ndarray:
    x = float(state["x"])
    y = float(state["y"])
    heading = math.radians(float(state.get("heading_deg", 0.0)))
    corners = np.asarray([
        [-length / 2, -width / 2],
        [length / 2, -width / 2],
        [length / 2, width / 2],
        [-length / 2, width / 2],
    ])
    rotation = np.asarray([
        [math.cos(heading), -math.sin(heading)],
        [math.sin(heading), math.cos(heading)],
    ])
    return corners @ rotation.T + np.asarray([x, y])


def render_frame(
    manifest: Mapping[str, Any],
    item: Mapping[str, Any],
    target_track: Mapping[str, Any],
    context_tracks: list[Mapping[str, Any]],
    visible_lanes: list[Mapping[str, Any]],
    frame: int,
    output_path: Path,
    dpi: int,
) -> None:
    target_states = valid_states(target_track)
    if not target_states:
        return
    selected = item.get("frame_range", {})
    start = int(selected.get("start_frame", 0))
    end = int(selected.get("end_frame", frame))
    target_xy = np.asarray([[float(s["x"]), float(s["y"])] for s in target_states])
    center = target_xy.mean(axis=0)
    span = max(32.0, float(np.ptp(target_xy[:, 0]) + 18.0), float(np.ptp(target_xy[:, 1]) + 18.0))
    x_min, y_min = center - span / 2.0
    x_max, y_max = center + span / 2.0

    figure, axis = plt.subplots(figsize=(6, 6), dpi=dpi)
    for lane in visible_lanes:
        polyline = np.asarray(lane.get("polyline", []), dtype=float)
        if polyline.ndim == 2 and len(polyline) >= 2:
            axis.plot(polyline[:, 0], polyline[:, 1], color="#c9c9c9", linewidth=0.55, zorder=1)

    for context in context_tracks:
        state = state_at(context, frame)
        if state is not None:
            axis.add_patch(Polygon(vehicle_polygon(state), closed=True, color="#c9caca", zorder=3))

    axis.plot(
        [float(s["x"]) for s in target_states],
        [float(s["y"]) for s in target_states],
        color="#2ca02c", linewidth=2.4, zorder=5, label=f"Vehicle {item.get('vehicle_id')}",
    )
    target_state = state_at(target_track, frame)
    if target_state is not None:
        axis.add_patch(Polygon(vehicle_polygon(target_state), closed=True, color="#2ca02c", zorder=8))
        heading = math.radians(float(target_state.get("heading_deg", 0.0)))
        axis.arrow(
            float(target_state["x"]), float(target_state["y"]),
            5.0 * math.cos(heading), 5.0 * math.sin(heading),
            color="#14532d", width=0.12, head_width=0.9,
            length_includes_head=True, zorder=9,
        )
        axis.text(
            float(target_state["x"]), float(target_state["y"]),
            f"  Vehicle {item.get('vehicle_id')}", color="#14532d", fontsize=9,
            fontweight="bold", zorder=10,
        )

    axis.set_xlim(x_min, x_max)
    axis.set_ylim(y_min, y_max)
    axis.set_aspect("equal")
    axis.set_axis_off()
    speed = target_state.get("speed_mps") if target_state else None
    speed_text = f"{float(speed):.1f} m/s" if speed is not None else "speed unavailable"
    axis.set_title(
        f"Vehicle {item.get('vehicle_id')} at Frame {frame} · {speed_text}\n"
        f"Supporting range {start}-{end} · other vehicles are context only",
        fontsize=12,
    )
    axis.legend(loc="upper left", framealpha=0.9, fontsize=9)
    figure.tight_layout(pad=0.5)
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--dpi", type=int, default=75)
    parser.add_argument("--max-vehicles", type=int)
    parser.add_argument("--vehicle-offset", type=int, default=0)
    parser.add_argument("--vehicle-stride", type=int, default=1)
    args = parser.parse_args()

    manifest = read_manifest(args.manifest)
    items_by_vehicle: dict[int, Mapping[str, Any]] = {}
    for item in manifest["items"]:
        if isinstance(item, Mapping) and "vehicle_id" in item:
            items_by_vehicle.setdefault(int(item["vehicle_id"]), item)
    vehicle_ids = sorted(items_by_vehicle)
    if args.max_vehicles is not None:
        vehicle_ids = vehicle_ids[: max(0, int(args.max_vehicles))]
    if args.vehicle_stride <= 0 or args.vehicle_offset < 0:
        raise ValueError("vehicle-stride must be positive and vehicle-offset non-negative")
    vehicle_ids = vehicle_ids[args.vehicle_offset :: args.vehicle_stride]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame_root = args.output_dir / "_frames"
    frame_root.mkdir(parents=True, exist_ok=True)
    scene_id = str(manifest.get("scene_id", "unknown"))
    for vehicle_id in vehicle_ids:
        item = items_by_vehicle[vehicle_id]
        target_track = item.get("target_track", {})
        context_tracks = item.get("context_tracks", [])
        if not isinstance(target_track, Mapping) or not isinstance(context_tracks, list):
            continue
        states = valid_states(target_track)
        if not states:
            continue
        target_xy = np.asarray([[float(s["x"]), float(s["y"])] for s in states])
        center = target_xy.mean(axis=0)
        span = max(32.0, float(np.ptp(target_xy[:, 0]) + 18.0), float(np.ptp(target_xy[:, 1]) + 18.0))
        x_min, y_min = center - span / 2.0
        x_max, y_max = center + span / 2.0
        visible_lanes = []
        for lane in manifest.get("map_lanes", []):
            polyline = np.asarray(lane.get("polyline", []), dtype=float)
            if polyline.ndim == 2 and len(polyline) >= 2 and np.any(
                (polyline[:, 0] >= x_min - 10.0)
                & (polyline[:, 0] <= x_max + 10.0)
                & (polyline[:, 1] >= y_min - 10.0)
                & (polyline[:, 1] <= y_max + 10.0)
            ):
                visible_lanes.append(lane)
        vehicle_frame_dir = frame_root / f"vehicle_{vehicle_id}"
        vehicle_frame_dir.mkdir(parents=True, exist_ok=True)
        frame_paths: list[Path] = []
        num_frames = int(manifest.get("num_frames", len(target_track.get("states", []))))
        for frame in range(num_frames):
            frame_path = vehicle_frame_dir / f"frame_{frame:04d}.png"
            render_frame(
                manifest, item, target_track, context_tracks, visible_lanes,
                frame, frame_path, args.dpi,
            )
            if frame_path.exists():
                frame_paths.append(frame_path)
        output_path = args.output_dir / f"{scene_id}__vehicle_{vehicle_id}.gif"
        images = [imageio.imread(path) for path in frame_paths]
        imageio.mimsave(output_path, images, duration=1.0 / args.fps, loop=0)
        print(json.dumps({"vehicle_id": vehicle_id, "frames": len(images), "output": str(output_path)}))
        for path in frame_paths:
            path.unlink(missing_ok=True)
        try:
            vehicle_frame_dir.rmdir()
        except OSError:
            pass
    try:
        frame_root.rmdir()
    except OSError:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
