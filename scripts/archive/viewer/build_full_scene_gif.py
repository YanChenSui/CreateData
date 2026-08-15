#!/usr/bin/env python3
"""Render a complete-scene GIF from the vehicle review manifest."""
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
    return [
        state for state in track.get("states", [])
        if isinstance(state, Mapping) and state.get("valid")
    ]


def state_at(track: Mapping[str, Any], frame: int) -> Mapping[str, Any] | None:
    states = track.get("states", [])
    if frame < 0 or frame >= len(states):
        return None
    state = states[frame]
    return state if isinstance(state, Mapping) and state.get("valid") else None


def polygon(state: Mapping[str, Any], length: float = 4.5, width: float = 1.8) -> np.ndarray:
    heading = math.radians(float(state.get("heading_deg", 0.0)))
    rotation = np.asarray([
        [math.cos(heading), -math.sin(heading)],
        [math.sin(heading), math.cos(heading)],
    ])
    corners = np.asarray([
        [-length / 2, -width / 2],
        [length / 2, -width / 2],
        [length / 2, width / 2],
        [-length / 2, width / 2],
    ])
    return corners @ rotation.T + np.asarray([float(state["x"]), float(state["y"])])


def render_frame(
    manifest: Mapping[str, Any],
    tracks: list[Mapping[str, Any]],
    frame: int,
    bounds: tuple[float, float, float, float],
    output: Path,
    dpi: int,
) -> dict[str, int]:
    x_min, x_max, y_min, y_max = bounds
    stationary = 0
    moving = 0
    valid_count = 0
    figure, axis = plt.subplots(figsize=(10, 8), dpi=dpi)

    for lane in manifest.get("map_lanes", []):
        line = np.asarray(lane.get("polyline", []), dtype=float)
        if line.ndim == 2 and len(line) >= 2:
            axis.plot(line[:, 0], line[:, 1], color="#d4d4d4", linewidth=0.45, zorder=1)

    for track in tracks:
        states = valid_states(track)
        if len(states) >= 2:
            axis.plot(
                [float(s["x"]) for s in states],
                [float(s["y"]) for s in states],
                color="#e5e7eb", linewidth=0.55, alpha=0.65, zorder=2,
            )
        state = state_at(track, frame)
        if state is None:
            continue
        valid_count += 1
        speed = float(state.get("speed_mps", 0.0))
        if speed <= 0.1:
            stationary += 1
            color = "#f59e0b"
            text_color = "#92400e"
        else:
            moving += 1
            color = "#2563eb"
            text_color = "#1e3a8a"
        axis.add_patch(Polygon(polygon(state), closed=True, color=color, alpha=0.9, zorder=5))
        heading = math.radians(float(state.get("heading_deg", 0.0)))
        axis.arrow(
            float(state["x"]), float(state["y"]),
            3.0 * math.cos(heading), 3.0 * math.sin(heading),
            color=text_color, width=0.06, head_width=0.45,
            length_includes_head=True, zorder=6,
        )
        axis.text(
            float(state["x"]), float(state["y"]),
            f"{track.get('vehicle_id')} ({speed:.1f})",
            fontsize=5.5, color=text_color, fontweight="bold", zorder=7,
        )

    axis.set_xlim(x_min, x_max)
    axis.set_ylim(y_min, y_max)
    axis.set_aspect("equal")
    axis.set_axis_off()
    axis.set_title(
        f"Complete scene · Frame {frame}\n"
        f"Valid vehicles: {valid_count} · stationary ≤0.1 m/s: {stationary} · moving: {moving}\n"
        "Orange = stationary · Blue = moving · grey = context trajectory",
        fontsize=13,
    )
    figure.tight_layout(pad=0.5)
    figure.savefig(output, dpi=dpi)
    plt.close(figure)
    return {"valid": valid_count, "stationary": stationary, "moving": moving}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--dpi", type=int, default=85)
    args = parser.parse_args()

    manifest = read_manifest(args.manifest)
    first_item = next(
        item for item in manifest["items"] if isinstance(item, Mapping)
    )
    tracks_by_id: dict[int, Mapping[str, Any]] = {}
    target = first_item.get("target_track")
    if isinstance(target, Mapping):
        tracks_by_id[int(target["vehicle_id"])] = target
    for context in first_item.get("context_tracks", []):
        if isinstance(context, Mapping):
            tracks_by_id[int(context["vehicle_id"])] = context
    tracks = [tracks_by_id[key] for key in sorted(tracks_by_id)]
    if not tracks:
        raise ValueError("manifest contains no vehicle tracks")

    all_points = np.asarray([
        [float(state["x"]), float(state["y"])]
        for track in tracks
        for state in valid_states(track)
    ])
    x_low, y_low = np.min(all_points, axis=0)
    x_high, y_high = np.max(all_points, axis=0)
    margin = max(20.0, 0.08 * max(x_high - x_low, y_high - y_low))
    bounds = (x_low - margin, x_high + margin, y_low - margin, y_high + margin)
    frames = int(manifest.get("num_frames", 0))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame_dir = args.output.parent / "_full_scene_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    frame_paths: list[Path] = []
    summary: list[dict[str, int]] = []
    for frame in range(frames):
        frame_path = frame_dir / f"frame_{frame:04d}.png"
        summary.append(render_frame(manifest, tracks, frame, bounds, frame_path, args.dpi))
        frame_paths.append(frame_path)
    imageio.mimsave(args.output, [imageio.imread(path) for path in frame_paths], duration=1.0 / args.fps, loop=0)
    for path in frame_paths:
        path.unlink(missing_ok=True)
    try:
        frame_dir.rmdir()
    except OSError:
        pass
    print(json.dumps({
        "output": str(args.output),
        "vehicle_tracks": len(tracks),
        "frames": frames,
        "stationary_counts_by_frame": [row["stationary"] for row in summary],
        "valid_counts_by_frame": [row["valid"] for row in summary],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
