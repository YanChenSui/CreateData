#!/usr/bin/env python3
"""Build target-only vehicle review data for the InterHub Streamlit viewer.

The generated JSON is an audit/render artifact.  It is derived from the raw
Waymo scene and full_vehicle_facts/vehicle descriptions; it never changes the
fact JSONL or the source timeline.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    # Current server layout.
    from scripts.facts import build_pair_timeline as facts_builder
except ModuleNotFoundError:
    # Compatibility with the checked-out repository layout.
    from scripts.facts import build_pair_timeline as facts_builder


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sources = [path] if path.is_file() else sorted(path.rglob("*.jsonl"))
    if not sources:
        raise FileNotFoundError(f"No JSONL facts file found under: {path}")
    for source in sources:
        with source.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    value = json.loads(line)
                    if isinstance(value, dict):
                        rows.append(value)
    return rows


def read_description(directory: Path, scene_id: str, vehicle_id: int) -> dict[str, Any]:
    exact = directory / f"{scene_id}__vehicle_{vehicle_id}_vehicle_qwen_description.json"
    candidates = [exact] if exact.exists() else sorted(
        directory.rglob(f"*vehicle_{vehicle_id}_vehicle_qwen_description.json")
    )
    if not candidates:
        return {}
    try:
        value = json.loads(candidates[0].read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def safe_scene_filename(scene_id: str) -> str:
    """Return a scene id that is safe to use as a local filename component."""
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(scene_id)).strip("._")
    if not value:
        raise ValueError(f"Invalid empty scene id for output filename: {scene_id!r}")
    return value


def scene_asset_dir(asset_dir: str, scene_id: str, *, batch: bool) -> str:
    """Resolve either a scene-specific asset dir or a batch asset root."""
    if not asset_dir:
        return ""
    root = Path(asset_dir)
    if batch and (root / scene_id).is_dir():
        return (root / scene_id).as_posix()
    return asset_dir.rstrip("/")


def load_scenarios_by_id(
    paths: list[str],
    scene_ids: list[str],
    compression_type: str = "",
) -> dict[str, Any]:
    """Load a set of scenarios with one TFRecord pass per shard.

    The single-scene loader scans the TFRecord from the beginning for every
    scene.  Batch mode uses this helper so a shard is scanned only once.
    """
    targets = {str(scene_id) for scene_id in scene_ids}
    found: dict[str, Any] = {}
    for path in paths:
        dataset = facts_builder.tf.data.TFRecordDataset(
            path,
            compression_type=compression_type,
        )
        for record in dataset:
            scenario = facts_builder._parse_scenario(bytes(record.numpy()))
            scene_id = str(scenario.scenario_id)
            if scene_id in targets:
                found[scene_id] = scenario
                if len(found) == len(targets):
                    return found
    missing = sorted(targets.difference(found))
    raise KeyError(
        f"{len(missing)} scene_id(s) from full facts were not found in TFRecord: "
        f"{missing[:10]}"
    )


def track_payload(track: Any) -> list[dict[str, Any]]:
    return [
        {
            "frame": int(frame),
            "valid": bool(track.valid[frame]),
            "x": round(float(track.xy[frame, 0]), 4),
            "y": round(float(track.xy[frame, 1]), 4),
            "z": round(float(track.z[frame]), 4),
            "heading_deg": round(float(track.yaw[frame] * 180.0 / 3.141592653589793), 3),
            "speed_mps": round(float(track.speed[frame]), 4),
        }
        for frame in range(track.T)
    ]


def lane_payload(lane_timeline: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for match, smooth, stable in zip(
        lane_timeline.matches,
        lane_timeline.smooth_group,
        lane_timeline.stabilized_group,
    ):
        rows.append({
            "frame": int(match.frame),
            "available": bool(match.available),
            "best_lane_id": match.best_lane_id,
            "best_continuous_lane_id": match.best_group_id,
            "smoothed_continuous_lane_id": smooth,
            "stabilized_continuous_lane_id": stable,
            "confidence": match.confidence,
            "distance_m": match.distance_m,
        })
    return rows


def build_manifest(
    scenario: Any,
    facts_rows: list[dict[str, Any]],
    descriptions_dir: Path,
    asset_dir: str = "",
) -> dict[str, Any]:
    scene_id = str(scenario.scenario_id)
    facts_by_vehicle = {
        int(row["vehicle_id"]): row
        for row in facts_rows
        if str(row.get("scene_id", scene_id)) == scene_id and "vehicle_id" in row
    }
    vehicle_tracks = {
        int(track.id): facts_builder.extract_agent_track(scenario, int(track.id))
        for track in scenario.tracks
        if int(track.object_type) == 1
    }
    lane_map = facts_builder.build_lane_map(scenario)
    items: list[dict[str, Any]] = []
    for vehicle_id, facts in sorted(facts_by_vehicle.items()):
        track = vehicle_tracks.get(vehicle_id)
        if track is None:
            continue
        description = read_description(descriptions_dir, scene_id, vehicle_id)
        ranges = description.get("supporting_frame_ranges", [])
        if not isinstance(ranges, list):
            ranges = []
        valid = [int(i) for i, value in enumerate(track.valid) if bool(value)]
        if not ranges and valid:
            ranges = [{
                "start_frame": valid[0],
                "end_frame": valid[-1],
                "description": "No generated supporting range; review full observed track.",
            }]
        try:
            lane_timeline = facts_builder.build_lane_timeline(track, lane_map)
            lane_rows = lane_payload(lane_timeline)
        except Exception as exc:
            lane_rows = [{"error": f"{type(exc).__name__}: {exc}"}]

        context_tracks = [
            {
                "vehicle_id": int(other_id),
                "states": track_payload(other_track),
            }
            for other_id, other_track in sorted(vehicle_tracks.items())
            if other_id != vehicle_id
        ]
        for range_index, frame_range in enumerate(ranges):
            if not isinstance(frame_range, dict):
                continue
            try:
                start = int(frame_range["start_frame"])
                end = int(frame_range["end_frame"])
            except (KeyError, TypeError, ValueError):
                continue
            if start < 0 or start > end or end >= track.T:
                continue
            items.append({
                "review_id": f"{scene_id}::vehicle_{vehicle_id}::range_{range_index}",
                "scene_id": scene_id,
                "vehicle_id": vehicle_id,
                "gif_path": (
                    f"{asset_dir.rstrip('/')}/{scene_id}__vehicle_{vehicle_id}.gif"
                    if asset_dir else ""
                ),
                "frame_range": {
                    "start_frame": start,
                    "end_frame": end,
                    "description": str(frame_range.get("description", "")),
                },
                "description": {
                    "short": str(description.get("description_short", "")),
                    "detailed": str(description.get("description_detailed", "")),
                    "uncertainty_notes": str(description.get("uncertainty_notes", "")),
                },
                "target_track": {
                    "vehicle_id": vehicle_id,
                    "states": track_payload(track),
                },
                "context_tracks": context_tracks,
                "lane_matching": lane_rows,
                "lane_transition": facts.get("vehicle", {}).get("route_transition", {}),
                "physical_facts": facts.get("vehicle", {}),
            })
    return {
        "schema_version": "vehicle_review_manifest_v1",
        "scene_id": scene_id,
        "num_frames": len(scenario.timestamps_seconds),
        "asset_dir": asset_dir,
        "full_scene_gif": f"{asset_dir.rstrip('/')}/full_scene.gif" if asset_dir else "",
        "map_lanes": [
            {
                "lane_id": int(lane_id),
                "polyline": [
                    [round(float(point[0]), 3), round(float(point[1]), 3)]
                    for point in segment.polyline
                ],
            }
            for lane_id, segment in sorted(lane_map.segments.items())
            if len(segment.polyline) >= 2
        ],
        "items": items,
        "context_policy": "Other vehicles are rendered only as neutral visual context, never as semantic evidence.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tfrecord", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--scene-id")
    group.add_argument("--record-index", type=int)
    group.add_argument(
        "--all-scenes",
        action="store_true",
        help="Build one manifest for every scene_id present in --full-facts",
    )
    parser.add_argument(
        "--full-facts",
        required=True,
        type=Path,
        help="full_vehicle_facts JSONL file, or a directory containing JSONL files",
    )
    parser.add_argument("--descriptions-dir", required=True, type=Path)
    parser.add_argument(
        "--asset-dir",
        default="",
        help=(
            "Single-scene mode: scene-specific GIF directory. "
            "Batch mode: root containing one scene subdirectory per scene."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output JSON path in single-scene mode, or output directory in batch mode",
    )
    parser.add_argument("--compression-type", default="")
    args = parser.parse_args()

    facts_builder._require_runtime_deps()
    facts_rows = read_jsonl(args.full_facts)
    paths = facts_builder.resolve_tfrecord_paths(args.tfrecord)

    if args.all_scenes:
        scene_ids = sorted({
            str(row["scene_id"])
            for row in facts_rows
            if str(row.get("scene_id", "")).strip()
        })
        if not scene_ids:
            raise ValueError(f"No scene_id found in full facts: {args.full_facts}")

        scenarios = load_scenarios_by_id(paths, scene_ids, args.compression_type)
        args.output.mkdir(parents=True, exist_ok=True)
        outputs: list[dict[str, Any]] = []
        for scene_id in scene_ids:
            scenario = scenarios[scene_id]
            manifest = build_manifest(
                scenario,
                facts_rows,
                args.descriptions_dir,
                scene_asset_dir(args.asset_dir, scene_id, batch=True),
            )
            output_path = args.output / (
                f"{safe_scene_filename(scene_id)}_vehicle_review_manifest.json"
            )
            output_path.write_text(
                json.dumps(manifest, ensure_ascii=False, allow_nan=False),
                encoding="utf-8",
            )
            outputs.append({
                "scene_id": manifest["scene_id"],
                "review_items": len(manifest["items"]),
                "output": str(output_path),
            })
        print(json.dumps({
            "mode": "all-scenes",
            "scene_count": len(outputs),
            "results": outputs,
        }, ensure_ascii=False))
        return 0

    if args.scene_id:
        scenario, _, _ = facts_builder.load_scenario_by_id(paths, args.scene_id, args.compression_type)
    else:
        scenario, _, _ = facts_builder.load_scenario_by_record_index(paths, int(args.record_index), args.compression_type)
    manifest = build_manifest(
        scenario,
        facts_rows,
        args.descriptions_dir,
        args.asset_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps({"scene_id": manifest["scene_id"], "review_items": len(manifest["items"]), "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
