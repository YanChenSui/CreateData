"""Build pair facts directly from a Waymo Motion Scenario protobuf.

InterHub supplies only ``scene_id``, ``agent_A`` and ``agent_B``.  The
complete trajectories are loaded from the raw Waymo Scenario record and fed
to the facts-only timeline builder.  InterHub interaction windows are not
used to select or extend a lane-change event.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .pair_behavior_timeline import build_pair_behavior_timeline
    from .waymo_scenario_loader import load_waymo_motion_record
except ImportError:  # Support direct execution from the repository checkout.
    from pair_behavior_timeline import build_pair_behavior_timeline
    from waymo_scenario_loader import load_waymo_motion_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build facts-only before/during/after facts from Waymo Scenario protobuf data."
    )
    parser.add_argument("--tfrecord", required=True, help="Waymo Scenario TFRecord file, directory, or glob.")
    scene_group = parser.add_mutually_exclusive_group(required=True)
    scene_group.add_argument("--scene-id", help="Exact Scenario.scenario_id.")
    scene_group.add_argument("--record-index", type=int, help="0-based index inside one TFRecord shard.")
    parser.add_argument("--agent-a", required=True)
    parser.add_argument("--agent-b", required=True)
    parser.add_argument("--lane-search-radius", type=float, default=8.0)
    parser.add_argument("--compression-type", default="")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    record, source_file = load_waymo_motion_record(
        spec=args.tfrecord,
        scene_id=args.scene_id,
        record_index=args.record_index,
        compression_type=args.compression_type,
        lane_search_radius=args.lane_search_radius,
    )
    result = build_pair_behavior_timeline(record, args.agent_a, args.agent_b)
    result["source"]["tfrecord"] = source_file

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print(f"[OK] scene: {result['scene_id']}")
    print(f"[OK] pair: {args.agent_a} - {args.agent_b}")
    print(f"[OK] event status: {result['analysis_window']['status']}")
    print(f"[OK] output: {output_path}")
    print(json.dumps(result["pair_facts"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
