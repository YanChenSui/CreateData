"""Batch-export every interaction row as an independent scene_motion_v3 JSON."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Dict, List

import pandas as pd
from trajdata import MapAPI

try:
    from ..semantic.behavior_analysis import BehaviorConfig, TTCConfig
    from .export_scene_json import export_scene_v3
    from .map_matching import MapMatchingConfig
except ImportError:  # pragma: no cover - supports direct script execution.
    from scripts.archive.semantic.behavior_analysis import BehaviorConfig, TTCConfig
    from scripts.archive.export.export_scene_json import export_scene_v3
    from scripts.archive.export.map_matching import MapMatchingConfig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export all interaction CSV rows as classified scene_motion_v3 JSON files."
    )
    parser.add_argument(
        "--interaction-csv",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--cache-root",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--context-before", type=int, default=30)
    parser.add_argument("--context-after", type=int, default=30)
    parser.add_argument(
        "--target-frames",
        type=int,
        help=(
            "target total exported frames per interaction; when set, "
            "context-before/after are computed per row. Rows whose interaction "
            "interval is longer than the target are recorded as failures."
        ),
    )
    parser.add_argument(
        "--all-frames",
        action="store_true",
        help="export all source frames for each interaction participant",
    )
    parser.add_argument("--map-distance-threshold", type=float, default=3.0)
    parser.add_argument("--map-heading-threshold-deg", type=float, default=22.5)
    parser.add_argument("--map-top-k", type=int, default=5)
    parser.add_argument("--behavior-min-stable-frames", type=int, default=3)
    parser.add_argument("--behavior-min-lane-match-confidence", type=float, default=0.5)
    parser.add_argument("--ttc-epsilon-m", type=float, default=1e-3)
    parser.add_argument("--ttc-closing-speed-epsilon-mps", type=float, default=1e-3)
    parser.add_argument("--ttc-default-safety-radius-m", type=float, default=2.0)
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--end-row", type=int)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="number of concurrent row-export workers (default: 1)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="keep an existing row JSON instead of regenerating it",
    )
    args = parser.parse_args()

    if args.target_frames is not None and args.target_frames <= 0:
        parser.error("--target-frames must be positive")
    if args.target_frames is not None and args.all_frames:
        parser.error("--target-frames and --all-frames cannot be combined")
    if args.context_before < 0 or args.context_after < 0:
        parser.error("--context-before and --context-after must be non-negative")
    if args.workers <= 0:
        parser.error("--workers must be positive")

    records = pd.read_csv(args.interaction_csv)
    start_row = max(0, args.start_row)
    end_row = len(records) if args.end_row is None else min(args.end_row, len(records))
    if start_row > end_row:
        parser.error("--start-row must be <= --end-row")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    map_api = MapAPI(args.cache_root)
    map_cache: Dict[str, object] = {}
    map_config = MapMatchingConfig(
        distance_threshold_m=args.map_distance_threshold,
        heading_threshold_rad=args.map_heading_threshold_deg * 3.141592653589793 / 180.0,
        top_k=max(1, args.map_top_k),
    )
    behavior_config = BehaviorConfig(
        min_stable_frames=max(1, args.behavior_min_stable_frames),
        min_lane_match_confidence=max(
            0.0, min(1.0, args.behavior_min_lane_match_confidence)
        ),
    )
    ttc_config = TTCConfig(
        epsilon_m=max(0.0, args.ttc_epsilon_m),
        closing_speed_epsilon_mps=max(0.0, args.ttc_closing_speed_epsilon_mps),
        default_safety_radius_m=max(0.0, args.ttc_default_safety_radius_m),
    )

    total = end_row - start_row
    success_count = 0
    skipped_count = 0
    failed_rows: List[Dict[str, object]] = []
    map_lock = Lock()

    def process_row(
        position: int, row_index: int, row: pd.Series
    ) -> tuple[int, str, Dict[str, object] | None]:
        dataset = str(row.get("dataset", "unknown"))
        scenario_value = row.get("scenario_idx", "unknown")
        track_id = str(row.get("track_id", ""))
        scene_index = scenario_value
        output_path = None
        try:
            scene_index = int(scenario_value)
            interaction_start = int(row["start"])
            interaction_end = int(row["end"])
            if interaction_end < interaction_start:
                raise ValueError(
                    f"invalid interaction interval: {interaction_start}-{interaction_end}"
                )
            interaction_frames = interaction_end - interaction_start + 1
            if args.target_frames is not None:
                if interaction_frames > args.target_frames:
                    raise ValueError(
                        "interaction interval has "
                        f"{interaction_frames} frames, exceeding target "
                        f"{args.target_frames}"
                    )
                remaining_frames = args.target_frames - interaction_frames
                context_before = remaining_frames // 2
                context_after = remaining_frames - context_before
            else:
                context_before = args.context_before
                context_after = args.context_after
            output_path = args.output_dir / (
                f"row_{row_index:03d}_scene_{scene_index}_v3_classified.json"
            )
            if args.skip_existing and output_path.exists():
                print(f"[{position}/{total}] skip {output_path.name}", flush=True)
                return position, "skipped", None

            scene_dir = args.cache_root / dataset / f"scene_{scene_index}"
            if not scene_dir.is_dir():
                raise FileNotFoundError(f"scene cache does not exist: {scene_dir}")

            map_id = f"{dataset}:{dataset}_{scene_index}"
            # MapAPI/map-cache initialization is shared and protected. The
            # expensive row export itself runs concurrently after this point.
            with map_lock:
                if map_id not in map_cache:
                    map_cache[map_id] = map_api.get_map(map_id)
                vector_map = map_cache[map_id]

            if args.all_frames:
                window_label = "window=all-source-frames"
            elif args.target_frames is not None:
                window_label = f"target_frames={args.target_frames}"
            else:
                window_label = f"context={context_before}+{context_after}"
            print(
                f"[{position}/{total}] row={row_index} scene={scene_index} "
                f"track_id={track_id} interaction_frames={interaction_frames} "
                f"{window_label}",
                flush=True,
            )
            payload = export_scene_v3(
                scene_dir=scene_dir,
                output_path=output_path,
                dt=args.dt,
                interaction_csv=args.interaction_csv,
                interaction_row=row_index,
                context_before=context_before,
                context_after=context_after,
                target_frames=args.target_frames,
                all_frames=args.all_frames,
                vector_map=vector_map,
                map_matching_config=map_config,
                behavior_config=behavior_config,
                ttc_config=ttc_config,
            )
            temporal = payload.get("temporal", {})
            print(
                f"[{position}/{total}] done row={row_index} "
                f"export={temporal.get('export_start_frame')}-"
                f"{temporal.get('export_end_frame')} "
                f"frames={temporal.get('num_export_frames')}",
                flush=True,
            )
            return position, "success", None
        except Exception as exc:
            failed_record = {
                "row_index": row_index,
                "dataset": dataset,
                "scenario_idx": scenario_value,
                "track_id": track_id,
                "output_path": str(output_path) if output_path is not None else "",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            print(
                f"[ERROR] row={row_index} scene={scenario_value}: {exc}",
                flush=True,
            )
            return position, "failed", failed_record

    def consume_result(result: tuple[int, str, Dict[str, object] | None]) -> None:
        nonlocal success_count, skipped_count
        _, status, failed_record = result
        if status == "success":
            success_count += 1
        elif status == "skipped":
            skipped_count += 1
        elif failed_record is not None:
            failed_rows.append(failed_record)

    row_items = [
        (position, row_index, records.iloc[row_index].copy())
        for position, row_index in enumerate(range(start_row, end_row), start=1)
    ]
    if args.workers == 1:
        for position, row_index, row in row_items:
            consume_result(process_row(position, row_index, row))
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process_row, position, row_index, row): position
                for position, row_index, row in row_items
            }
            for future in as_completed(futures):
                consume_result(future.result())

    failed_rows_path = args.output_dir / "failed_rows.csv"
    failed_rows_columns = [
        "row_index",
        "dataset",
        "scenario_idx",
        "track_id",
        "output_path",
        "error_type",
        "error",
    ]
    failed_rows.sort(key=lambda item: int(item["row_index"]))
    pd.DataFrame(failed_rows, columns=failed_rows_columns).to_csv(
        failed_rows_path, index=False, encoding="utf-8-sig"
    )
    print(f"success: {success_count}")
    print(f"skipped: {skipped_count}")
    print(f"failed: {len(failed_rows)}")
    print(f"failed_rows_csv: {failed_rows_path}")
    print(f"output_dir: {args.output_dir}")


if __name__ == "__main__":
    main()
