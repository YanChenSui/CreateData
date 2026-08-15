#!/usr/bin/env python3
"""Build vehicle review GIFs using InterHub's existing draw_pic renderer."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Mapping

import imageio.v2 as imageio
from trajdata import UnifiedDataset

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.visualize_utils import draw_pic


def read_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("items"), list):
        raise ValueError("vehicle review manifest must contain an items list")
    return value


def vehicle_ids(manifest: Mapping[str, Any]) -> list[str]:
    first_item = next(
        item for item in manifest["items"] if isinstance(item, Mapping)
    )
    values: set[str] = set()
    target = first_item.get("target_track")
    if isinstance(target, Mapping):
        values.add(str(target["vehicle_id"]))
    for context in first_item.get("context_tracks", []):
        if isinstance(context, Mapping):
            values.add(str(context["vehicle_id"]))
    return sorted(values, key=int)


def make_gif(frame_paths: list[Path], output: Path, fps: float) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(
        output,
        [imageio.imread(path) for path in frame_paths],
        duration=1.0 / fps,
        loop=0,
    )


def render_one(
    desired_scene: Any,
    dataset: Any,
    id_rawid: dict[int, int],
    raw_scene_index: int,
    all_agents: list[str],
    all_vehicle_ids: list[str],
    all_timesteps: list[int],
    dt: float,
    output_dir: Path,
    scene_id: str,
    target_id: str | None,
    dpi: int,
    fps: float,
) -> None:
    if target_id is None:
        output_stem = "full_scene"
        key_agents: list[str] = []
        title = "Complete vehicle scene"
    else:
        output_stem = f"{scene_id}__vehicle_{target_id}"
        key_agents = [target_id]
        title = f"Vehicle {target_id} highlighted"
    frame_dir = output_dir / "_interhub_frames" / output_stem
    frame_dir.mkdir(parents=True, exist_ok=True)
    frame_paths: list[Path] = []
    for timestamp in all_timesteps:
        draw_pic(
            desired_scene,
            all_agents,
            all_timesteps,
            dataset,
            id_rawid,
            raw_scene_index,
            "vehicle",
            all_vehicle_ids,
            key_agents,
            timestamp,
            0,
            len(all_timesteps) - 1,
            dt,
            5.0,
            str(frame_dir),
            ego_id=None,
            dpi=dpi,
            render_agent_ids=all_vehicle_ids,
            context_only=True,
            output_stem=output_stem,
            title_prefix=title,
        )
        frame_paths.append(frame_dir / f"{output_stem}_{timestamp}.png")
    make_gif(
        frame_paths,
        output_dir / f"{output_stem}.gif",
        fps,
    )
    shutil.rmtree(frame_dir, ignore_errors=True)
    print(json.dumps({
        "target_vehicle_id": target_id,
        "frames": len(frame_paths),
        "output": str(output_dir / f"{output_stem}.gif"),
    }, ensure_ascii=False))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--cache-location", required=True)
    parser.add_argument("--dataset", default="waymo_train")
    parser.add_argument("--raw-scene-index", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mode", choices=("full", "target", "both"), default="both")
    parser.add_argument("--vehicle-offset", type=int, default=0)
    parser.add_argument("--vehicle-stride", type=int, default=1)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--dpi", type=int, default=160)
    args = parser.parse_args()

    manifest = read_manifest(args.manifest)
    scene_id = str(manifest.get("scene_id", "unknown"))
    ids = vehicle_ids(manifest)
    if args.vehicle_stride <= 0 or args.vehicle_offset < 0:
        raise ValueError("vehicle-stride must be positive and vehicle-offset non-negative")
    ids = ids[args.vehicle_offset :: args.vehicle_stride]

    cache_location = Path(args.cache_location)
    if cache_location.name == args.dataset:
        cache_location = cache_location.parent
    dataset = UnifiedDataset(
        desired_data=[args.dataset],
        standardize_data=False,
        rebuild_cache=False,
        rebuild_maps=False,
        centric="scene",
        verbose=True,
        cache_location=str(cache_location),
        num_workers=1,
        incl_vector_map=True,
        data_dirs={args.dataset: ""},
    )
    id_rawid = {
        scene.raw_data_idx: index
        for index, scene in enumerate(dataset.scenes())
    }
    if args.raw_scene_index not in id_rawid:
        raise KeyError(f"raw scene index not found: {args.raw_scene_index}")
    desired_scene = dataset.get_scene(id_rawid[args.raw_scene_index])
    all_agents = [str(agent.name) for agent in desired_scene.agents]
    all_vehicle_ids = [value for value in vehicle_ids(manifest) if value in all_agents]
    all_timesteps = list(range(desired_scene.length_timesteps))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode in ("full", "both"):
        render_one(
            desired_scene, dataset, id_rawid, args.raw_scene_index,
            all_agents, all_vehicle_ids, all_timesteps, float(desired_scene.dt),
            args.output_dir, scene_id, None, args.dpi, args.fps,
        )
    if args.mode in ("target", "both"):
        for target_id in ids:
            if target_id not in all_agents:
                print(json.dumps({"target_vehicle_id": target_id, "skipped": "not in trajdata scene"}))
                continue
            render_one(
                desired_scene, dataset, id_rawid, args.raw_scene_index,
                all_agents, all_vehicle_ids, all_timesteps, float(desired_scene.dt),
                args.output_dir, scene_id, target_id, args.dpi, args.fps,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
