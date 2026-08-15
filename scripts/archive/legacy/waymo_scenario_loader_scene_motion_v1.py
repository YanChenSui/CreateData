"""Load Waymo Motion Scenario protobuf records into timeline-compatible data.

Waymo Motion v1.3 data4 stores serialized ``scenario_pb2.Scenario`` messages
inside TFRecord records.  It is not a ``tf.train.Example``.  This module keeps
the rest of the pair-timeline code independent from that storage detail:

    raw TFRecord record -> Scenario protobuf -> normalized motion record

The normalized record intentionally contains observable states only.  Lane
ids are assigned by nearest lane-centerline matching from ``map_features``;
they are not read from an InterHub interaction window.
"""

from __future__ import annotations

import glob
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


def resolve_waymo_paths(spec: str) -> List[str]:
    """Resolve one Scenario TFRecord, a directory, or a glob."""
    path = Path(spec)
    if path.is_file():
        return [str(path)]
    if path.is_dir():
        matches = sorted(
            str(item)
            for item in path.iterdir()
            if item.is_file() and "tfrecord" in item.name
        )
    else:
        matches = sorted(glob.glob(spec))
    if not matches:
        raise FileNotFoundError(f"No Waymo TFRecord matched: {spec}")
    return matches


def _scenario_pb2():
    """Import Waymo protobuf lazily so JSON-only tests need no Waymo install."""
    try:
        from waymo_open_dataset.protos import scenario_pb2
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "Loading raw Scenario protobuf requires waymo_open_dataset. "
            "Install the Waymo protobuf package in the runtime environment."
        ) from exc
    return scenario_pb2


def _tf_record_dataset(path: str, compression_type: str = ""):
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "Reading Waymo TFRecord shards requires tensorflow in the runtime environment."
        ) from exc
    return tf.data.TFRecordDataset(path, compression_type=compression_type)


def _scenario_id(scenario: Any) -> str:
    value = getattr(scenario, "scenario_id", "")
    return str(value)


def load_waymo_scenario(
    spec: str,
    scene_id: Optional[str] = None,
    record_index: Optional[int] = None,
    compression_type: str = "",
) -> Tuple[Any, str]:
    """Load one raw ``scenario_pb2.Scenario`` from Waymo TFRecord data.

    ``scene_id`` searches the Scenario's own ``scenario_id`` field.  When
    ``record_index`` is supplied, exactly one resolved shard must be given and
    the index is interpreted as that shard's local record index.
    """
    paths = resolve_waymo_paths(spec)
    if scene_id is None and record_index is None:
        raise ValueError("Provide scene_id or record_index")
    if scene_id is not None and record_index is not None:
        raise ValueError("scene_id and record_index are mutually exclusive")
    if record_index is not None:
        if len(paths) != 1:
            raise ValueError("record_index requires exactly one TFRecord shard")
        if record_index < 0:
            raise ValueError("record_index must be >= 0")

    scenario_pb2 = _scenario_pb2()
    for path in paths:
        for index, record in enumerate(_tf_record_dataset(path, compression_type)):
            if record_index is not None and index != record_index:
                continue

            # Important: this is the Scenario protobuf loader.  Do not replace
            # this with tf.io.parse_single_example; data4 is not TFExample.
            scenario = scenario_pb2.Scenario()
            scenario.ParseFromString(record.numpy())

            if scene_id is None or _scenario_id(scenario) == str(scene_id):
                return scenario, path

        if record_index is not None:
            break

    if record_index is not None:
        raise IndexError(f"record index {record_index} not found in {paths[0]}")
    raise KeyError(f"Scenario id {scene_id!r} not found in {len(paths)} TFRecord file(s)")


def _point_xy(point: Any) -> Optional[np.ndarray]:
    x = getattr(point, "x", None)
    y = getattr(point, "y", None)
    try:
        x, y = float(x), float(y)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    return np.asarray([x, y], dtype=np.float64)


def _lane_polylines(scenario: Any) -> List[Tuple[str, np.ndarray]]:
    """Extract lane centerlines without depending on generated proto version."""
    lanes: List[Tuple[str, np.ndarray]] = []
    for feature in getattr(scenario, "map_features", []):
        lane = getattr(feature, "lane", None)
        if lane is None:
            continue
        points = []
        for point in getattr(lane, "polyline", []):
            xy = _point_xy(point)
            if xy is not None:
                points.append(xy)
        if len(points) >= 2:
            feature_id = str(getattr(feature, "id", ""))
            lanes.append((feature_id, np.asarray(points, dtype=np.float64)))
    return lanes


class _UnionFind:
    def __init__(self, items: Iterable[str]):
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, first: str, second: str) -> None:
        first_root, second_root = self.find(first), self.find(second)
        if first_root != second_root:
            self.parent[second_root] = first_root


def _lane_group_map(
    lanes: Sequence[Tuple[str, np.ndarray]],
    connect_distance: float = 2.5,
    heading_dot_threshold: float = 0.94,
    lateral_threshold: float = 1.75,
) -> Dict[str, str]:
    """Approximate physical lanes from adjacent centerline feature segments."""
    descriptors: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for lane_id, polyline in lanes:
        direction = polyline[1:] - polyline[:-1]
        lengths = np.linalg.norm(direction, axis=1)
        valid = lengths > 1e-6
        if not np.any(valid):
            continue
        direction = direction[valid] / lengths[valid, None]
        mean_direction = direction.mean(axis=0)
        norm = np.linalg.norm(mean_direction)
        if norm <= 1e-6:
            continue
        mean_direction /= norm
        descriptors[lane_id] = (polyline[0], polyline[-1], mean_direction)

    uf = _UnionFind(descriptors)
    for first_id, (first_start, first_end, first_direction) in descriptors.items():
        for second_id, (second_start, _second_end, second_direction) in descriptors.items():
            if first_id == second_id:
                continue
            delta = second_start - first_end
            if float(np.linalg.norm(delta)) > connect_distance:
                continue
            if float(np.dot(first_direction, second_direction)) < heading_dot_threshold:
                continue
            if abs(float(first_direction[0] * delta[1] - first_direction[1] * delta[0])) > lateral_threshold:
                continue
            if float(np.dot(delta, first_direction)) < -1.0:
                continue
            uf.union(first_id, second_id)

    roots = sorted({uf.find(lane_id) for lane_id in descriptors})
    root_to_group = {root: str(index) for index, root in enumerate(roots)}
    return {lane_id: root_to_group[uf.find(lane_id)] for lane_id in descriptors}


def _nearest_lane_id(
    xy: np.ndarray,
    heading: float,
    lanes: Sequence[Tuple[str, np.ndarray]],
    search_radius: float = 8.0,
    heading_dot_threshold: float = 0.0,
) -> Optional[str]:
    if not lanes or not np.isfinite(xy).all() or not math.isfinite(heading):
        return None
    heading_vec = np.asarray([math.cos(heading), math.sin(heading)])
    best: Optional[Tuple[float, str]] = None
    for lane_id, polyline in lanes:
        segments = polyline[1:] - polyline[:-1]
        lengths = np.linalg.norm(segments, axis=1)
        valid = lengths > 1e-6
        if not np.any(valid):
            continue
        segment_indices = np.flatnonzero(valid)
        directions = segments[valid] / lengths[valid, None]
        starts = polyline[:-1][valid]
        relative = xy[None, :] - starts
        projection = np.sum(relative * directions, axis=1)
        projection = np.clip(projection, 0.0, lengths[valid] - 1e-9)
        closest = starts + projection[:, None] * directions
        distances = np.linalg.norm(xy[None, :] - closest, axis=1)
        heading_dots = directions @ heading_vec
        compatible = heading_dots >= heading_dot_threshold
        if not np.any(compatible):
            continue
        local = int(np.argmin(np.where(compatible, distances, np.inf)))
        distance = float(distances[local])
        if distance > search_radius:
            continue
        # A small heading penalty avoids assigning a nearby opposite-direction
        # lane when two carriageways are close together.
        score = distance + 1.0 * (1.0 - float(heading_dots[local]))
        candidate = (score, lane_id)
        if best is None or candidate < best:
            best = candidate
    return best[1] if best is not None else None


def _state_value(state: Any, name: str, default: float = 0.0) -> float:
    value = getattr(state, name, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def scenario_to_motion_record(
    scenario: Any,
    lane_search_radius: float = 8.0,
) -> Dict[str, Any]:
    """Convert Scenario protobuf tracks into the existing timeline schema."""
    lanes = _lane_polylines(scenario)
    lane_groups = _lane_group_map(lanes)
    agents: List[Dict[str, Any]] = []
    sdc_index = getattr(scenario, "sdc_track_index", -1)
    tracks_to_predict = {
        int(index)
        for index in getattr(scenario, "tracks_to_predict", [])
    }

    for track_index, track in enumerate(getattr(scenario, "tracks", [])):
        agent_id = str(getattr(track, "id", ""))
        states: List[Dict[str, Any]] = []
        for frame, raw_state in enumerate(getattr(track, "states", [])):
            valid = bool(getattr(raw_state, "valid", False))
            x = _state_value(raw_state, "center_x", float("nan"))
            y = _state_value(raw_state, "center_y", float("nan"))
            z = _state_value(raw_state, "center_z", float("nan"))
            vx = _state_value(raw_state, "velocity_x", float("nan"))
            vy = _state_value(raw_state, "velocity_y", float("nan"))
            heading = _state_value(raw_state, "heading", float("nan"))
            matched_lane_id = (
                _nearest_lane_id(
                    np.asarray([x, y], dtype=np.float64),
                    heading,
                    lanes,
                    search_radius=lane_search_radius,
                )
                if valid
                else None
            )
            lane_id = lane_groups.get(matched_lane_id, matched_lane_id)
            speed = math.hypot(vx, vy) if math.isfinite(vx) and math.isfinite(vy) else None
            states.append(
                {
                    "frame": frame,
                    "valid": valid,
                    "position": {"x": x, "y": y, "z": z},
                    "velocity": {"x": vx, "y": vy, "speed": speed},
                    "heading_rad": heading,
                    "map": {"lane_id": lane_id},
                }
            )
        agents.append(
            {
                "agent_id": agent_id,
                "agent_type": int(getattr(track, "object_type", 0)),
                "is_sdc": track_index == sdc_index,
                "track_to_predict": track_index in tracks_to_predict,
                "states": states,
            }
        )

    return {
        "scene_id": _scenario_id(scenario),
        "source": {
            "format": "waymo_scenario_proto",
            "num_tracks": len(agents),
            "num_lane_features": len(lanes),
            "num_lane_groups": len(set(lane_groups.values())),
        },
        "agents": agents,
    }


def load_waymo_motion_record(
    spec: str,
    scene_id: Optional[str] = None,
    record_index: Optional[int] = None,
    compression_type: str = "",
    lane_search_radius: float = 8.0,
) -> Tuple[Dict[str, Any], str]:
    """Load and normalize one Scenario record for pair timeline analysis."""
    scenario, source_file = load_waymo_scenario(
        spec=spec,
        scene_id=scene_id,
        record_index=record_index,
        compression_type=compression_type,
    )
    return scenario_to_motion_record(scenario, lane_search_radius), source_file
