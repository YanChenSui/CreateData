"""Trajectory-to-lane map matching for the scene_motion_v3 exporter.

The matcher deliberately keeps map matching separate from InterHub's source
labels.  It uses centreline geometry for the emission cost and the lane graph
for the transition cost, then solves each continuous observed segment with a
Viterbi dynamic program.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


MATCH_METHOD = "position_heading_topology_viterbi"


@dataclass(frozen=True)
class MapMatchingConfig:
    """Thresholds and costs are intentionally explicit and reproducible."""

    distance_threshold_m: float = 3.0
    heading_threshold_rad: float = math.pi / 8.0
    top_k: int = 5
    distance_weight: float = 1.0
    heading_weight: float = 3.0
    same_lane_cost: float = 0.0
    connected_lane_cost: float = 0.25
    adjacent_lane_cost: float = 1.0
    unrelated_lane_cost: float = 20.0
    stability_max_frames: int = 2


@dataclass(frozen=True)
class LaneMatchCandidate:
    lane_id: str
    centerline_distance_m: float
    heading_error_rad: float
    lane_s_m: float
    lateral_offset_m: float
    emission_cost: float


def _lane_id(lane: Any) -> str:
    value = getattr(lane, "id", lane)
    return str(value)


def _lane_items(vector_map: Any) -> List[Any]:
    lanes = getattr(vector_map, "lanes", [])
    if isinstance(lanes, Mapping):
        return list(lanes.values())
    return list(lanes)


def _lane_index(vector_map: Any) -> Dict[str, Any]:
    return {_lane_id(lane): lane for lane in _lane_items(vector_map)}


def _as_xy_points(lane: Any) -> np.ndarray:
    center = getattr(lane, "center", lane)
    points = getattr(center, "points", None)
    if points is None:
        points = getattr(center, "xyz", None)
    if points is None:
        raise ValueError("lane centreline has neither points nor xyz")
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError("lane centreline must be an N x 2+ array")
    points = points[:, :2]
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    if len(points) < 2:
        raise ValueError("lane centreline must contain at least two finite points")
    keep = np.ones(len(points), dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-8
    return points[keep]


def _project_to_polyline(point_xy: np.ndarray, points: np.ndarray) -> Tuple[float, float, float, float]:
    """Return distance, arc length, signed lateral offset and tangent heading."""
    starts = points[:-1]
    vectors = points[1:] - starts
    lengths = np.linalg.norm(vectors, axis=1)
    valid = lengths > 1e-8
    if not np.any(valid):
        raise ValueError("lane centreline has no non-zero segment")

    safe_lengths_sq = np.where(valid, lengths * lengths, 1.0)
    t = np.sum((point_xy - starts) * vectors, axis=1) / safe_lengths_sq
    t = np.clip(t, 0.0, 1.0)
    projections = starts + t[:, None] * vectors
    distances = np.linalg.norm(projections - point_xy[None, :], axis=1)
    distances[~valid] = np.inf
    segment_index = int(np.argmin(distances))
    tangent = vectors[segment_index] / lengths[segment_index]
    projection = projections[segment_index]
    prefix = float(np.sum(lengths[:segment_index]))
    lane_s = prefix + float(t[segment_index] * lengths[segment_index])
    signed_offset = float(
        tangent[0] * (point_xy[1] - projection[1])
        - tangent[1] * (point_xy[0] - projection[0])
    )
    tangent_heading = math.atan2(float(tangent[1]), float(tangent[0]))
    return float(distances[segment_index]), lane_s, signed_offset, tangent_heading


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _candidate_lane_objects(
    vector_map: Any, x: float, y: float, z: float, search_distance_m: float
) -> Iterable[Any]:
    """Use trajdata's spatial index when available, with a safe full-map fallback."""
    get_lanes = getattr(vector_map, "get_lanes_within", None)
    if get_lanes is not None:
        try:
            candidates = get_lanes(
                np.asarray([x, y, z], dtype=float), search_distance_m
            )
            if candidates is not None and len(candidates) > 0:
                return candidates
        except (AttributeError, KeyError, TypeError, ValueError, IndexError):
            pass
    return _lane_items(vector_map)


def _candidate_for_lane(
    lane: Any,
    x: float,
    y: float,
    heading: float,
    config: MapMatchingConfig,
) -> Optional[LaneMatchCandidate]:
    try:
        points = _as_xy_points(lane)
        distance, lane_s, lateral_offset, tangent_heading = _project_to_polyline(
            np.asarray([x, y], dtype=float), points
        )
    except (ValueError, TypeError, IndexError, FloatingPointError):
        return None
    heading_error = abs(_wrap_angle(float(heading) - tangent_heading))
    if distance > config.distance_threshold_m or heading_error > config.heading_threshold_rad:
        return None
    emission = (
        config.distance_weight * distance
        + config.heading_weight * heading_error
    )
    return LaneMatchCandidate(
        lane_id=_lane_id(lane),
        centerline_distance_m=distance,
        heading_error_rad=heading_error,
        lane_s_m=lane_s,
        lateral_offset_m=lateral_offset,
        emission_cost=emission,
    )


def find_lane_candidates(
    vector_map: Any,
    x: float,
    y: float,
    heading: float,
    config: MapMatchingConfig = MapMatchingConfig(),
    z: float = 0.0,
) -> List[LaneMatchCandidate]:
    """Generate filtered Top-K candidates for one vehicle state."""
    unique: Dict[str, LaneMatchCandidate] = {}
    for lane in _candidate_lane_objects(
        vector_map, x, y, z, config.distance_threshold_m
    ):
        candidate = _candidate_for_lane(lane, x, y, heading, config)
        if candidate is None:
            continue
        previous = unique.get(candidate.lane_id)
        if previous is None or candidate.emission_cost < previous.emission_cost:
            unique[candidate.lane_id] = candidate
    return sorted(unique.values(), key=lambda item: item.emission_cost)[: max(1, config.top_k)]


def _id_set(lane: Any, attribute: str) -> set:
    values = getattr(lane, attribute, set()) or set()
    return {_lane_id(value) for value in values}


def _transition_cost(
    previous_id: str,
    current_id: str,
    lanes: Mapping[str, Any],
    config: MapMatchingConfig,
) -> float:
    if previous_id == current_id:
        return config.same_lane_cost
    previous = lanes.get(previous_id)
    current = lanes.get(current_id)
    if previous is None or current is None:
        return config.unrelated_lane_cost
    if (
        current_id in _id_set(previous, "next_lanes")
        or previous_id in _id_set(current, "prev_lanes")
    ):
        return config.connected_lane_cost
    if (
        current_id in _id_set(previous, "adj_lanes_left")
        or current_id in _id_set(previous, "adj_lanes_right")
        or previous_id in _id_set(current, "adj_lanes_left")
        or previous_id in _id_set(current, "adj_lanes_right")
    ):
        return config.adjacent_lane_cost
    return config.unrelated_lane_cost


def _unmatched() -> Dict[str, Any]:
    return {
        "lane_id": None,
        "lane_match_confidence": None,
        "match_method": MATCH_METHOD,
        "centerline_distance_m": None,
        "heading_error_rad": None,
        "lane_s_m": None,
        "lateral_offset_m": None,
        "match_status": "unmatched",
    }


def _candidate_record(candidate: LaneMatchCandidate, config: MapMatchingConfig) -> Dict[str, Any]:
    # This is a quality heuristic, not a calibrated probability.
    quality = math.exp(
        -0.5
        * (
            candidate.centerline_distance_m / max(config.distance_threshold_m, 1e-6)
            + candidate.heading_error_rad / max(config.heading_threshold_rad, 1e-6)
        )
    )
    return {
        "lane_id": candidate.lane_id,
        "lane_match_confidence": round(float(np.clip(quality, 0.0, 1.0)), 6),
        "match_method": MATCH_METHOD,
        "centerline_distance_m": round(candidate.centerline_distance_m, 6),
        "heading_error_rad": round(candidate.heading_error_rad, 6),
        "lane_s_m": round(candidate.lane_s_m, 6),
        "lateral_offset_m": round(candidate.lateral_offset_m, 6),
        "match_status": "matched",
    }


def _stabilize_short_excursions(
    selected: List[Optional[LaneMatchCandidate]],
    candidates_by_frame: Sequence[List[LaneMatchCandidate]],
    config: MapMatchingConfig,
) -> List[Optional[LaneMatchCandidate]]:
    """Suppress one/two-frame jumps only when the stable lane is also a candidate."""
    result = list(selected)
    index = 0
    while index < len(result):
        candidate = result[index]
        lane_id = candidate.lane_id if candidate is not None else None
        end = index + 1
        while end < len(result):
            next_id = result[end].lane_id if result[end] is not None else None
            if next_id != lane_id:
                break
            end += 1
        run_length = end - index
        if (
            lane_id is not None
            and run_length <= config.stability_max_frames
            and index > 0
            and end < len(result)
            and result[index - 1] is not None
            and result[end] is not None
            and result[index - 1].lane_id == result[end].lane_id
        ):
            stable_id = result[index - 1].lane_id
            for frame_index in range(index, end):
                replacement = next(
                    (item for item in candidates_by_frame[frame_index] if item.lane_id == stable_id),
                    None,
                )
                if replacement is not None:
                    result[frame_index] = replacement
        index = end
    return result


def _viterbi_segment(
    candidates_by_frame: Sequence[List[LaneMatchCandidate]],
    lanes: Mapping[str, Any],
    config: MapMatchingConfig,
) -> List[Optional[LaneMatchCandidate]]:
    if not candidates_by_frame:
        return []
    costs: List[List[float]] = [[candidate.emission_cost for candidate in frame] for frame in candidates_by_frame]
    backpointers: List[List[int]] = [[-1] * len(frame) for frame in candidates_by_frame]
    for frame_index in range(1, len(candidates_by_frame)):
        for current_index, current in enumerate(candidates_by_frame[frame_index]):
            options = []
            for previous_index, previous in enumerate(candidates_by_frame[frame_index - 1]):
                options.append(
                    costs[frame_index - 1][previous_index]
                    + _transition_cost(previous.lane_id, current.lane_id, lanes, config)
                )
            best_previous = int(np.argmin(options))
            costs[frame_index][current_index] += options[best_previous]
            backpointers[frame_index][current_index] = best_previous
    index = int(np.argmin(costs[-1]))
    selected: List[Optional[LaneMatchCandidate]] = [None] * len(candidates_by_frame)
    for frame_index in range(len(candidates_by_frame) - 1, -1, -1):
        selected[frame_index] = candidates_by_frame[frame_index][index]
        index = backpointers[frame_index][index]
        if index < 0:
            break
    return selected


def match_trajectory(
    vector_map: Any,
    observations: Sequence[Mapping[str, Any]],
    config: MapMatchingConfig = MapMatchingConfig(),
) -> List[Dict[str, Any]]:
    """Match one ordered trajectory; missing/invalid states remain unmatched."""
    results = [_unmatched() for _ in observations]
    candidates_by_index: List[List[LaneMatchCandidate]] = [[] for _ in observations]
    for index, observation in enumerate(observations):
        if not observation.get("valid", True):
            continue
        try:
            values = [observation[key] for key in ("x", "y", "heading")]
            if not all(math.isfinite(float(value)) for value in values):
                continue
            candidates_by_index[index] = find_lane_candidates(
                vector_map,
                float(observation["x"]),
                float(observation["y"]),
                float(observation["heading"]),
                config=config,
                z=float(observation.get("z", 0.0) or 0.0),
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            continue

    lane_index = _lane_index(vector_map)
    index = 0
    while index < len(observations):
        if not candidates_by_index[index]:
            index += 1
            continue
        end = index + 1
        while end < len(observations) and candidates_by_index[end]:
            end += 1
        selected = _viterbi_segment(candidates_by_index[index:end], lane_index, config)
        selected = _stabilize_short_excursions(
            selected, candidates_by_index[index:end], config
        )
        for offset, candidate in enumerate(selected, start=index):
            if candidate is not None:
                results[offset] = _candidate_record(candidate, config)
        index = end
    return results
