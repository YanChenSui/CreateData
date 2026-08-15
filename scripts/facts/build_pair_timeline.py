#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_pair_timeline.py
===============================

Conservative physical-fact extraction for a selected pair in the Waymo Open
Motion Dataset (WOMD).

This revision intentionally stops before free-form behavior descriptions.  The
builder emits only conservative, evidence-backed maneuver structures such as
physical lane changes, confirmed turns, U-turn evidence, and main-road merge
evidence.  It does not emit cut-in / overtake / follow / yielding labels and
does not use an InterHub interaction window to crop or decide facts.

It answers only four questions from the full WOMD scenario:

1. Did either vehicle show a physically supported lateral maneuver?
2. Are the two vehicles in the same travel channel, a different channel, or is
   the relation uncertain?
3. Which vehicle is ahead / behind when a longitudinal comparison is valid?
4. Did either vehicle show a clear acceleration or deceleration episode?
5. What is the pair's heading relation, closest approach, and distance evolution?
6. How did each individual vehicle's heading evolve over its valid track?

Important policy
----------------
* Raw lane IDs are map evidence only.  The lane fact layer first normalizes
  matched lane segments through WOMD entry/exit topology; a lateral neighbor
  relation is promoted to a directional lane-change fact only when the
  trajectory also contains physical lane-relative lateral motion.
* Physical lateral motion is measured as sustained change in signed position
  relative to the local lane centerline while the vehicle remains in one
  stabilized physical lane group.  A map-segment switch or world-coordinate
  displacement is not, by itself, a lateral-motion fact.
* Same/different travel-channel facts combine conservative map matching with
  pair geometry.  Disagreement is serialized as ``uncertain`` rather than
  forced into a label.
* Ahead/behind is emitted only when the pair is approximately same-direction.
* Acceleration/deceleration comes only from the smoothed WOMD speed time series;
  no causal statement about the other vehicle is made.
* InterHub start/end are retained only as audit metadata.
* A main-road merge requires a directed lane continuation plus incoming-lane
  topology; it is not inferred from a raw lane-ID change or lateral motion.

The output is intended for small-batch visual validation before any semantic or
language-generation stage is re-enabled.
"""


from __future__ import annotations

import argparse
import glob
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Any

import numpy as np

try:
    import tensorflow as tf
except Exception:  # pragma: no cover - allows --help / syntax checks without TF
    tf = None

try:
    from waymo_open_dataset.protos import scenario_pb2
except Exception:  # pragma: no cover
    scenario_pb2 = None


# =============================================================================
# Configuration
# =============================================================================

# Motion state.
MIN_MOVE_SPEED = 0.50
DELTA_SPEED_THRESHOLD = 0.50
SPEED_RESPONSE_THRESHOLD = 0.80

# Pair geometry.
LONGITUDINAL_HYSTERESIS_M = 2.0
LATERAL_ALIGN_THRESHOLD_M = 2.0
SAME_DIRECTION_MAX_DEG = 45.0
PAIR_DISTANCE_RELEVANT_M = 30.0

# Lane matching.
LANE_SEARCH_RADIUS_M = 10.0
LANE_Z_TOLERANCE_M = 3.0
LANE_MIN_HEADING_DOT = 0.25
LANE_SCORE_EPS = 0.25
LANE_TOP_K = 3
LANE_CONFIDENCE_MIN = 0.30

# Segment-level map-matching denoising.  This is deliberately separate from
# continuous-lane dwell filtering: a short segment is removed only when its
# local topology and confidence indicate a matching flicker.
LANE_SEGMENT_FLICKER_MAX_FRAMES = 2
LANE_SEGMENT_FLICKER_CONFIDENCE_MARGIN = 0.10
LANE_SEGMENT_FLICKER_CONFIDENCE_RATIO = 0.80

# Temporal lane-state stabilization.
LANE_SMOOTH_RADIUS = 2
LANE_MIN_DWELL_FRAMES = 3

# Conservative continuous-lane union using official lane topology.
CONT_ENDPOINT_DIST_M = 8.0
CONT_ENDPOINT_LATERAL_M = 2.0
CONT_HEADING_DOT = 0.92
CONT_REQUIRE_RECIPROCAL = True
CONT_REQUIRE_UNAMBIGUOUS_TOPOLOGY = False  # v2 uses reciprocal mutual-best continuation

# Event windows.
DEFAULT_CONTEXT_FRAMES = 10
DEFAULT_EVENT_HALF_WIDTH = 1  # 3-frame event window, matching current tests
EVENT_SEARCH_EDGE_MARGIN = 2

# Event selection.
EVENT_MIN_SCORE = 2.50
EVENT_DEDUP_RADIUS = 2
INTERHUB_PRIOR_MAX_BONUS = 1.50
INTERHUB_PRIOR_DECAY_FRAMES = 10.0

# v3 event-channel policy.
# Each channel uses the same evidence threshold as v2, but event TYPE rather
# than mere threshold crossing determines whether it is pair-level.
CHANNEL_EVENT_MIN_SCORE = EVENT_MIN_SCORE

# InterHub scope categories. These affect audit/interpretation only; they do
# not hard-reject an event.
INTERHUB_SCOPE_NEAR_FRAMES = 10
INTERHUB_SCOPE_MODERATE_FRAMES = 20
INTERHUB_SCOPE_FAR_FRAMES = 40


# =============================================================================
# Conservative physical-fact policy (facts only; no behavior semantics)
# =============================================================================

# Local lane-relative lateral-motion confirmation.  A map lane/group switch is
# never enough by itself.
PHYS_LATERAL_MIN_MAP_CONFIDENCE = 0.30
PHYS_LATERAL_OFFSET_RATE_THRESHOLD_MPS = 0.25
PHYS_LATERAL_MIN_DURATION_FRAMES = 3
PHYS_LATERAL_MIN_CENTERLINE_OFFSET_CHANGE_M = 0.75
PHYS_LATERAL_MIN_SIGN_CONSISTENCY = 0.65
PHYS_LATERAL_SMOOTH_RADIUS = 1
PHYS_LATERAL_MAP_SWITCH_ASSOC_RADIUS = 8
PHYS_LATERAL_MIN_COVERAGE_FOR_NONE = 0.35

# Pair travel-channel relation.  The two thresholds deliberately leave a gray
# zone that becomes ``uncertain``.
PHYS_PAIR_SAME_DIRECTION_MAX_DEG = 45.0
PHYS_CHANNEL_SAME_LATERAL_M = 1.25
PHYS_CHANNEL_SAME_GROUP_MAX_LATERAL_M = 2.40
PHYS_CHANNEL_DIFFERENT_LATERAL_M = 2.80

# Orthogonal pair geometry.  These facts describe direction and distance
# geometry only; they do not assign interaction or behavior semantics.
PHYS_HEADING_ALIGNED_MAX_DEG = 10.0
PHYS_HEADING_ROUGHLY_ALIGNED_MAX_DEG = PHYS_PAIR_SAME_DIRECTION_MAX_DEG
PHYS_HEADING_OPPOSING_MIN_DEG = 135.0
PHYS_HEADING_MIN_COMMON_FRAMES = 2
PHYS_DISTANCE_EVOLUTION_WINDOW_FRAMES = 10
PHYS_DISTANCE_EVOLUTION_MIN_POINTS = 5
PHYS_DISTANCE_EVOLUTION_EDGE_POINTS = 3
PHYS_DISTANCE_EVOLUTION_DELTA_THRESHOLD_M = 0.30

# Route/path/map facts.  These thresholds are deliberately geometric and are
# not behavior labels.
PHYS_PATH_DOWNSAMPLE_STEP = 2
PHYS_PATH_MIN_POINTS = 2
PHYS_PATH_SPATIAL_OVERLAP_DISTANCE_M = 2.0
PHYS_PATH_EVOLUTION_EDGE_POINTS = 3
PHYS_PATH_EVOLUTION_DELTA_THRESHOLD_M = 0.30
PHYS_MAP_CONTEXT_NEAR_DISTANCE_M = 30.0
PHYS_MAP_INTERSECTION_DISTANCE_M = 15.0
PHYS_ROUTE_HEADING_WINDOW_FRAMES = 5

# Single-agent heading motion.  These thresholds identify sustained heading
# change in the track itself; they do not classify a turn or maneuver.
PHYS_HEADING_MOTION_SMOOTH_RADIUS = 2
PHYS_HEADING_EPISODE_MIN_RATE_DEG_PER_FRAME = 1.0
PHYS_HEADING_EPISODE_MIN_DURATION_FRAMES = 3
PHYS_HEADING_EPISODE_MIN_NET_CHANGE_DEG = 10.0

# Turn maneuver semantics require independent track and lane-path evidence.
# ``extract_heading_motion_facts`` remains a heading-evolution fact and keeps
# its looser 10-degree threshold; these gates are deliberately stricter.
PHYS_TURN_MIN_LANE_HEADING_CHANGE_DEG = 20.0
PHYS_TURN_MIN_TRACK_HEADING_CHANGE_DEG = 25.0
PHYS_TURN_DIRECTION_AGREEMENT = True
PHYS_TURN_MIN_ENTRY_EXIT_FRAMES = 5

# Trajectory quality is an evidence flag only.  It never repairs, deletes, or
# interpolates a Track state.  The thresholds are intentionally conservative:
# a flag should make downstream consumers cautious before speed, turn, or lane
# facts are interpreted.
PHYS_TRAJECTORY_QUALITY_HEADING_JUMP_DEG = 30.0
# A single frame-to-frame speed change in this range is retained as a warning
# rather than making the whole trajectory ineligible for caption generation.
PHYS_TRAJECTORY_QUALITY_SPEED_JUMP_WARNING_MPS2 = 8.0
PHYS_TRAJECTORY_QUALITY_SPEED_JUMP_MODERATE_MPS2 = 15.0
PHYS_TRAJECTORY_QUALITY_SPEED_JUMP_SEVERE_MPS2 = 30.0
PHYS_TRAJECTORY_QUALITY_SPEED_JUMP_REPEAT_WINDOW_FRAMES = 5
PHYS_TRAJECTORY_QUALITY_SPEED_JUMP_REPEAT_MIN_COUNT = 2
PHYS_TRAJECTORY_QUALITY_MAX_POSITION_VELOCITY_ERROR_M = 2.0
PHYS_TRAJECTORY_QUALITY_POSITION_VELOCITY_ERROR_RATIO = 0.75
PHYS_TRAJECTORY_QUALITY_DIRECTION_MISMATCH_DEG = 60.0
PHYS_TRAJECTORY_QUALITY_MIN_DIRECTION_SPEED_MPS = 0.5
PHYS_TRAJECTORY_QUALITY_LONG_GAP_FRAMES = 5

# U-turn evidence is deliberately stricter than an ordinary heading episode.
# It requires a large global reversal, sustained same-sign rotation, and a
# trajectory whose entry/exit directions visibly reverse.  These are physical
# evidence gates; they do not force a U-turn caption by themselves.
PHYS_U_TURN_MIN_GLOBAL_HEADING_CHANGE_DEG = 135.0
PHYS_U_TURN_MIN_ROTATION_FRAMES = 8
PHYS_U_TURN_MIN_ROTATION_CONSISTENCY = 0.75
PHYS_U_TURN_MIN_EDGE_DIRECTION_REVERSAL_DOT = -0.35
PHYS_U_TURN_MIN_PATH_TO_ENDPOINT_RATIO = 1.15
PHYS_U_TURN_EDGE_WINDOW_FRAMES = 5

# Partial U-turn evidence is intentionally a separate status.  It never
# weakens the complete-U-turn gate above: it captures a large, sustained
# heading change whose observation ends before a stable exit heading or full
# entry/exit reversal is visible.
PHYS_U_TURN_PARTIAL_MIN_GLOBAL_HEADING_CHANGE_DEG = 90.0
PHYS_U_TURN_PARTIAL_MIN_ROTATION_CONSISTENCY = 0.70
PHYS_U_TURN_PARTIAL_MIN_ROTATION_FRAMES = 8
PHYS_U_TURN_PARTIAL_MAX_EDGE_DIRECTION_DOT = 0.55
PHYS_U_TURN_PARTIAL_MIN_PATH_TO_ENDPOINT_RATIO = 1.05
PHYS_U_TURN_PARTIAL_TAIL_WINDOW_FRAMES = 12
PHYS_U_TURN_PARTIAL_STABLE_EXIT_MAX_CHANGE_DEG = 8.0
PHYS_U_TURN_PARTIAL_STABLE_EXIT_MAX_ACTIVE_FRACTION = 0.35

# Longitudinal relation.
PHYS_LONGITUDINAL_DEADBAND_M = 2.0

# Speed-change confirmation.
PHYS_SPEED_SMOOTH_RADIUS = 2
PHYS_ACCEL_THRESHOLD_MPS2 = 0.80
PHYS_SPEED_EVENT_MIN_DURATION_FRAMES = 5
PHYS_SPEED_EVENT_MERGE_GAP_FRAMES = 5
PHYS_ACCEL_MIN_SPEED_DELTA_MPS = 0.60
# This is only used while forming raw threshold-crossing episodes.  Semantic
# events use PHYS_SPEED_EVENT_MIN_DURATION_FRAMES and the consolidation pass
# below; keeping the raw bridge separate prevents a threshold blip from
# becoming a final behavior event by itself.
PHYS_ACCEL_MAX_BRIDGED_GAP_FRAMES = 1


# =============================================================================
# Small helpers
# =============================================================================

def _require_runtime_deps() -> None:
    missing = []
    if tf is None:
        missing.append("tensorflow")
    if scenario_pb2 is None:
        missing.append("waymo_open_dataset")
    if missing:
        raise ImportError(
            "Missing runtime dependencies: " + ", ".join(missing) + ". "
            "Run this script in the same environment used to read WOMD "
            "Scenario-protobuf TFRecords."
        )


def _normalize_angle(x: np.ndarray | float) -> np.ndarray | float:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def _heading_vec(yaw: np.ndarray | float) -> np.ndarray:
    return np.stack([np.cos(yaw), np.sin(yaw)], axis=-1)


def _cross2d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _safe_mean(x: np.ndarray) -> Optional[float]:
    if len(x) == 0:
        return None
    v = float(np.mean(x))
    return v if np.isfinite(v) else None


def _round(x: Optional[float], ndigits: int = 3) -> Optional[float]:
    if x is None:
        return None
    x = float(x)
    if not np.isfinite(x):
        return None
    return round(x, ndigits)


def _mode(values: Sequence[Optional[int]]) -> Optional[int]:
    clean = [int(v) for v in values if v is not None]
    if not clean:
        return None
    counts: Dict[int, int] = {}
    for v in clean:
        counts[v] = counts.get(v, 0) + 1
    return max(counts.items(), key=lambda kv: (kv[1], -kv[0]))[0]


def _clip_window(center: int, radius: int, total: int) -> List[int]:
    return list(range(max(0, center - radius), min(total, center + radius + 1)))


def _contiguous_runs(values: Sequence[Optional[int]]) -> List[Tuple[int, int, Optional[int]]]:
    """Return inclusive runs: (start, end, value)."""
    if not values:
        return []
    runs: List[Tuple[int, int, Optional[int]]] = []
    start = 0
    cur = values[0]
    for i in range(1, len(values)):
        if values[i] != cur:
            runs.append((start, i - 1, cur))
            start = i
            cur = values[i]
    runs.append((start, len(values) - 1, cur))
    return runs


def _jsonable(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays to JSON-safe Python objects."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


# =============================================================================
# WOMD Scenario loader
# =============================================================================

def resolve_tfrecord_paths(spec: str) -> List[str]:
    p = Path(spec)
    if p.is_file():
        return [str(p)]
    if p.is_dir():
        files = sorted(str(x) for x in p.iterdir() if x.is_file() and "tfrecord" in x.name)
        if not files:
            raise FileNotFoundError(f"No TFRecord files found under {spec}")
        return files
    matches = sorted(glob.glob(spec))
    if not matches:
        raise FileNotFoundError(f"No TFRecord matched: {spec}")
    return matches


def _parse_scenario(serialized: bytes):
    _require_runtime_deps()
    scenario = scenario_pb2.Scenario()
    scenario.ParseFromString(serialized)
    return scenario


def load_scenario_by_id(
    paths: Sequence[str],
    scenario_id: str,
    compression_type: str = "",
):
    target = str(scenario_id)
    for path in paths:
        ds = tf.data.TFRecordDataset(path, compression_type=compression_type)
        for record_index, rec in enumerate(ds):
            scenario = _parse_scenario(bytes(rec.numpy()))
            if scenario.scenario_id == target:
                return scenario, path, record_index
    raise KeyError(f"scenario_id={target!r} not found in {len(paths)} TFRecord file(s)")


def load_scenario_by_record_index(
    paths: Sequence[str],
    record_index: int,
    compression_type: str = "",
):
    if len(paths) != 1:
        raise ValueError("--record-index requires --tfrecord to resolve to exactly one shard")
    if record_index < 0:
        raise ValueError("--record-index must be >= 0")
    path = paths[0]
    ds = tf.data.TFRecordDataset(path, compression_type=compression_type)
    for i, rec in enumerate(ds):
        if i == record_index:
            return _parse_scenario(bytes(rec.numpy())), path, i
    raise IndexError(f"record index {record_index} not found in {path}")


# =============================================================================
# Agent tracks
# =============================================================================

@dataclass
class AgentTrack:
    agent_id: int
    track_index: int
    object_type: int
    is_sdc: bool
    is_object_of_interest: bool
    track_to_predict: bool
    timestamps: np.ndarray
    xy: np.ndarray
    z: np.ndarray
    velocity: np.ndarray
    speed: np.ndarray
    yaw: np.ndarray
    valid: np.ndarray

    @property
    def T(self) -> int:
        return int(len(self.valid))


def extract_trajectory_quality(track: AgentTrack) -> Dict[str, Any]:
    """Flag suspicious raw Track transitions without modifying the track.

    Quality checks are deliberately upstream of all motion semantics.  They
    inspect only adjacent valid states and never bridge invalid-frame gaps,
    interpolate values, or remove an observation.  Downstream builders decide
    whether a suspicious track is eligible for captions or needs review.
    """
    valid = np.asarray(track.valid, dtype=bool)
    xy = np.asarray(track.xy, dtype=np.float64)
    velocity = np.asarray(track.velocity, dtype=np.float64)
    yaw = np.asarray(track.yaw, dtype=np.float64)
    timestamps = np.asarray(track.timestamps, dtype=np.float64)
    valid_indices = np.flatnonzero(valid)
    issues: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    speed_jump_warnings: List[Dict[str, Any]] = []
    checked_pairs = 0

    for frame in valid_indices:
        frame = int(frame)
        finite = (
            np.all(np.isfinite(xy[frame]))
            and np.all(np.isfinite(velocity[frame]))
            and np.isfinite(yaw[frame])
        )
        if not finite:
            issues.append({
                "type": "nonfinite_state",
                "frame": frame,
            })

    for previous_frame, current_frame in zip(valid_indices[:-1], valid_indices[1:]):
        previous_frame = int(previous_frame)
        current_frame = int(current_frame)
        gap_frames = current_frame - previous_frame - 1
        if gap_frames > PHYS_TRAJECTORY_QUALITY_LONG_GAP_FRAMES:
            issues.append({
                "type": "long_validity_gap",
                "frame": current_frame,
                "gap_start_frame": previous_frame + 1,
                "gap_end_frame": current_frame - 1,
                "gap_frames": gap_frames,
            })

        # Never compare across a missing-state gap.  A jump across such a gap
        # is an observation-boundary issue, not evidence of a per-frame jump.
        if gap_frames != 0:
            continue
        dt = float(timestamps[current_frame] - timestamps[previous_frame])
        if dt <= 1e-6:
            continue
        if not (
            np.all(np.isfinite(xy[previous_frame]))
            and np.all(np.isfinite(xy[current_frame]))
            and np.all(np.isfinite(velocity[previous_frame]))
            and np.all(np.isfinite(velocity[current_frame]))
            and np.isfinite(yaw[previous_frame])
            and np.isfinite(yaw[current_frame])
        ):
            continue
        checked_pairs += 1

        heading_delta_deg = math.degrees(float(_normalize_angle(
            yaw[current_frame] - yaw[previous_frame]
        )))
        if abs(heading_delta_deg) >= PHYS_TRAJECTORY_QUALITY_HEADING_JUMP_DEG:
            issues.append({
                "type": "heading_jump",
                "frame": current_frame,
                "delta_deg": _round(heading_delta_deg, 3),
            })

        previous_speed = float(np.linalg.norm(velocity[previous_frame]))
        current_speed = float(np.linalg.norm(velocity[current_frame]))
        speed_delta = current_speed - previous_speed
        speed_change_rate = abs(speed_delta) / dt
        if speed_change_rate >= PHYS_TRAJECTORY_QUALITY_SPEED_JUMP_WARNING_MPS2:
            speed_jump = {
                "type": "velocity_jump",
                "frame": current_frame,
                "delta_speed_mps": _round(speed_delta, 3),
                "rate_mps2": _round(speed_change_rate, 3),
                "severity": (
                    "severe"
                    if speed_change_rate >= PHYS_TRAJECTORY_QUALITY_SPEED_JUMP_SEVERE_MPS2
                    else (
                        "moderate"
                        if speed_change_rate >= PHYS_TRAJECTORY_QUALITY_SPEED_JUMP_MODERATE_MPS2
                        else "warning"
                    )
                ),
            }
            speed_jump_warnings.append(speed_jump)

        displacement = xy[current_frame] - xy[previous_frame]
        displacement_m = float(np.linalg.norm(displacement))
        expected_displacement_m = 0.5 * (previous_speed + current_speed) * dt
        position_velocity_error_m = abs(
            displacement_m - expected_displacement_m
        )
        mismatch_limit = max(
            PHYS_TRAJECTORY_QUALITY_MAX_POSITION_VELOCITY_ERROR_M,
            PHYS_TRAJECTORY_QUALITY_POSITION_VELOCITY_ERROR_RATIO
            * expected_displacement_m,
        )
        if position_velocity_error_m >= mismatch_limit:
            issues.append({
                "type": "position_velocity_mismatch",
                "frame": current_frame,
                "observed_displacement_m": _round(displacement_m, 3),
                "expected_displacement_m": _round(expected_displacement_m, 3),
                "error_m": _round(position_velocity_error_m, 3),
            })

        average_velocity = 0.5 * (
            velocity[previous_frame] + velocity[current_frame]
        )
        average_speed = float(np.linalg.norm(average_velocity))
        if (
            displacement_m >= 0.2
            and average_speed >= PHYS_TRAJECTORY_QUALITY_MIN_DIRECTION_SPEED_MPS
        ):
            cosine = float(np.dot(displacement, average_velocity)) / (
                displacement_m * average_speed
            )
            cosine = float(np.clip(cosine, -1.0, 1.0))
            direction_delta_deg = math.degrees(math.acos(cosine))
            if direction_delta_deg >= PHYS_TRAJECTORY_QUALITY_DIRECTION_MISMATCH_DEG:
                issues.append({
                    "type": "position_velocity_direction_mismatch",
                    "frame": current_frame,
                    "delta_deg": _round(direction_delta_deg, 3),
                })

    # Speed jumps are warning-level by default.  Promote the trajectory to
    # suspicious when they are individually severe, repeated in a short
    # window, or accompanied by another quality issue.  Suspicious status is
    # an audit flag and does not block caption generation.
    warnings.extend(speed_jump_warnings)
    severe_speed_jumps = [
        item
        for item in speed_jump_warnings
        if float(item["rate_mps2"]) >= PHYS_TRAJECTORY_QUALITY_SPEED_JUMP_SEVERE_MPS2
    ]
    repeated_speed_jumps = (
        len(speed_jump_warnings)
        >= PHYS_TRAJECTORY_QUALITY_SPEED_JUMP_REPEAT_MIN_COUNT
        and any(
            right["frame"] - left["frame"]
            <= PHYS_TRAJECTORY_QUALITY_SPEED_JUMP_REPEAT_WINDOW_FRAMES
            for left, right in zip(speed_jump_warnings, speed_jump_warnings[1:])
        )
    )
    if severe_speed_jumps or repeated_speed_jumps or issues:
        issues.extend(speed_jump_warnings)
        warnings = []

    status = "suspicious" if issues else ("warning" if warnings else "clean")
    return {
        "status": status,
        "issues": issues,
        "warnings": warnings,
        "summary": {
            "valid_frame_count": int(len(valid_indices)),
            "checked_adjacent_pairs": int(checked_pairs),
        },
        "policy": {
            "heading_jump_threshold_deg": PHYS_TRAJECTORY_QUALITY_HEADING_JUMP_DEG,
            # Keep the legacy field as the warning threshold for consumers
            # that still read it, while exposing the tiered policy explicitly.
            "max_speed_change_mps2": PHYS_TRAJECTORY_QUALITY_SPEED_JUMP_WARNING_MPS2,
            "speed_jump_warning_mps2": PHYS_TRAJECTORY_QUALITY_SPEED_JUMP_WARNING_MPS2,
            "speed_jump_moderate_mps2": PHYS_TRAJECTORY_QUALITY_SPEED_JUMP_MODERATE_MPS2,
            "speed_jump_severe_mps2": PHYS_TRAJECTORY_QUALITY_SPEED_JUMP_SEVERE_MPS2,
            "speed_jump_repeat_window_frames": PHYS_TRAJECTORY_QUALITY_SPEED_JUMP_REPEAT_WINDOW_FRAMES,
            "speed_jump_repeat_min_count": PHYS_TRAJECTORY_QUALITY_SPEED_JUMP_REPEAT_MIN_COUNT,
            "max_position_velocity_error_m": PHYS_TRAJECTORY_QUALITY_MAX_POSITION_VELOCITY_ERROR_M,
            "position_velocity_error_ratio": PHYS_TRAJECTORY_QUALITY_POSITION_VELOCITY_ERROR_RATIO,
            "direction_mismatch_threshold_deg": PHYS_TRAJECTORY_QUALITY_DIRECTION_MISMATCH_DEG,
            "long_validity_gap_frames": PHYS_TRAJECTORY_QUALITY_LONG_GAP_FRAMES,
            "repairs_applied": False,
        },
    }


def extract_agent_track(scenario, agent_id: int) -> AgentTrack:
    matches = [(i, tr) for i, tr in enumerate(scenario.tracks) if int(tr.id) == int(agent_id)]
    if not matches:
        raise KeyError(f"Agent id {agent_id} not found in scenario {scenario.scenario_id}")
    if len(matches) > 1:
        raise RuntimeError(f"Duplicate agent id {agent_id}")

    idx, tr = matches[0]
    T = len(scenario.timestamps_seconds)
    if len(tr.states) != T:
        raise ValueError(
            f"Track {agent_id} has {len(tr.states)} states but scenario has {T} timestamps"
        )

    x = np.zeros(T, dtype=np.float64)
    y = np.zeros(T, dtype=np.float64)
    z = np.zeros(T, dtype=np.float64)
    vx = np.zeros(T, dtype=np.float64)
    vy = np.zeros(T, dtype=np.float64)
    yaw = np.zeros(T, dtype=np.float64)
    valid = np.zeros(T, dtype=bool)

    for t, s in enumerate(tr.states):
        valid[t] = bool(s.valid)
        x[t] = float(s.center_x)
        y[t] = float(s.center_y)
        z[t] = float(s.center_z)
        vx[t] = float(s.velocity_x)
        vy[t] = float(s.velocity_y)
        yaw[t] = float(s.heading)

    xy = np.stack([x, y], axis=-1)
    velocity = np.stack([vx, vy], axis=-1)
    speed = np.linalg.norm(velocity, axis=1)

    # These are descriptive metadata only.  Vehicle-level extraction must not
    # require OOI membership or use it as a selection gate.
    ooi_ids = set(int(x) for x in getattr(scenario, "objects_of_interest", []))
    ttp_indices = set(
        int(x.track_index) for x in getattr(scenario, "tracks_to_predict", [])
    )

    return AgentTrack(
        agent_id=int(agent_id),
        track_index=int(idx),
        object_type=int(tr.object_type),
        is_sdc=(int(idx) == int(scenario.sdc_track_index)),
        is_object_of_interest=(int(agent_id) in ooi_ids),
        track_to_predict=(int(idx) in ttp_indices),
        timestamps=np.asarray(scenario.timestamps_seconds, dtype=np.float64),
        xy=xy,
        z=z,
        velocity=velocity,
        speed=speed,
        yaw=yaw,
        valid=valid,
    )


# =============================================================================
# Lane map and continuous-lane grouping
# =============================================================================

@dataclass(frozen=True)
class LaneNeighborSegment:
    """A longitudinally bounded neighbor relation from WOMD map_features."""

    neighbor_id: int
    self_start_index: Optional[int] = None
    self_end_index: Optional[int] = None
    neighbor_start_index: Optional[int] = None
    neighbor_end_index: Optional[int] = None


@dataclass
class LaneSegment:
    lane_id: int
    lane_type: int
    polyline: np.ndarray          # [N,3]
    tangent_xy: np.ndarray        # [N,2]
    entry_lanes: List[int]
    exit_lanes: List[int]
    left_neighbors: List[int]
    right_neighbors: List[int]
    # Keep the old ID lists for compatibility, but retain the WOMD neighbor
    # ranges for topology queries at the vehicle's longitudinal position.
    left_neighbor_segments: List[LaneNeighborSegment] = field(default_factory=list)
    right_neighbor_segments: List[LaneNeighborSegment] = field(default_factory=list)

    @property
    def xy(self) -> np.ndarray:
        return self.polyline[:, :2]

    @property
    def z(self) -> np.ndarray:
        return self.polyline[:, 2]

    @property
    def start_xy(self) -> np.ndarray:
        return self.xy[0]

    @property
    def end_xy(self) -> np.ndarray:
        return self.xy[-1]

    @property
    def start_dir(self) -> np.ndarray:
        return self.tangent_xy[0]

    @property
    def end_dir(self) -> np.ndarray:
        return self.tangent_xy[-1]


class UnionFind:
    def __init__(self, items: Iterable[int]):
        self.parent = {int(x): int(x) for x in items}
        self.rank = {int(x): 0 for x in items}

    def find(self, x: int) -> int:
        x = int(x)
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


@dataclass
class LaneMap:
    segments: Dict[int, LaneSegment]
    segment_to_group: Dict[int, int]
    group_to_segments: Dict[int, List[int]]

    # Undirected adjacency retained for the existing event score.
    group_neighbors: Dict[int, List[int]]

    # Directed lateral topology retained for factual maneuver direction.
    # These come from Waymo lane.left_neighbors / lane.right_neighbors after
    # conservative continuous-lane grouping.
    group_left_neighbors: Dict[int, List[int]]
    group_right_neighbors: Dict[int, List[int]]

    continuation_edges: List[Tuple[int, int]]

    # Flattened points for fast matching.
    point_xy: np.ndarray = field(repr=False)
    point_z: np.ndarray = field(repr=False)
    point_dir: np.ndarray = field(repr=False)
    point_lane_id: np.ndarray = field(repr=False)


def _polyline_tangents(polyline_xyz: np.ndarray) -> np.ndarray:
    xy = polyline_xyz[:, :2]
    n = len(xy)
    if n == 1:
        return np.array([[1.0, 0.0]], dtype=np.float64)

    d = np.zeros_like(xy, dtype=np.float64)
    d[0] = xy[1] - xy[0]
    d[-1] = xy[-1] - xy[-2]
    if n > 2:
        d[1:-1] = xy[2:] - xy[:-2]

    norm = np.linalg.norm(d, axis=1, keepdims=True)
    good = norm[:, 0] > 1e-8
    d[good] /= norm[good]
    d[~good] = np.array([1.0, 0.0])
    return d


def _continuation_geometry_ok(a: LaneSegment, b: LaneSegment) -> bool:
    delta = b.start_xy - a.end_xy
    dist = float(np.linalg.norm(delta))
    if dist > CONT_ENDPOINT_DIST_M:
        return False

    dot = float(np.dot(a.end_dir, b.start_dir))
    if dot < CONT_HEADING_DOT:
        return False

    lateral = abs(float(_cross2d(a.end_dir, delta)))
    if lateral > CONT_ENDPOINT_LATERAL_M:
        return False

    # Allow small overlap/noise but reject a clearly backwards connection.
    longitudinal = float(np.dot(delta, a.end_dir))
    if longitudinal < -2.0:
        return False

    return True


def build_lane_map(scenario) -> LaneMap:
    segments: Dict[int, LaneSegment] = {}

    for feat in scenario.map_features:
        if feat.WhichOneof("feature_data") != "lane":
            continue
        lane = feat.lane
        if len(lane.polyline) == 0:
            continue

        poly = np.array(
            [[float(p.x), float(p.y), float(p.z)] for p in lane.polyline],
            dtype=np.float64,
        )
        tangents = _polyline_tangents(poly)

        def _neighbor_segments(neighbors) -> List[LaneNeighborSegment]:
            result: List[LaneNeighborSegment] = []
            for neighbor in neighbors:
                result.append(
                    LaneNeighborSegment(
                        neighbor_id=int(neighbor.feature_id),
                        self_start_index=(
                            int(neighbor.self_start_index)
                            if hasattr(neighbor, "self_start_index") else None
                        ),
                        self_end_index=(
                            int(neighbor.self_end_index)
                            if hasattr(neighbor, "self_end_index") else None
                        ),
                        neighbor_start_index=(
                            int(neighbor.neighbor_start_index)
                            if hasattr(neighbor, "neighbor_start_index") else None
                        ),
                        neighbor_end_index=(
                            int(neighbor.neighbor_end_index)
                            if hasattr(neighbor, "neighbor_end_index") else None
                        ),
                    )
                )
            return result

        left_neighbor_segments = _neighbor_segments(lane.left_neighbors)
        right_neighbor_segments = _neighbor_segments(lane.right_neighbors)
        segments[int(feat.id)] = LaneSegment(
            lane_id=int(feat.id),
            lane_type=int(lane.type),
            polyline=poly,
            tangent_xy=tangents,
            entry_lanes=[int(x) for x in lane.entry_lanes],
            exit_lanes=[int(x) for x in lane.exit_lanes],
            left_neighbors=[int(n.feature_id) for n in lane.left_neighbors],
            right_neighbors=[int(n.feature_id) for n in lane.right_neighbors],
            left_neighbor_segments=left_neighbor_segments,
            right_neighbor_segments=right_neighbor_segments,
        )

    if not segments:
        raise ValueError("Scenario contains no usable lane-center map features")

    uf = UnionFind(segments.keys())
    continuation_edges: List[Tuple[int, int]] = []

    # Build every reciprocal topology edge that is also geometrically plausible.
    # At real forks/merges there may be multiple entry/exit IDs. Instead of
    # rejecting all non-1:1 topology, select only mutual-best geometric
    # continuations. This preserves segment chains while avoiding aggressive
    # union across branches.
    continuation_candidates: List[Tuple[float, int, int]] = []

    for a_id, a in segments.items():
        for b_id in [x for x in a.exit_lanes if x in segments]:
            b = segments[b_id]
            if CONT_REQUIRE_RECIPROCAL and a_id not in b.entry_lanes:
                continue
            if not _continuation_geometry_ok(a, b):
                continue

            delta = b.start_xy - a.end_xy
            dist = float(np.linalg.norm(delta))
            lateral = abs(float(_cross2d(a.end_dir, delta)))
            heading_dot = float(np.dot(a.end_dir, b.start_dir))
            # Lower is better. Heading mismatch is expressed in meter-like
            # penalty units so a visibly straighter continuation wins.
            cost = dist + 2.0 * lateral + 8.0 * (1.0 - heading_dot)
            continuation_candidates.append((cost, a_id, b_id))

    best_out: Dict[int, Tuple[float, int]] = {}
    best_in: Dict[int, Tuple[float, int]] = {}
    for cost, a_id, b_id in continuation_candidates:
        if a_id not in best_out or cost < best_out[a_id][0]:
            best_out[a_id] = (cost, b_id)
        if b_id not in best_in or cost < best_in[b_id][0]:
            best_in[b_id] = (cost, a_id)

    for cost, a_id, b_id in continuation_candidates:
        if best_out.get(a_id, (None, None))[1] != b_id:
            continue
        if best_in.get(b_id, (None, None))[1] != a_id:
            continue
        uf.union(a_id, b_id)
        continuation_edges.append((a_id, b_id))

    # Stable group IDs: minimum segment id inside each union component.
    root_members: Dict[int, List[int]] = {}
    for lid in segments:
        root_members.setdefault(uf.find(lid), []).append(lid)

    root_to_group = {root: min(members) for root, members in root_members.items()}
    segment_to_group = {lid: root_to_group[uf.find(lid)] for lid in segments}

    group_to_segments: Dict[int, List[int]] = {}
    for lid, gid in segment_to_group.items():
        group_to_segments.setdefault(gid, []).append(lid)
    for gid in group_to_segments:
        group_to_segments[gid].sort()

    # Adjacent lanes remain distinct groups. Keep an undirected graph for the
    # existing event score, plus directed left/right graphs for factual
    # lane-change direction. Adjacent groups are NEVER unioned here.
    neigh: Dict[int, set] = {gid: set() for gid in group_to_segments}
    left_neigh: Dict[int, set] = {gid: set() for gid in group_to_segments}
    right_neigh: Dict[int, set] = {gid: set() for gid in group_to_segments}

    for lid, seg in segments.items():
        g = segment_to_group[lid]

        for nid in seg.left_neighbors:
            if nid not in segment_to_group:
                continue
            ng = segment_to_group[nid]
            if ng == g:
                continue
            left_neigh[g].add(ng)
            neigh[g].add(ng)
            neigh.setdefault(ng, set()).add(g)

        for nid in seg.right_neighbors:
            if nid not in segment_to_group:
                continue
            ng = segment_to_group[nid]
            if ng == g:
                continue
            right_neigh[g].add(ng)
            neigh[g].add(ng)
            neigh.setdefault(ng, set()).add(g)

    group_neighbors = {
        g: sorted(int(x) for x in s)
        for g, s in neigh.items()
    }
    group_left_neighbors = {
        g: sorted(int(x) for x in s)
        for g, s in left_neigh.items()
    }
    group_right_neighbors = {
        g: sorted(int(x) for x in s)
        for g, s in right_neigh.items()
    }

    point_xy = []
    point_z = []
    point_dir = []
    point_lane_id = []
    for lid, seg in segments.items():
        point_xy.append(seg.xy)
        point_z.append(seg.z)
        point_dir.append(seg.tangent_xy)
        point_lane_id.append(np.full(len(seg.xy), lid, dtype=np.int64))

    return LaneMap(
        segments=segments,
        segment_to_group=segment_to_group,
        group_to_segments=group_to_segments,
        group_neighbors=group_neighbors,
        group_left_neighbors=group_left_neighbors,
        group_right_neighbors=group_right_neighbors,
        continuation_edges=continuation_edges,
        point_xy=np.concatenate(point_xy, axis=0),
        point_z=np.concatenate(point_z, axis=0),
        point_dir=np.concatenate(point_dir, axis=0),
        point_lane_id=np.concatenate(point_lane_id, axis=0),
    )


# =============================================================================
# Per-frame lane matching and temporal stabilization
# =============================================================================

@dataclass
class LaneFrameMatch:
    frame: int
    available: bool
    best_lane_id: Optional[int]
    best_group_id: Optional[int]
    confidence: Optional[float]
    distance_m: Optional[float]
    top_candidates: List[Dict[str, Any]]


@dataclass
class LaneTimeline:
    matches: List[LaneFrameMatch]
    raw_group: List[Optional[int]]
    smooth_group: List[Optional[int]]
    stabilized_group: List[Optional[int]]


def match_lane_at_frame(
    track: AgentTrack,
    frame: int,
    lane_map: LaneMap,
) -> LaneFrameMatch:
    if frame < 0 or frame >= track.T or not track.valid[frame]:
        return LaneFrameMatch(frame, False, None, None, None, None, [])

    pos = track.xy[frame]
    z = track.z[frame]
    h = np.array([math.cos(track.yaw[frame]), math.sin(track.yaw[frame])], dtype=np.float64)

    delta = lane_map.point_xy - pos[None, :]
    dist2 = np.sum(delta * delta, axis=1)
    dist = np.sqrt(dist2)
    heading_dot = lane_map.point_dir @ h

    mask = (
        (dist <= LANE_SEARCH_RADIUS_M)
        & (np.abs(lane_map.point_z - z) <= LANE_Z_TOLERANCE_M)
        & (heading_dot >= LANE_MIN_HEADING_DOT)
    )
    if not np.any(mask):
        return LaneFrameMatch(frame, False, None, None, None, None, [])

    ids = lane_map.point_lane_id[mask]
    d2 = dist2[mask]
    dd = dist[mask]
    align = np.clip(heading_dot[mask], 0.0, 1.0)
    point_score = (0.20 + 0.80 * align) / (d2 + LANE_SCORE_EPS)

    score_by_lane: Dict[int, float] = {}
    min_dist_by_lane: Dict[int, float] = {}
    for lid in np.unique(ids):
        m = ids == lid
        lid_int = int(lid)
        score_by_lane[lid_int] = float(np.sum(point_score[m]))
        min_dist_by_lane[lid_int] = float(np.min(dd[m]))

    ranked = sorted(score_by_lane.items(), key=lambda kv: kv[1], reverse=True)
    total = sum(s for _, s in ranked)
    best_lane, best_score = ranked[0]
    confidence = float(best_score / total) if total > 0 else 0.0

    top = []
    for lid, score in ranked[:LANE_TOP_K]:
        top.append(
            {
                "lane_id": int(lid),
                "continuous_lane_id": int(lane_map.segment_to_group[lid]),
                "score": round(float(score), 6),
                "score_fraction": round(float(score / total), 4) if total > 0 else None,
                "nearest_distance_m": round(min_dist_by_lane[lid], 3),
            }
        )

    return LaneFrameMatch(
        frame=int(frame),
        available=True,
        best_lane_id=int(best_lane),
        best_group_id=int(lane_map.segment_to_group[best_lane]),
        confidence=round(confidence, 4),
        distance_m=round(min_dist_by_lane[best_lane], 3),
        top_candidates=top,
    )


def _smooth_group_sequence(raw: List[Optional[int]], radius: int) -> List[Optional[int]]:
    out: List[Optional[int]] = []
    n = len(raw)
    for i in range(n):
        values = raw[max(0, i - radius) : min(n, i + radius + 1)]
        out.append(_mode(values))
    return out


def _remove_short_group_runs(
    seq: List[Optional[int]],
    min_dwell: int,
) -> List[Optional[int]]:
    out = list(seq)
    changed = True
    # A few iterations are enough to absorb single-frame / short flicker runs.
    for _ in range(4):
        if not changed:
            break
        changed = False
        runs = _contiguous_runs(out)
        for r_i, (s, e, value) in enumerate(runs):
            if value is None or (e - s + 1) >= min_dwell:
                continue
            prev_v = runs[r_i - 1][2] if r_i > 0 else None
            next_v = runs[r_i + 1][2] if r_i + 1 < len(runs) else None
            replacement = None
            if prev_v is not None and prev_v == next_v:
                replacement = prev_v
            elif prev_v is not None and next_v is None:
                replacement = prev_v
            elif next_v is not None and prev_v is None:
                replacement = next_v
            if replacement is not None:
                for j in range(s, e + 1):
                    out[j] = replacement
                changed = True
    return out


def build_lane_timeline(track: AgentTrack, lane_map: LaneMap) -> LaneTimeline:
    matches = [match_lane_at_frame(track, t, lane_map) for t in range(track.T)]
    raw_group = [
        m.best_group_id if (m.available and (m.confidence or 0.0) >= LANE_CONFIDENCE_MIN) else None
        for m in matches
    ]
    smooth = _smooth_group_sequence(raw_group, LANE_SMOOTH_RADIUS)

    # Never create lane state on invalid track frames.
    for t in range(track.T):
        if not track.valid[t]:
            smooth[t] = None

    stable = _remove_short_group_runs(smooth, LANE_MIN_DWELL_FRAMES)
    return LaneTimeline(matches, raw_group, smooth, stable)


def _short_lane_run_confidence_is_lower(
    confidences: Sequence[Optional[float]],
    start: int,
    end: int,
    before_start: int,
    before_end: int,
    after_start: int,
    after_end: int,
) -> bool:
    """Require candidate confidence to be lower than both neighboring runs."""
    candidate_values = [
        float(value)
        for value in confidences[start:end + 1]
        if value is not None and math.isfinite(float(value))
    ]
    before_values = [
        float(value)
        for value in confidences[before_start:before_end + 1]
        if value is not None and math.isfinite(float(value))
    ]
    after_values = [
        float(value)
        for value in confidences[after_start:after_end + 1]
        if value is not None and math.isfinite(float(value))
    ]
    if not candidate_values or not before_values or not after_values:
        return False
    candidate_confidence = float(np.median(candidate_values))
    before_confidence = float(np.median(before_values))
    after_confidence = float(np.median(after_values))
    return (
        candidate_confidence + LANE_SEGMENT_FLICKER_CONFIDENCE_MARGIN
        <= before_confidence
        and candidate_confidence + LANE_SEGMENT_FLICKER_CONFIDENCE_MARGIN
        <= after_confidence
        and candidate_confidence
        <= before_confidence * LANE_SEGMENT_FLICKER_CONFIDENCE_RATIO
        and candidate_confidence
        <= after_confidence * LANE_SEGMENT_FLICKER_CONFIDENCE_RATIO
    )


def _filter_short_lane_segment_flickers(
    raw: List[Optional[int]],
    confidences: Sequence[Optional[float]],
) -> Tuple[List[Optional[int]], List[Dict[str, Any]]]:
    """Remove only short segment runs supported as matching flickers.

    The first conservative version only removes the ``A -> B -> A`` pattern:
    B must be short and its matching confidence must be clearly below both
    surrounding A runs.  Any ``A -> B -> C`` transition is preserved, even if
    B is short or its topology is not yet fully explained, because B may be a
    real lateral-transition or intermediate-lane observation.  Runs at the
    beginning/end and runs separated by an unavailable frame are preserved.
    """
    stable = list(raw)
    filtered: List[Dict[str, Any]] = []
    for _ in range(4):
        changed = False
        runs = _contiguous_runs(stable)
        for run_index, (start, end, candidate) in enumerate(runs):
            duration = end - start + 1
            if (
                candidate is None
                or duration > LANE_SEGMENT_FLICKER_MAX_FRAMES
                or run_index == 0
                or run_index + 1 >= len(runs)
            ):
                continue
            before_start, before_end, before = runs[run_index - 1]
            after_start, after_end, after = runs[run_index + 1]
            if before is None or after is None or before == candidate or after == candidate:
                continue

            # Do not delete A -> B -> C.  In particular, B may be the target
            # or transition lane in a real lateral topology sequence.
            if before != after:
                continue
            lower_confidence = _short_lane_run_confidence_is_lower(
                confidences,
                start,
                end,
                before_start,
                before_end,
                after_start,
                after_end,
            )
            if not lower_confidence:
                continue

            for frame in range(start, end + 1):
                stable[frame] = int(before)
            filtered.append({
                "start_frame": int(start),
                "end_frame": int(end),
                "removed_lane_id": int(candidate),
                "replacement_lane_id": int(before),
                "previous_lane_id": int(before),
                "next_lane_id": int(after),
                "reason": "short_return_run_with_lower_matching_confidence",
            })
            changed = True
            break
        if not changed:
            break
    return stable, filtered


def _lane_segment_runs(
    track: AgentTrack,
    lane_timeline: LaneTimeline,
) -> Tuple[
    List[Optional[int]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    ]:
    """Return raw-audit and reliable topology-stabilized lane-segment runs.

    Continuous-lane groups are intentionally not used here.  A group is a
    useful compatibility/audit abstraction.  ``raw`` keeps every available
    best match; ``reliable`` applies ``LANE_CONFIDENCE_MIN`` before it becomes
    input to the map topology normalization below.
    """
    # Raw matching is retained for audit, including low-confidence matches.
    raw: List[Optional[int]] = []
    reliable: List[Optional[int]] = []
    for frame, match in enumerate(lane_timeline.matches):
        if (
            track.valid[frame]
            and match.available
            and match.best_lane_id is not None
        ):
            raw.append(int(match.best_lane_id))
            reliable.append(
                int(match.best_lane_id)
                if (match.confidence or 0.0) >= LANE_CONFIDENCE_MIN
                else None
            )
        else:
            raw.append(None)
            reliable.append(None)

    confidences = [
        (
            float(match.confidence)
            if match.available and match.confidence is not None
            else None
        )
        for match in lane_timeline.matches
    ]
    stable, filtered_runs = _filter_short_lane_segment_flickers(
        reliable, confidences
    )
    confidence_rejected_runs: List[Dict[str, Any]] = []
    frame = 0
    while frame < len(raw):
        if raw[frame] is None or reliable[frame] is not None:
            frame += 1
            continue
        start = frame
        lane_id = int(raw[frame])
        values = []
        while (
            frame < len(raw)
            and raw[frame] == lane_id
            and reliable[frame] is None
        ):
            if confidences[frame] is not None:
                values.append(float(confidences[frame]))
            frame += 1
        confidence_rejected_runs.append({
            "start_frame": int(start),
            "end_frame": int(frame - 1),
            "lane_id": lane_id,
            "confidence_median": (
                _round(float(np.median(values)), 4) if values else None
            ),
            "reason": "below_lane_confidence_min",
        })
    unavailable_runs: List[Dict[str, Any]] = []
    frame = 0
    while frame < len(raw):
        if raw[frame] is not None:
            frame += 1
            continue
        start = frame
        while frame < len(raw) and raw[frame] is None:
            frame += 1
        valid_frames = [
            index for index in range(start, frame)
            if bool(track.valid[index])
        ]
        unavailable_runs.append({
            "start_frame": int(start),
            "end_frame": int(frame - 1),
            "reason": (
                "match_unavailable_or_no_candidate"
                if valid_frames
                else "invalid_track_frame"
            ),
        })
    runs = []
    for start, end, lane_id in _contiguous_runs(stable):
        if lane_id is None:
            continue
        start_match = lane_timeline.matches[start]
        runs.append({
            "lane_id": int(lane_id),
            "continuous_lane_id": (
                int(start_match.best_group_id)
                if start_match.best_group_id is not None else None
            ),
            "start_frame": int(start),
            "end_frame": int(end),
            "duration_frames": int(end - start + 1),
        })
    return raw, runs, filtered_runs, confidence_rejected_runs, unavailable_runs


def _lane_s_at_position(position: np.ndarray, segment: LaneSegment) -> Optional[float]:
    """Project a point to a lane polyline and return its longitudinal s."""
    xy = np.asarray(segment.xy, dtype=np.float64)
    if len(xy) == 0:
        return None
    if len(xy) == 1:
        return 0.0
    best_dist2 = float("inf")
    best_s = 0.0
    cumulative = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    )
    point = np.asarray(position, dtype=np.float64)
    for index in range(len(xy) - 1):
        delta = xy[index + 1] - xy[index]
        length2 = float(np.dot(delta, delta))
        if length2 <= 1e-12:
            continue
        ratio = float(np.clip(np.dot(point - xy[index], delta) / length2, 0.0, 1.0))
        projected = xy[index] + ratio * delta
        distance2 = float(np.dot(point - projected, point - projected))
        if distance2 < best_dist2:
            best_dist2 = distance2
            best_s = float(cumulative[index] + ratio * math.sqrt(length2))
    return best_s


def _neighbor_side_at_position(
    lane_map: LaneMap,
    from_lane_id: int,
    to_lane_id: int,
    source_s: Optional[float],
) -> Optional[str]:
    """Return left/right only when the WOMD neighbor relation is applicable.

    If the installed WOMD proto does not expose neighbor ranges, the legacy
    ID lists are used as a compatibility fallback.  This keeps old synthetic
    fixtures usable without weakening datasets that do provide ranges.
    """
    source = lane_map.segments.get(int(from_lane_id))
    if source is None:
        return None

    for side, ids, bounded in (
        ("left", source.left_neighbors, source.left_neighbor_segments),
        ("right", source.right_neighbors, source.right_neighbor_segments),
    ):
        for relation in bounded:
            if int(relation.neighbor_id) != int(to_lane_id):
                continue
            if (
                source_s is None
                or relation.self_start_index is None
                or relation.self_end_index is None
            ):
                return side
            start_index = max(0, min(int(relation.self_start_index), len(source.xy) - 1))
            end_index = max(0, min(int(relation.self_end_index), len(source.xy) - 1))
            start_s = _lane_s_at_index(source, start_index)
            end_s = _lane_s_at_index(source, end_index)
            if start_s is not None and end_s is not None:
                low_s, high_s = sorted((start_s, end_s))
                if low_s <= source_s <= high_s:
                    return side

        # Fixtures and older map protos may only expose feature IDs.
        if int(to_lane_id) in {int(value) for value in ids} and not bounded:
            return side
    return None


def _lane_s_at_index(segment: LaneSegment, index: int) -> Optional[float]:
    xy = np.asarray(segment.xy, dtype=np.float64)
    if len(xy) == 0:
        return None
    index = max(0, min(int(index), len(xy) - 1))
    if index == 0:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(xy[: index + 1], axis=0), axis=1)))


def build_physical_lane_chain(
    track: AgentTrack,
    lane_timeline: LaneTimeline,
    lane_map: LaneMap,
    physical_lateral_facts: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Normalize matched lane segments into topology-aware vehicle facts.

    Raw lane-ID changes are retained as traceable map-matching evidence.  They
    become ``lane_keep`` only when the map declares an entry/exit
    continuation, and become a directional lane-change fact only when a
    position-valid left/right neighbor relation is paired with the already
    extracted physical lateral-motion evidence.  No priority/overlap behavior
    classifier is used here.
    """
    (
        raw_lane_ids,
        lane_runs,
        filtered_flicker_runs,
        confidence_rejected_runs,
        unavailable_runs,
    ) = _lane_segment_runs(
        track, lane_timeline
    )
    raw_runs = [
        {
            "lane_id": int(value),
            "start_frame": int(start),
            "end_frame": int(end),
            "duration_frames": int(end - start + 1),
        }
        for start, end, value in _contiguous_runs(raw_lane_ids)
        if value is not None
    ]
    base: Dict[str, Any] = {
        "status": "unknown" if not lane_runs else "available",
        # The detailed segment sequence retains separate reliable windows;
        # the compact chain does not duplicate the same lane across a rejected
        # confidence gap.
        "lane_chain": [],
        "lane_segment_sequence": lane_runs,
        "stable_lane_segment_sequence": lane_runs,
        "raw_lane_segment_sequence": raw_runs,
        "segment_flicker_filtering": {
            "removed_runs": filtered_flicker_runs,
            "confidence_rejected_runs": confidence_rejected_runs,
            "unavailable_runs": unavailable_runs,
            "policy": {
                "max_removed_run_frames": LANE_SEGMENT_FLICKER_MAX_FRAMES,
                "stable_sequence_confidence_min": LANE_CONFIDENCE_MIN,
                "short_run_removal_pattern": "A_to_B_to_A_only",
                "distinct_before_after_segments_are_preserved": True,
                "confidence_margin": LANE_SEGMENT_FLICKER_CONFIDENCE_MARGIN,
                "confidence_ratio": LANE_SEGMENT_FLICKER_CONFIDENCE_RATIO,
            },
        },
        "topological_relations": [],
        "lane_change_evidence": [],
        "behavior": "unknown" if not lane_runs else "lane_keep",
        "policy": {
            "lane_identity": "raw matched lane segments normalized by map topology",
            "raw_sequence_for_audit_only": True,
            "stable_sequence_confidence_min": LANE_CONFIDENCE_MIN,
            "raw_lane_id_change_is_core_behavior": False,
            "lane_change_requires": [
                "position-valid left/right neighbor relation",
                "sustained physical lane-relative lateral-motion evidence",
            ],
        },
    }
    for run in lane_runs:
        lane_id = int(run["lane_id"])
        if not base["lane_chain"] or base["lane_chain"][-1] != lane_id:
            base["lane_chain"].append(lane_id)

    unresolved_lane_gaps: List[Dict[str, Any]] = []
    for before, after in zip(lane_runs, lane_runs[1:]):
        gap_start = int(before["end_frame"]) + 1
        gap_end = int(after["start_frame"]) - 1
        if gap_start > gap_end:
            continue

        before_lane_id = int(before["lane_id"])
        after_lane_id = int(after["lane_id"])
        if before_lane_id == after_lane_id:
            continue

        rejected = [
            item for item in confidence_rejected_runs
            if int(item["end_frame"]) >= gap_start
            and int(item["start_frame"]) <= gap_end
        ]
        unavailable = [
            item for item in unavailable_runs
            if int(item["end_frame"]) >= gap_start
            and int(item["start_frame"]) <= gap_end
        ]
        if not rejected and not unavailable:
            continue

        before_group = lane_map.segment_to_group.get(before_lane_id)
        after_group = lane_map.segment_to_group.get(after_lane_id)
        same_continuous_lane = (
            before_group is not None
            and after_group is not None
            and int(before_group) == int(after_group)
        )
        if same_continuous_lane:
            continue

        gap_types = []
        if rejected:
            gap_types.append("confidence_rejected")
        if unavailable:
            gap_types.append("match_unavailable")
        unresolved_lane_gaps.append({
            "start_frame": gap_start,
            "end_frame": gap_end,
            "before_lane_id": before_lane_id,
            "after_lane_id": after_lane_id,
            "before_continuous_lane_id": before_group,
            "after_continuous_lane_id": after_group,
            "confidence_rejected_runs": rejected,
            "unavailable_runs": unavailable,
            "gap_types": gap_types,
            "reason": "distinct_reliable_lane_runs_separated_by_unresolved_lane_gap",
        })
    base["unresolved_lane_gaps"] = unresolved_lane_gaps
    # Compatibility name retained for consumers that already read this field;
    # it now includes both confidence-rejected and unavailable gaps.
    base["unresolved_confidence_gaps"] = unresolved_lane_gaps
    if unresolved_lane_gaps:
        base["behavior"] = "unknown"
    if len(lane_runs) < 2:
        return base

    lateral_episodes = []
    if isinstance(physical_lateral_facts, Mapping):
        candidate_episodes = physical_lateral_facts.get("episodes", [])
        if isinstance(candidate_episodes, list):
            lateral_episodes = [
                episode for episode in candidate_episodes
                if isinstance(episode, Mapping)
            ]

    transitions: List[Dict[str, Any]] = []
    for before, after in zip(lane_runs, lane_runs[1:]):
        if int(before["end_frame"]) + 1 != int(after["start_frame"]):
            continue
        from_lane_id = int(before["lane_id"])
        to_lane_id = int(after["lane_id"])
        if from_lane_id == to_lane_id:
            continue
        source_segment = lane_map.segments.get(from_lane_id)
        source_s = None
        source_frame = int(before["end_frame"])
        if source_segment is not None and 0 <= source_frame < track.T:
            source_s = _lane_s_at_position(track.xy[source_frame], source_segment)

        relation_evidence = []
        if source_segment is not None and to_lane_id in source_segment.exit_lanes:
            relation_evidence.append("source_exit_lane")
        target_segment = lane_map.segments.get(to_lane_id)
        if target_segment is not None and from_lane_id in target_segment.entry_lanes:
            relation_evidence.append("target_entry_lane")

        neighbor_side = _neighbor_side_at_position(
            lane_map, from_lane_id, to_lane_id, source_s
        )
        if relation_evidence:
            relation = "continuation"
        elif neighbor_side is not None:
            relation = f"{neighbor_side}_neighbor"
            relation_evidence.append(f"source_{neighbor_side}_neighbor")
        elif (
            lane_map.segment_to_group.get(from_lane_id)
            == lane_map.segment_to_group.get(to_lane_id)
        ):
            relation = "same_continuous_lane_group"
            relation_evidence.append("continuous_lane_group_audit_only")
        else:
            relation = "unknown"

        transition_frame = int(after["start_frame"])
        episode_indices = [
            index for index, episode in enumerate(lateral_episodes)
            if int(episode.get("start_frame", transition_frame))
            <= transition_frame + PHYS_LATERAL_MAP_SWITCH_ASSOC_RADIUS
            and int(episode.get("end_frame", transition_frame))
            >= transition_frame - PHYS_LATERAL_MAP_SWITCH_ASSOC_RADIUS
        ]
        physical_directions = sorted({
            str(lateral_episodes[index].get("direction_relative_to_local_lane"))
            for index in episode_indices
            if lateral_episodes[index].get("direction_relative_to_local_lane")
        })
        direction_matches = bool(
            neighbor_side is not None
            and neighbor_side in physical_directions
        )
        physically_supported = bool(
            neighbor_side is not None and direction_matches and episode_indices
        )
        transition = {
            "transition_frame": transition_frame,
            "from_lane_id": from_lane_id,
            "to_lane_id": to_lane_id,
            "from_continuous_lane_id": lane_map.segment_to_group.get(from_lane_id),
            "to_continuous_lane_id": lane_map.segment_to_group.get(to_lane_id),
            "from_frame": int(before["start_frame"]),
            "to_frame": int(after["end_frame"]),
            "relation": relation,
            "relation_evidence": relation_evidence,
            "source_lane_s": _round(source_s, 3),
            "neighbor_side": neighbor_side,
            "physical_lateral_episode_indices": episode_indices,
            "physical_direction_matches_neighbor_side": direction_matches,
            "physical_lane_change_supported": physically_supported,
        }
        transitions.append(transition)
        if physically_supported:
            base["lane_change_evidence"].append({
                "transition_frame": transition_frame,
                "from_lane_id": from_lane_id,
                "to_lane_id": to_lane_id,
                "direction": neighbor_side,
                "physical_lateral_episode_indices": episode_indices,
                "evidence": "neighbor_topology_plus_physical_lateral_motion",
            })

    base["topological_relations"] = transitions
    confirmed_sides = sorted({
        str(item["neighbor_side"])
        for item in transitions
        if item["physical_lane_change_supported"]
    })
    if unresolved_lane_gaps:
        base["behavior"] = "unknown"
    elif len(confirmed_sides) == 1:
        base["behavior"] = f"{confirmed_sides[0]}_lane_change"
    elif len(confirmed_sides) > 1:
        base["behavior"] = "mixed_lane_behavior"
    elif any(item["relation"] == "unknown" for item in transitions):
        base["behavior"] = "unknown"
    elif any(item["relation"] in {"left_neighbor", "right_neighbor"} for item in transitions):
        base["behavior"] = "unknown"
    else:
        base["behavior"] = "lane_keep"
    return base


def lane_group_at(
    timeline: LaneTimeline,
    frame: int,
    radius: int = 2,
) -> Optional[int]:
    if frame < 0 or frame >= len(timeline.stabilized_group):
        return None
    values = timeline.stabilized_group[
        max(0, frame - radius) : min(len(timeline.stabilized_group), frame + radius + 1)
    ]
    return _mode(values)


# =============================================================================
# Conservative physical facts
# =============================================================================

def _median_dt_seconds(track: AgentTrack) -> float:
    ts = np.asarray(track.timestamps, dtype=np.float64)
    if len(ts) < 2:
        return 0.1
    d = np.diff(ts)
    d = d[np.isfinite(d) & (d > 1e-6)]
    return float(np.median(d)) if len(d) else 0.1


def _rolling_nanmedian(values: np.ndarray, radius: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    out = np.full_like(values, np.nan, dtype=np.float64)
    n = len(values)
    for i in range(n):
        w = values[max(0, i - radius): min(n, i + radius + 1)]
        good = w[np.isfinite(w)]
        if len(good):
            out[i] = float(np.median(good))
    return out


def _bridge_short_boolean_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    """Fill short False gaps bounded by True runs."""
    out = np.asarray(mask, dtype=bool).copy()
    if max_gap <= 0 or len(out) == 0:
        return out
    i = 0
    n = len(out)
    while i < n:
        if out[i]:
            i += 1
            continue
        s = i
        while i < n and not out[i]:
            i += 1
        e = i - 1
        gap_len = e - s + 1
        left_true = s - 1 >= 0 and out[s - 1]
        right_true = i < n and out[i]
        if left_true and right_true and gap_len <= max_gap:
            out[s:e + 1] = True
    return out


def _boolean_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    runs: List[Tuple[int, int]] = []
    i = 0
    n = len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        s = i
        while i + 1 < n and mask[i + 1]:
            i += 1
        runs.append((s, i))
        i += 1
    return runs


def _consolidate_speed_episodes(
    raw_eps: Sequence[Mapping[str, Any]],
    *,
    smooth_speed: np.ndarray,
    sign: int,
) -> List[Dict[str, Any]]:
    """Filter and consolidate raw same-sign speed episodes.

    A short threshold crossing is retained as raw evidence, but is not a
    semantic speed event unless it survives this pass.  Nearby episodes are
    merged only when the recomputed end-to-end speed trend still has the
    expected sign and the intervening frames do not form a stable plateau.
    """
    if sign not in {-1, 1}:
        raise ValueError(f"speed episode sign must be -1 or 1, got {sign}")
    if not raw_eps:
        return []

    speed = np.asarray(smooth_speed, dtype=np.float64)
    ordered = sorted(
        (dict(ep) for ep in raw_eps),
        key=lambda ep: (int(ep["start_frame"]), int(ep["end_frame"])),
    )

    def _endpoint_delta(start: int, end: int) -> Optional[float]:
        pre = max(0, start - 1)
        post = min(len(speed) - 1, end + 1)
        if pre >= len(speed) or post < 0:
            return None
        if not (np.isfinite(speed[pre]) and np.isfinite(speed[post])):
            return None
        return float(speed[post] - speed[pre])

    def _is_stable_plateau(left_end: int, right_start: int) -> bool:
        gap_start = left_end + 1
        gap_end = right_start - 1
        gap_len = gap_end - gap_start + 1
        # Four-frame gaps are still treated as a bridge.  Only a gap of at
        # least five frames can become a separate stable phase, as in 70--74
        # for the Vehicle 14 example.
        if gap_len < 5:
            return False
        values = speed[gap_start:gap_end + 1]
        values = values[np.isfinite(values)]
        if len(values) < 4:
            return False
        net = float(values[-1] - values[0])
        spread = float(np.max(values) - np.min(values))
        return abs(net) <= 0.15 and spread <= 0.25

    def _merge(left: Dict[str, Any], right: Mapping[str, Any]) -> Dict[str, Any]:
        start = int(left["start_frame"])
        end = int(right["end_frame"])
        delta = _endpoint_delta(start, end)
        merged = dict(left)
        merged.update({
            "start_frame": start,
            "end_frame": end,
            "duration_frames": end - start + 1,
            "speed_before_mps": _round(float(speed[max(0, start - 1)]), 3)
            if np.isfinite(speed[max(0, start - 1)]) else None,
            "speed_after_mps": _round(float(speed[min(len(speed) - 1, end + 1)]), 3)
            if np.isfinite(speed[min(len(speed) - 1, end + 1)]) else None,
            "speed_delta_mps": _round(delta, 3) if delta is not None else None,
            "median_acceleration_mps2": _round(float(np.average(
                [
                    float(left.get("median_acceleration_mps2", 0.0)),
                    float(right.get("median_acceleration_mps2", 0.0)),
                ],
                weights=[
                    max(1, int(left.get("duration_frames", 1))),
                    max(1, int(right.get("duration_frames", 1))),
                ],
            )), 3),
            "max_abs_acceleration_mps2": _round(float(max(
                float(left.get("max_abs_acceleration_mps2", 0.0)),
                float(right.get("max_abs_acceleration_mps2", 0.0)),
            )), 3),
        })
        return merged

    consolidated: List[Dict[str, Any]] = []
    current = ordered[0]
    for candidate in ordered[1:]:
        gap = int(candidate["start_frame"]) - int(current["end_frame"]) - 1
        can_merge = gap <= PHYS_SPEED_EVENT_MERGE_GAP_FRAMES
        if can_merge and _is_stable_plateau(
            int(current["end_frame"]), int(candidate["start_frame"])
        ):
            can_merge = False
        if can_merge:
            delta = _endpoint_delta(
                int(current["start_frame"]), int(candidate["end_frame"])
            )
            if delta is None or (sign > 0 and delta < PHYS_ACCEL_MIN_SPEED_DELTA_MPS) \
                    or (sign < 0 and delta > -PHYS_ACCEL_MIN_SPEED_DELTA_MPS):
                can_merge = False
        if can_merge:
            current = _merge(current, candidate)
        else:
            consolidated.append(current)
            current = candidate
    consolidated.append(current)

    final: List[Dict[str, Any]] = []
    for episode in consolidated:
        duration = int(episode["end_frame"]) - int(episode["start_frame"]) + 1
        delta = episode.get("speed_delta_mps")
        if duration < PHYS_SPEED_EVENT_MIN_DURATION_FRAMES:
            continue
        if delta is None:
            continue
        if sign > 0 and float(delta) < PHYS_ACCEL_MIN_SPEED_DELTA_MPS:
            continue
        if sign < 0 and float(delta) > -PHYS_ACCEL_MIN_SPEED_DELTA_MPS:
            continue
        final.append(episode)
    return final


def _categorical_runs(
    frame_value_pairs: Sequence[Tuple[int, str]],
) -> List[Dict[str, Any]]:
    """Consecutive-frame runs for a categorical fact."""
    if not frame_value_pairs:
        return []
    pairs = sorted((int(f), str(v)) for f, v in frame_value_pairs)
    out: List[Dict[str, Any]] = []
    s_f, prev_f = pairs[0][0], pairs[0][0]
    value = pairs[0][1]
    for f, v in pairs[1:]:
        if f == prev_f + 1 and v == value:
            prev_f = f
            continue
        out.append({
            "start_frame": int(s_f),
            "end_frame": int(prev_f),
            "duration_frames": int(prev_f - s_f + 1),
            "value": value,
        })
        s_f = prev_f = f
        value = v
    out.append({
        "start_frame": int(s_f),
        "end_frame": int(prev_f),
        "duration_frames": int(prev_f - s_f + 1),
        "value": value,
    })
    return out


def _dominant_categorical(
    values: Sequence[str],
    uncertain_value: str = "uncertain",
) -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    known = {k: v for k, v in counts.items() if k != uncertain_value}
    dominant = None
    dominance = None
    if known:
        dominant, n = max(known.items(), key=lambda kv: (kv[1], kv[0]))
        denom = sum(known.values())
        dominance = float(n / denom) if denom else None
    return {
        "dominant": dominant,
        "dominance_fraction_among_known": _round(dominance, 4),
        "frame_counts": counts,
        "num_frames": int(len(values)),
    }


def _local_lane_frame(
    track: AgentTrack,
    lane_timeline: LaneTimeline,
    lane_map: LaneMap,
    frame: int,
) -> Optional[Dict[str, Any]]:
    """Return local matched-lane tangent and signed centerline offset."""
    if not (0 <= frame < track.T) or not track.valid[frame]:
        return None
    stabilized_group = lane_timeline.stabilized_group[frame]
    if stabilized_group is None:
        return None
    m = lane_timeline.matches[frame]
    if (
        not m.available
        or m.best_lane_id is None
        or m.confidence is None
        or float(m.confidence) < PHYS_LATERAL_MIN_MAP_CONFIDENCE
        or m.best_group_id != stabilized_group
    ):
        return None
    seg = lane_map.segments.get(int(m.best_lane_id))
    if seg is None or len(seg.xy) == 0:
        return None

    pos = track.xy[frame]
    d = seg.xy - pos[None, :]
    idx = int(np.argmin(np.sum(d * d, axis=1)))
    tangent = np.asarray(seg.tangent_xy[idx], dtype=np.float64).copy()
    norm = float(np.linalg.norm(tangent))
    if norm <= 1e-8:
        return None
    tangent /= norm

    vel = track.velocity[frame]
    if float(np.linalg.norm(vel)) >= MIN_MOVE_SPEED and float(np.dot(tangent, vel)) < 0.0:
        tangent *= -1.0

    center = seg.xy[idx]
    offset = float(_cross2d(tangent, pos - center))
    v_long = float(np.dot(track.velocity[frame], tangent))
    v_lat = float(_cross2d(tangent, track.velocity[frame]))
    return {
        "lane_id": int(m.best_lane_id),
        "continuous_lane_id": int(stabilized_group),
        "confidence": _round(m.confidence, 4),
        "nearest_distance_m": _round(m.distance_m, 3),
        "tangent_xy": tangent,
        "signed_centerline_offset_m": _round(offset, 3),
        "lane_relative_longitudinal_velocity_mps": _round(v_long, 3),
        "lane_relative_lateral_velocity_mps": _round(v_lat, 3),
    }


def _map_group_switches(
    track: AgentTrack,
    lane_timeline: LaneTimeline,
) -> List[Dict[str, Any]]:
    """Map-matching switches only.  No physical maneuver claim is made."""
    seq = lane_timeline.stabilized_group
    switches: List[Dict[str, Any]] = []
    for t in range(1, len(seq)):
        old = seq[t - 1]
        new = seq[t]
        if old is None or new is None or old == new or not track.valid[t]:
            continue
        before = seq[max(0, t - LANE_MIN_DWELL_FRAMES): t]
        after = seq[t: min(len(seq), t + LANE_MIN_DWELL_FRAMES)]
        if sum(v == old for v in before) < min(LANE_MIN_DWELL_FRAMES, len(before)):
            continue
        if sum(v == new for v in after) < min(LANE_MIN_DWELL_FRAMES, len(after)):
            continue
        switches.append({
            "frame": int(t),
            "old_continuous_lane_id": int(old),
            "new_continuous_lane_id": int(new),
            "interpretation": "map_match_group_switch_only",
        })
    return switches


def extract_physical_lateral_facts(
    track: AgentTrack,
    lane_timeline: LaneTimeline,
    lane_map: LaneMap,
) -> Dict[str, Any]:
    """Extract lane-relative lateral-motion evidence, not a maneuver label.

    The signed offset is measured from the matched lane centerline in the lane's
    local tangent frame.  Candidate episodes are never allowed to cross a
    stabilized continuous-lane-group change or an unavailable measurement gap.
    This deliberately does not use world-coordinate displacement as a shift
    criterion.  ``status=confirmed`` means that reliable lateral-motion
    evidence exists; it does not mean that a lane change or shift occurred.
    """
    T = track.T
    offsets = np.full(T, np.nan, dtype=np.float64)
    lateral_velocity = np.full(T, np.nan, dtype=np.float64)
    groups: List[Optional[int]] = [None] * T
    reliable = np.zeros(T, dtype=bool)

    for f in range(T):
        local = _local_lane_frame(track, lane_timeline, lane_map, f)
        if local is None:
            continue
        offsets[f] = float(local["signed_centerline_offset_m"])
        lateral_velocity[f] = float(local["lane_relative_lateral_velocity_mps"])
        groups[f] = int(local["continuous_lane_id"])
        reliable[f] = True

    dt = _median_dt_seconds(track)
    smooth_offsets = _rolling_nanmedian(offsets, PHYS_LATERAL_SMOOTH_RADIUS)
    offset_rate = np.full(T, np.nan, dtype=np.float64)
    same_group = np.zeros(T, dtype=bool)
    for f in range(1, T):
        if (
            np.isfinite(smooth_offsets[f - 1])
            and np.isfinite(smooth_offsets[f])
            and groups[f] is not None
            and groups[f] == groups[f - 1]
        ):
            offset_rate[f] = (smooth_offsets[f] - smooth_offsets[f - 1]) / dt
            same_group[f] = True

    active = same_group & np.isfinite(offset_rate)
    active &= np.abs(offset_rate) >= PHYS_LATERAL_OFFSET_RATE_THRESHOLD_MPS

    episodes: List[Dict[str, Any]] = []
    for s, e in _boolean_runs(active):
        start_frame = max(0, s - 1)
        end_frame = e
        frames = [
            f for f in range(start_frame, end_frame + 1)
            if np.isfinite(smooth_offsets[f]) and groups[f] == groups[start_frame]
        ]
        if len(frames) < PHYS_LATERAL_MIN_DURATION_FRAMES:
            continue
        group_offsets = np.asarray(
            [smooth_offsets[f] for f in frames], dtype=np.float64
        )
        signed_disp = float(group_offsets[-1] - group_offsets[0])
        offset_steps = np.diff(group_offsets)
        absolute_integral = float(np.sum(np.abs(offset_steps)))
        sign_consistency = (
            abs(signed_disp) / absolute_integral if absolute_integral > 1e-9 else 0.0
        )
        if abs(signed_disp) < PHYS_LATERAL_MIN_CENTERLINE_OFFSET_CHANGE_M:
            continue
        if sign_consistency < PHYS_LATERAL_MIN_SIGN_CONSISTENCY:
            continue

        before_candidates = [
            f for f in range(max(0, start_frame - 3), start_frame + 1)
            if np.isfinite(offsets[f]) and groups[f] == groups[start_frame]
        ]
        after_candidates = [
            f for f in range(end_frame, min(T, end_frame + 4))
            if np.isfinite(offsets[f]) and groups[f] == groups[start_frame]
        ]
        offset_before = (
            float(np.median(offsets[before_candidates])) if before_candidates else None
        )
        offset_after = (
            float(np.median(offsets[after_candidates])) if after_candidates else None
        )
        episodes.append({
            "start_frame": int(start_frame),
            "end_frame": int(end_frame),
            "duration_frames": int(end_frame - start_frame + 1),
            "continuous_lane_id": int(groups[start_frame]),
            "direction_relative_to_local_lane": "left" if signed_disp > 0 else "right",
            "centerline_offset_change_m": _round(signed_disp, 3),
            "integrated_lateral_displacement_m": _round(signed_disp, 3),
            "absolute_centerline_offset_change_m": _round(absolute_integral, 3),
            "absolute_lateral_motion_integral_m": _round(absolute_integral, 3),
            "sign_consistency": _round(sign_consistency, 3),
            "max_abs_lateral_velocity_mps": _round(
                float(np.nanmax(np.abs(lateral_velocity[start_frame:end_frame + 1]))), 3
            ),
            "median_abs_lateral_velocity_mps": _round(
                float(np.nanmedian(np.abs(lateral_velocity[start_frame:end_frame + 1]))), 3
            ),
            "max_abs_centerline_offset_rate_mps": _round(
                float(np.max(np.abs(offset_rate[s:e + 1]))), 3
            ),
            "median_abs_centerline_offset_rate_mps": _round(
                float(np.median(np.abs(offset_rate[s:e + 1]))), 3
            ),
            "centerline_offset_before_m": _round(offset_before, 3),
            "centerline_offset_after_m": _round(offset_after, 3),
            "evidence": "sustained_same_physical_lane_centerline_offset_change",
        })

    switches = _map_group_switches(track, lane_timeline)
    for sw in switches:
        f = int(sw["frame"])
        linked = [
            i for i, ep in enumerate(episodes)
            if ep["start_frame"] - PHYS_LATERAL_MAP_SWITCH_ASSOC_RADIUS
            <= f
            <= ep["end_frame"] + PHYS_LATERAL_MAP_SWITCH_ASSOC_RADIUS
        ]
        sw["physically_supported_by_episode_indices"] = linked
        sw["physical_confirmation"] = bool(linked)

    valid_count = int(np.sum(track.valid))
    reliable_count = int(np.sum(reliable & track.valid))
    coverage = float(reliable_count / valid_count) if valid_count else 0.0
    if episodes:
        status = "confirmed"
    elif valid_count == 0 or reliable_count < 5 or coverage < PHYS_LATERAL_MIN_COVERAGE_FOR_NONE:
        status = "unknown"
    else:
        status = "none_detected"

    return {
        "fact_type": "lateral_motion_evidence",
        "status": status,
        "episodes": episodes,
        "num_confirmed_episodes": int(len(episodes)),
        "map_match_group_switches_audit_only": switches,
        "num_map_switches_without_physical_confirmation": int(
            sum(not sw["physical_confirmation"] for sw in switches)
        ),
        "measurement_coverage": {
            "valid_track_frames": valid_count,
            "reliable_lane_relative_frames": reliable_count,
            "coverage_fraction": _round(coverage, 4),
        },
        "policy": {
            "map_lane_group_switch_is_sufficient": False,
            "minimum_centerline_offset_rate_mps": PHYS_LATERAL_OFFSET_RATE_THRESHOLD_MPS,
            "minimum_duration_frames": PHYS_LATERAL_MIN_DURATION_FRAMES,
            "minimum_centerline_offset_change_m": PHYS_LATERAL_MIN_CENTERLINE_OFFSET_CHANGE_M,
            "minimum_sign_consistency": PHYS_LATERAL_MIN_SIGN_CONSISTENCY,
            "requires_same_stabilized_physical_lane": True,
            "world_coordinate_shift_threshold_used": False,
            "lateral_velocity_used_as_decision": False,
        },
    }


def extract_speed_change_facts(track: AgentTrack) -> Dict[str, Any]:
    T = track.T
    speed = np.where(track.valid, track.speed, np.nan).astype(np.float64)
    smooth = _rolling_nanmedian(speed, PHYS_SPEED_SMOOTH_RADIUS)
    accel = np.full(T, np.nan, dtype=np.float64)
    ts = np.asarray(track.timestamps, dtype=np.float64)

    for t in range(1, T - 1):
        if not (track.valid[t - 1] and track.valid[t] and track.valid[t + 1]):
            continue
        if not (np.isfinite(smooth[t - 1]) and np.isfinite(smooth[t + 1])):
            continue
        dt = float(ts[t + 1] - ts[t - 1])
        if dt <= 1e-6:
            continue
        accel[t] = float((smooth[t + 1] - smooth[t - 1]) / dt)
    accel = _rolling_nanmedian(accel, 1)

    def _episodes_for_sign(sign: int) -> List[Dict[str, Any]]:
        if sign > 0:
            active = np.isfinite(accel) & (accel >= PHYS_ACCEL_THRESHOLD_MPS2)
            kind = "acceleration"
        else:
            active = np.isfinite(accel) & (accel <= -PHYS_ACCEL_THRESHOLD_MPS2)
            kind = "deceleration"
        active &= track.valid
        active = _bridge_short_boolean_gaps(active, PHYS_ACCEL_MAX_BRIDGED_GAP_FRAMES)
        # Never let the raw bridge turn an invalid Track state into a valid
        # speed-event frame.
        active &= track.valid
        eps: List[Dict[str, Any]] = []
        for s, e in _boolean_runs(active):
            frames = [f for f in range(s, e + 1) if np.isfinite(accel[f])]
            if not frames:
                continue
            pre = max(0, s - 1)
            post = min(T - 1, e + 1)
            if not (np.isfinite(smooth[pre]) and np.isfinite(smooth[post])):
                continue
            delta = float(smooth[post] - smooth[pre])
            vals = np.asarray([accel[f] for f in frames], dtype=np.float64)
            eps.append({
                "type": kind,
                "start_frame": int(s),
                "end_frame": int(e),
                "duration_frames": int(e - s + 1),
                "speed_before_mps": _round(float(smooth[pre]), 3),
                "speed_after_mps": _round(float(smooth[post]), 3),
                "speed_delta_mps": _round(delta, 3),
                "median_acceleration_mps2": _round(float(np.median(vals)), 3),
                "max_abs_acceleration_mps2": _round(float(np.max(np.abs(vals))), 3),
            })
        return eps

    raw_accel_eps = _episodes_for_sign(+1)
    raw_decel_eps = _episodes_for_sign(-1)
    accel_eps = _consolidate_speed_episodes(
        raw_accel_eps,
        smooth_speed=smooth,
        sign=+1,
    )
    decel_eps = _consolidate_speed_episodes(
        raw_decel_eps,
        smooth_speed=smooth,
        sign=-1,
    )
    valid_speeds = smooth[np.isfinite(smooth)]
    return {
        "has_clear_acceleration": bool(accel_eps),
        "has_clear_deceleration": bool(decel_eps),
        "raw_acceleration_episodes": raw_accel_eps,
        "raw_deceleration_episodes": raw_decel_eps,
        "acceleration_episodes": accel_eps,
        "deceleration_episodes": decel_eps,
        "speed_summary": {
            "median_mps": _round(float(np.median(valid_speeds)), 3) if len(valid_speeds) else None,
            "min_mps": _round(float(np.min(valid_speeds)), 3) if len(valid_speeds) else None,
            "max_mps": _round(float(np.max(valid_speeds)), 3) if len(valid_speeds) else None,
        },
        "policy": {
            "smoothed_speed_radius_frames": PHYS_SPEED_SMOOTH_RADIUS,
            "acceleration_threshold_mps2": PHYS_ACCEL_THRESHOLD_MPS2,
            "raw_bridge_gap_frames": PHYS_ACCEL_MAX_BRIDGED_GAP_FRAMES,
            "semantic_minimum_duration_frames": PHYS_SPEED_EVENT_MIN_DURATION_FRAMES,
            "semantic_merge_gap_frames": PHYS_SPEED_EVENT_MERGE_GAP_FRAMES,
            "minimum_speed_delta_mps": PHYS_ACCEL_MIN_SPEED_DELTA_MPS,
            "causal_interpretation_emitted": False,
        },
    }


def _u_turn_evidence_for_run(
    track: AgentTrack,
    start_frame: int,
    end_frame: int,
) -> Dict[str, Any]:
    """Assess global U-turn evidence on one contiguous valid trajectory run."""
    raw_rad = np.asarray(track.yaw[start_frame:end_frame + 1], dtype=np.float64)
    xy = np.asarray(track.xy[start_frame:end_frame + 1], dtype=np.float64)
    if (
        len(raw_rad) < 2
        or not np.all(np.isfinite(raw_rad))
        or not np.all(np.isfinite(xy))
    ):
        return {
            "status": "insufficient_data",
            "start_frame": int(start_frame),
            "end_frame": int(end_frame),
            "global_heading_change_deg": None,
            "rotation_direction": "uncertain",
            "rotation_consistency": None,
            "trajectory_reversal": "uncertain",
            "path_to_endpoint_ratio": None,
            "evidence": [],
        }

    unwrapped_deg = np.degrees(np.unwrap(raw_rad))
    smooth_deg = np.degrees(
        _rolling_nanmedian(
            np.unwrap(raw_rad),
            PHYS_HEADING_MOTION_SMOOTH_RADIUS,
        )
    )
    net_change = float(unwrapped_deg[-1] - unwrapped_deg[0])
    abs_change = abs(net_change)
    rotation_direction = (
        "increasing_ccw"
        if net_change > PHYS_HEADING_EPISODE_MIN_NET_CHANGE_DEG
        else "decreasing_cw"
        if net_change < -PHYS_HEADING_EPISODE_MIN_NET_CHANGE_DEG
        else "roughly_stable"
    )

    increments = np.diff(smooth_deg)
    signs = np.zeros(len(increments), dtype=np.int8)
    signs[increments >= PHYS_HEADING_EPISODE_MIN_RATE_DEG_PER_FRAME] = 1
    signs[increments <= -PHYS_HEADING_EPISODE_MIN_RATE_DEG_PER_FRAME] = -1
    active_signs = signs[signs != 0]
    if len(active_signs):
        positive_count = int(np.sum(active_signs > 0))
        negative_count = int(np.sum(active_signs < 0))
        dominant_sign = 1 if positive_count >= negative_count else -1
        dominant_count = max(positive_count, negative_count)
        rotation_consistency = float(dominant_count / len(active_signs))
    else:
        dominant_sign = 0
        dominant_count = 0
        rotation_consistency = 0.0

    longest_dominant_run = 0
    current_run = 0
    for sign in signs:
        if int(sign) == dominant_sign and dominant_sign != 0:
            current_run += 1
            longest_dominant_run = max(longest_dominant_run, current_run)
        else:
            current_run = 0

    edge_window = min(
        PHYS_U_TURN_EDGE_WINDOW_FRAMES,
        max(1, (end_frame - start_frame) // 3),
    )
    start_vector = xy[edge_window] - xy[0]
    end_vector = xy[-1] - xy[-1 - edge_window]
    start_norm = float(np.linalg.norm(start_vector))
    end_norm = float(np.linalg.norm(end_vector))
    if start_norm > 1e-6 and end_norm > 1e-6:
        edge_direction_dot = float(
            np.dot(start_vector, end_vector) / (start_norm * end_norm)
        )
        trajectory_reversal = (
            "supported"
            if edge_direction_dot <= PHYS_U_TURN_MIN_EDGE_DIRECTION_REVERSAL_DOT
            else "not_supported"
        )
    else:
        edge_direction_dot = None
        trajectory_reversal = "insufficient_data"

    segment_lengths = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    path_length = float(np.sum(segment_lengths))
    endpoint_displacement = float(np.linalg.norm(xy[-1] - xy[0]))
    path_to_endpoint_ratio = (
        path_length / endpoint_displacement
        if endpoint_displacement > 1e-6
        else None
    )
    path_shape_supported = (
        path_to_endpoint_ratio is not None
        and path_to_endpoint_ratio >= PHYS_U_TURN_MIN_PATH_TO_ENDPOINT_RATIO
    )

    observation_reaches_track_boundary = bool(
        end_frame >= track.T - 1
        or not np.any(np.asarray(track.valid[end_frame + 1:], dtype=bool))
    )
    tail_window = min(
        PHYS_U_TURN_PARTIAL_TAIL_WINDOW_FRAMES,
        max(1, len(smooth_deg) - 1),
    )
    tail_start = max(0, len(smooth_deg) - 1 - tail_window)
    tail_change = float(smooth_deg[-1] - smooth_deg[tail_start])
    tail_increments = increments[tail_start:]
    tail_active = np.abs(tail_increments) >= PHYS_HEADING_EPISODE_MIN_RATE_DEG_PER_FRAME
    tail_active_fraction = (
        float(np.mean(tail_active)) if len(tail_active) else 0.0
    )
    stable_exit_heading = bool(
        abs(tail_change) <= PHYS_U_TURN_PARTIAL_STABLE_EXIT_MAX_CHANGE_DEG
        and tail_active_fraction <= PHYS_U_TURN_PARTIAL_STABLE_EXIT_MAX_ACTIVE_FRACTION
    )
    partial_shape_supported = bool(
        (
            edge_direction_dot is not None
            and edge_direction_dot <= PHYS_U_TURN_PARTIAL_MAX_EDGE_DIRECTION_DOT
        )
        or (
            path_to_endpoint_ratio is not None
            and path_to_endpoint_ratio >= PHYS_U_TURN_PARTIAL_MIN_PATH_TO_ENDPOINT_RATIO
        )
    )

    heading_supported = abs_change >= PHYS_U_TURN_MIN_GLOBAL_HEADING_CHANGE_DEG
    rotation_supported = (
        len(active_signs) >= PHYS_U_TURN_MIN_ROTATION_FRAMES
        and rotation_consistency >= PHYS_U_TURN_MIN_ROTATION_CONSISTENCY
        and longest_dominant_run >= PHYS_U_TURN_MIN_ROTATION_FRAMES
        and dominant_sign == (1 if net_change > 0 else -1)
    )
    trajectory_supported = (
        trajectory_reversal == "supported" and path_shape_supported
    )
    partial_heading_supported = (
        abs_change >= PHYS_U_TURN_PARTIAL_MIN_GLOBAL_HEADING_CHANGE_DEG
    )
    partial_rotation_supported = (
        len(active_signs) >= PHYS_U_TURN_PARTIAL_MIN_ROTATION_FRAMES
        and rotation_consistency >= PHYS_U_TURN_PARTIAL_MIN_ROTATION_CONSISTENCY
        and longest_dominant_run >= PHYS_U_TURN_PARTIAL_MIN_ROTATION_FRAMES
        and dominant_sign == (1 if net_change > 0 else -1)
    )
    partial_boundary_supported = bool(
        observation_reaches_track_boundary and not stable_exit_heading
    )
    partial_supported = bool(
        partial_heading_supported
        and partial_rotation_supported
        and partial_shape_supported
        and partial_boundary_supported
    )
    evidence = []
    if heading_supported:
        evidence.append("large_global_heading_reversal")
    if rotation_supported:
        evidence.append("sustained_same_direction_rotation")
    if trajectory_supported:
        evidence.append("trajectory_entry_exit_direction_reversal")
        evidence.append("curved_return_path_shape")
    partial_evidence = []
    if partial_heading_supported:
        partial_evidence.append("large_partial_global_heading_change")
    if partial_rotation_supported:
        partial_evidence.append("sustained_same_direction_partial_rotation")
    if partial_shape_supported:
        partial_evidence.append("partial_trajectory_return_shape")
    if partial_boundary_supported:
        partial_evidence.append("observation_reaches_track_boundary_without_stable_exit")

    status = (
        "supported"
        if heading_supported and rotation_supported and trajectory_supported
        else "partial_supported"
        if partial_supported
        else "not_supported"
    )

    return {
        "status": status,
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
        "global_heading_change_deg": _round(net_change, 3),
        "rotation_direction": rotation_direction,
        "rotation_consistency": _round(rotation_consistency, 3),
        "longest_same_direction_rotation_frames": int(longest_dominant_run),
        "trajectory_reversal": trajectory_reversal,
        "edge_direction_dot": _round(edge_direction_dot, 3),
        "path_length_m": _round(path_length, 3),
        "endpoint_displacement_m": _round(endpoint_displacement, 3),
        "path_to_endpoint_ratio": _round(path_to_endpoint_ratio, 3),
        "observation_reaches_track_boundary": observation_reaches_track_boundary,
        "stable_exit_heading": stable_exit_heading,
        "tail_heading_change_deg": _round(tail_change, 3),
        "tail_active_rotation_fraction": _round(tail_active_fraction, 3),
        "evidence": evidence,
        "partial_evidence": partial_evidence,
        "policy": {
            "global_heading_change_min_deg": PHYS_U_TURN_MIN_GLOBAL_HEADING_CHANGE_DEG,
            "rotation_consistency_min": PHYS_U_TURN_MIN_ROTATION_CONSISTENCY,
            "minimum_same_direction_rotation_frames": PHYS_U_TURN_MIN_ROTATION_FRAMES,
            "edge_direction_reversal_dot_max": PHYS_U_TURN_MIN_EDGE_DIRECTION_REVERSAL_DOT,
            "path_to_endpoint_ratio_min": PHYS_U_TURN_MIN_PATH_TO_ENDPOINT_RATIO,
            "partial_global_heading_change_min_deg": PHYS_U_TURN_PARTIAL_MIN_GLOBAL_HEADING_CHANGE_DEG,
            "partial_rotation_consistency_min": PHYS_U_TURN_PARTIAL_MIN_ROTATION_CONSISTENCY,
            "partial_edge_direction_dot_max": PHYS_U_TURN_PARTIAL_MAX_EDGE_DIRECTION_DOT,
            "partial_path_to_endpoint_ratio_min": PHYS_U_TURN_PARTIAL_MIN_PATH_TO_ENDPOINT_RATIO,
            "partial_tail_window_frames": PHYS_U_TURN_PARTIAL_TAIL_WINDOW_FRAMES,
            "partial_stable_exit_max_change_deg": PHYS_U_TURN_PARTIAL_STABLE_EXIT_MAX_CHANGE_DEG,
            "partial_stable_exit_max_active_fraction": PHYS_U_TURN_PARTIAL_STABLE_EXIT_MAX_ACTIVE_FRACTION,
            "source": "complete longest contiguous valid trajectory run",
            "behavior_semantics_emitted": False,
            "partial_status_is_evidence_only": True,
        },
    }


def extract_u_turn_evidence(track: AgentTrack) -> Dict[str, Any]:
    """Return conservative global physical evidence for a U-turn."""
    valid_frames = np.flatnonzero(track.valid)
    runs = _boolean_runs(track.valid)
    usable_runs = [run for run in runs if run[1] - run[0] + 1 >= 2]
    if not usable_runs:
        return {
            "status": "insufficient_data",
            "valid_frame_count": int(len(valid_frames)),
            "evidence": [],
            "policy": {
                "source": "complete longest contiguous valid trajectory run",
                "behavior_semantics_emitted": False,
            },
        }
    start_frame, end_frame = max(
        usable_runs,
        key=lambda run: (run[1] - run[0] + 1, -run[0]),
    )
    result = _u_turn_evidence_for_run(track, start_frame, end_frame)
    result["valid_frame_count"] = int(len(valid_frames))
    return result


def extract_heading_motion_facts(track: AgentTrack) -> Dict[str, Any]:
    """Describe one track's heading evolution without turning semantics."""
    valid_frames = np.flatnonzero(track.valid)
    if len(valid_frames) < 2:
        return {
            "status": "insufficient_data",
            "heading_start_deg": None,
            "heading_end_deg": None,
            "net_heading_change_deg": None,
            "heading_change_direction": "uncertain",
            "turning_episode": {
                "status": "insufficient_data",
                "start_frame": None,
                "end_frame": None,
                "direction": "uncertain",
                "net_heading_change_deg": None,
            },
            "start_frame": None,
            "end_frame": None,
            "valid_frame_count": int(len(valid_frames)),
            "policy": {
                "source": "Track.heading on valid frames",
                "heading_change_is_behavior_semantics": False,
            },
        }

    # Use the longest contiguous valid run.  This avoids fabricating a large
    # heading change across a missing-data gap; ties choose the earliest run.
    runs = _boolean_runs(track.valid)
    usable_runs = [run for run in runs if run[1] - run[0] + 1 >= 2]
    if not usable_runs:
        return {
            "status": "insufficient_data",
            "heading_start_deg": None,
            "heading_end_deg": None,
            "net_heading_change_deg": None,
            "heading_change_direction": "uncertain",
            "turning_episode": {
                "status": "insufficient_data",
                "start_frame": None,
                "end_frame": None,
                "direction": "uncertain",
                "net_heading_change_deg": None,
            },
            "start_frame": None,
            "end_frame": None,
            "valid_frame_count": int(len(valid_frames)),
            "policy": {
                "source": "Track.heading on valid frames",
                "heading_change_is_behavior_semantics": False,
            },
        }
    start_frame, end_frame = max(
        usable_runs,
        key=lambda run: (run[1] - run[0] + 1, -run[0]),
    )
    raw_rad = np.asarray(track.yaw[start_frame:end_frame + 1], dtype=np.float64)
    unwrapped_rad = np.unwrap(raw_rad)
    smooth_rad = _rolling_nanmedian(unwrapped_rad, PHYS_HEADING_MOTION_SMOOTH_RADIUS)
    smooth_deg = np.degrees(smooth_rad)
    raw_deg = np.degrees(unwrapped_rad)
    net_change = float(raw_deg[-1] - raw_deg[0])

    if net_change > PHYS_HEADING_EPISODE_MIN_NET_CHANGE_DEG:
        overall_direction = "increasing_ccw"
    elif net_change < -PHYS_HEADING_EPISODE_MIN_NET_CHANGE_DEG:
        overall_direction = "decreasing_cw"
    else:
        overall_direction = "roughly_stable"

    increments = np.diff(smooth_deg)
    episodes: List[Dict[str, Any]] = []
    current_sign: Optional[int] = None
    current_start: Optional[int] = None

    def flush_episode(last_increment_index: int) -> None:
        nonlocal current_sign, current_start
        if current_sign is None or current_start is None:
            return
        episode_start = start_frame + current_start
        episode_end = start_frame + last_increment_index + 1
        episode_change = float(smooth_deg[last_increment_index + 1] - smooth_deg[current_start])
        duration = episode_end - episode_start + 1
        if (
            duration >= PHYS_HEADING_EPISODE_MIN_DURATION_FRAMES
            and abs(episode_change) >= PHYS_HEADING_EPISODE_MIN_NET_CHANGE_DEG
        ):
            episodes.append({
                "status": "available",
                "start_frame": int(episode_start),
                "end_frame": int(episode_end),
                "duration_frames": int(duration),
                "direction": "increasing_ccw" if current_sign > 0 else "decreasing_cw",
                "net_heading_change_deg": _round(episode_change, 3),
            })
        current_sign = None
        current_start = None

    for idx, increment in enumerate(increments):
        sign = 0
        if increment >= PHYS_HEADING_EPISODE_MIN_RATE_DEG_PER_FRAME:
            sign = 1
        elif increment <= -PHYS_HEADING_EPISODE_MIN_RATE_DEG_PER_FRAME:
            sign = -1
        if sign == 0:
            flush_episode(idx - 1)
            continue
        if current_sign is None:
            current_sign = sign
            current_start = idx
        elif sign != current_sign:
            flush_episode(idx - 1)
            current_sign = sign
            current_start = idx
    flush_episode(len(increments) - 1)

    strongest_episode = None
    if episodes:
        strongest_episode = max(episodes, key=lambda item: abs(float(item["net_heading_change_deg"])))
    return {
        "status": "available",
        "heading_start_deg": _round(float(_normalize_angle(raw_rad[0]) * 180.0 / np.pi), 3),
        "heading_end_deg": _round(float(_normalize_angle(raw_rad[-1]) * 180.0 / np.pi), 3),
        "net_heading_change_deg": _round(net_change, 3),
        "heading_change_direction": overall_direction,
        "turning_episode": strongest_episode or {
            "status": "no_clear_heading_change_episode",
            "start_frame": None,
            "end_frame": None,
            "direction": "uncertain",
            "net_heading_change_deg": None,
        },
        "turning_episodes": episodes,
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
        "valid_frame_count": int(len(valid_frames)),
        "policy": {
            "source": "Track.heading on the longest contiguous valid run",
            "unwrap_before_smoothing": True,
            "smoothing_radius_frames": PHYS_HEADING_MOTION_SMOOTH_RADIUS,
            "episode_min_rate_deg_per_frame": PHYS_HEADING_EPISODE_MIN_RATE_DEG_PER_FRAME,
            "episode_min_duration_frames": PHYS_HEADING_EPISODE_MIN_DURATION_FRAMES,
            "episode_min_net_change_deg": PHYS_HEADING_EPISODE_MIN_NET_CHANGE_DEG,
            "positive_heading_direction": "increasing_ccw",
            "negative_heading_direction": "decreasing_cw",
            "heading_change_is_behavior_semantics": False,
            "turning_episode_is_geometry_only": True,
        },
    }


def _unit_heading_from_yaw(yaw: float) -> np.ndarray:
    return np.array([math.cos(float(yaw)), math.sin(float(yaw))], dtype=np.float64)


def _pair_common_direction(
    a: AgentTrack,
    b: AgentTrack,
    a_lane: LaneTimeline,
    b_lane: LaneTimeline,
    lane_map: LaneMap,
    frame: int,
) -> Tuple[Optional[np.ndarray], Optional[float]]:
    if not (a.valid[frame] and b.valid[frame]):
        return None, None
    diff = abs(float(_normalize_angle(a.yaw[frame] - b.yaw[frame])))
    diff_deg = math.degrees(diff)
    if diff_deg > PHYS_PAIR_SAME_DIRECTION_MAX_DEG:
        return None, diff_deg

    la = _local_lane_frame(a, a_lane, lane_map, frame)
    lb = _local_lane_frame(b, b_lane, lane_map, frame)
    if la is not None and lb is not None:
        ha = np.asarray(la["tangent_xy"], dtype=np.float64)
        hb = np.asarray(lb["tangent_xy"], dtype=np.float64)
        if float(np.dot(ha, hb)) < 0.0:
            hb = -hb
        h = ha + hb
    else:
        ha = _unit_heading_from_yaw(a.yaw[frame])
        hb = _unit_heading_from_yaw(b.yaw[frame])
        if float(np.dot(ha, hb)) < 0.0:
            hb = -hb
        h = ha + hb

    norm = float(np.linalg.norm(h))
    if norm <= 1e-8:
        return None, diff_deg
    return h / norm, diff_deg


def pair_frame_physical_relation(
    a: AgentTrack,
    b: AgentTrack,
    a_lane: LaneTimeline,
    b_lane: LaneTimeline,
    lane_map: LaneMap,
    frame: int,
) -> Optional[Dict[str, Any]]:
    if not (0 <= frame < a.T and 0 <= frame < b.T):
        return None
    if not (a.valid[frame] and b.valid[frame]):
        return None

    h, heading_diff_deg = _pair_common_direction(
        a, b, a_lane, b_lane, lane_map, frame
    )
    delta_b_from_a = b.xy[frame] - a.xy[frame]
    distance = float(np.linalg.norm(delta_b_from_a))

    if h is None:
        return {
            "frame": int(frame),
            "distance_m": _round(distance, 3),
            "heading_difference_deg": _round(heading_diff_deg, 2),
            "travel_channel_relation": "uncertain",
            "longitudinal_relation": "uncertain",
            "longitudinal_gap_m": None,
            "lateral_gap_m": None,
            "agent_A_continuous_lane_id": lane_group_at(a_lane, frame, radius=1),
            "agent_B_continuous_lane_id": lane_group_at(b_lane, frame, radius=1),
        }

    long_b = float(np.dot(delta_b_from_a, h))
    lat_b = float(_cross2d(h, delta_b_from_a))
    abs_lat = abs(lat_b)
    ga = lane_group_at(a_lane, frame, radius=1)
    gb = lane_group_at(b_lane, frame, radius=1)

    if ga is not None and gb is not None and ga == gb:
        channel = (
            "same_travel_channel"
            if abs_lat <= PHYS_CHANNEL_SAME_GROUP_MAX_LATERAL_M
            else "uncertain"
        )
    elif ga is not None and gb is not None and ga != gb:
        if abs_lat >= PHYS_CHANNEL_DIFFERENT_LATERAL_M:
            channel = "different_travel_channel"
        else:
            # Map and geometry do not agree strongly enough: do not guess.
            channel = "uncertain"
    else:
        if abs_lat <= PHYS_CHANNEL_SAME_LATERAL_M:
            channel = "same_travel_channel"
        elif abs_lat >= PHYS_CHANNEL_DIFFERENT_LATERAL_M:
            channel = "different_travel_channel"
        else:
            channel = "uncertain"

    if long_b > PHYS_LONGITUDINAL_DEADBAND_M:
        longitudinal = "agent_B_ahead_of_agent_A"
    elif long_b < -PHYS_LONGITUDINAL_DEADBAND_M:
        longitudinal = "agent_A_ahead_of_agent_B"
    else:
        longitudinal = "side_by_side_or_overlap"

    return {
        "frame": int(frame),
        "distance_m": _round(distance, 3),
        "heading_difference_deg": _round(heading_diff_deg, 2),
        "travel_channel_relation": channel,
        "longitudinal_relation": longitudinal,
        "longitudinal_gap_m": _round(long_b, 3),
        "lateral_gap_m": _round(lat_b, 3),
        "agent_A_continuous_lane_id": int(ga) if ga is not None else None,
        "agent_B_continuous_lane_id": int(gb) if gb is not None else None,
    }


def extract_pair_relation_facts(
    a: AgentTrack,
    b: AgentTrack,
    a_lane: LaneTimeline,
    b_lane: LaneTimeline,
    lane_map: LaneMap,
) -> Dict[str, Any]:
    per_frame: List[Dict[str, Any]] = []
    for f in range(min(a.T, b.T)):
        r = pair_frame_physical_relation(a, b, a_lane, b_lane, lane_map, f)
        if r is not None:
            per_frame.append(r)

    channel_pairs = [(r["frame"], r["travel_channel_relation"]) for r in per_frame]
    long_pairs = [(r["frame"], r["longitudinal_relation"]) for r in per_frame]
    channel_values = [v for _, v in channel_pairs]
    long_values = [v for _, v in long_pairs]

    return {
        "common_valid_frame_count": int(len(per_frame)),
        "travel_channel": {
            **_dominant_categorical(channel_values),
            "runs": _categorical_runs(channel_pairs),
            "policy": (
                "same/different uses conservative continuous-lane matching plus "
                "same-direction pair geometry; disagreement becomes uncertain"
            ),
        },
        "longitudinal": {
            **_dominant_categorical(long_values),
            "runs": _categorical_runs(long_pairs),
            "policy": (
                "ahead/behind uses projection onto a common local travel direction; "
                "cross-direction frames become uncertain"
            ),
        },
        "per_frame": per_frame,
    }


def _heading_relation_from_difference(heading_difference_deg: Optional[float]) -> str:
    """Map absolute heading difference to a conservative geometry relation."""
    if heading_difference_deg is None or not np.isfinite(float(heading_difference_deg)):
        return "uncertain"
    value = abs(float(heading_difference_deg))
    if value <= PHYS_HEADING_ALIGNED_MAX_DEG:
        return "aligned"
    if value <= PHYS_HEADING_ROUGHLY_ALIGNED_MAX_DEG:
        return "roughly_aligned"
    if value >= PHYS_HEADING_OPPOSING_MIN_DEG:
        return "opposing"
    return "cross_direction"


def _heading_relation_facts(a: AgentTrack, b: AgentTrack) -> Dict[str, Any]:
    frame_values: List[Tuple[int, str]] = []
    differences: List[float] = []
    for frame in range(min(a.T, b.T)):
        if not (a.valid[frame] and b.valid[frame]):
            continue
        diff_rad = abs(float(_normalize_angle(a.yaw[frame] - b.yaw[frame])))
        diff_deg = math.degrees(diff_rad)
        if not np.isfinite(diff_deg):
            frame_values.append((frame, "uncertain"))
            continue
        differences.append(float(diff_deg))
        frame_values.append((frame, _heading_relation_from_difference(diff_deg)))

    summary = {
        "median": _round(float(np.median(differences)), 3) if differences else None,
        "min": _round(float(np.min(differences)), 3) if differences else None,
        "max": _round(float(np.max(differences)), 3) if differences else None,
    }
    result = _dominant_categorical([value for _, value in frame_values])
    if len(frame_values) < PHYS_HEADING_MIN_COMMON_FRAMES:
        # A single jointly valid frame is not enough for a temporal relation.
        # Keep the observed frame accounting, but expose the conservative
        # result as uncertain/insufficient rather than a stable label.
        result["dominant"] = None
        result["dominance_fraction_among_known"] = None
    result.update({
        "status": (
            "available"
            if len(frame_values) >= PHYS_HEADING_MIN_COMMON_FRAMES
            else "insufficient_data"
        ),
        "runs": _categorical_runs(frame_values),
        "heading_difference_summary_deg": summary,
        "policy": {
            "aligned_max_deg": PHYS_HEADING_ALIGNED_MAX_DEG,
            "roughly_aligned_max_deg": PHYS_HEADING_ROUGHLY_ALIGNED_MAX_DEG,
            "opposing_min_deg": PHYS_HEADING_OPPOSING_MIN_DEG,
            "minimum_common_valid_frames": PHYS_HEADING_MIN_COMMON_FRAMES,
            "cross_direction_interval_deg": [
                PHYS_HEADING_ROUGHLY_ALIGNED_MAX_DEG,
                PHYS_HEADING_OPPOSING_MIN_DEG,
            ],
            "source": "absolute wrapped Track.heading difference on jointly valid frames",
            "behavior_semantics_emitted": False,
        },
    })
    return result


def _distance_samples(
    a: AgentTrack,
    b: AgentTrack,
    pair_relation: Mapping[str, Any],
) -> List[Tuple[int, float]]:
    """Reuse pair_relation distances rather than serializing another timeline."""
    samples: List[Tuple[int, float]] = []
    per_frame = pair_relation.get("per_frame", [])
    if isinstance(per_frame, list):
        for item in per_frame:
            if not isinstance(item, Mapping):
                continue
            try:
                frame = int(item["frame"])
                distance = float(item["distance_m"])
            except (KeyError, TypeError, ValueError):
                continue
            if 0 <= frame < min(a.T, b.T) and np.isfinite(distance):
                samples.append((frame, distance))
    return sorted(samples, key=lambda item: item[0])


def _compute_exact_closest_approach(
    pair_relation: Mapping[str, Any],
    timestamps: np.ndarray,
) -> Dict[str, Any]:
    """Return the exact argmin over existing per-frame distances.

    The pair timeline is the source of truth here.  No tolerance is applied to
    move the selected frame; Python's stable minimum selection keeps the first
    frame when exact equal minima occur.
    """
    candidates: List[Tuple[int, float]] = []
    per_frame = pair_relation.get("per_frame", [])
    if isinstance(per_frame, list):
        for item in per_frame:
            if not isinstance(item, Mapping):
                continue
            distance_value = item.get("distance_m")
            if distance_value is None:
                continue
            try:
                frame = int(item["frame"])
                distance = float(distance_value)
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(distance):
                candidates.append((frame, distance))

    if not candidates:
        return {
            "status": "unavailable",
            "frame": None,
            "timestamp_s": None,
            "distance_m": None,
            "minimum_distance_m": None,
            "common_valid_frame_count": 0,
            "policy": {
                "selection": "exact minimum distance and stable argmin order",
                "behavior_semantics_emitted": False,
            },
        }

    selected_frame, selected_distance = min(candidates, key=lambda item: item[1])
    timestamp = float(timestamps[selected_frame]) if selected_frame < len(timestamps) else None
    return {
        "status": "available",
        "frame": int(selected_frame),
        "timestamp_s": _round(timestamp, 4),
        "distance_m": _round(selected_distance, 3),
        "minimum_distance_m": _round(selected_distance, 3),
        "common_valid_frame_count": int(len(candidates)),
        "policy": {
            "selection": "exact minimum distance and stable argmin order",
            "behavior_semantics_emitted": False,
        },
    }


def _distance_window_facts(
    samples: Sequence[Tuple[int, float]],
    label: str,
) -> Dict[str, Any]:
    base = {
        "status": "uncertain",
        "label": "uncertain",
        "start_frame": None,
        "end_frame": None,
        "distance_start_m": None,
        "distance_end_m": None,
        "delta_m": None,
        "num_points": int(len(samples)),
    }
    if len(samples) < PHYS_DISTANCE_EVOLUTION_MIN_POINTS:
        base["status"] = "insufficient_data"
        return base

    edge_points = min(PHYS_DISTANCE_EVOLUTION_EDGE_POINTS, len(samples) // 2)
    start_values = [distance for _, distance in samples[:edge_points]]
    end_values = [distance for _, distance in samples[-edge_points:]]
    start_distance = float(np.median(start_values))
    end_distance = float(np.median(end_values))
    delta = end_distance - start_distance
    if delta <= -PHYS_DISTANCE_EVOLUTION_DELTA_THRESHOLD_M:
        direction = "approaching"
    elif delta >= PHYS_DISTANCE_EVOLUTION_DELTA_THRESHOLD_M:
        direction = "separating"
    else:
        direction = "roughly_stable"
    base.update({
        "status": "available",
        "label": direction,
        "start_frame": int(samples[0][0]),
        "end_frame": int(samples[-1][0]),
        "distance_start_m": _round(start_distance, 3),
        "distance_end_m": _round(end_distance, 3),
        "delta_m": _round(delta, 3),
    })
    return base


def _distance_evolution_facts(
    samples: Sequence[Tuple[int, float]],
    closest_frame: Optional[int],
) -> Dict[str, Any]:
    if not samples or closest_frame is None:
        return {
            "status": "insufficient_data",
            "overall": "uncertain",
            "before_closest": {"label": "uncertain", "status": "insufficient_data"},
            "after_closest": {"label": "uncertain", "status": "insufficient_data"},
            "pre_window": None,
            "post_window": None,
            "policy": {
                "window_frames": PHYS_DISTANCE_EVOLUTION_WINDOW_FRAMES,
                "minimum_points": PHYS_DISTANCE_EVOLUTION_MIN_POINTS,
                "edge_points_for_robust_delta": PHYS_DISTANCE_EVOLUTION_EDGE_POINTS,
                "delta_threshold_m": PHYS_DISTANCE_EVOLUTION_DELTA_THRESHOLD_M,
                "method": "median of edge points in stable windows around closest approach",
                "behavior_semantics_emitted": False,
            },
        }

    before = [item for item in samples if item[0] < closest_frame]
    after = [item for item in samples if item[0] > closest_frame]
    before_window = before[-PHYS_DISTANCE_EVOLUTION_WINDOW_FRAMES:]
    after_window = after[:PHYS_DISTANCE_EVOLUTION_WINDOW_FRAMES]
    before_facts = _distance_window_facts(before_window, "before_closest")
    after_facts = _distance_window_facts(after_window, "after_closest")
    before_label = before_facts["label"]
    after_label = after_facts["label"]

    if before_label == "approaching" and after_label == "separating":
        overall = "approaching_then_separating"
    elif before_label == "approaching":
        overall = "approaching_only"
    elif after_label == "separating":
        overall = "separating_only"
    elif before_label == "roughly_stable" and after_label == "roughly_stable":
        overall = "roughly_stable"
    else:
        overall = "uncertain"

    return {
        "status": "available" if before_facts["status"] == "available" or after_facts["status"] == "available" else "insufficient_data",
        "overall": overall,
        "before_closest": before_facts,
        "after_closest": after_facts,
        "pre_window": before_facts,
        "post_window": after_facts,
        "policy": {
            "window_frames": PHYS_DISTANCE_EVOLUTION_WINDOW_FRAMES,
            "minimum_points": PHYS_DISTANCE_EVOLUTION_MIN_POINTS,
            "edge_points_for_robust_delta": PHYS_DISTANCE_EVOLUTION_EDGE_POINTS,
            "delta_threshold_m": PHYS_DISTANCE_EVOLUTION_DELTA_THRESHOLD_M,
            "method": "median of edge points in stable windows around closest approach",
            "behavior_semantics_emitted": False,
        },
    }


def _point_segment_closest_points(
    point: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    segment = end - start
    denom = float(np.dot(segment, segment))
    if denom <= 1e-12:
        closest = start.copy()
    else:
        ratio = float(np.dot(point - start, segment) / denom)
        closest = start + np.clip(ratio, 0.0, 1.0) * segment
    return point.copy(), closest, float(np.linalg.norm(point - closest))


def _segment_closest_points(
    a0: np.ndarray,
    a1: np.ndarray,
    b0: np.ndarray,
    b1: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Closest points between two 2-D line segments."""
    da = a1 - a0
    db = b1 - b0
    denom = float(_cross2d(da, db))
    if abs(denom) > 1e-12:
        offset = b0 - a0
        ta = float(_cross2d(offset, db) / denom)
        tb = float(_cross2d(offset, da) / denom)
        if 0.0 <= ta <= 1.0 and 0.0 <= tb <= 1.0:
            return a0 + ta * da, b0 + tb * db, 0.0

    candidates = [
        _point_segment_closest_points(a0, b0, b1),
        _point_segment_closest_points(a1, b0, b1),
    ]
    p, q, d = min(candidates, key=lambda item: item[2])
    reverse_candidates = [
        _point_segment_closest_points(b0, a0, a1),
        _point_segment_closest_points(b1, a0, a1),
    ]
    q2, p2, d2 = min(reverse_candidates, key=lambda item: item[2])
    if d2 < d:
        return p2, q2, d2
    return p, q, d


def _compute_path_geometry(
    a: AgentTrack,
    b: AgentTrack,
) -> Dict[str, Any]:
    """Compare the two jointly observed trajectory polylines."""
    if not hasattr(a, "xy") or not hasattr(b, "xy"):
        return {
            "status": "insufficient_data",
            "min_path_distance_m": None,
            "closest_path_points": None,
            "spatial_overlap": "uncertain",
            "convergence_status": "uncertain",
            "common_valid_frame_count": 0,
            "policy": {
                "downsample_step_frames": PHYS_PATH_DOWNSAMPLE_STEP,
                "behavior_semantics_emitted": False,
            },
        }
    common_frames = [
        frame
        for frame in range(min(a.T, b.T))
        if a.valid[frame] and b.valid[frame]
        and np.all(np.isfinite(a.xy[frame]))
        and np.all(np.isfinite(b.xy[frame]))
    ]
    if len(common_frames) < PHYS_PATH_MIN_POINTS:
        return {
            "status": "insufficient_data",
            "min_path_distance_m": None,
            "closest_path_points": None,
            "spatial_overlap": "uncertain",
            "convergence_status": "uncertain",
            "common_valid_frame_count": int(len(common_frames)),
            "policy": {
                "downsample_step_frames": PHYS_PATH_DOWNSAMPLE_STEP,
                "behavior_semantics_emitted": False,
            },
        }

    sampled_frames = common_frames[::PHYS_PATH_DOWNSAMPLE_STEP]
    if sampled_frames[-1] != common_frames[-1]:
        sampled_frames.append(common_frames[-1])
    a_points = np.asarray([a.xy[f] for f in sampled_frames], dtype=np.float64)
    b_points = np.asarray([b.xy[f] for f in sampled_frames], dtype=np.float64)

    best_a = None
    best_b = None
    best_distance = float("inf")
    for i in range(len(a_points) - 1):
        for j in range(len(b_points) - 1):
            point_a, point_b, distance = _segment_closest_points(
                a_points[i], a_points[i + 1], b_points[j], b_points[j + 1]
            )
            if distance < best_distance:
                best_a, best_b, best_distance = point_a, point_b, distance

    # Degenerate two-point trajectories still have a well-defined point
    # distance, even if no segment was created.
    if best_a is None:
        distances = np.linalg.norm(a_points[:, None, :] - b_points[None, :, :], axis=2)
        i, j = np.unravel_index(int(np.argmin(distances)), distances.shape)
        best_a, best_b = a_points[i], b_points[j]
        best_distance = float(distances[i, j])

    edge_count = min(PHYS_PATH_EVOLUTION_EDGE_POINTS, len(sampled_frames) // 2)
    start_distance = np.linalg.norm(a_points[:edge_count] - b_points[:edge_count], axis=1)
    end_distance = np.linalg.norm(a_points[-edge_count:] - b_points[-edge_count:], axis=1)
    delta = float(np.median(end_distance) - np.median(start_distance))
    if delta <= -PHYS_PATH_EVOLUTION_DELTA_THRESHOLD_M:
        convergence_status = "paths_converge"
    elif delta >= PHYS_PATH_EVOLUTION_DELTA_THRESHOLD_M:
        convergence_status = "paths_diverge"
    else:
        convergence_status = "uncertain"

    return {
        "status": "available",
        "min_path_distance_m": _round(best_distance, 3),
        "closest_path_points": {
            "agent_A": {"x": _round(best_a[0], 3), "y": _round(best_a[1], 3)},
            "agent_B": {"x": _round(best_b[0], 3), "y": _round(best_b[1], 3)},
        },
        "spatial_overlap": (
            "paths_have_spatial_overlap"
            if best_distance <= PHYS_PATH_SPATIAL_OVERLAP_DISTANCE_M
            else "no_clear_overlap"
        ),
        "convergence_status": convergence_status,
        "common_valid_frame_count": int(len(common_frames)),
        "sampled_point_count": int(len(sampled_frames)),
        "start_distance_m": _round(float(np.median(start_distance)), 3),
        "end_distance_m": _round(float(np.median(end_distance)), 3),
        "delta_m": _round(delta, 3),
        "policy": {
            "downsample_step_frames": PHYS_PATH_DOWNSAMPLE_STEP,
            "spatial_overlap_distance_m": PHYS_PATH_SPATIAL_OVERLAP_DISTANCE_M,
            "edge_points_for_robust_delta": PHYS_PATH_EVOLUTION_EDGE_POINTS,
            "delta_threshold_m": PHYS_PATH_EVOLUTION_DELTA_THRESHOLD_M,
            "method": "minimum distance between downsampled 2-D trajectory segments",
            "behavior_semantics_emitted": False,
        },
    }


def _lane_heading_at_frame(
    track: AgentTrack,
    lane_timeline: LaneTimeline,
    lane_map: LaneMap,
    frame: int,
) -> Optional[float]:
    if frame < 0 or frame >= track.T:
        return None
    match = lane_timeline.matches[frame]
    if not match.available or match.best_lane_id not in lane_map.segments:
        return None
    segment = lane_map.segments[match.best_lane_id]
    index = int(np.argmin(np.linalg.norm(segment.xy - track.xy[frame], axis=1)))
    tangent = segment.tangent_xy[index]
    return _round(math.degrees(math.atan2(float(tangent[1]), float(tangent[0]))) % 360.0, 3)


def _lane_heading_window_median(
    track: AgentTrack,
    lane_timeline: LaneTimeline,
    lane_map: LaneMap,
    start_frame: int,
    end_frame: int,
    expected_group: int,
    reverse: bool = False,
) -> Optional[float]:
    """Circular median of reliable lane tangents within one stable run."""
    frames = list(range(start_frame, end_frame + 1))
    if reverse:
        frames.reverse()
    frames = frames[:PHYS_ROUTE_HEADING_WINDOW_FRAMES]
    headings = []
    for frame in frames:
        if frame < 0 or frame >= track.T or not track.valid[frame]:
            continue
        if lane_timeline.stabilized_group[frame] != expected_group:
            continue
        match = lane_timeline.matches[frame]
        if (
            not match.available
            or match.best_group_id != expected_group
            or (match.confidence or 0.0) < LANE_CONFIDENCE_MIN
        ):
            continue
        heading = _lane_heading_at_frame(track, lane_timeline, lane_map, frame)
        if heading is not None:
            headings.append(float(heading))
    if not headings:
        return None

    # Unwrap around the first reliable heading before taking the median, so
    # headings such as 359° and 1° remain neighbours around 0°.
    reference = headings[0]
    unwrapped = [
        reference + math.degrees(
            float(_normalize_angle(math.radians(heading - reference)))
        )
        for heading in headings
    ]
    return _round(float(np.median(unwrapped)) % 360.0, 3)


def extract_turn_maneuver_facts(
    track: AgentTrack,
    lane_timeline: LaneTimeline,
    lane_map: LaneMap,
    heading_motion: Mapping[str, Any],
    physical_lane_chain: Mapping[str, Any],
) -> Dict[str, Any]:
    """Classify a turn only when track and lane-path evidence agree.

    This is intentionally separate from ``route_transition``.  A route
    transition describes map-declared continuation; this function evaluates
    the direction change along the reliable lane sequence actually traversed
    by the vehicle.  A lane-ID or continuous-lane-group change is never used
    as a turn decision by itself.
    """
    result: Dict[str, Any] = {
        "status": "turn_uncertain",
        "candidate_direction": "uncertain",
        "track_heading_change_deg": None,
        "lane_path_heading_change_deg": None,
        "track_direction": "uncertain",
        "lane_path_direction": "uncertain",
        "direction_agreement": False,
        "intersection_like_topology": False,
        "entry": None,
        "exit": None,
        "evidence": [],
        "reasons": [],
        "policy": {
            "min_lane_heading_change_deg": PHYS_TURN_MIN_LANE_HEADING_CHANGE_DEG,
            "min_track_heading_change_deg": PHYS_TURN_MIN_TRACK_HEADING_CHANGE_DEG,
            "direction_agreement_required": PHYS_TURN_DIRECTION_AGREEMENT,
            "min_entry_exit_frames": PHYS_TURN_MIN_ENTRY_EXIT_FRAMES,
            "heading_source": "heading_motion.net_heading_change_deg",
            "lane_heading_source": "entry/exit tangents on stable lane sequence",
            "route_transition_used_as_turn_gate": False,
        },
    }

    quality = extract_trajectory_quality(track)
    result["trajectory_quality_status"] = quality.get("status")
    if quality.get("status") == "suspicious":
        result["reasons"].append("trajectory_quality_suspicious")

    if not isinstance(heading_motion, Mapping):
        result["reasons"].append("heading_motion_unavailable")
        return result
    track_delta = heading_motion.get("net_heading_change_deg")
    if track_delta is None:
        result["reasons"].append("track_heading_change_unavailable")
        return result
    try:
        track_delta = float(track_delta)
    except (TypeError, ValueError):
        result["reasons"].append("track_heading_change_unavailable")
        return result
    if not np.isfinite(track_delta):
        result["reasons"].append("track_heading_change_unavailable")
        return result
    result["track_heading_change_deg"] = _round(track_delta, 3)
    track_supported = abs(track_delta) >= PHYS_TURN_MIN_TRACK_HEADING_CHANGE_DEG
    if track_supported:
        result["track_direction"] = "left" if track_delta > 0 else "right"
        result["candidate_direction"] = result["track_direction"]
        result["evidence"].append("sustained_track_heading_change")
    else:
        result["reasons"].append("track_heading_change_below_turn_threshold")

    if not isinstance(physical_lane_chain, Mapping):
        result["reasons"].append("physical_lane_chain_unavailable")
        return result
    # Use the reliable lane sequence as the primary source.  The compact
    # stable sequence is only a compatibility fallback for older records.
    lane_runs = physical_lane_chain.get("lane_segment_sequence")
    if not isinstance(lane_runs, list) or not lane_runs:
        lane_runs = physical_lane_chain.get("stable_lane_segment_sequence")
    lane_runs = [run for run in (lane_runs or []) if isinstance(run, Mapping)]
    if not lane_runs:
        result["reasons"].append("reliable_lane_sequence_unavailable")
        return result

    lane_runs = sorted(
        lane_runs,
        key=lambda run: (
            int(run.get("start_frame", 0)),
            int(run.get("end_frame", 0)),
        ),
    )

    turning_episode = heading_motion.get("turning_episode")
    turn_start = None
    turn_end = None
    if isinstance(turning_episode, Mapping):
        try:
            candidate_start = int(turning_episode["start_frame"])
            candidate_end = int(turning_episode["end_frame"])
            if candidate_start <= candidate_end:
                turn_start = candidate_start
                turn_end = candidate_end
                result["turning_episode"] = {
                    "start_frame": candidate_start,
                    "end_frame": candidate_end,
                    "source": "heading_motion.turning_episode",
                }
        except (KeyError, TypeError, ValueError):
            pass

    def _run_bounds(run: Mapping[str, Any]) -> Optional[Tuple[int, int]]:
        try:
            start = int(run["start_frame"])
            end = int(run["end_frame"])
        except (KeyError, TypeError, ValueError):
            return None
        if start > end:
            return None
        return start, end

    def _run_duration(run: Mapping[str, Any]) -> int:
        bounds = _run_bounds(run)
        if bounds is None:
            return 0
        start, end = bounds
        return end - start + 1

    entry_run = None
    exit_run = None
    entry_selection = "full_lane_sequence_start"
    exit_selection = "full_lane_sequence_end"
    if turn_start is not None and turn_end is not None:
        eligible = [
            run
            for run in lane_runs
            if _run_duration(run) >= PHYS_TURN_MIN_ENTRY_EXIT_FRAMES
        ]
        before_turn = [
            run
            for run in eligible
            if _run_bounds(run)[1] < turn_start
        ]
        after_turn = [
            run
            for run in eligible
            if _run_bounds(run)[0] > turn_end
        ]
        entry_run = max(
            before_turn,
            key=lambda run: _run_bounds(run)[1],
            default=None,
        )
        exit_run = min(
            after_turn,
            key=lambda run: _run_bounds(run)[0],
            default=None,
        )

        # If the turn reaches the observation boundary, use the last stable
        # run covering the end of the turning episode as a partial exit.  This
        # preserves cases such as a turn that is still completing at frame T.
        if exit_run is None:
            boundary_exit = [
                run
                for run in eligible
                if _run_bounds(run)[0] <= turn_end <= _run_bounds(run)[1]
            ]
            exit_run = max(
                boundary_exit,
                key=lambda run: _run_bounds(run)[1],
                default=None,
            )
            if exit_run is not None:
                exit_selection = "turning_episode_boundary_exit"

        # If the final lane run is shorter than the minimum dwell but the
        # turn continues to the observation boundary, use the last stable run
        # that overlaps the tail of the turning episode.  This avoids falling
        # back to a one-to-four-frame terminal lane when a preceding stable
        # run still provides a reliable exit tangent.
        if exit_run is None:
            partial_tail_exit = [
                run
                for run in eligible
                if _run_bounds(run)[1] >= turn_start
                and _run_bounds(run)[0] > turn_start
            ]
            exit_run = max(
                partial_tail_exit,
                key=lambda run: _run_bounds(run)[1],
                default=None,
            )
            if exit_run is not None:
                exit_selection = "turning_episode_partial_tail_exit"

        if entry_run is not None:
            entry_selection = "stable_run_before_turning_episode"
        if exit_run is not None and exit_selection == "full_lane_sequence_end":
            exit_selection = "stable_run_after_turning_episode"

    if entry_run is None:
        entry_run = lane_runs[0]
    if exit_run is None:
        exit_run = lane_runs[-1]

    entry_duration = int(entry_run.get("duration_frames", 0))
    exit_duration = int(exit_run.get("duration_frames", 0))
    result["entry"] = {
        "lane_id": entry_run.get("lane_id"),
        "start_frame": entry_run.get("start_frame"),
        "end_frame": entry_run.get("end_frame"),
        "duration_frames": entry_duration,
    }
    result["exit"] = {
        "lane_id": exit_run.get("lane_id"),
        "start_frame": exit_run.get("start_frame"),
        "end_frame": exit_run.get("end_frame"),
        "duration_frames": exit_duration,
    }
    result["entry"]["selection"] = entry_selection
    result["exit"]["selection"] = exit_selection
    if (
        entry_duration < PHYS_TURN_MIN_ENTRY_EXIT_FRAMES
        or exit_duration < PHYS_TURN_MIN_ENTRY_EXIT_FRAMES
    ):
        result["reasons"].append("entry_or_exit_lane_run_too_short")
        return result

    try:
        entry_lane_id = int(entry_run["lane_id"])
        exit_lane_id = int(exit_run["lane_id"])
        entry_group = int(lane_map.segment_to_group[entry_lane_id])
        exit_group = int(lane_map.segment_to_group[exit_lane_id])
    except (KeyError, TypeError, ValueError):
        result["reasons"].append("entry_or_exit_lane_group_unavailable")
        return result

    entry_heading = _lane_heading_window_median(
        track,
        lane_timeline,
        lane_map,
        int(entry_run["start_frame"]),
        int(entry_run["end_frame"]),
        entry_group,
        reverse=False,
    )
    exit_heading = _lane_heading_window_median(
        track,
        lane_timeline,
        lane_map,
        int(exit_run["start_frame"]),
        int(exit_run["end_frame"]),
        exit_group,
        reverse=True,
    )
    result["entry"]["heading_deg"] = entry_heading
    result["exit"]["heading_deg"] = exit_heading
    if entry_heading is None or exit_heading is None:
        result["reasons"].append("entry_or_exit_lane_heading_unavailable")
        return result

    lane_delta = math.degrees(float(_normalize_angle(
        math.radians(float(exit_heading) - float(entry_heading))
    )))
    result["lane_path_heading_change_deg"] = _round(lane_delta, 3)
    lane_supported = abs(lane_delta) >= PHYS_TURN_MIN_LANE_HEADING_CHANGE_DEG
    if lane_supported:
        result["lane_path_direction"] = "left" if lane_delta > 0 else "right"
        result["evidence"].append("sustained_lane_path_heading_change")
    else:
        result["reasons"].append("lane_path_heading_change_below_turn_threshold")

    # Use branching in the lane graph near the entry/exit portions of the
    # traversed path.  This avoids treating an isolated highway bend as a
    # confirmed intersection turn.
    anchor_points = []
    for run, take_start in ((entry_run, True), (exit_run, False)):
        lane_id = int(run["lane_id"])
        segment = lane_map.segments.get(lane_id)
        if segment is None or len(segment.xy) == 0:
            continue
        points = segment.xy[:8] if take_start else segment.xy[-8:]
        anchor_points.append(np.asarray(points, dtype=np.float64))
    nearby_lane_ids = set()
    if anchor_points:
        anchors = np.concatenate(anchor_points, axis=0)
        for lane_id, segment in lane_map.segments.items():
            if _point_distance_to_positions(segment.xy, anchors) > PHYS_MAP_INTERSECTION_DISTANCE_M:
                continue
            nearby_lane_ids.add(int(lane_id))
    branching_lane_ids = []
    for lane_id in sorted(nearby_lane_ids):
        segment = lane_map.segments[lane_id]
        valid_entries = [x for x in segment.entry_lanes if x in lane_map.segments]
        valid_exits = [x for x in segment.exit_lanes if x in lane_map.segments]
        if len(set(valid_entries)) > 1 or len(set(valid_exits)) > 1:
            branching_lane_ids.append(lane_id)
    result["intersection_like_topology"] = bool(branching_lane_ids)
    result["branching_lane_ids"] = branching_lane_ids
    if result["intersection_like_topology"]:
        result["evidence"].append("intersection_like_lane_graph_branching")
    else:
        result["reasons"].append("intersection_like_topology_not_found")

    directions_agree = bool(
        track_supported
        and lane_supported
        and result["track_direction"] == result["lane_path_direction"]
    )
    result["direction_agreement"] = directions_agree
    if PHYS_TURN_DIRECTION_AGREEMENT and not directions_agree:
        result["reasons"].append("track_and_lane_directions_disagree")

    if (
        quality.get("status") != "suspicious"
        and track_supported
        and lane_supported
        and directions_agree
        and result["intersection_like_topology"]
    ):
        result["status"] = f"{result['lane_path_direction']}_turn"
        result["evidence"].append("track_lane_topology_agreement")
    return result


def _groups_connected_in_lane_graph(
    lane_map: LaneMap,
    before_group: int,
    after_group: int,
) -> bool:
    if before_group == after_group:
        return True
    for segment in lane_map.segments.values():
        group = lane_map.segment_to_group[segment.lane_id]
        if group == before_group:
            if any(
                lane_map.segment_to_group.get(next_id) == after_group
                for next_id in segment.exit_lanes
            ):
                return True
        if group == after_group:
            if any(
                lane_map.segment_to_group.get(prev_id) == before_group
                for prev_id in segment.entry_lanes
            ):
                return True
    return False


def _compute_route_transition(
    track: AgentTrack,
    lane_timeline: LaneTimeline,
    lane_map: LaneMap,
) -> Dict[str, Any]:
    """Describe map-declared continuation transitions only.

    A raw lane-ID change is no longer sufficient for this fact.  The canonical
    source is ``physical_lane_chain.topological_relations``; lateral neighbor
    transitions are kept in that new fact and are not promoted to a route
    continuation.
    """
    lane_chain = build_physical_lane_chain(track, lane_timeline, lane_map)
    transition_pairs = [
        relation for relation in lane_chain["topological_relations"]
        if relation.get("relation") == "continuation"
    ]

    base = {
        "status": "no_route_transition",
        "route_transition": False,
        "transitions": [],
        "primary_transition": None,
        # Compatibility summary fields.  They mirror primary_transition when
        # one exists, or the first observed transition otherwise.
        "lane_before": None,
        "lane_after": None,
        "incoming_lane_heading_deg": None,
        "outgoing_lane_heading_deg": None,
        "lane_heading_change_deg": None,
        "connected_in_lane_graph": None,
        "transition_frame": None,
        "policy": {
            "lane_identity": "map-declared topological continuation",
            "heading_window_frames": PHYS_ROUTE_HEADING_WINDOW_FRAMES,
            "heading_window_method": "circular median of reliable lane tangents within each stable run",
            "raw_lane_id_change_is_core_behavior": False,
            "behavior_semantics_emitted": False,
        },
    }
    if not transition_pairs:
        return base

    transitions = []
    for relation in transition_pairs:
        before_group = int(relation["from_continuous_lane_id"])
        after_group = int(relation["to_continuous_lane_id"])
        incoming_heading = _lane_heading_window_median(
            track,
            lane_timeline,
            lane_map,
            int(relation["from_frame"]),
            int(relation["transition_frame"]) - 1,
            before_group,
            reverse=True,
        )
        outgoing_heading = _lane_heading_window_median(
            track,
            lane_timeline,
            lane_map,
            int(relation["transition_frame"]),
            int(relation["to_frame"]),
            after_group,
        )
        connected = True
        heading_change = None
        if incoming_heading is not None and outgoing_heading is not None:
            heading_change = _round(
                math.degrees(
                    float(_normalize_angle(math.radians(outgoing_heading - incoming_heading)))
                ),
                3,
            )
        transitions.append({
            "transition_frame": int(relation["transition_frame"]),
            "lane_before": before_group,
            "lane_after": after_group,
            "lane_segment_before": int(relation["from_lane_id"]),
            "lane_segment_after": int(relation["to_lane_id"]),
            "incoming_lane_heading_deg": incoming_heading,
            "outgoing_lane_heading_deg": outgoing_heading,
            "lane_heading_change_deg": heading_change,
            "connected_in_lane_graph": connected,
        })

    connected_transitions = [
        item for item in transitions if item["connected_in_lane_graph"] is True
    ]
    primary = None
    if connected_transitions:
        primary = max(
            connected_transitions,
            key=lambda item: (
                abs(float(item["lane_heading_change_deg"]))
                if item["lane_heading_change_deg"] is not None else -1.0,
                -int(item["transition_frame"]),
            ),
        )
    else:
        # Preserve auditable evidence even when no transition is graph-
        # connected, but do not promote it to route_transition=true.
        primary = None
    summary = primary or transitions[0]
    has_connected = bool(connected_transitions)
    base.update({
        "status": "available" if has_connected else "unconnected_lane_transition_evidence",
        "route_transition": has_connected,
        "transitions": transitions,
        "primary_transition": primary,
        "lane_before": summary["lane_before"],
        "lane_after": summary["lane_after"],
        "incoming_lane_heading_deg": summary["incoming_lane_heading_deg"],
        "outgoing_lane_heading_deg": summary["outgoing_lane_heading_deg"],
        "lane_heading_change_deg": summary["lane_heading_change_deg"],
        "connected_in_lane_graph": summary["connected_in_lane_graph"],
        "transition_frame": summary["transition_frame"],
    })
    return base


def _point_distance_to_positions(point: np.ndarray, positions: np.ndarray) -> float:
    if len(positions) == 0:
        return float("inf")
    point_array = np.asarray(point, dtype=np.float64)
    if point_array.ndim == 1:
        point_array = point_array[None, :]
    distances = np.linalg.norm(point_array[:, None, :] - positions[None, :, :], axis=2)
    return float(np.min(distances))


def _compute_vehicle_map_context(
    scenario: Any,
    track: AgentTrack,
    lane_map: LaneMap,
) -> Dict[str, Any]:
    """Extract map context around one vehicle trajectory only."""
    return _compute_map_context(scenario, track, lane_map, other_track=None)


def _compute_map_context(
    scenario: Any,
    a: AgentTrack,
    lane_map: LaneMap,
    other_track: Optional[AgentTrack] = None,
) -> Dict[str, Any]:
    """Extract nearby map objects and lane-graph branching evidence only.

    ``other_track`` is optional so the same objective map-context extractor can
    serve both pair and vehicle-centered fact records.  It never performs pair
    relation or pair geometry inference.
    """
    position_arrays = [a.xy[a.valid]]
    if other_track is not None:
        position_arrays.append(other_track.xy[other_track.valid])
    positions = np.concatenate(position_arrays, axis=0)
    if len(positions) == 0:
        return {
            "status": "insufficient_data",
            "near_intersection": None,
            "intersection_evidence": {
                "lane_graph_branching": None,
                "nearby_crosswalk": None,
                "nearby_traffic_signal": None,
                "nearby_stop_sign": None,
            },
            "nearby_crosswalk": None,
            "nearby_stop_sign": None,
            "nearby_traffic_signal": None,
            "policy": {"behavior_semantics_emitted": False},
        }

    nearby_lane_ids = {
        int(lane_id)
        for lane_id, segment in lane_map.segments.items()
        if _point_distance_to_positions(segment.xy, positions) <= PHYS_MAP_CONTEXT_NEAR_DISTANCE_M
    }
    branching_lane_ids = []
    for lane_id in sorted(nearby_lane_ids):
        segment = lane_map.segments[lane_id]
        if _point_distance_to_positions(segment.xy, positions) > PHYS_MAP_INTERSECTION_DISTANCE_M:
            continue
        entry_count = len(set(x for x in segment.entry_lanes if x in lane_map.segments))
        exit_count = len(set(x for x in segment.exit_lanes if x in lane_map.segments))
        if entry_count > 1 or exit_count > 1:
            branching_lane_ids.append(lane_id)

    nearby_crosswalk = False
    nearby_stop_sign = False
    nearby_signal = False
    for feature in scenario.map_features:
        feature_kind = feature.WhichOneof("feature_data")
        if feature_kind == "crosswalk":
            points = np.asarray([[float(p.x), float(p.y)] for p in feature.crosswalk.polygon])
            nearby_crosswalk = nearby_crosswalk or any(
                _point_distance_to_positions(point, positions) <= PHYS_MAP_CONTEXT_NEAR_DISTANCE_M
                for point in points
            )
        elif feature_kind == "stop_sign":
            point = np.asarray([
                float(feature.stop_sign.position.x),
                float(feature.stop_sign.position.y),
            ])
            nearby_stop_sign = nearby_stop_sign or (
                _point_distance_to_positions(point, positions) <= PHYS_MAP_CONTEXT_NEAR_DISTANCE_M
            )

    for dynamic_state in getattr(scenario, "dynamic_map_states", []):
        for lane_state in dynamic_state.lane_states:
            if int(lane_state.lane) in nearby_lane_ids:
                nearby_signal = True
                break
        if nearby_signal:
            break

    lane_graph_branching = bool(branching_lane_ids)
    near_intersection = bool(
        lane_graph_branching or nearby_crosswalk or nearby_signal
    )
    return {
        "status": "available",
        "near_intersection": near_intersection,
        "intersection_evidence": {
            "lane_graph_branching": lane_graph_branching,
            "nearby_crosswalk": bool(nearby_crosswalk),
            "nearby_traffic_signal": bool(nearby_signal),
            "nearby_stop_sign": bool(nearby_stop_sign),
        },
        "nearby_crosswalk": bool(nearby_crosswalk),
        "nearby_stop_sign": bool(nearby_stop_sign),
        "nearby_traffic_signal": bool(nearby_signal),
        "evidence": {
            "nearby_lane_count": int(len(nearby_lane_ids)),
            "branching_lane_ids": branching_lane_ids,
        },
        "policy": {
            "near_distance_m": PHYS_MAP_CONTEXT_NEAR_DISTANCE_M,
            "intersection_evidence_distance_m": PHYS_MAP_INTERSECTION_DISTANCE_M,
            "near_intersection_definition": "lane graph branching, nearby crosswalk, or nearby traffic signal evidence",
            "near_intersection_logic": "lane_graph_branching OR nearby_traffic_signal OR nearby_crosswalk",
            "behavior_semantics_emitted": False,
        },
    }


def extract_pair_geometry_facts(
    a: AgentTrack,
    b: AgentTrack,
    pair_relation: Mapping[str, Any],
) -> Dict[str, Any]:
    """Return orthogonal direction/distance facts without semantic labels."""
    heading = _heading_relation_facts(a, b)
    samples = _distance_samples(a, b, pair_relation)
    closest = _compute_exact_closest_approach(pair_relation, a.timestamps)
    evolution = _distance_evolution_facts(samples, closest.get("frame"))
    path_geometry = _compute_path_geometry(a, b)
    return {
        "heading_relation": heading,
        "closest_approach": closest,
        "distance_evolution": evolution,
        "path_geometry": path_geometry,
        "policy": {
            "scope": "orthogonal pair geometry facts only",
            "source": "jointly valid Track states and existing pair_relation.per_frame distances",
            "behavior_semantics_emitted": False,
            "causal_interpretation_emitted": False,
        },
    }


def _lane_group_runs_for_audit(
    track: AgentTrack,
    lane_timeline: LaneTimeline,
) -> List[Dict[str, Any]]:
    values: List[Tuple[int, str]] = []
    for f, g in enumerate(lane_timeline.stabilized_group):
        if not track.valid[f]:
            continue
        values.append((f, "unknown" if g is None else str(int(g))))
    runs = _categorical_runs(values)
    for run in runs:
        if run["value"] != "unknown":
            run["continuous_lane_id"] = int(run.pop("value"))
        else:
            run["continuous_lane_id"] = None
            run.pop("value", None)
    return runs


def _agent_metadata(track: AgentTrack) -> Dict[str, Any]:
    valid_frames = np.flatnonzero(track.valid)
    return {
        "id": int(track.agent_id),
        "track_index": int(track.track_index),
        "object_type": int(track.object_type),
        "is_sdc": bool(track.is_sdc),
        "is_object_of_interest": bool(track.is_object_of_interest),
        "track_to_predict": bool(track.track_to_predict),
        "valid_frame_count": int(len(valid_frames)),
        "first_valid_frame": int(valid_frames[0]) if len(valid_frames) else None,
        "last_valid_frame": int(valid_frames[-1]) if len(valid_frames) else None,
    }


# =============================================================================
# Main timeline builder
# =============================================================================

def build_pair_timeline_v3(
    scenario,
    agent_a_id: int,
    agent_b_id: int,
    interhub_start: Optional[int] = None,
    interhub_end: Optional[int] = None,
    context_frames: int = DEFAULT_CONTEXT_FRAMES,
    event_half_width: int = DEFAULT_EVENT_HALF_WIDTH,
) -> Dict[str, Any]:
    """
    Build a facts-only pair record from the full WOMD scenario.

    ``context_frames`` and ``event_half_width`` are accepted only for CLI/backward
    compatibility in this facts-only revision; they do not crop the timeline.
    """
    a = extract_agent_track(scenario, agent_a_id)
    b = extract_agent_track(scenario, agent_b_id)
    if a.T != b.T:
        raise ValueError("Pair tracks have different timeline lengths")

    lane_map = build_lane_map(scenario)
    a_lane = build_lane_timeline(a, lane_map)
    b_lane = build_lane_timeline(b, lane_map)

    a_lateral = extract_physical_lateral_facts(a, a_lane, lane_map)
    b_lateral = extract_physical_lateral_facts(b, b_lane, lane_map)
    a_lane_chain = build_physical_lane_chain(a, a_lane, lane_map, a_lateral)
    b_lane_chain = build_physical_lane_chain(b, b_lane, lane_map, b_lateral)
    a_speed = extract_speed_change_facts(a)
    b_speed = extract_speed_change_facts(b)
    a_heading_motion = extract_heading_motion_facts(a)
    b_heading_motion = extract_heading_motion_facts(b)
    a_turn_maneuver = extract_turn_maneuver_facts(
        a, a_lane, lane_map, a_heading_motion, a_lane_chain
    )
    b_turn_maneuver = extract_turn_maneuver_facts(
        b, b_lane, lane_map, b_heading_motion, b_lane_chain
    )
    a_u_turn_evidence = extract_u_turn_evidence(a)
    b_u_turn_evidence = extract_u_turn_evidence(b)
    a_route_transition = _compute_route_transition(a, a_lane, lane_map)
    b_route_transition = _compute_route_transition(b, b_lane, lane_map)
    pair_relation = extract_pair_relation_facts(a, b, a_lane, b_lane, lane_map)
    pair_geometry = extract_pair_geometry_facts(a, b, pair_relation)
    map_context = _compute_map_context(scenario, a, lane_map, other_track=b)

    result = {
        "schema_version": "pair_physical_facts_v1",
        "implementation_revision": "facts_only_canonical_pair_geometry_route_context",
        "scene_id": str(scenario.scenario_id),
        "timeline": {
            "num_frames": int(a.T),
            "current_time_index": int(scenario.current_time_index),
            "timestamps_seconds": [round(float(x), 4) for x in scenario.timestamps_seconds],
            "full_scenario_used": True,
        },
        "pair": {
            "agent_A": _agent_metadata(a),
            "agent_B": _agent_metadata(b),
        },
        "physical_facts": {
            "agent_A": {
                "trajectory_quality": extract_trajectory_quality(a),
                "lateral_motion_evidence": a_lateral,
                "physical_lane_chain": a_lane_chain,
                "speed_change": a_speed,
                "heading_motion": a_heading_motion,
                "turn_maneuver": a_turn_maneuver,
                "u_turn_evidence": a_u_turn_evidence,
                "route_transition": a_route_transition,
            },
            "agent_B": {
                "trajectory_quality": extract_trajectory_quality(b),
                "lateral_motion_evidence": b_lateral,
                "physical_lane_chain": b_lane_chain,
                "speed_change": b_speed,
                "heading_motion": b_heading_motion,
                "turn_maneuver": b_turn_maneuver,
                "u_turn_evidence": b_u_turn_evidence,
                "route_transition": b_route_transition,
            },
            "pair_relation": pair_relation,
            "pair_geometry": pair_geometry,
            "map_context": map_context,
        },
        "map_matching_audit_only": {
            "agent_A_continuous_lane_runs": _lane_group_runs_for_audit(a, a_lane),
            "agent_B_continuous_lane_runs": _lane_group_runs_for_audit(b, b_lane),
            "agent_A_physical_lane_chain": a_lane_chain,
            "agent_B_physical_lane_chain": b_lane_chain,
            "num_lane_segments": int(len(lane_map.segments)),
            "num_continuous_lane_groups": int(len(lane_map.group_to_segments)),
            "policy": (
                "Lane/group identity is map evidence only. A group switch never "
                "creates a physical lateral-maneuver fact by itself."
            ),
        },
        "interhub_window_audit_only": {
            "start_frame": interhub_start,
            "end_frame": interhub_end,
            "used_as_hard_crop": False,
            "used_for_fact_decision": False,
        },
        "facts_only_policy": {
            "questions_answered": [
                "lateral_motion_evidence",
                "same_or_different_travel_channel",
                "ahead_or_behind",
                "clear_acceleration_or_deceleration",
                "pair_heading_relation",
                "closest_approach_distance",
                "distance_evolution",
                "single_agent_heading_motion",
                "single_agent_route_transition",
                "physical_lane_chain_topology",
                "pair_path_geometry",
                "map_context",
            ],
            "behavior_semantics_emitted": False,
            "unknown_is_allowed": True,
            "note": (
                "This builder is intentionally a physical fact extractor. "
                "Behavior semantics and language generation belong downstream "
                "after small-batch visual validation."
            ),
            "compatibility_arguments_ignored_for_cropping": {
                "context_frames": int(context_frames),
                "event_half_width": int(event_half_width),
            },
        },
    }
    return _jsonable(result)


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build conservative facts-only pair records from full WOMD Scenario TFRecords."
    )
    p.add_argument("--tfrecord", required=True, help="TFRecord file, directory, or glob")

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--scene-id", help="Exact raw WOMD Scenario.scenario_id")
    g.add_argument(
        "--record-index",
        type=int,
        help="0-based record index inside exactly one TFRecord shard",
    )

    p.add_argument("--agent-a", type=int, required=True)
    p.add_argument("--agent-b", type=int, required=True)

    p.add_argument(
        "--interhub-start",
        type=int,
        default=None,
        help="InterHub start frame; audit metadata only",
    )
    p.add_argument(
        "--interhub-end",
        type=int,
        default=None,
        help="InterHub end frame; audit metadata only",
    )
    p.add_argument("--context-frames", type=int, default=DEFAULT_CONTEXT_FRAMES)
    p.add_argument("--event-half-width", type=int, default=DEFAULT_EVENT_HALF_WIDTH)
    p.add_argument("--compression-type", default="")
    p.add_argument("--output", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _require_runtime_deps()

    if (args.interhub_start is None) != (args.interhub_end is None):
        raise ValueError("Provide both --interhub-start and --interhub-end, or neither")

    paths = resolve_tfrecord_paths(args.tfrecord)
    if args.scene_id is not None:
        scenario, source_file, record_index = load_scenario_by_id(
            paths, args.scene_id, args.compression_type
        )
    else:
        scenario, source_file, record_index = load_scenario_by_record_index(
            paths, args.record_index, args.compression_type
        )

    result = build_pair_timeline_v3(
        scenario=scenario,
        agent_a_id=args.agent_a,
        agent_b_id=args.agent_b,
        interhub_start=args.interhub_start,
        interhub_end=args.interhub_end,
        context_frames=args.context_frames,
        event_half_width=args.event_half_width,
    )
    result["source"] = {
        "tfrecord": source_file,
        "record_index_in_shard": int(record_index),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] scene: {result['scene_id']}")
    print(f"[OK] pair : {args.agent_a} / {args.agent_b}")
    pf = result["physical_facts"]
    print("[PHYSICAL FACTS]")
    for role in ("agent_A", "agent_B"):
        agent_id = result["pair"][role]["id"]
        lat = pf[role]["lateral_motion_evidence"]
        spd = pf[role]["speed_change"]
        print(
            f"  {role} id={agent_id}: lateral={lat['status']} "
            f"accel={spd['has_clear_acceleration']} "
            f"decel={spd['has_clear_deceleration']}"
        )

    rel = pf["pair_relation"]
    print(
        "[PAIR] travel_channel=",
        rel["travel_channel"]["dominant"],
        " longitudinal=",
        rel["longitudinal"]["dominant"],
        sep="",
    )
    print(f"[OK] output: {out}")


if __name__ == "__main__":
    main()
