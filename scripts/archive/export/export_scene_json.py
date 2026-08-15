"""Export cached InterHub motion data as v2 or scene_motion_v3 JSON.

The v3 exporter separates source data, InterHub labels, derived kinematics,
and unfinished map/behavior processing. It keeps the v2 exporter available
through ``--output-schema scene_motion_v2`` and never overwrites old files.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.signal import savgol_filter
except ImportError:  # pragma: no cover - the project requirements include scipy.
    savgol_filter = None

try:
    from .map_matching import MapMatchingConfig, match_trajectory
except ImportError:  # pragma: no cover - supports direct script execution.
    from map_matching import MapMatchingConfig, match_trajectory

try:
    from ..semantic.behavior_analysis import (
        BehaviorConfig,
        TTCConfig,
        build_agent_behaviors,
        collect_behavior_events,
        associate_behavior_events,
        select_caption_behavior_event,
        classify_interaction_behavior,
        compute_ttc_metrics,
    )
except ImportError:  # pragma: no cover - supports direct script execution.
    from scripts.archive.semantic.behavior_analysis import (
        BehaviorConfig,
        TTCConfig,
        build_agent_behaviors,
        collect_behavior_events,
        associate_behavior_events,
        select_caption_behavior_event,
        classify_interaction_behavior,
        compute_ttc_metrics,
    )


REQUIRED_COLUMNS = {
    "agent_id",
    "scene_ts",
    "x",
    "y",
    "z",
    "vx",
    "vy",
    "heading",
}
SMOOTHING_WINDOW = 7
SMOOTHING_POLYORDER = 2
PENDING_MAP_STATUS = "pending"


def empty_map_record(status: str = PENDING_MAP_STATUS) -> Dict[str, Any]:
    return {
        "lane_id": None,
        "lane_match_confidence": None,
        "match_method": None,
        "centerline_distance_m": None,
        "heading_error_rad": None,
        "lane_s_m": None,
        "lateral_offset_m": None,
        "match_status": status,
    }


def finite_or_none(value: Any) -> Any:
    """Convert NumPy/Pandas values to JSON-safe finite values or ``None``."""
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    value = value.item() if hasattr(value, "item") else value
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def normalize_agent_id(value: Any) -> Optional[str]:
    """Normalize CSV/NumPy scalar agent IDs to comparable strings."""
    value = finite_or_none(value)
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if numeric.is_integer():
            return str(int(numeric))
    text = str(value).strip()
    return text or None


def parse_agent_ids(value: Any) -> List[str]:
    """Parse the semicolon-separated IDs used by InterHub CSV files."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    return [item.strip() for item in str(value).split(";") if item.strip()]


def parse_listish(value: Any) -> List[str]:
    """Parse a real list or a legacy string representation of a list."""
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)):
            return [str(item) for item in parsed]
    except (SyntaxError, ValueError):
        pass
    return [text]


def infer_scene_index(scene_id: str) -> Optional[int]:
    match = re.search(r"(?:scene_|train_)(\d+)$", scene_id)
    return int(match.group(1)) if match else None


def split_dataset_key(dataset_key: str) -> Tuple[str, Optional[str]]:
    """Split keys such as ``waymo_train`` into dataset and split."""
    known_splits = {"train", "val", "test", "multi", "single"}
    if "_" in dataset_key:
        dataset, suffix = dataset_key.rsplit("_", 1)
        if suffix in known_splits:
            return dataset, suffix
    return dataset_key, None


def load_interaction_record(
    interaction_csv: Path, interaction_row: int
) -> Dict[str, Any]:
    records = pd.read_csv(interaction_csv)
    if interaction_row < 0 or interaction_row >= len(records):
        raise IndexError(
            f"interaction row {interaction_row} is outside 0..{len(records) - 1}"
        )

    row = records.iloc[interaction_row].to_dict()
    row["participants"] = parse_agent_ids(row.get("track_id", ""))
    row["key_agents_list"] = parse_agent_ids(row.get("key_agents", ""))
    row["start"] = int(row["start"])
    row["end"] = int(row["end"])
    if row["end"] < row["start"]:
        raise ValueError("interaction end frame must be >= start frame")
    if not row["participants"]:
        raise ValueError("the selected interaction row has no participant IDs")
    if not row["key_agents_list"]:
        row["key_agents_list"] = row["participants"][:2]
    return row


def load_scene_data(scene_dir: Path) -> Tuple[pd.DataFrame, Path, int, int]:
    feather_files = sorted(scene_dir.glob("agent_data_dt*.feather"))
    if not feather_files:
        raise FileNotFoundError(f"No agent_data_dt*.feather found in {scene_dir}")
    data_path = feather_files[0]
    frame_df = pd.read_feather(data_path)
    missing = sorted(REQUIRED_COLUMNS - set(frame_df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    frame_df = frame_df.sort_values(["agent_id", "scene_ts"], kind="stable")
    return (
        frame_df,
        data_path,
        int(frame_df["scene_ts"].min()),
        int(frame_df["scene_ts"].max()),
    )


def resolve_window(
    frame_df: pd.DataFrame,
    source_start: int,
    source_end: int,
    interaction: Optional[Mapping[str, Any]],
    context_before: int,
    context_after: int,
    target_frames: Optional[int] = None,
    all_frames: bool = False,
) -> Tuple[pd.DataFrame, int, int]:
    if interaction is None:
        return frame_df, source_start, source_end

    if all_frames:
        selected_agents = set(interaction["participants"])
        filtered = frame_df[
            frame_df["agent_id"].astype(str).isin(selected_agents)
            & frame_df["scene_ts"].between(source_start, source_end)
        ]
        if filtered.empty:
            raise ValueError("No agent states remain in the full source scene")
        return filtered, source_start, source_end

    interaction_start = int(interaction["start"])
    interaction_end = int(interaction["end"])
    if interaction_end < interaction_start:
        raise ValueError(
            f"invalid interaction interval: {interaction_start}-{interaction_end}"
        )

    if target_frames is not None:
        if target_frames <= 0:
            raise ValueError("target_frames must be positive")
        interaction_frames = interaction_end - interaction_start + 1
        source_frames = source_end - source_start + 1
        if interaction_frames > target_frames:
            raise ValueError(
                "interaction interval has "
                f"{interaction_frames} frames, exceeding target {target_frames}"
            )
        if source_frames < target_frames:
            raise ValueError(
                f"source scene has {source_frames} frames, shorter than target {target_frames}"
            )

        # Keep the interaction inside the target window.  If the preferred
        # centered window reaches a scene boundary, shift the whole window
        # instead of allowing the boundary clamp to shorten it.
        preferred_start = interaction_start - max(0, context_before)
        minimum_start = max(source_start, interaction_end - target_frames + 1)
        maximum_start = min(interaction_start, source_end - target_frames + 1)
        if minimum_start > maximum_start:
            raise ValueError(
                f"cannot place {target_frames}-frame window around "
                f"interaction {interaction_start}-{interaction_end}"
            )
        export_start = min(max(preferred_start, minimum_start), maximum_start)
        export_end = export_start + target_frames - 1
    else:
        export_start = max(source_start, interaction_start - max(0, context_before))
        export_end = min(source_end, interaction_end + max(0, context_after))
    selected_agents = set(interaction["participants"])
    filtered = frame_df[
        frame_df["agent_id"].astype(str).isin(selected_agents)
        & frame_df["scene_ts"].between(export_start, export_end)
    ]
    if filtered.empty:
        raise ValueError("No agent states remain after interaction filtering")
    return filtered, export_start, export_end


def _contiguous_segments(frames: Sequence[int]) -> Iterable[Tuple[int, int]]:
    if not frames:
        return
    start = 0
    for index in range(1, len(frames) + 1):
        if index == len(frames) or frames[index] != frames[index - 1] + 1:
            yield start, index
            start = index


def _contiguous_value_segments(
    frames: Sequence[int], values: Sequence[Optional[float]]
) -> Iterable[Tuple[int, int]]:
    """Yield runs that are frame-contiguous and contain only non-null values."""
    if len(frames) != len(values):
        raise ValueError("frames and values must have equal length")
    start: Optional[int] = None
    for index, value in enumerate(values):
        if value is None:
            if start is not None:
                yield start, index
                start = None
            continue
        if start is None:
            start = index
        elif frames[index] != frames[index - 1] + 1:
            yield start, index
            start = index
    if start is not None:
        yield start, len(values)


def smooth_acceleration(
    frames: Sequence[int], values: Sequence[Optional[float]]
) -> List[Optional[float]]:
    """Smooth only finite, contiguous runs; never alter the raw values."""
    result: List[Optional[float]] = [None] * len(values)
    if savgol_filter is None:
        return result

    for start, end in _contiguous_value_segments(list(frames), values):
        segment = values[start:end]
        if len(segment) < SMOOTHING_WINDOW:
            continue
        window = min(SMOOTHING_WINDOW, len(segment))
        if window % 2 == 0:
            window -= 1
        if window <= SMOOTHING_POLYORDER:
            continue
        smoothed = savgol_filter(
            np.asarray(segment, dtype=float),
            window_length=window,
            polyorder=min(SMOOTHING_POLYORDER, window - 1),
            mode="interp",
        )
        for offset, value in enumerate(smoothed, start=start):
            result[offset] = float(value)
    return result


def state_speed(row: Mapping[str, Any]) -> Optional[float]:
    vx = finite_or_none(row.get("vx"))
    vy = finite_or_none(row.get("vy"))
    if vx is None or vy is None:
        return None
    return math.sqrt(float(vx) ** 2 + float(vy) ** 2)


def _acceleration_with_fallback(
    observed_frames: Sequence[int],
    observed: Mapping[int, Mapping[str, Any]],
    acceleration_key: str,
    velocity_key: str,
    dt: float,
) -> Tuple[List[Optional[float]], List[Optional[float]], List[str]]:
    """Return raw acceleration, effective acceleration, and per-frame source."""
    raw_values: List[Optional[float]] = []
    effective_values: List[Optional[float]] = []
    sources: List[str] = []
    previous_frame: Optional[int] = None
    previous_velocity: Optional[float] = None

    for frame in observed_frames:
        row = observed[frame]
        raw_value = finite_or_none(row.get(acceleration_key))
        velocity = finite_or_none(row.get(velocity_key))
        derived_value = None
        if (
            velocity is not None
            and previous_velocity is not None
            and previous_frame is not None
            and frame > previous_frame
            and dt > 0
        ):
            elapsed = (frame - previous_frame) * dt
            derived_value = (float(velocity) - float(previous_velocity)) / elapsed

        if raw_value is not None:
            effective_value = float(raw_value)
            source = "source"
        elif derived_value is not None:
            effective_value = float(derived_value)
            source = "derived_from_velocity"
        else:
            effective_value = None
            source = "insufficient_data"

        raw_values.append(raw_value)
        effective_values.append(effective_value)
        sources.append(source)
        if velocity is not None:
            previous_frame = frame
            previous_velocity = float(velocity)
        else:
            previous_frame = None
            previous_velocity = None

    return raw_values, effective_values, sources


def _combined_acceleration_source(ax_source: str, ay_source: str) -> str:
    used_sources = {
        source
        for source in (ax_source, ay_source)
        if source != "insufficient_data"
    }
    if not used_sources:
        return "insufficient_data"
    if used_sources == {"source"}:
        return "source"
    if used_sources == {"derived_from_velocity"}:
        return "derived_from_velocity"
    return "mixed_source_and_velocity"


def _valid_motion_values(row: Mapping[str, Any]) -> bool:
    return all(
        finite_or_none(row.get(key)) is not None
        for key in ("x", "y", "vx", "vy", "heading")
    )


def _dimensions_from_row(row: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Map cached vehicle dimensions to the canonical V3 state schema."""
    if row is None:
        return {"length_m": None, "width_m": None}
    return {
        "length_m": finite_or_none(row.get("length")),
        "width_m": finite_or_none(row.get("width")),
    }


def _build_v3_agent(
    agent_id: str,
    agent_df: pd.DataFrame,
    export_start: int,
    export_end: int,
    interaction_start: int,
    dt: float,
    control_type: Optional[str],
) -> Dict[str, Any]:
    observed = {
        int(row.scene_ts): row._asdict()
        for row in agent_df.itertuples(index=False)
    }
    observed_frames = sorted(observed)
    raw_ax, effective_ax, source_ax = _acceleration_with_fallback(
        observed_frames, observed, "ax", "vx", dt
    )
    raw_ay, effective_ay, source_ay = _acceleration_with_fallback(
        observed_frames, observed, "ay", "vy", dt
    )
    smooth_ax = smooth_acceleration(observed_frames, effective_ax)
    smooth_ay = smooth_acceleration(observed_frames, effective_ay)
    smoothed_by_frame = {
        frame: (smooth_ax[index], smooth_ay[index])
        for index, frame in enumerate(observed_frames)
    }

    states = []
    for frame in range(export_start, export_end + 1):
        row = observed.get(frame)
        if row is None:
            states.append(
                {
                    "frame": frame,
                    "time_s": round(frame * dt, 6),
                    "relative_time_s": round((frame - interaction_start) * dt, 6),
                    "position": {"x": None, "y": None, "z": None},
                    "velocity": {"vx": None, "vy": None, "speed": None},
                    "acceleration": {
                        "ax_raw": None,
                        "ay_raw": None,
                        "ax_effective": None,
                        "ay_effective": None,
                        "ax_smoothed": None,
                        "ay_smoothed": None,
                        "source": "insufficient_data",
                    },
                    "heading_rad": None,
                    "dimensions": _dimensions_from_row(None),
                    "map": empty_map_record(),
                    "valid": False,
                }
            )
            continue

        speed = state_speed(row)
        smoothed = smoothed_by_frame.get(frame, (None, None))
        frame_index = observed_frames.index(frame)
        states.append(
            {
                "frame": frame,
                "time_s": round(frame * dt, 6),
                "relative_time_s": round((frame - interaction_start) * dt, 6),
                "position": {
                    "x": finite_or_none(row.get("x")),
                    "y": finite_or_none(row.get("y")),
                    "z": finite_or_none(row.get("z")),
                },
                "velocity": {
                    "vx": finite_or_none(row.get("vx")),
                    "vy": finite_or_none(row.get("vy")),
                    "speed": speed,
                },
                "acceleration": {
                    "ax_raw": raw_ax[frame_index],
                    "ay_raw": raw_ay[frame_index],
                    "ax_effective": effective_ax[frame_index],
                    "ay_effective": effective_ay[frame_index],
                    "ax_smoothed": smoothed[0],
                    "ay_smoothed": smoothed[1],
                    "source": _combined_acceleration_source(
                        source_ax[frame_index], source_ay[frame_index]
                    ),
                },
                "heading_rad": finite_or_none(row.get("heading")),
                "dimensions": _dimensions_from_row(row),
                "map": empty_map_record(),
                "valid": _valid_motion_values(row),
            }
        )

    valid_states = [state for state in states if state["valid"]]
    speeds = [state["velocity"]["speed"] for state in valid_states]
    summary = {
        "initial_speed_mps": speeds[0] if speeds else None,
        "final_speed_mps": speeds[-1] if speeds else None,
        "max_speed_mps": max(speeds) if speeds else None,
        "min_speed_mps": min(speeds) if speeds else None,
        "initial_lane_id": None,
        "final_lane_id": None,
        "lane_change_detected": None,
    }
    return {
        "agent_id": str(agent_id),
        "agent_type": "vehicle",
        "control_type": control_type,
        "first_frame": observed_frames[0] if observed_frames else None,
        "last_frame": observed_frames[-1] if observed_frames else None,
        "state_count": len(states),
        "observed_state_count": len(observed_frames),
        "summary": summary,
        "states": states,
    }


def apply_map_matching(
    agents: List[Dict[str, Any]],
    vector_map: Any,
    config: MapMatchingConfig,
) -> None:
    """Populate per-state map fields and lane summary fields in-place."""
    for agent in agents:
        observations = []
        for state in agent["states"]:
            observations.append(
                {
                    "frame": state["frame"],
                    "valid": state["valid"],
                    "x": state["position"]["x"],
                    "y": state["position"]["y"],
                    "z": state["position"]["z"],
                    "heading": state["heading_rad"],
                }
            )
        match_records = match_trajectory(vector_map, observations, config)
        for state, match_record in zip(agent["states"], match_records):
            state["map"] = match_record

        matched = [
            state["map"]["lane_id"]
            for state in agent["states"]
            if state["map"]["lane_id"] is not None
        ]
        agent["summary"]["initial_lane_id"] = matched[0] if matched else None
        agent["summary"]["final_lane_id"] = matched[-1] if matched else None
        # A lane-ID change alone is not enough to label a lane change. The
        # behavior layer must additionally inspect topology and lateral motion.
        agent["summary"]["lane_change_detected"] = None


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def compute_pairwise_features(
    agent_i: Mapping[str, Any],
    agent_j: Mapping[str, Any],
    dt_seconds: float,
    interaction_start_frame: int,
    pet_seconds: Optional[float],
    ttc_config: TTCConfig = TTCConfig(),
) -> Dict[str, Any]:
    """Compute geometry, relative motion, and explicit TTC status."""
    states_i = {state["frame"]: state for state in agent_i["states"]}
    states_j = {state["frame"]: state for state in agent_j["states"]}
    pair_states = []
    for frame in sorted(set(states_i) & set(states_j)):
        left, right = states_i[frame], states_j[frame]
        if not left["valid"] or not right["valid"]:
            pair_states.append(
                {
                    "frame": frame,
                    "distance_m": None,
                    "speed_difference_mps": None,
                    "relative_velocity_norm_mps": None,
                    "relative_speed_mps": None,
                    "relative_heading_rad": None,
                    "closing_speed_mps": None,
                    "ttc_seconds": None,
                    "ttc_status": "insufficient_data",
                    "same_lane": None,
                }
            )
            continue
        pi, pj = left["position"], right["position"]
        vi, vj = left["velocity"], right["velocity"]
        distance = math.hypot(pi["x"] - pj["x"], pi["y"] - pj["y"])
        dvx = vi["vx"] - vj["vx"]
        dvy = vi["vy"] - vj["vy"]
        speed_difference = vi["speed"] - vj["speed"]
        relative_velocity_norm = math.hypot(dvx, dvy)
        relative_heading = _wrap_angle(
            left["heading_rad"] - right["heading_rad"]
        )
        ttc = compute_ttc_metrics(left, right, ttc_config)
        pair_states.append(
            {
                "frame": frame,
                "distance_m": distance,
                "speed_difference_mps": speed_difference,
                "relative_velocity_norm_mps": relative_velocity_norm,
                "relative_speed_mps": relative_velocity_norm,
                "relative_heading_rad": relative_heading,
                "closing_speed_mps": ttc["closing_speed_mps"],
                "ttc_seconds": ttc["ttc_seconds"],
                "ttc_status": ttc["ttc_status"],
                "same_lane": (
                    left["map"]["lane_id"] == right["map"]["lane_id"]
                    if left["map"]["lane_id"] is not None
                    and right["map"]["lane_id"] is not None
                    else None
                ),
            }
        )

    start_state = next(
        (state for state in pair_states if state["frame"] == interaction_start_frame),
        None,
    )
    distance_states = [
        state for state in pair_states if state["distance_m"] is not None
    ]
    ttc_states = [
        state for state in pair_states if state["ttc_status"] == "valid"
    ]
    minimum = min(distance_states, key=lambda state: state["distance_m"], default=None)
    minimum_ttc = min(ttc_states, key=lambda state: state["ttc_seconds"], default=None)
    return {
        "agent_i": agent_i["agent_id"],
        "agent_j": agent_j["agent_id"],
        "relative_speed_definition": "relative_velocity_norm_mps",
        "summary": {
            "initial_distance_m": pair_states[0]["distance_m"] if pair_states else None,
            "minimum_distance_m": minimum["distance_m"] if minimum else None,
            "minimum_distance_frame": minimum["frame"] if minimum else None,
            "relative_speed_at_interaction_start_mps": (
                start_state["relative_speed_mps"] if start_state else None
            ),
            "relative_heading_at_interaction_start_rad": (
                start_state["relative_heading_rad"] if start_state else None
            ),
            "pet_seconds": pet_seconds,
            "minimum_ttc_seconds": (
                minimum_ttc["ttc_seconds"] if minimum_ttc else None
            ),
            "minimum_ttc_frame": minimum_ttc["frame"] if minimum_ttc else None,
            "valid_ttc_frame_count": len(ttc_states),
        },
        "states": pair_states,
    }


def _interaction_v3(
    interaction: Mapping[str, Any],
    participant_ids: List[str],
    key_agent_ids: List[str],
) -> Dict[str, Any]:
    control_types = parse_listish(interaction.get("vehicle_type"))
    if len(control_types) != len(participant_ids):
        control_types = [None] * len(participant_ids)
    priority_raw = finite_or_none(interaction.get("priority_label"))
    priority_agent = normalize_agent_id(priority_raw)
    if priority_agent not in participant_ids:
        priority_agent = None
    # Keep the scene cardinality separate from the agent pair used for
    # behavior/TTC analysis. A multi-agent interaction can still have exactly
    # two key agents selected for pairwise analysis.
    scene_scale = "pairwise" if len(participant_ids) == 2 else "multi_agent"
    analysis_scale = "pairwise" if len(key_agent_ids) == 2 else "multi_agent"
    return {
        "participant_ids": participant_ids,
        "key_agent_ids": key_agent_ids,
        "num_participants": len(participant_ids),
        "scene_scale": scene_scale,
        "analysis_scale": analysis_scale,
        # Backward-compatible alias: old consumers interpreted this field as
        # the scale of the participant set. New code must use scene_scale or
        # analysis_scale explicitly.
        "interaction_scale": scene_scale,
        "intensity": {
            "value": finite_or_none(interaction.get("intensity")),
            "definition": "interhub_intensity",
            "source": "interhub",
        },
        "pet_seconds": finite_or_none(interaction.get("PET")),
        "source_labels": {
            "path_category": finite_or_none(interaction.get("path_category")),
            "path_relation": finite_or_none(interaction.get("path_relation")),
            "turn_label": finite_or_none(interaction.get("turn_label")),
            "priority_agent_id": priority_agent,
            "priority_label_raw": priority_raw,
            "participant_control_types": control_types,
            "av_included": finite_or_none(interaction.get("AV_included")),
        },
        "interhub": {
            "candidate_type": "interaction_risk_candidate",
            "window": {
                "start_frame": int(interaction["start"]),
                "end_frame": int(interaction["end"]),
            },
            "source": "interhub",
        },
        # Kept as a backward-compatible pairwise classifier field for the
        # existing v3 contract. New consumers must use the top-level
        # behavior_events, behavior_associations, and caption fields below;
        # this field is not the InterHub interaction type.
        "behavior": {
            "type": None,
            "subtype": None,
            "subject_agent_id": None,
            "reference_agent_id": None,
            "status": "unlabeled",
            "source": None,
            "rule_version": None,
            "confidence": None,
            "evidence": {},
        },
    }


def _ttc_safety_radius_source(
    agents: Sequence[Mapping[str, Any]],
    key_agent_ids: Sequence[str],
) -> str:
    """Describe which dimensions are available to the exported TTC states."""
    selected_ids = {str(agent_id) for agent_id in key_agent_ids}
    selected_agents = [
        agent for agent in agents if str(agent.get("agent_id")) in selected_ids
    ]
    states = [
        state
        for agent in selected_agents
        for state in agent.get("states", [])
        if state.get("valid")
    ]
    if not states:
        return "configured_default"

    dimension_states = [
        state
        for state in states
        if isinstance(state.get("dimensions"), Mapping)
        and finite_or_none(state["dimensions"].get("length_m")) is not None
        and finite_or_none(state["dimensions"].get("width_m")) is not None
        and float(state["dimensions"]["length_m"]) > 0
        and float(state["dimensions"]["width_m"]) > 0
    ]
    if len(dimension_states) == len(states):
        return "state_dimensions"
    if dimension_states:
        return "state_dimensions_or_configured_default"
    return "configured_default"


def _assert_finite_tree(value: Any, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _assert_finite_tree(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_finite_tree(child, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite numeric value at {path}")


def validate_scene_motion_v3(payload: Mapping[str, Any]) -> None:
    """Programmatic validation for the v3 invariants."""
    if payload.get("schema_version") != "scene_motion_v3":
        raise ValueError("schema_version must be scene_motion_v3")
    temporal = payload["temporal"]
    if temporal["export_end_frame"] - temporal["export_start_frame"] + 1 != temporal[
        "num_export_frames"
    ]:
        raise ValueError("num_export_frames is inconsistent")
    if temporal["interaction_end_frame"] < temporal["interaction_start_frame"]:
        raise ValueError("interaction frame order is invalid")
    behavior_start = temporal.get("behavior_start_frame")
    behavior_end = temporal.get("behavior_end_frame")
    caption_start = temporal.get("caption_start_frame")
    caption_end = temporal.get("caption_end_frame")
    if (behavior_start is None) != (behavior_end is None):
        raise ValueError("behavior window must have both bounds or neither")
    if behavior_start is not None and behavior_end < behavior_start:
        raise ValueError("behavior frame order is invalid")
    if (caption_start is None) != (caption_end is None):
        raise ValueError("caption window must have both bounds or neither")
    if caption_start is not None and caption_end < caption_start:
        raise ValueError("caption frame order is invalid")
    if (behavior_start, behavior_end) != (caption_start, caption_end):
        raise ValueError("phase-1 caption window must equal behavior window")
    dt = float(temporal["dt_seconds"])
    interaction_start = temporal["interaction_start_frame"]

    interaction = payload["interaction"]
    participants = interaction["participant_ids"]
    keys = interaction["key_agent_ids"]
    if not participants or not set(keys).issubset(set(participants)):
        raise ValueError("participant/key agent IDs are inconsistent")
    if interaction["num_participants"] != len(participants):
        raise ValueError("num_participants is inconsistent")
    if interaction["behavior"]["status"] == "unlabeled" and interaction["behavior"][
        "type"
    ] is not None:
        raise ValueError("unlabeled behavior cannot have a type")

    for agent in payload["agents"]:
        if agent["state_count"] != len(agent["states"]):
            raise ValueError(f"state_count mismatch for {agent['agent_id']}")
        frames = [state["frame"] for state in agent["states"]]
        if frames != sorted(frames):
            raise ValueError(f"frames are not monotonic for {agent['agent_id']}")
        for state in agent["states"]:
            if abs(state["time_s"] - state["frame"] * dt) >= 1e-6:
                raise ValueError(f"time_s mismatch at frame {state['frame']}")
            expected_relative = (state["frame"] - interaction_start) * dt
            if abs(state["relative_time_s"] - expected_relative) >= 1e-6:
                raise ValueError(f"relative_time_s mismatch at frame {state['frame']}")
            if payload["processing_status"]["map_matching"] == "pending":
                if state["map"]["lane_id"] is not None:
                    raise ValueError("lane_id must be null while map matching is pending")

    scene_scale = interaction.get("scene_scale")
    analysis_scale = interaction.get("analysis_scale")
    expected_scene_scale = "pairwise" if len(participants) == 2 else "multi_agent"
    expected_analysis_scale = "pairwise" if len(keys) == 2 else "multi_agent"
    if scene_scale != expected_scene_scale:
        raise ValueError("scene_scale is inconsistent with participant_ids")
    if analysis_scale != expected_analysis_scale:
        raise ValueError("analysis_scale is inconsistent with key_agent_ids")
    if interaction.get("interaction_scale") != scene_scale:
        raise ValueError("legacy interaction_scale is inconsistent with scene_scale")

    pairwise = payload.get("pairwise")
    if analysis_scale == "pairwise" and pairwise is None:
        raise ValueError("pairwise analysis requires pairwise features")
    if pairwise is not None:
        if len(keys) != 2:
            raise ValueError("pairwise output requires exactly two key agents")
        pair_agents = [str(pairwise.get("agent_i")), str(pairwise.get("agent_j"))]
        if pair_agents != [str(keys[0]), str(keys[1])]:
            raise ValueError("pairwise agent order must match key_agent_ids")
    _assert_finite_tree(payload)


def export_scene_v3(
    scene_dir: Path,
    output_path: Path,
    dt: float,
    interaction_csv: Path,
    interaction_row: int = 0,
    context_before: int = 30,
    context_after: int = 30,
    target_frames: Optional[int] = None,
    all_frames: bool = False,
    vector_map: Any = None,
    map_matching_config: Optional[MapMatchingConfig] = None,
    behavior_config: Optional[BehaviorConfig] = None,
    ttc_config: Optional[TTCConfig] = None,
) -> Dict[str, Any]:
    frame_df, data_path, source_start, source_end = load_scene_data(scene_dir)
    interaction = load_interaction_record(interaction_csv, interaction_row)
    filtered, export_start, export_end = resolve_window(
        frame_df,
        source_start,
        source_end,
        interaction,
        context_before,
        context_after,
        target_frames,
        all_frames,
    )
    participant_ids = interaction["participants"]
    # Preserve InterHub's semantic subject/reference order while removing
    # duplicate IDs. Do not sort numeric-looking strings lexicographically.
    key_agent_ids = list(dict.fromkeys(interaction["key_agents_list"]))
    if not set(key_agent_ids).issubset(set(participant_ids)):
        raise ValueError("key_agents must be a subset of track_id participants")
    if not key_agent_ids:
        raise ValueError("key_agents must not be empty")

    dataset_key = str(interaction.get("dataset", "unknown"))
    dataset, split = split_dataset_key(dataset_key)
    scene_id = str(interaction.get("original_scene_id") or scene_dir.name)
    scenario_index = interaction.get("scenario_idx", infer_scene_index(scene_id))
    scenario_index = int(scenario_index) if scenario_index is not None else None
    control_types = parse_listish(interaction.get("vehicle_type"))
    if len(control_types) != len(participant_ids):
        control_types = [None] * len(participant_ids)
    control_by_agent = dict(zip(participant_ids, control_types))

    agents = []
    for agent_id in participant_ids:
        agent_df = filtered[filtered["agent_id"].astype(str) == agent_id]
        agents.append(
            _build_v3_agent(
                agent_id,
                agent_df,
                export_start,
                export_end,
                int(interaction["start"]),
                dt,
                control_by_agent.get(agent_id),
            )
        )

    map_matching_completed = vector_map is not None
    behavior_config = behavior_config or BehaviorConfig()
    ttc_config = ttc_config or TTCConfig()
    if map_matching_completed:
        apply_map_matching(
            agents,
            vector_map,
            map_matching_config or MapMatchingConfig(),
        )
    agent_behaviors = build_agent_behaviors(
        agents,
        vector_map if map_matching_completed else None,
        behavior_config,
    )
    for agent in agents:
        behavior = agent_behaviors[str(agent["agent_id"])]
        agent["summary"]["lane_change_detected"] = behavior[
            "lane_change_detected"
        ]
        agent["summary"]["lane_change_events"] = behavior["lane_change_events"]

    interaction_id = (
        f"{dataset_key}_{scene_id}_{'_'.join(key_agent_ids)}_"
        f"{int(interaction['start'])}_{int(interaction['end'])}"
    )
    interaction_v3 = _interaction_v3(interaction, participant_ids, key_agent_ids)
    pairwise = None
    if interaction_v3["analysis_scale"] == "pairwise":
        agent_by_id = {str(agent["agent_id"]): agent for agent in agents}
        if any(agent_id not in agent_by_id for agent_id in key_agent_ids):
            raise ValueError("key_agent_ids must be present in exported agents")
        pairwise = compute_pairwise_features(
            agent_by_id[key_agent_ids[0]],
            agent_by_id[key_agent_ids[1]],
            dt,
            int(interaction["start"]),
            finite_or_none(interaction.get("PET")),
            ttc_config,
        )

    behavior_events = collect_behavior_events(agent_behaviors)
    behavior_associations = associate_behavior_events(
        behavior_events,
        agents,
        key_agent_ids,
        int(interaction["start"]),
        int(interaction["end"]),
        pairwise,
        vector_map if map_matching_completed else None,
        behavior_config,
    )
    selected_caption = select_caption_behavior_event(
        behavior_events,
        behavior_associations,
    )

    if map_matching_completed:
        classification_input = dict(interaction)
        classification_input["key_agent_ids"] = key_agent_ids
        behavior_result = classify_interaction_behavior(
            classification_input,
            agents,
            agent_behaviors,
            pairwise,
            vector_map,
            behavior_config,
        )
        interaction_v3["behavior"] = behavior_result["behavior"]
        behavior_classification_status = behavior_result["processing_status"]
    else:
        behavior_classification_status = "pending"

    ttc_safety_radius_source = _ttc_safety_radius_source(agents, key_agent_ids)
    payload: Dict[str, Any] = {
        "schema_version": "scene_motion_v3",
        "interaction_id": interaction_id,
        "source": {
            "dataset": dataset,
            "split": split,
            "dataset_key": dataset_key,
            "folder": finite_or_none(interaction.get("folder")),
            "scene_id": scene_id,
            "scenario_index": scenario_index,
            "source_file": data_path.name,
        },
        "temporal": {
            "dt_seconds": dt,
            "source_start_frame": source_start,
            "source_end_frame": source_end,
            "export_start_frame": export_start,
            "export_end_frame": export_end,
            "num_export_frames": export_end - export_start + 1,
            "interaction_start_frame": int(interaction["start"]),
            "interaction_end_frame": int(interaction["end"]),
            "num_interaction_frames": int(interaction["end"])
            - int(interaction["start"])
            + 1,
            "interaction_start_time_s": round(int(interaction["start"]) * dt, 6),
            "interaction_end_time_s": round(int(interaction["end"]) * dt, 6),
            "behavior_start_frame": (
                int(selected_caption["event"]["start_frame"])
                if selected_caption is not None
                else None
            ),
            "behavior_end_frame": (
                int(selected_caption["event"]["end_frame"])
                if selected_caption is not None
                else None
            ),
            "caption_start_frame": (
                int(selected_caption["event"]["start_frame"])
                if selected_caption is not None
                else None
            ),
            "caption_end_frame": (
                int(selected_caption["event"]["end_frame"])
                if selected_caption is not None
                else None
            ),
            "context_before_frames": max(
                0, int(interaction["start"]) - export_start
            ),
            "context_after_frames": max(
                0, export_end - int(interaction["end"])
            ),
            "relative_time_anchor": {
                "type": "interaction_start",
                "frame": int(interaction["start"]),
                "time_s": round(int(interaction["start"]) * dt, 6),
            },
        },
        "coordinate_system": {
            "frame": "dataset_global",
            "position_unit": "m",
            "velocity_unit": "m/s",
            "acceleration_unit": "m/s2",
            "heading_unit": "rad",
            "heading_convention": "dataset_native",
            "velocity_frame": "dataset_global",
        },
        "interaction": interaction_v3,
        "agent_behaviors": agent_behaviors,
        "behavior_events": behavior_events,
        "behavior_associations": behavior_associations,
        "selected_behavior_event": selected_caption,
        "caption": {
            "status": "ready" if selected_caption is not None else "unavailable",
            "scope": selected_caption["caption_scope"] if selected_caption else None,
            "behavior_event_id": (
                selected_caption["event"].get("event_id")
                if selected_caption
                else None
            ),
            "behavior_window": (
                {
                    "start_frame": int(selected_caption["event"]["start_frame"]),
                    "end_frame": int(selected_caption["event"]["end_frame"]),
                }
                if selected_caption
                else None
            ),
            # Phase 1 intentionally uses the behavior window directly.
            "caption_window": (
                {
                    "start_frame": int(selected_caption["event"]["start_frame"]),
                    "end_frame": int(selected_caption["event"]["end_frame"]),
                }
                if selected_caption
                else None
            ),
        },
        "processing_status": {
            "trajectory_export": "completed",
            "map_matching": "completed" if map_matching_completed else "pending",
            "pairwise_features": "completed" if pairwise is not None else "pending",
            "behavior_classification": behavior_classification_status,
            "text_generation": "pending",
        },
        "kinematics_processing": {
            "acceleration_source": "source_or_derived_from_velocity",
            "acceleration_source_policy": "use_source_ax_ay_else_backward_velocity_difference",
            "smoothing_enabled": True,
            "smoothing_method": "savitzky_golay",
            "window_frames": SMOOTHING_WINDOW,
            "polyorder": SMOOTHING_POLYORDER,
            "fallback_policy": "null_when_run_is_short_or_noncontiguous",
        },
        "agents": agents,
        "pairwise": pairwise,
        "provenance": {
            "original_schema_version": "scene_motion_v2",
            "original_scene_id": scene_id,
            "original_track_id": finite_or_none(interaction.get("track_id")),
            "original_interaction_row": interaction_row,
            "behavior_ttc_config": {
                "behavior_rule_version": "v2",
                "min_stable_frames": behavior_config.min_stable_frames,
                "min_lane_match_confidence": behavior_config.min_lane_match_confidence,
                "min_valid_match_ratio": behavior_config.min_valid_match_ratio,
                "min_lateral_delta_m": behavior_config.min_lateral_delta_m,
                "evidence_window_frames": behavior_config.evidence_window_frames,
                "classification_padding_frames": behavior_config.classification_padding_frames,
                "lane_group_max_hops": behavior_config.lane_group_max_hops,
                "min_same_lane_frames": behavior_config.min_same_lane_frames,
                "min_same_lane_ratio": behavior_config.min_same_lane_ratio,
                "same_lane_distance_threshold_m": behavior_config.same_lane_distance_threshold_m,
                "cut_in_gap_threshold_m": behavior_config.cut_in_gap_threshold_m,
                "deceleration_threshold_mps": behavior_config.deceleration_threshold_mps,
                "gap_decrease_threshold_m": behavior_config.gap_decrease_threshold_m,
                "ttc_epsilon_m": ttc_config.epsilon_m,
                "ttc_closing_speed_epsilon_mps": ttc_config.closing_speed_epsilon_mps,
                "ttc_default_safety_radius_m": ttc_config.default_safety_radius_m,
                "ttc_lateral_not_applicable_ratio": ttc_config.lateral_not_applicable_ratio,
                "ttc_safety_radius_source": ttc_safety_radius_source,
            },
        },
    }
    validate_scene_motion_v3(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return payload


def export_scene_v2(
    scene_dir: Path,
    output_path: Path,
    dt: float,
    interaction_csv: Optional[Path] = None,
    interaction_row: int = 0,
    context_before: int = 0,
    context_after: int = 0,
) -> Dict[str, Any]:
    """Keep the original v2 top-level and nested-agent structure unchanged."""
    frame_df, data_path, source_start, source_end = load_scene_data(scene_dir)
    interaction = (
        load_interaction_record(interaction_csv, interaction_row)
        if interaction_csv is not None
        else None
    )
    filtered, export_start, export_end = resolve_window(
        frame_df,
        source_start,
        source_end,
        interaction,
        context_before,
        context_after,
    )
    agents = []
    participant_ids = interaction["participants"] if interaction else sorted(
        filtered["agent_id"].astype(str).unique()
    )
    for agent_id in participant_ids:
        agent_df = filtered[filtered["agent_id"].astype(str) == agent_id]
        states = []
        for row in agent_df.itertuples(index=False):
            speed = state_speed(row._asdict())
            states.append(
                {
                    "frame": int(row.scene_ts),
                    "time_s": round(int(row.scene_ts) * dt, 6),
                    "x": finite_or_none(row.x),
                    "y": finite_or_none(row.y),
                    "z": finite_or_none(row.z),
                    "vx": finite_or_none(row.vx),
                    "vy": finite_or_none(row.vy),
                    "speed": speed,
                    "ax": finite_or_none(row.ax),
                    "ay": finite_or_none(row.ay),
                    "heading": finite_or_none(row.heading),
                    "lane_id": None,
                    "interaction_time_s": (
                        round((int(row.scene_ts) - interaction["start"]) * dt, 6)
                        if interaction is not None
                        else None
                    ),
                }
            )
        if states:
            agents.append(
                {
                    "agent_id": agent_id,
                    "first_frame": states[0]["frame"],
                    "last_frame": states[-1]["frame"],
                    "state_count": len(states),
                    "states": states,
                }
            )
    payload = {
        "schema_version": "scene_motion_v2",
        "scene_id": scene_dir.name,
        "source_file": data_path.name,
        "dt_seconds": dt,
        "frame_start": export_start,
        "frame_end": export_end,
        "num_frames": export_end - export_start + 1,
        "source_frame_start": source_start,
        "source_frame_end": source_end,
        "scope": "interaction_window" if interaction else "full_scene",
        "agent_count": len(agents),
        "state_count": sum(agent["state_count"] for agent in agents),
        "lane_id_status": "pending_map_matching",
        "pairwise_features_status": "not_exported_in_v1",
        "agents": agents,
    }
    if interaction:
        payload["interaction"] = {
            key: finite_or_none(value)
            for key, value in interaction.items()
            if key not in {"participants", "key_agents_list"}
        }
        payload["interaction"]["participants"] = interaction["participants"]
        payload["interaction"]["key_agents"] = interaction["key_agents_list"]
        payload["context_before_frames"] = max(0, context_before)
        payload["context_after_frames"] = max(0, context_after)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return payload


# Backwards-compatible function name used by the first exporter version.
def export_scene(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    return export_scene_v2(*args, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--output-schema",
        choices=("scene_motion_v2", "scene_motion_v3"),
        default="scene_motion_v3",
    )
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--interaction-csv", type=Path)
    parser.add_argument("--interaction-row", type=int, default=0)
    parser.add_argument("--context-before", type=int, default=30)
    parser.add_argument("--context-after", type=int, default=30)
    parser.add_argument(
        "--map-cache-root",
        type=Path,
        help="trajdata UnifiedCache root containing <env>/maps",
    )
    parser.add_argument(
        "--map-id",
        help="trajdata map id, for example waymo_train:waymo_train_16",
    )
    parser.add_argument("--map-distance-threshold", type=float, default=3.0)
    parser.add_argument(
        "--map-heading-threshold-deg", type=float, default=22.5
    )
    parser.add_argument("--map-top-k", type=int, default=5)
    parser.add_argument("--behavior-min-stable-frames", type=int, default=3)
    parser.add_argument("--behavior-min-lane-match-confidence", type=float, default=0.5)
    parser.add_argument("--ttc-epsilon-m", type=float, default=1e-3)
    parser.add_argument("--ttc-closing-speed-epsilon-mps", type=float, default=1e-3)
    parser.add_argument("--ttc-default-safety-radius-m", type=float, default=2.0)
    args = parser.parse_args()

    if args.output_schema == "scene_motion_v3" and args.interaction_csv is None:
        parser.error("scene_motion_v3 requires --interaction-csv")

    if (args.map_cache_root is None) != (args.map_id is None):
        parser.error("--map-cache-root and --map-id must be provided together")
    vector_map = None
    map_config = None
    behavior_config = BehaviorConfig(
        min_stable_frames=max(1, args.behavior_min_stable_frames),
        min_lane_match_confidence=max(0.0, min(1.0, args.behavior_min_lane_match_confidence)),
    )
    ttc_config = TTCConfig(
        epsilon_m=max(0.0, args.ttc_epsilon_m),
        closing_speed_epsilon_mps=max(0.0, args.ttc_closing_speed_epsilon_mps),
        default_safety_radius_m=max(0.0, args.ttc_default_safety_radius_m),
    )
    if args.map_cache_root is not None:
        if args.output_schema != "scene_motion_v3":
            parser.error("map matching is currently supported for scene_motion_v3 only")
        try:
            from trajdata import MapAPI

            vector_map = MapAPI(args.map_cache_root).get_map(args.map_id)
        except Exception as exc:
            parser.error(f"failed to load map {args.map_id!r}: {exc}")
        map_config = MapMatchingConfig(
            distance_threshold_m=args.map_distance_threshold,
            heading_threshold_rad=math.radians(args.map_heading_threshold_deg),
            top_k=max(1, args.map_top_k),
        )

    if args.output_schema == "scene_motion_v3":
        result = export_scene_v3(
            args.scene_dir,
            args.output,
            args.dt,
            args.interaction_csv,
            args.interaction_row,
            args.context_before,
            args.context_after,
            vector_map,
            map_config,
            behavior_config,
            ttc_config,
        )
        summary = {
            "output": str(args.output),
            "schema_version": result["schema_version"],
            "interaction_id": result["interaction_id"],
            "export_frames": result["temporal"]["num_export_frames"],
            "participants": result["interaction"]["participant_ids"],
            "map_matching": result["processing_status"]["map_matching"],
            "behavior_classification": result["processing_status"]["behavior_classification"],
            "behavior_type": result["interaction"]["behavior"]["type"],
        }
    else:
        result = export_scene_v2(
            args.scene_dir,
            args.output,
            args.dt,
            args.interaction_csv,
            args.interaction_row,
            args.context_before,
            args.context_after,
        )
        summary = {
            "output": str(args.output),
            "schema_version": result["schema_version"],
            "scene_id": result["scene_id"],
            "num_frames": result["num_frames"],
            "state_count": result["state_count"],
        }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
