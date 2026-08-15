#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate Qwen descriptions for one vehicle from full_vehicle_facts.jsonl.

Unlike the pair caption runner, this script has no reference vehicle and no
pair prompt.  It asks Qwen to describe only the subject vehicle's own
physical motion: lateral maneuver, speed change, heading motion, route
transition, and vehicle-centered map context.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping

from openai import OpenAI


MODEL_NAME = os.getenv("QWEN_MODEL", "qwen-3.6")
BASE_URL = os.getenv("QWEN_BASE_URL", "http://172.17.0.1:60200/v1")
API_KEY = os.getenv("QWEN_API_KEY", "EMPTY")
TEMPERATURE = 0.2
MAX_TOKENS = 900
MAX_RETRIES = 3
RETRY_WAIT_SECONDS = 3
REQUEST_TIMEOUT_SECONDS = float(os.getenv("QWEN_REQUEST_TIMEOUT_SECONDS", "30"))
TEMPORAL_GROUNDING_MIN_EPISODE_OVERLAP = 0.5
TURN_PREPARATION_MAX_LANE_CHANGE_GAP_FRAMES = 10
OUTPUT_SCHEMA_VERSION = "vehicle_qwen_description_v1"
INPUT_SCHEMA_VERSION = "full_vehicle_facts_v1"
VEHICLE_JSON_DIRNAME = "vehicle_json"
STATIONARY_JSON_DIRNAME = "stationary"
SUSPICIOUS_JSON_DIRNAME = "suspicious"
LOG_DIRNAME = "log"
DESCRIPTION_JSONL_NAME = "vehicle_qwen_descriptions.jsonl"
SUMMARY_CSV_NAME = "vehicle_qwen_descriptions_summary.csv"
STATIONARY_SPEED_THRESHOLD_MPS = 0.1
STATIONARY_P95_SPEED_THRESHOLD_MPS = 0.1
STATIONARY_ENDPOINT_DISPLACEMENT_M = 1.0
STATIONARY_SPATIAL_EXTENT_M = 1.5
STATIONARY_PATH_LENGTH_M = 2.0
MIN_STATIONARY_VALID_FRAMES = 5
INSUFFICIENT_MOTION_STATUSES = {
    "too_few_valid_frames",
    "insufficient_motion_evidence",
}

PAIR_RELATION_RE = re.compile(
    r"\b(?:follows?|followed|yields?|yielded|"
    r"overtakes?|overtook|cuts?\s+in|cut\s+in|"
    r"merges?\s+with|interacts?|gives?\s+way)\b",
    re.IGNORECASE,
)
OTHER_VEHICLE_RE = re.compile(r"\bvehicle\s+(\d+)\b", re.IGNORECASE)
ORDINARY_SHIFT_RE = re.compile(
    r"\b(?:shift|shifts|shifted|shifting|"
    r"drift|drifts|drifted|drifting)\b",
    re.IGNORECASE,
)
TURN_LEFT_RE = re.compile(
    r"\b(?:turn|turns|turned|turning)\s+(?:to\s+the\s+)?left\b"
    r"|\bleft\s+turn\b",
    re.IGNORECASE,
)
TURN_RIGHT_RE = re.compile(
    r"\b(?:turn|turns|turned|turning)\s+(?:to\s+the\s+)?right\b"
    r"|\bright\s+turn\b",
    re.IGNORECASE,
)
LANE_LEFT_RE = re.compile(
    r"\b(?:change|changes|changed|changing)\s+lanes?\s+(?:to\s+the\s+)?left\b"
    r"|\bleft\s+lane\s+change\b",
    re.IGNORECASE,
)
LANE_RIGHT_RE = re.compile(
    r"\b(?:change|changes|changed|changing)\s+lanes?\s+(?:to\s+the\s+)?right\b"
    r"|\bright\s+lane\s+change\b",
    re.IGNORECASE,
)
ACCELERATION_RE = re.compile(
    r"\b(?:accelerat(?:e|es|ed|ing)|speed(?:s)?\s+up|gains?\s+speed)\b",
    re.IGNORECASE,
)
DECELERATION_RE = re.compile(
    r"\b(?:slow(?:s|ed|ing)?|decelerat(?:e|es|ed|ing)|reduces?\s+speed)\b",
    re.IGNORECASE,
)
U_TURN_RE = re.compile(r"\b(?:u[- ]turn|turns?\s+around)\b", re.IGNORECASE)
FORWARD_MOTION_RE = re.compile(
    r"\b(?:continues?|continued|continuing|moves?|moving|"
    r"travels?|traveling|proceeds?|proceeding)\s+(?:straight|forward)\b",
    re.IGNORECASE,
)
STEADY_MOTION_RE = re.compile(
    r"\b(?:maintains?|maintaining|keeps?|keeping)\s+(?:a\s+)?"
    r"(?:steady|constant)\s+(?:speed|pace)\b"
    r"|\bsteady\s+speed\b",
    re.IGNORECASE,
)
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?")

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    timeout=REQUEST_TIMEOUT_SECONDS,
    max_retries=0,
)
logger = logging.getLogger(__name__)

ORDINARY_LATERAL_MOTION_POLICY = (
    "Do not describe ordinary lateral displacement as 'shift', 'shifts', "
    "'shifted', 'drift', or 'drifting'.  A centerline offset or lateral-motion "
    "episode alone is not a captionable shift/drift behavior.  Use lateral "
    "evidence for 'changes lanes to the left/right' only when compatible lane "
    "topology supports a lane change; otherwise omit the ordinary lateral "
    "motion and preserve supported turn, acceleration, deceleration, or "
    "forward-motion descriptions."
)


def load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"input JSON is not an object: {path}")
    return value


def default_prompt_path() -> Path:
    return Path(__file__).resolve().parent / "prompt" / "qwen_generate_vehicle_descriptions_prompt.txt"


def load_prompt(path: Path) -> str:
    value = path.read_text(encoding="utf-8").strip()
    if not value:
        raise ValueError(f"prompt file is empty: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean_json(value: Any) -> Any:
    """Keep only JSON-safe facts and remove accidental generation metadata."""
    if isinstance(value, Mapping):
        return {str(k): _clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean_json(v) for v in value]
    return value


def prompt_input(data: Mapping[str, Any]) -> dict[str, Any]:
    if data.get("schema_version") != INPUT_SCHEMA_VERSION:
        raise ValueError(
            f"expected {INPUT_SCHEMA_VERSION}, got {data.get('schema_version')!r}"
        )
    vehicle_id = data.get("vehicle_id")
    record_id = data.get("vehicle_record_id")
    scene_id = data.get("scene_id")
    if vehicle_id is None or record_id is None or scene_id is None:
        raise ValueError("full vehicle facts require vehicle_record_id, scene_id, and vehicle_id")
    timeline = data.get("timeline")
    vehicle = data.get("vehicle")
    map_context = data.get("map_context")
    track_motion_summary = data.get("track_motion_summary")
    if not isinstance(timeline, Mapping) or not isinstance(vehicle, Mapping):
        raise ValueError("full vehicle facts require timeline and vehicle objects")
    num_frames = timeline.get("num_frames")
    if isinstance(num_frames, bool):
        raise ValueError("timeline.num_frames must be a positive integer")
    try:
        num_frames = int(num_frames)
    except (TypeError, ValueError) as exc:
        raise ValueError("timeline.num_frames must be a positive integer") from exc
    if num_frames <= 0:
        raise ValueError("timeline.num_frames must be a positive integer")
    return {
        "vehicle_record_id": str(record_id),
        "scene_id": str(scene_id),
        "vehicle_id": int(vehicle_id),
        "timeline": {**_clean_json(timeline), "num_frames": num_frames},
        "trajectory_quality": _clean_json(
            data.get("trajectory_quality", {})
            if isinstance(data.get("trajectory_quality", {}), Mapping)
            else {}
        ),
        "track_motion_summary": _clean_json(
            track_motion_summary if isinstance(track_motion_summary, Mapping) else {}
        ),
        "vehicle": _clean_json(vehicle),
        "map_context": _clean_json(map_context if isinstance(map_context, Mapping) else {}),
    }


def _episode_span(episodes: Any) -> tuple[int | None, int | None]:
    """Return the union of valid episode frames, or ``(None, None)``."""
    if not isinstance(episodes, list):
        return None, None
    spans = []
    for episode in episodes:
        if not isinstance(episode, Mapping):
            continue
        try:
            start = int(episode["start_frame"])
            end = int(episode["end_frame"])
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= start <= end:
            spans.append((start, end))
    if not spans:
        return None, None
    return min(start for start, _ in spans), max(end for _, end in spans)


def _baseline_motion_episode(facts: Mapping[str, Any]) -> dict[str, Any] | None:
    """Build a forward/steady baseline only for a genuinely moving track."""
    summary = facts.get("track_motion_summary")
    timeline = facts.get("timeline")
    if not isinstance(summary, Mapping) or not isinstance(timeline, Mapping):
        return None
    try:
        valid_count = int(summary.get("valid_frame_count"))
        max_speed = float(summary.get("max_speed_mps"))
        endpoint = float(summary.get("endpoint_displacement_m"))
        path_length = float(summary.get("path_length_m"))
    except (TypeError, ValueError):
        return None
    if valid_count < MIN_STATIONARY_VALID_FRAMES:
        return None
    if not all(
        value == value and abs(value) != float("inf")
        for value in (max_speed, endpoint, path_length)
    ):
        return None
    moving = (
        max_speed > STATIONARY_SPEED_THRESHOLD_MPS
        or endpoint > STATIONARY_ENDPOINT_DISPLACEMENT_M
        or path_length > STATIONARY_PATH_LENGTH_M
    )
    if not moving:
        return None
    valid_indices = []
    for value in timeline.get("valid_frame_indices", []):
        try:
            valid_indices.append(int(value))
        except (TypeError, ValueError):
            continue
    if not valid_indices:
        return None

    vehicle = facts.get("vehicle", {})
    speed_summary = (
        vehicle.get("speed_change", {})
        if isinstance(vehicle, Mapping)
        else {}
    )
    speed_summary = (
        speed_summary.get("speed_summary", {})
        if isinstance(speed_summary, Mapping)
        else {}
    )
    try:
        speed_span = float(speed_summary["max_mps"]) - float(speed_summary["min_mps"])
    except (KeyError, TypeError, ValueError):
        speed_span = None
    motion_type = (
        "steady_motion"
        if speed_span is not None and speed_span <= 1.0
        else "forward_motion"
    )
    return {
        "type": motion_type,
        "start_frame": min(valid_indices),
        "end_frame": max(valid_indices),
        "source": "track_motion_summary.baseline_motion",
    }


def extract_required_behavior_episodes(
    facts: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Extract confirmed behavior episodes for temporal grounding."""
    vehicle = facts.get("vehicle", {})
    if not isinstance(vehicle, Mapping):
        return []

    candidates: list[dict[str, Any]] = []
    turn_preparation_ranges: list[tuple[int, int, str | None]] = []
    u_turn = vehicle.get("u_turn_evidence", {})
    u_turn_supported = (
        isinstance(u_turn, Mapping)
        and u_turn.get("status") == "supported"
    )
    if u_turn_supported:
        start, end = _episode_span([u_turn])
        candidates.append({
            "type": "u_turn",
            "start_frame": start,
            "end_frame": end,
            "source": "vehicle.u_turn_evidence",
        })
        if start is not None and end is not None:
            turn_preparation_ranges.append((start, end, None))

    turn = vehicle.get("turn_maneuver", {})
    if not u_turn_supported and isinstance(turn, Mapping):
        turn_status = str(turn.get("status", ""))
        if turn_status in {"left_turn", "right_turn"}:
            heading = vehicle.get("heading_motion", {})
            heading_episode = (
                heading.get("turning_episode")
                if isinstance(heading, Mapping)
                else None
            )
            start = end = None
            temporal_source = "vehicle.turn_maneuver.entry_exit_fallback"
            if isinstance(heading_episode, Mapping):
                try:
                    candidate_start = int(heading_episode["start_frame"])
                    candidate_end = int(heading_episode["end_frame"])
                except (KeyError, TypeError, ValueError):
                    candidate_start = candidate_end = None
                if (
                    candidate_start is not None
                    and candidate_end is not None
                    and 0 <= candidate_start <= candidate_end
                ):
                    start = candidate_start
                    end = candidate_end
                    temporal_source = "vehicle.heading_motion.turning_episode"
            if start is None or end is None:
                entry = turn.get("entry")
                exit_ = turn.get("exit")
                ranges = [item for item in (entry, exit_) if isinstance(item, Mapping)]
                start, end = _episode_span(ranges)
            candidates.append({
                "type": turn_status,
                "start_frame": start,
                "end_frame": end,
                "source": temporal_source,
            })
            if start is not None and end is not None:
                turn_direction = (
                    "left" if turn_status == "left_turn" else "right"
                )
                turn_preparation_ranges.append(
                    (start, end, turn_direction)
                )

    lane_chain = vehicle.get("physical_lane_chain", {})
    if isinstance(lane_chain, Mapping):
        lane_type = {
            "left_lane_change": "lane_change_left",
            "right_lane_change": "lane_change_right",
        }.get(str(lane_chain.get("behavior")))
        if lane_type:
            transitions = lane_chain.get("lane_change_evidence", [])
            transition_added = False
            transition_suppressed = False
            if isinstance(transitions, list):
                for transition_index, transition in enumerate(transitions):
                    if not isinstance(transition, Mapping):
                        continue
                    try:
                        transition_frame = int(transition["transition_frame"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    lane_direction = (
                        "left" if lane_type == "lane_change_left" else "right"
                    )
                    is_turn_preparation = any(
                        (
                            turn_direction is None
                            or lane_direction == turn_direction
                        )
                        and (
                            turn_start
                            - TURN_PREPARATION_MAX_LANE_CHANGE_GAP_FRAMES
                            <= transition_frame <= turn_end
                        )
                        for turn_start, turn_end, turn_direction
                        in turn_preparation_ranges
                    )
                    if is_turn_preparation:
                        transition_suppressed = True
                        continue
                    candidates.append({
                        "type": lane_type,
                        "start_frame": transition_frame,
                        "end_frame": transition_frame,
                        "source": (
                            "vehicle.physical_lane_chain.lane_change_evidence"
                            f"[{transition_index}]"
                        ),
                    })
                    transition_added = True
            if not transition_added and not transition_suppressed:
                candidates.append({
                    "type": lane_type,
                    "start_frame": None,
                    "end_frame": None,
                    "source": "vehicle.physical_lane_chain",
                })

    speed = vehicle.get("speed_change", {})
    if isinstance(speed, Mapping):
        for field, behavior_type in (
            ("acceleration_episodes", "acceleration"),
            ("deceleration_episodes", "deceleration"),
        ):
            episodes = speed.get(field)
            if not isinstance(episodes, list):
                continue
            for episode in episodes:
                if not isinstance(episode, Mapping):
                    continue
                try:
                    start = int(episode["start_frame"])
                    end = int(episode["end_frame"])
                except (KeyError, TypeError, ValueError):
                    continue
                if not 0 <= start <= end:
                    continue
                candidates.append({
                    "type": behavior_type,
                    "start_frame": start,
                    "end_frame": end,
                    "source": f"vehicle.speed_change.{field}",
                })

    specific_types = {
        "u_turn",
        "left_turn",
        "right_turn",
        "lane_change_left",
        "lane_change_right",
        "acceleration",
        "deceleration",
    }
    if not any(item["type"] in specific_types for item in candidates):
        baseline = _baseline_motion_episode(facts)
        if baseline is not None:
            candidates.append(baseline)

    return sorted(
        candidates,
        key=lambda item: (
            item["start_frame"] is None,
            item["start_frame"] if item["start_frame"] is not None else 0,
            item["type"],
        ),
    )


def extract_required_major_behaviors(
    facts: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Extract type-level caption requirements from confirmed episodes."""
    episodes = extract_required_behavior_episodes(facts)
    grouped: dict[str, dict[str, Any]] = {}
    for episode in episodes:
        behavior_type = str(episode["type"])
        current = grouped.get(behavior_type)
        if current is None:
            grouped[behavior_type] = {
                "type": behavior_type,
                "start_frame": episode.get("start_frame"),
                "end_frame": episode.get("end_frame"),
                "source": "aggregated_from_required_behavior_episodes",
            }
            continue
        start = episode.get("start_frame")
        end = episode.get("end_frame")
        if start is not None:
            current["start_frame"] = (
                start
                if current["start_frame"] is None
                else min(int(current["start_frame"]), int(start))
            )
        if end is not None:
            current["end_frame"] = (
                end
                if current["end_frame"] is None
                else max(int(current["end_frame"]), int(end))
            )
    return sorted(
        grouped.values(),
        key=lambda item: (
            item["start_frame"] is None,
            item["start_frame"] if item["start_frame"] is not None else 0,
            item["type"],
        ),
    )


def build_vehicle_prompt(
    data: Mapping[str, Any],
    required_major_behaviors: list[Mapping[str, Any]] | None = None,
    required_behavior_episodes: list[Mapping[str, Any]] | None = None,
) -> str:
    facts = prompt_input(data)
    required = required_major_behaviors
    if required is None:
        required = extract_required_major_behaviors(facts)
    episodes = required_behavior_episodes
    if episodes is None:
        episodes = extract_required_behavior_episodes(facts)
    return (
        "Grounded single-vehicle facts (the only source of truth):\n"
        f"{json.dumps(facts, ensure_ascii=False, indent=2, allow_nan=False)}\n\n"
        "Required major behaviors extracted deterministically from the facts "
        "(the short caption must express every item; do not add or remove items):\n"
        f"{json.dumps(required, ensure_ascii=False, indent=2, allow_nan=False)}\n\n"
        "Required behavior episodes for temporal grounding of supporting ranges "
        "and behavior segments:\n"
        f"{json.dumps(episodes, ensure_ascii=False, indent=2, allow_nan=False)}\n\n"
        "Additional hard lateral-motion rule:\n"
        f"{ORDINARY_LATERAL_MOTION_POLICY}\n\n"
        "Generation policy: behavior_segments are distinct informative behavior "
        "phases, not repetitions of every supporting range. Every confirmed major "
        "behavior should have at least one corresponding segment.\n\n"
        "Write the JSON response now."
    )


def _valid_frame_bounds(data: Mapping[str, Any]) -> tuple[int | None, int | None]:
    facts = prompt_input(data)
    timeline = facts["timeline"]
    indices = timeline.get("valid_frame_indices", [])
    valid = []
    if isinstance(indices, list):
        for value in indices:
            try:
                frame = int(value)
            except (TypeError, ValueError):
                continue
            if 0 <= frame < int(timeline["num_frames"]):
                valid.append(frame)
    if valid:
        return min(valid), max(valid)
    return None, None


def derive_track_motion_status(facts: Mapping[str, Any]) -> dict[str, Any]:
    """Classify observation sufficiency before stationary/Qwen routing.

    This only interprets the existing track_motion_summary fields.  It does
    not infer a vehicle behavior or add a second motion detector.
    """
    summary = facts.get("track_motion_summary")
    if not isinstance(summary, Mapping):
        return {"status": "insufficient_motion_evidence"}

    raw_valid_count = summary.get("valid_frame_count")
    try:
        valid_count = int(raw_valid_count)
    except (TypeError, ValueError):
        return {
            **_clean_json(summary),
            "status": "insufficient_motion_evidence",
        }
    if valid_count < MIN_STATIONARY_VALID_FRAMES:
        return {
            **_clean_json(summary),
            "status": "too_few_valid_frames",
        }

    required_metrics = (
        "max_speed_mps",
        "p95_speed_mps",
        "endpoint_displacement_m",
        "spatial_extent_m",
        "path_length_m",
    )
    try:
        metrics = {
            name: float(summary.get(name))
            for name in required_metrics
        }
    except (TypeError, ValueError):
        return {
            **_clean_json(summary),
            "status": "insufficient_motion_evidence",
        }
    if not all(value == value and abs(value) != float("inf") for value in metrics.values()):
        return {
            **_clean_json(summary),
            "status": "insufficient_motion_evidence",
        }
    return {
        **_clean_json(summary),
        "status": "available",
    }


def is_track_stationary(motion_status: Mapping[str, Any]) -> bool:
    """Return true only for sufficiently observed, near-zero-motion tracks."""
    if motion_status.get("status") != "available":
        return False
    try:
        return (
            float(motion_status["max_speed_mps"]) <= STATIONARY_SPEED_THRESHOLD_MPS
            and float(motion_status["p95_speed_mps"]) <= STATIONARY_P95_SPEED_THRESHOLD_MPS
            and float(motion_status["endpoint_displacement_m"]) <= STATIONARY_ENDPOINT_DISPLACEMENT_M
            and float(motion_status["spatial_extent_m"]) <= STATIONARY_SPATIAL_EXTENT_M
            and float(motion_status["path_length_m"]) <= STATIONARY_PATH_LENGTH_M
        )
    except (TypeError, ValueError):
        return False


def build_stationary_caption(vehicle_id: str | int) -> dict[str, Any]:
    """Build the single fixed stationary caption without calling the LLM."""
    return {
        "description_short": "The target vehicle remains stationary.",
        "description_detailed": (
            f"Vehicle {vehicle_id} remains stationary throughout the valid observation."
        ),
        "supporting_frame_ranges": [],
        "behavior_segments": [],
        "uncertainty_notes": "",
    }


def build_insufficient_observation_result(vehicle_id: str | int) -> dict[str, Any]:
    """Build a fixed result when temporal evidence is insufficient."""
    return {
        "description_short": "Insufficient observation.",
        "description_detailed": (
            f"Vehicle {vehicle_id} has insufficient temporal observation "
            "for a reliable motion description."
        ),
        "supporting_frame_ranges": [],
        "behavior_segments": [],
        "uncertainty_notes": (
            "Too few valid observations are available for reliable temporal interpretation."
        ),
    }


def clean_json(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def has_pair_semantics(text: str, vehicle_id: int) -> bool:
    if PAIR_RELATION_RE.search(text):
        return True
    return any(
        int(match.group(1)) != int(vehicle_id)
        for match in OTHER_VEHICLE_RE.finditer(text)
    )


def reject_ordinary_shift_drift(text: str, field_name: str) -> None:
    """Reject forbidden ordinary lateral-motion wording in generated text."""
    if ORDINARY_SHIFT_RE.search(text):
        raise ValueError(
            f"ordinary shift/drift wording is forbidden in {field_name}"
        )


def expected_turn_direction(data: Mapping[str, Any]) -> str | None:
    """Return the confirmed turn direction, with legacy heading fallback."""
    vehicle = data.get("vehicle", {})
    if not isinstance(vehicle, Mapping):
        return None
    turn = vehicle.get("turn_maneuver", {})
    if isinstance(turn, Mapping):
        status = turn.get("status")
        if status == "left_turn":
            return "left"
        if status == "right_turn":
            return "right"
    heading = vehicle.get("heading_motion", {})
    if not isinstance(heading, Mapping):
        return None
    episode = heading.get("turning_episode", {})
    if not isinstance(episode, Mapping):
        return None
    direction = episode.get("direction")
    if direction == "increasing_ccw":
        return "left"
    if direction == "decreasing_cw":
        return "right"
    return None


def reject_conflicting_turn_direction(
    text: str,
    field_name: str,
    expected_direction: str | None,
) -> None:
    """Reject only an opposite turn label; do not require the model to mention a turn."""
    if expected_direction not in {"left", "right"}:
        return
    opposite = "right" if expected_direction == "left" else "left"
    pattern = TURN_RIGHT_RE if opposite == "right" else TURN_LEFT_RE
    if pattern.search(text):
        heading_direction = (
            "increasing_ccw" if expected_direction == "left" else "decreasing_cw"
        )
        raise ValueError(
            "turn direction conflicts with heading_motion: "
            f"{heading_direction} requires {expected_direction} turn"
            f" in {field_name}"
        )


def _major_behavior_patterns(behavior_type: str) -> tuple[re.Pattern[str], ...]:
    patterns = {
        "u_turn": (U_TURN_RE,),
        "left_turn": (TURN_LEFT_RE,),
        "right_turn": (TURN_RIGHT_RE,),
        "lane_change_left": (LANE_LEFT_RE,),
        "lane_change_right": (LANE_RIGHT_RE,),
        "acceleration": (ACCELERATION_RE,),
        "deceleration": (DECELERATION_RE,),
        "forward_motion": (FORWARD_MOTION_RE, STEADY_MOTION_RE),
        "steady_motion": (FORWARD_MOTION_RE, STEADY_MOTION_RE),
    }
    return patterns.get(behavior_type, ())


def major_behaviors_covered(
    text: str,
    required_major_behaviors: list[Mapping[str, Any]],
) -> list[str]:
    covered = []
    for behavior in required_major_behaviors:
        behavior_type = str(behavior.get("type", ""))
        if any(pattern.search(text) for pattern in _major_behavior_patterns(behavior_type)):
            covered.append(behavior_type)
    return covered


def validate_major_behavior_coverage(
    text: str,
    required_major_behaviors: list[Mapping[str, Any]],
    field_name: str = "description_short",
) -> dict[str, Any]:
    required = [str(item.get("type", "")) for item in required_major_behaviors]
    covered = major_behaviors_covered(text, required_major_behaviors)
    missing = [behavior_type for behavior_type in required if behavior_type not in covered]
    if missing:
        raise ValueError(
            f"{field_name} does not cover required major behaviors: {', '.join(missing)}"
        )
    return {
        "required": required,
        "covered": covered,
        "missing": missing,
        "all_required_covered": not missing,
    }


def reject_unrequired_lane_change(
    text: str,
    field_name: str,
    required_major_behaviors: list[Mapping[str, Any]],
) -> None:
    """Reject lane-change wording that is not a required behavior."""
    required_types = {
        str(item.get("type", ""))
        for item in required_major_behaviors
    }
    if (
        LANE_LEFT_RE.search(text)
        and "lane_change_left" not in required_types
    ):
        raise ValueError(
            f"unrequired left lane change detected in {field_name}"
        )
    if (
        LANE_RIGHT_RE.search(text)
        and "lane_change_right" not in required_types
    ):
        raise ValueError(
            f"unrequired right lane change detected in {field_name}"
        )


def _segment_behavior_signature(
    description: str,
    required_major_behaviors: list[Mapping[str, Any]],
) -> tuple[str, ...]:
    return tuple(major_behaviors_covered(description, required_major_behaviors))


def segments_are_duplicate(
    a: Mapping[str, Any],
    b: Mapping[str, Any],
    required_behavior_episodes: list[Mapping[str, Any]],
) -> bool:
    if _segment_behavior_signature(
        str(a["description"]), required_behavior_episodes
    ) != _segment_behavior_signature(
        str(b["description"]), required_behavior_episodes
    ):
        return False
    overlap_start = max(int(a["start_frame"]), int(b["start_frame"]))
    overlap_end = min(int(a["end_frame"]), int(b["end_frame"]))
    if overlap_start > overlap_end:
        return False
    overlap = overlap_end - overlap_start + 1
    a_length = int(a["end_frame"]) - int(a["start_frame"]) + 1
    b_length = int(b["end_frame"]) - int(b["start_frame"]) + 1
    shorter = min(a_length, b_length)
    return shorter > 0 and overlap / shorter >= 0.8


def filter_behavior_segments(
    segments: list[dict[str, Any]],
    required_behavior_episodes: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Keep confirmed behavior phases and remove only temporal duplicates."""
    filtered: list[dict[str, Any]] = []
    for segment in segments:
        signature = _segment_behavior_signature(
            str(segment["description"]), required_behavior_episodes
        )
        if not signature:
            continue
        if any(
            segments_are_duplicate(segment, existing, required_behavior_episodes)
            for existing in filtered
        ):
            continue
        filtered.append(segment)
    return sorted(
        filtered,
        key=lambda item: (
            int(item["start_frame"]),
            int(item["end_frame"]),
        ),
    )


def _required_episode_overlap_ratio(
    candidate: Mapping[str, Any],
    required: Mapping[str, Any],
) -> float:
    """Return candidate/required temporal overlap, normalized by the episode."""
    required_start = required.get("start_frame")
    required_end = required.get("end_frame")
    if required_start is None or required_end is None:
        return 1.0
    try:
        overlap_start = max(int(required_start), int(candidate["start_frame"]))
        overlap_end = min(int(required_end), int(candidate["end_frame"]))
    except (KeyError, TypeError, ValueError):
        return 0.0
    if overlap_start > overlap_end:
        return 0.0
    required_length = int(required_end) - int(required_start) + 1
    if required_length <= 0:
        return 0.0
    return (overlap_end - overlap_start + 1) / required_length


def validate_behavior_segment_coverage(
    segments: list[Mapping[str, Any]],
    required_behavior_episodes: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Require one semantically and temporally matching segment per episode."""
    required = [
        {
            "index": index,
            "type": str(item.get("type", "")),
            "start_frame": item.get("start_frame"),
            "end_frame": item.get("end_frame"),
        }
        for index, item in enumerate(required_behavior_episodes)
    ]
    covered: list[int] = []
    episode_matches: list[dict[str, Any]] = []
    used_segment_type_pairs: set[tuple[int, str]] = set()

    for required_item in required:
        required_index = int(required_item["index"])
        behavior_type = str(required_item["type"])
        candidates: list[tuple[float, int]] = []
        for segment_index, segment in enumerate(segments):
            if (segment_index, behavior_type) in used_segment_type_pairs:
                continue
            if behavior_type not in _segment_behavior_signature(
                str(segment["description"]), required_behavior_episodes
            ):
                continue
            start = required_item["start_frame"]
            end = required_item["end_frame"]
            if start is None or end is None:
                candidates.append((1.0, segment_index))
                continue
            ratio = _required_episode_overlap_ratio(segment, required_item)
            if ratio >= TEMPORAL_GROUNDING_MIN_EPISODE_OVERLAP:
                candidates.append((ratio, segment_index))

        if candidates:
            ratio, segment_index = max(candidates, key=lambda item: item[0])
            used_segment_type_pairs.add((segment_index, behavior_type))
            covered.append(required_index)
            episode_matches.append({
                "required_index": required_index,
                "segment_index": segment_index,
                "overlap_ratio": round(ratio, 3),
            })

    missing = [
        int(item["index"])
        for item in required
        if int(item["index"]) not in covered
    ]
    if missing:
        missing_labels = [
            f"{item['type']}[{item['index']}]"
            for item in required
            if int(item["index"]) in missing
        ]
        raise ValueError(
            "behavior_segments do not cover required major behaviors: "
            + ", ".join(missing_labels)
        )
    return {
        "required": required,
        "covered_episode_indices": sorted(covered),
        "episode_matches": episode_matches,
        "missing": missing,
        "all_required_covered": not missing,
    }


def validate_supporting_frame_range_grounding(
    supporting_ranges: list[Mapping[str, Any]],
    required_behavior_episodes: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Ground major-behavior supporting ranges against upstream episodes."""
    grounded_ranges: list[dict[str, Any]] = []
    for range_index, frame_range in enumerate(supporting_ranges):
        signature = _segment_behavior_signature(
            str(frame_range["description"]),
            required_behavior_episodes,
        )
        if not signature:
            continue
        declared_behavior_types = set(signature)
        matched_behavior_types: set[str] = set()
        matches = []
        for required_index, required in enumerate(required_behavior_episodes):
            behavior_type = str(required.get("type", ""))
            if behavior_type not in signature:
                continue
            ratio = _required_episode_overlap_ratio(frame_range, required)
            if ratio >= TEMPORAL_GROUNDING_MIN_EPISODE_OVERLAP:
                matched_behavior_types.add(behavior_type)
                matches.append({
                    "required_index": required_index,
                    "overlap_ratio": round(ratio, 3),
                })
        missing_behavior_types = sorted(
            declared_behavior_types - matched_behavior_types
        )
        if missing_behavior_types:
            raise ValueError(
                "supporting_frame_ranges[{}] is not temporally grounded for "
                "declared behaviors: {}".format(
                    range_index,
                    ", ".join(missing_behavior_types),
                )
            )
        grounded_ranges.append({
            "range_index": range_index,
            "matches": matches,
        })
    return {
        "min_episode_overlap": TEMPORAL_GROUNDING_MIN_EPISODE_OVERLAP,
        "checked_ranges": grounded_ranges,
        "all_checked_ranges_grounded": True,
    }


def validate_behavior_segments(
    segments: Any,
    supporting_ranges: list[Mapping[str, Any]],
    num_frames: int,
    expected_direction: str | None = None,
) -> list[dict[str, Any]]:
    """Keep schema-valid LLM segments without inferring or merging behavior."""
    if segments is None:
        return []
    if not isinstance(segments, list):
        logger.warning("Dropping behavior_segments because it is not a list")
        return []

    expected_keys = {
        "start_frame",
        "end_frame",
        "description",
        "supporting_range_indices",
    }
    valid_segments: list[dict[str, Any]] = []

    def drop(index: int, reason: str) -> None:
        logger.warning("Dropping invalid behavior segment %d: %s", index, reason)

    for index, segment in enumerate(segments):
        if not isinstance(segment, Mapping):
            drop(index, "segment is not an object")
            continue
        if set(segment) != expected_keys:
            drop(index, "segment keys must be exactly the behavior segment schema")
            continue
        start = segment.get("start_frame")
        end = segment.get("end_frame")
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
        ):
            drop(index, "start_frame and end_frame must be integers")
            continue
        if not (0 <= start <= end < int(num_frames)):
            drop(index, "segment frame bounds are outside the timeline")
            continue
        description = segment.get("description")
        if not isinstance(description, str) or not description.strip():
            drop(index, "segment description is empty")
            continue
        reject_ordinary_shift_drift(
            description,
            f"behavior_segments[{index}].description",
        )
        reject_conflicting_turn_direction(
            description,
            f"behavior_segments[{index}].description",
            expected_direction,
        )
        range_indices = segment.get("supporting_range_indices")
        if not isinstance(range_indices, list) or not range_indices:
            drop(index, "supporting_range_indices must be a non-empty list")
            continue
        if any(
            isinstance(range_index, bool) or not isinstance(range_index, int)
            for range_index in range_indices
        ):
            drop(index, "supporting_range_indices must contain integers")
            continue
        if any(
            range_index < 0 or range_index >= len(supporting_ranges)
            for range_index in range_indices
        ):
            drop(index, "supporting_range_indices references missing evidence")
            continue
        evidence_start = min(
            int(supporting_ranges[range_index]["start_frame"])
            for range_index in range_indices
        )
        evidence_end = max(
            int(supporting_ranges[range_index]["end_frame"])
            for range_index in range_indices
        )
        if start > evidence_start or end < evidence_end:
            drop(index, "segment does not cover its referenced evidence")
            continue
        valid_segments.append({
            "start_frame": start,
            "end_frame": end,
            "description": description.strip(),
            "supporting_range_indices": list(range_indices),
        })
    return valid_segments


def validate_model_result(
    result: Any,
    vehicle_id: int,
    num_frames: int,
    short_word_range: tuple[int, int] = (4, 18),
    expected_direction: str | None = None,
    required_major_behaviors: list[Mapping[str, Any]] | None = None,
    required_behavior_episodes: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if not isinstance(result, Mapping):
        raise ValueError("Qwen output is not a JSON object")
    short = result.get("description_short")
    detailed = result.get("description_detailed")
    ranges = result.get("supporting_frame_ranges")
    behavior_segments = result.get("behavior_segments", [])
    uncertainty = result.get("uncertainty_notes")
    if not isinstance(short, str) or not short.strip():
        raise ValueError("missing description_short")
    reject_ordinary_shift_drift(short, "description_short")
    reject_conflicting_turn_direction(
        short,
        "description_short",
        expected_direction,
    )
    short_count = word_count(short.strip())
    min_words, max_words = short_word_range
    if not min_words <= short_count <= max_words:
        raise ValueError(
            f"description_short must contain {min_words}-{max_words} words, got {short_count}"
        )
    if not re.search(r"\bthe\s+target\s+vehicle\b", short, flags=re.IGNORECASE):
        raise ValueError("description_short must refer to the target vehicle")
    required = required_major_behaviors or []
    grounding_requirements = (
        required_behavior_episodes
        if required_behavior_episodes is not None
        else required
    )
    reject_unrequired_lane_change(short, "description_short", required)
    coverage = validate_major_behavior_coverage(short, required)
    if has_pair_semantics(short, vehicle_id) or has_pair_semantics(str(detailed or ""), vehicle_id):
        raise ValueError("pair-level semantic wording detected")
    if not isinstance(detailed, str) or not detailed.strip():
        raise ValueError("missing description_detailed")
    reject_unrequired_lane_change(detailed, "description_detailed", required)
    reject_ordinary_shift_drift(detailed, "description_detailed")
    reject_conflicting_turn_direction(
        detailed,
        "description_detailed",
        expected_direction,
    )
    if not re.search(
        rf"\bVehicle\s+{int(vehicle_id)}\b", detailed, flags=re.IGNORECASE
    ):
        raise ValueError(
            f"description_detailed must use Vehicle {int(vehicle_id)}"
        )
    if not isinstance(ranges, list):
        raise ValueError("supporting_frame_ranges must be an array")
    for index, item in enumerate(ranges):
        if isinstance(item, Mapping) and isinstance(item.get("description"), str):
            reject_ordinary_shift_drift(
                item["description"],
                f"supporting_frame_ranges[{index}].description",
            )
            reject_conflicting_turn_direction(
                item["description"],
                f"supporting_frame_ranges[{index}].description",
                expected_direction,
            )
            reject_unrequired_lane_change(
                item["description"],
                f"supporting_frame_ranges[{index}].description",
                required,
            )
    if isinstance(behavior_segments, list):
        for index, item in enumerate(behavior_segments):
            if isinstance(item, Mapping) and isinstance(item.get("description"), str):
                reject_ordinary_shift_drift(
                    item["description"],
                    f"behavior_segments[{index}].description",
                )
                reject_conflicting_turn_direction(
                    item["description"],
                    f"behavior_segments[{index}].description",
                    expected_direction,
                )
                reject_unrequired_lane_change(
                    item["description"],
                    f"behavior_segments[{index}].description",
                    required,
                )
    normalized_ranges: list[dict[str, Any]] = []
    for item in ranges:
        if not isinstance(item, Mapping):
            raise ValueError("supporting_frame_ranges item is not an object")
        try:
            start = int(item["start_frame"])
            end = int(item["end_frame"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("supporting_frame_ranges requires integer start/end frames") from exc
        if not (0 <= start <= end < int(num_frames)):
            raise ValueError(
                "supporting frame range must satisfy "
                f"0 <= start_frame <= end_frame < {int(num_frames)}"
            )
        description = item.get("description")
        if not isinstance(description, str) or not description.strip():
            raise ValueError("supporting frame description is missing")
        reject_ordinary_shift_drift(
            description,
            f"supporting_frame_ranges[{len(normalized_ranges)}].description",
        )
        reject_conflicting_turn_direction(
            description,
            f"supporting_frame_ranges[{len(normalized_ranges)}].description",
            expected_direction,
        )
        if has_pair_semantics(description, vehicle_id):
            raise ValueError("pair-level semantic wording detected in supporting range")
        if not re.search(
            rf"\bVehicle\s+{int(vehicle_id)}\b",
            description,
            flags=re.IGNORECASE,
        ):
            raise ValueError(
                "supporting frame description must use "
                f"Vehicle {int(vehicle_id)}"
            )
        normalized_ranges.append({
            "start_frame": start,
            "end_frame": end,
            "description": description.strip(),
        })
    if not isinstance(uncertainty, str):
        raise ValueError("uncertainty_notes must be a string")
    supporting_grounding = validate_supporting_frame_range_grounding(
        normalized_ranges,
        grounding_requirements,
    )
    normalized_segments = validate_behavior_segments(
        behavior_segments,
        normalized_ranges,
        int(num_frames),
        expected_direction=expected_direction,
    )
    normalized_segments = filter_behavior_segments(
        normalized_segments,
        grounding_requirements,
    )
    segment_coverage = validate_behavior_segment_coverage(
        normalized_segments,
        grounding_requirements,
    )
    return {
        "description_short": short.strip(),
        "description_detailed": detailed.strip(),
        "supporting_frame_ranges": normalized_ranges,
        "behavior_segments": normalized_segments,
        "uncertainty_notes": uncertainty.strip(),
        "major_behavior_coverage": coverage,
        "behavior_segment_coverage": segment_coverage,
        "supporting_frame_grounding": supporting_grounding,
    }


def call_qwen(data: Mapping[str, Any], template: str) -> dict[str, Any]:
    facts = prompt_input(data)
    required_major_behaviors = extract_required_major_behaviors(facts)
    required_behavior_episodes = extract_required_behavior_episodes(facts)
    prompt = build_vehicle_prompt(
        data,
        required_major_behaviors,
        required_behavior_episodes,
    )
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            request_kwargs = {
                "model": MODEL_NAME,
                "messages": [
                    {"role": "system", "content": template},
                    {"role": "user", "content": prompt},
                ],
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS,
                "stream": False,
                "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            }
            thinking_type = os.getenv("QWEN_THINKING_TYPE", "").strip()
            if thinking_type:
                request_kwargs["extra_body"]["thinking"] = {"type": thinking_type}
            response = client.chat.completions.create(
                **request_kwargs,
            )
            message = response.choices[0].message
            content = message.content
            if not content:
                raise ValueError("Qwen returned empty content")
            return validate_model_result(
                json.loads(clean_json(content)),
                int(facts["vehicle_id"]),
                int(facts["timeline"]["num_frames"]),
                expected_direction=expected_turn_direction(facts),
                required_major_behaviors=required_major_behaviors,
                required_behavior_episodes=required_behavior_episodes,
            )
        except Exception as exc:
            last_error = exc
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_WAIT_SECONDS * attempt)
    raise RuntimeError(f"Qwen request failed after {MAX_RETRIES} attempts: {last_error}")


def output_name(data: Mapping[str, Any]) -> str:
    facts = prompt_input(data)
    return "{}__vehicle_{}_vehicle_qwen_description.json".format(
        re.sub(r"[^A-Za-z0-9_.-]+", "_", str(facts["scene_id"])).strip("._") or "unknown",
        int(facts["vehicle_id"]),
    )


def process_record(
    data: Mapping[str, Any],
    output_path: Path,
    template: str,
    input_file: str,
    input_line: int,
    prompt_file: str,
    prompt_hash: str,
) -> str:
    facts = prompt_input(data)
    required_major_behaviors = extract_required_major_behaviors(facts)
    required_behavior_episodes = extract_required_behavior_episodes(facts)
    trajectory_quality = facts.get("trajectory_quality", {})
    quality_status = (
        trajectory_quality.get("status")
        if isinstance(trajectory_quality, Mapping)
        else None
    )
    suspicious = quality_status == "suspicious"
    result = None
    generation_mode = ""
    motion_status = derive_track_motion_status(facts)
    if result is None:
        if motion_status["status"] in INSUFFICIENT_MOTION_STATUSES:
            result = build_insufficient_observation_result(facts["vehicle_id"])
            generation_mode = "insufficient_observation"
        elif is_track_stationary(motion_status):
            result = build_stationary_caption(facts["vehicle_id"])
            generation_mode = "stationary_template"
        else:
            result = call_qwen(data, template)
            generation_mode = "qwen_suspicious_trajectory" if suspicious else "qwen"
    stationary = generation_mode == "stationary_template"
    output = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "vehicle_record_id": facts["vehicle_record_id"],
        "scene_id": facts["scene_id"],
        "vehicle_id": facts["vehicle_id"],
        "generation_status": result.get("generation_status", "generated"),
        "trajectory_quality": facts.get("trajectory_quality", {}),
        "required_major_behaviors": required_major_behaviors,
        "required_behavior_episodes": required_behavior_episodes,
        "track_motion_summary": motion_status,
        "description_short": result["description_short"],
        "description_detailed": result["description_detailed"],
        "supporting_frame_ranges": result["supporting_frame_ranges"],
        "behavior_segments": result["behavior_segments"],
        "uncertainty_notes": result["uncertainty_notes"],
        "generation": {
            "model": MODEL_NAME,
            "base_url": BASE_URL,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
            "thinking": "disabled",
            "prompt_file": prompt_file,
            "prompt_sha256": prompt_hash,
            "input_file": input_file,
            "input_line": int(input_line),
            "generation_mode": generation_mode,
            "trajectory_quality_action": (
                "generated_with_suspicious_flag" if suspicious else "normal"
            ),
            "generation_status": result.get("generation_status", "generated"),
            "required_major_behaviors": required_major_behaviors,
            "required_behavior_episodes": required_behavior_episodes,
            "major_behavior_coverage": result.get("major_behavior_coverage", {}),
            "behavior_segment_coverage": result.get("behavior_segment_coverage", {}),
            "supporting_frame_grounding": result.get("supporting_frame_grounding", {}),
            "temporal_grounding_min_episode_overlap": TEMPORAL_GROUNDING_MIN_EPISODE_OVERLAP,
            "behavior_segments_policy": (
                "confirmed_major_behavior_phases_only;_deduplicate_only_when_same_semantics_and_temporally_overlapping"
            ),
            "stationary_policy": "template",
            "stationary_speed_threshold_mps": STATIONARY_SPEED_THRESHOLD_MPS,
            "stationary_detected": stationary,
            "caption_source": generation_mode,
        },
    }
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return generation_mode


SUMMARY_FIELDS = [
    "input_file",
    "input_line",
    "output_file",
    "scene_id",
    "vehicle_id",
    "generation_mode",
    "generation_status",
    "required_major_behaviors",
    "required_behavior_episodes",
    "major_behavior_coverage",
    "behavior_segment_coverage",
    "description_short",
    "description_detailed",
    "supporting_frame_ranges",
    "behavior_segments",
    "uncertainty_notes",
]


def iter_description_json_paths(vehicle_json_dir: Path) -> list[Path]:
    """Return generated JSON files, preferring the new stationary subdir."""
    candidates = sorted(
        vehicle_json_dir.rglob("*_vehicle_qwen_description.json"),
        key=lambda path: (
            0 if path.parent.name == STATIONARY_JSON_DIRNAME else 1,
            str(path),
        ),
    )
    paths: list[Path] = []
    seen_names: set[str] = set()
    for path in candidates:
        # A previous run may have left a stationary file directly under
        # vehicle_json. Prefer the new stationary/<file> copy during export.
        if path.name in seen_names:
            continue
        seen_names.add(path.name)
        paths.append(path)
    return paths


def export_summary_csv(vehicle_json_dir: Path, summary_path: Path) -> int:
    rows: list[dict[str, str]] = []
    for path in iter_description_json_paths(vehicle_json_dir):
        try:
            data = load_json(path)
            rows.append({
                "input_file": str(data.get("generation", {}).get("input_file", "")),
                "input_line": str(data.get("generation", {}).get("input_line", "")),
                "output_file": path.name,
                "scene_id": str(data.get("scene_id", "")),
                "vehicle_id": str(data.get("vehicle_id", "")),
                "generation_mode": str(
                    data.get("generation", {}).get("generation_mode", "")
                ),
                "generation_status": str(data.get("generation_status", "")),
                "required_major_behaviors": json.dumps(
                    data.get("required_major_behaviors", []), ensure_ascii=False
                ),
                "required_behavior_episodes": json.dumps(
                    data.get("required_behavior_episodes", []), ensure_ascii=False
                ),
                "major_behavior_coverage": json.dumps(
                    data.get("generation", {}).get("major_behavior_coverage", {}),
                    ensure_ascii=False,
                ),
                "behavior_segment_coverage": json.dumps(
                    data.get("generation", {}).get("behavior_segment_coverage", {}),
                    ensure_ascii=False,
                ),
                "description_short": str(data.get("description_short", "")),
                "description_detailed": str(data.get("description_detailed", "")),
                "supporting_frame_ranges": json.dumps(
                    data.get("supporting_frame_ranges", []), ensure_ascii=False
                ),
                "behavior_segments": json.dumps(
                    data.get("behavior_segments", []), ensure_ascii=False
                ),
                "uncertainty_notes": str(data.get("uncertainty_notes", "")),
            })
        except Exception as exc:
            print(f"[SUMMARY-SKIP] {path.name}: {exc}", flush=True)
    with summary_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def export_description_jsonl(vehicle_json_dir: Path, output_path: Path) -> int:
    """Write one compact JSON record per generated vehicle description."""
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for path in iter_description_json_paths(vehicle_json_dir):
            data = load_json(path)
            handle.write(json.dumps(data, ensure_ascii=False, allow_nan=False) + "\n")
            count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", "--input-jsonl", dest="input_jsonl", required=True, type=Path,
        help="full_vehicle_facts.jsonl containing one full_vehicle_facts_v1 object per line",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="Vehicle prompt file; defaults to prompt/qwen_generate_vehicle_descriptions_prompt.txt",
    )
    parser.add_argument("--max-files", "--max-rows", dest="max_rows", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--stationary-policy",
        choices=("template",),
        default="template",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--summary-csv", type=Path, default=None)
    args = parser.parse_args()
    args.workers = max(1, args.workers)
    if not args.input_jsonl.is_file():
        raise SystemExit(f"Input JSONL not found: {args.input_jsonl}")
    prompt_file = args.prompt_file or default_prompt_path()
    if not prompt_file.is_file():
        raise SystemExit(f"Prompt file not found: {prompt_file}")

    template = load_prompt(prompt_file)
    prompt_hash = sha256(prompt_file)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    vehicle_json_dir = args.output_dir / VEHICLE_JSON_DIRNAME
    vehicle_json_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.output_dir / LOG_DIRNAME
    log_dir.mkdir(parents=True, exist_ok=True)
    prompt_copy = args.output_dir / prompt_file.name
    if prompt_copy.resolve() != prompt_file.resolve():
        shutil.copy2(prompt_file, prompt_copy)
    jobs: list[tuple[int, Mapping[str, Any], Path]] = []
    failed: list[dict[str, str]] = []
    skipped = 0
    suspicious_trajectory_generated = 0
    stationary_template = 0
    insufficient_observation = 0
    qwen_generated = 0
    input_count = 0
    seen_outputs: set[Path] = set()
    with args.input_jsonl.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            if args.max_rows is not None and args.max_rows > 0 and input_count >= args.max_rows:
                break
            input_count += 1
            output_path: Path | None = None
            try:
                data = json.loads(line)
                if not isinstance(data, Mapping):
                    raise ValueError("input JSONL row is not an object")
                # Validate the source schema before creating a Qwen job.
                facts = prompt_input(data)
                motion_status = derive_track_motion_status(facts)
                trajectory_quality = facts.get("trajectory_quality", {})
                suspicious = (
                    isinstance(trajectory_quality, Mapping)
                    and trajectory_quality.get("status") == "suspicious"
                )
                stationary = is_track_stationary(motion_status) and not suspicious
                if suspicious:
                    output_subdir = vehicle_json_dir / SUSPICIOUS_JSON_DIRNAME
                elif stationary:
                    output_subdir = vehicle_json_dir / STATIONARY_JSON_DIRNAME
                else:
                    output_subdir = vehicle_json_dir
                output_subdir.mkdir(parents=True, exist_ok=True)
                output_path = output_subdir / output_name(data)
                if output_path in seen_outputs:
                    raise ValueError(f"duplicate output path for vehicle record: {output_path.name}")
                seen_outputs.add(output_path)
                if args.skip_existing and output_path.exists():
                    skipped += 1
                    continue
                jobs.append((line_number, data, output_path))
            except Exception as exc:
                failed.append({
                    "input_file": f"{args.input_jsonl}:line_{line_number}",
                    "output_file": str(output_path) if output_path else "",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                })

    success = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_record,
                data,
                output_path,
                template,
                str(args.input_jsonl),
                index,
                prompt_copy.name,
                prompt_hash,
            ): (index, data, output_path)
            for index, data, output_path in jobs
        }
        for future in as_completed(futures):
            index, data, output_path = futures[future]
            try:
                generation_mode = future.result()
                success += 1
                trajectory_quality = data.get("trajectory_quality", {})
                if (
                    isinstance(trajectory_quality, Mapping)
                    and trajectory_quality.get("status") == "suspicious"
                ):
                    suspicious_trajectory_generated += 1
                if generation_mode == "stationary_template":
                    stationary_template += 1
                elif generation_mode == "insufficient_observation":
                    insufficient_observation += 1
                elif generation_mode == "qwen_suspicious_trajectory":
                    qwen_generated += 1
                elif generation_mode == "qwen":
                    qwen_generated += 1
                print(
                    f"[line {index}] {generation_mode}: {output_path.name}",
                    flush=True,
                )
            except Exception as exc:
                failed.append({
                    "input_file": f"{args.input_jsonl}:line_{index}",
                    "output_file": str(output_path),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                })
                print(f"[line {index}] failed: {output_path.name}: {exc}", flush=True)

    failed_csv = log_dir / "failed_files.csv"
    with failed_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["input_file", "output_file", "error_type", "error"])
        writer.writeheader()
        writer.writerows(failed)
    summary_path = args.summary_csv or (args.output_dir / SUMMARY_CSV_NAME)
    summary_rows = export_summary_csv(vehicle_json_dir, summary_path)
    description_jsonl = args.output_dir / DESCRIPTION_JSONL_NAME
    description_rows = export_description_jsonl(vehicle_json_dir, description_jsonl)
    print(json.dumps({
        "input_rows": input_count,
        "queued_rows": len(jobs),
        "success": success,
        "skipped": skipped,
        "suspicious_trajectory_generated": suspicious_trajectory_generated,
        "stationary_template": stationary_template,
        "insufficient_observation": insufficient_observation,
        "qwen": qwen_generated,
        "failed": len(failed),
        "summary_rows": summary_rows,
        "description_jsonl_rows": description_rows,
        "vehicle_json_dir": str(vehicle_json_dir),
        "log_dir": str(log_dir),
        "description_jsonl": str(description_jsonl),
        "summary_csv": str(summary_path),
        "prompt_copy": str(prompt_copy),
        "output_dir": str(args.output_dir),
        "model": MODEL_NAME,
        "base_url": BASE_URL,
        "stationary_policy": "template",
        "stationary_speed_threshold_mps": STATIONARY_SPEED_THRESHOLD_MPS,
    }, ensure_ascii=False))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
