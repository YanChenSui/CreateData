"""Build grounded LLM inputs from classified scenes and behavior records.

The extractor does not infer a new behavior label.  It carries the InterHub
candidate pair into the generation input and organizes already-computed
evidence into subject/reference, Before/During/After, distance, and Motion
context so that the LLM can describe the interaction rather than only the
subject vehicle.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

try:
    from ..semantic.semantic_behavior_graph import (
        build_fact_policy_audit,
        build_semantic_behavior_graph,
        render_grounded_caption,
        select_grounded_facts,
    )
except ImportError:  # pragma: no cover - direct script execution
    from semantic_behavior_graph import (
        build_fact_policy_audit,
        build_semantic_behavior_graph,
        render_grounded_caption,
        select_grounded_facts,
    )


LLM_INPUT_SCHEMA_VERSION = "scene_motion_llm_input_v6"
EXTRACTOR_VERSION = "v6_semantic_graph"
CONTEXT_SAMPLE_SECONDS = 1.0


def get_path(obj: Any, *keys: str, default: Any = None) -> Any:
    for key in keys:
        if not isinstance(obj, dict):
            return default
        obj = obj.get(key)
    return obj


def as_agent_id(value: Any) -> str | None:
    return None if value is None else str(value)


def finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def number(value: Any, digits: int = 2) -> float | int | None:
    value = finite_float(value)
    if value is None:
        return None
    result = round(value, digits)
    return int(result) if digits == 0 else result


def normalize_window(value: Any) -> tuple[int | None, int | None]:
    if isinstance(value, dict):
        start, end = value.get("start_frame"), value.get("end_frame")
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        start, end = value
    else:
        return None, None
    return (start, end) if isinstance(start, int) and isinstance(end, int) else (None, None)


def get_agents_by_id(scene: dict[str, Any]) -> dict[str, dict[str, Any]]:
    agents = scene.get("agents", [])
    if isinstance(agents, dict):
        return {
            str(agent_id): value
            for agent_id, value in agents.items()
            if isinstance(value, dict)
        }
    return {
        str(agent.get("agent_id")): agent
        for agent in agents
        if isinstance(agent, dict) and agent.get("agent_id") is not None
    }


def get_agent_behaviors_by_id(scene: dict[str, Any]) -> dict[str, dict[str, Any]]:
    behaviors = scene.get("agent_behaviors", {})
    if not isinstance(behaviors, dict):
        return {}
    return {
        str(agent_id): value
        for agent_id, value in behaviors.items()
        if isinstance(value, dict)
    }


def exact_state(agent: dict[str, Any], frame: int | None) -> dict[str, Any]:
    states = agent.get("states", [])
    if frame is None or not isinstance(states, list):
        return {}
    for state in states:
        if isinstance(state, dict) and state.get("frame") == frame:
            return state
    return {}


def state_speed(state: dict[str, Any]) -> float | int | None:
    return number(get_path(state, "velocity", "speed"))


def mean_state_speed(agent: dict[str, Any], start_frame: int, end_frame: int) -> float | int | None:
    """Average speed over an explicit frame interval."""
    states = agent.get("states", [])
    if not isinstance(states, list):
        return None
    values = [
        finite_float(get_path(state, "velocity", "speed"))
        for state in states
        if isinstance(state, dict)
        and isinstance(state.get("frame"), int)
        and start_frame <= state["frame"] <= end_frame
    ]
    values = [value for value in values if value is not None]
    return number(sum(values) / len(values)) if values else None


def state_lane(state: dict[str, Any]) -> str | None:
    return as_agent_id(get_path(state, "map", "lane_id"))


def context_offset_frames(scene: dict[str, Any]) -> int:
    dt = finite_float(get_path(scene, "temporal", "dt_seconds"))
    if dt is None or dt <= 0:
        return 10
    return max(1, int(round(CONTEXT_SAMPLE_SECONDS / dt)))


def find_behavior_event(scene: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    event_id = record.get("behavior_event_id")
    events = scene.get("behavior_events", [])
    if isinstance(events, list):
        for event in events:
            if isinstance(event, dict) and event.get("event_id") == event_id:
                return event

    behavior = get_path(scene, "interaction", "behavior", default={})
    evidence = record.get("evidence", {})
    if not isinstance(evidence, dict):
        evidence = {}
    lane_event = get_path(behavior, "evidence", "lane_change_event", default={})
    if not isinstance(lane_event, dict):
        lane_event = {}
    window_start, window_end = normalize_window(record.get("behavior_window"))
    return {
        "event_id": event_id,
        "agent_id": record.get("subject_agent_id") or get_path(behavior, "subject_agent_id"),
        "type": record.get("behavior_type") or behavior.get("type"),
        "subtype": record.get("behavior_subtype") or behavior.get("subtype"),
        "start_frame": lane_event.get("start_frame", window_start),
        "transition_frame": lane_event.get("transition_frame", evidence.get("transition_frame")),
        "end_frame": lane_event.get("end_frame", window_end),
        "from_lane_id": lane_event.get("from_lane_id", evidence.get("from_lane_id")),
        "to_lane_id": lane_event.get("to_lane_id", evidence.get("to_lane_id")),
        "direction": lane_event.get("direction", evidence.get("direction")),
    }


def caption_participant_ids(scene: dict[str, Any], record: dict[str, Any]) -> list[str]:
    """Recover the pair from the behavior record and InterHub key agents.

    `caption_scope=pair` is the authority for exposing the reference to the
    LLM.  The key pair is retained separately even when a record is agent-only.
    """
    subject = as_agent_id(record.get("subject_agent_id"))
    reference = as_agent_id(record.get("reference_agent_id"))
    scope = record.get("caption_scope")
    if scope == "pair":
        pair = [value for value in (subject, reference) if value is not None]
        if len(pair) == 2:
            return pair

    if scope is None:
        pair = [value for value in (subject, reference) if value is not None]
        if len(pair) == 2:
            return pair

    return [subject] if subject is not None else []


def key_agent_ids(scene: dict[str, Any], record: dict[str, Any]) -> list[str]:
    values = get_path(scene, "interaction", "key_agent_ids", default=[])
    if not isinstance(values, list) or not values:
        values = record.get("candidate_pair_agent_ids", [])
    if not isinstance(values, list):
        values = []
    values = [as_agent_id(value) for value in values]
    values = [value for value in values if value is not None]
    for value in (record.get("subject_agent_id"), record.get("reference_agent_id")):
        value = as_agent_id(value)
        if value is not None and value not in values:
            values.append(value)
    return list(dict.fromkeys(values))


def resolve_reference_agent(
    scene: dict[str, Any], record: dict[str, Any], subject_id: str | None
) -> tuple[str | None, str]:
    """Recover the pair reference without searching unrelated scene agents.

    The behavior record is preferred.  The scene-level InterHub behavior is the
    next source, followed by the two-agent InterHub key pair.  The final
    fallback is intentionally restricted to lane_change and an exact two-agent
    key pair, so a generic nearby vehicle can never become a reference.
    """
    direct = as_agent_id(record.get("reference_agent_id"))
    if direct is not None:
        return direct, "behavior_record"
    scene_behavior = get_path(scene, "interaction", "behavior", default={})
    scene_reference = as_agent_id(get_path(scene_behavior, "reference_agent_id"))
    if scene_reference is not None and scene_reference != subject_id:
        return scene_reference, "scene_interaction_behavior"
    behavior_type = record.get("behavior_type") or get_path(scene_behavior, "type")
    pair = key_agent_ids(scene, record)
    if behavior_type == "lane_change" and subject_id is not None and len(pair) == 2:
        other = [agent_id for agent_id in pair if agent_id != subject_id]
        if len(other) == 1:
            return other[0], "interhub_key_agent_pair"
    return None, "unresolved"


def _speed_relation(before: Any, during: Any) -> str:
    before_value, during_value = finite_float(before), finite_float(during)
    if before_value is None or during_value is None:
        return "unknown"
    delta = during_value - before_value
    if delta >= 0.3:
        return "accelerating"
    if delta <= -0.3:
        return "decelerating"
    return "stable"


def motion_summary(
    scene: dict[str, Any],
    record: dict[str, Any],
    participant_ids: list[str],
    start_frame: int | None,
    end_frame: int | None,
    transition_frame: int | None,
) -> dict[str, dict[str, Any]]:
    agents = get_agents_by_id(scene)
    if not isinstance(start_frame, int) or not isinstance(end_frame, int):
        return {}
    offset = context_offset_frames(scene)
    during = transition_frame if isinstance(transition_frame, int) else (start_frame + end_frame) // 2
    summaries: dict[str, dict[str, Any]] = {}
    for agent_id in participant_ids:
        agent = agents.get(agent_id, {})
        before_state = exact_state(agent, start_frame - offset)
        during_state = exact_state(agent, during)
        after_state = exact_state(agent, end_frame + offset)
        # Speed relations use the same windows as build_behavior_records:
        # mean speed during the one-second context before the caption window
        # versus mean speed across the complete caption window.  The transition
        # frame remains useful for lane context, but never for speed comparison.
        before_speed = mean_state_speed(agent, start_frame - offset, start_frame - 1)
        during_speed = mean_state_speed(agent, start_frame, end_frame)
        after_speed = mean_state_speed(agent, end_frame + 1, end_frame + offset)
        summaries[agent_id] = {
            "speed_before_mps": before_speed,
            "speed_during_mps": during_speed,
            "speed_after_mps": after_speed,
            "speed_relation": _speed_relation(before_speed, during_speed),
            "lane_before": state_lane(before_state),
            "lane_during": state_lane(during_state),
            "lane_after": state_lane(after_state),
            "caption_start_frame": start_frame,
            "caption_end_frame": end_frame,
            "during_frame": during,
        }
    return summaries


def pairwise_states(scene: dict[str, Any]) -> list[dict[str, Any]]:
    states = get_path(scene, "pairwise", "states", default=[])
    return [state for state in states if isinstance(state, dict)] if isinstance(states, list) else []


def distance_context(
    scene: dict[str, Any], record: dict[str, Any], start_frame: int | None, end_frame: int | None
) -> dict[str, Any]:
    minimum = number(record.get("minimum_distance_m"))
    trend = record.get("distance_trend") or "unknown"
    summary = get_path(scene, "pairwise", "summary", default={})
    if minimum is None and isinstance(summary, dict):
        minimum = number(summary.get("minimum_distance_m"))
    if trend == "unknown" and isinstance(start_frame, int) and isinstance(end_frame, int):
        values = sorted(
            (state.get("frame"), finite_float(state.get("distance_m")))
            for state in pairwise_states(scene)
            if isinstance(state.get("frame"), int) and finite_float(state.get("distance_m")) is not None
        )
        before = [value for frame, value in values if frame < start_frame]
        during = [value for frame, value in values if start_frame <= frame <= end_frame]
        after = [value for frame, value in values if frame > end_frame]
        if before and during and after:
            minimum_during = min(during)
            if before[-1] > minimum_during + 0.25 and after[0] > minimum_during + 0.25:
                trend = "decreasing_then_increasing"
            elif before[-1] > after[0] + 0.25:
                trend = "decreasing"
            elif after[0] > before[-1] + 0.25:
                trend = "increasing"
            else:
                trend = "stable"
    pet_s = number(get_path(scene, "interaction", "pet_seconds"))
    minimum_ttc = None
    if isinstance(summary, dict):
        minimum_ttc = number(
            summary.get("minimum_ttc_seconds", summary.get("minimum_ttc_s"))
        )
    if minimum_ttc is None:
        ttc_values = [
            finite_float(state.get("ttc_seconds"))
            for state in pairwise_states(scene)
            if state.get("ttc_status") == "valid"
            and (
                not isinstance(start_frame, int)
                or not isinstance(end_frame, int)
                or start_frame <= state.get("frame", start_frame - 1) <= end_frame
            )
        ]
        ttc_values = [value for value in ttc_values if value is not None]
        if ttc_values:
            minimum_ttc = number(min(ttc_values))

    # Conservative interaction-strength semantics.  A decreasing distance by
    # itself is weak evidence: vehicles in neighboring lanes can close their
    # longitudinal gap while remaining a non-conflicting pair.  Stronger
    # evidence requires either a very small separation or a jointly small PET
    # and TTC, so the LLM does not turn "decreasing" into "approaching".
    if minimum is not None and minimum <= 5.0:
        intensity = "strong"
    elif pet_s is not None and minimum_ttc is not None and pet_s <= 1.0 and minimum_ttc <= 2.0:
        intensity = "strong"
    elif minimum is not None and minimum <= 10.0:
        intensity = "moderate"
    elif pet_s is not None and minimum_ttc is not None and pet_s <= 2.0 and minimum_ttc <= 3.0:
        intensity = "moderate"
    else:
        intensity = "weak"

    return {
        "minimum_distance_m": minimum,
        "distance_trend": trend,
        "pet_s": pet_s,
        "minimum_ttc_s": minimum_ttc,
        "interaction_intensity": intensity,
    }


def _lane_relation_text(value: Any) -> str:
    mapping = {
        "same_lane": "the same lane",
        "different_lane": "different lanes",
        "adjacent": "adjacent lanes",
        "adjacent_lane": "adjacent lanes",
        "unknown": "an unspecified lane relationship",
        None: "an unspecified lane relationship",
    }
    return mapping.get(value, str(value).replace("_", " "))


def _position_text(value: Any) -> str:
    if value in {"ahead", "behind", "alongside"}:
        return str(value)
    return "nearby"


def _can_use_longitudinal_position(lane_relation: Any) -> bool:
    """Avoid presenting an across-lane ordering as a following relation."""
    return lane_relation in {"same_lane", "adjacent", "adjacent_lane"}


def _before_relation_sentence(
    subject: str | None,
    reference: str | None,
    before: dict[str, Any],
) -> str:
    if subject is None:
        return "Before: The subject vehicle's initial relation is not specified."
    subject_lane = before.get("subject_lane")
    reference_lane = before.get("reference_lane")
    lane_relation = before.get("lane_relation")
    subject_part = f"Vehicle {subject} travels"
    if subject_lane is not None:
        subject_part += f" in lane {subject_lane}"
    if reference is None:
        return f"Before: {subject_part}."

    reference_part = f"Vehicle {reference}"
    if reference_lane is not None:
        reference_part += f" in lane {reference_lane}"
    if lane_relation in {"adjacent", "adjacent_lane"}:
        relation_part = "the vehicles occupy adjacent lanes"
    elif lane_relation == "same_lane":
        relation_part = "the vehicles occupy the same lane"
    elif lane_relation == "different_lane":
        relation_part = "the vehicles occupy different lanes"
    else:
        relation_part = "the vehicles are nearby"

    # Longitudinal order is useful only when the lane relation makes it
    # semantically meaningful.  In particular, do not emit "ahead of" for a
    # pair that is merely in different lanes.
    position = before.get("relative_position")
    if _can_use_longitudinal_position(lane_relation) and position in {"ahead", "behind", "alongside"}:
        return f"Before: {subject_part} {_position_text(position)} {reference_part}; {relation_part}."
    if reference_lane is not None:
        reference_sentence = f"Vehicle {reference} travels in lane {reference_lane}"
    else:
        reference_sentence = f"Vehicle {reference} travels nearby"
    return f"Before: {subject_part}, while {reference_sentence}; {relation_part}."


def _after_relation_sentence(
    subject: str | None,
    reference: str | None,
    after: dict[str, Any],
) -> str:
    if subject is None:
        return "After: The final relation is not specified."
    if reference is None:
        lane = after.get("subject_lane")
        suffix = f" in lane {lane}" if lane is not None else ""
        return f"After: Vehicle {subject} continues{suffix}."
    lane_relation = after.get("lane_relation")
    if lane_relation in {"adjacent", "adjacent_lane"}:
        relation = "remain in adjacent lanes"
    elif lane_relation == "same_lane":
        relation = "continue in the same lane"
    elif lane_relation == "different_lane":
        relation = "remain in different lanes"
    else:
        relation = "remain nearby with the final lane relation unspecified"
    return f"After: Vehicle {subject} and Vehicle {reference} {relation}."


def _relation_fact_lines(
    title: str,
    subject: str | None,
    reference: str | None,
    relation: dict[str, Any],
) -> list[str]:
    """Render Before/After as atomic facts rather than caption sentences."""
    lines = [f"{title} facts:"]
    if relation.get("subject_lane") is not None:
        lines.append(f"- subject lane: {relation['subject_lane']}")
    if reference is not None and relation.get("reference_lane") is not None:
        lines.append(f"- reference lane: {relation['reference_lane']}")
    if relation.get("lane_relation") is not None:
        lines.append(f"- lane relation: {relation['lane_relation']}")
    if relation.get("relative_position") is not None:
        lines.append(f"- relative position: {relation['relative_position']}")
    if len(lines) == 1:
        lines.append("- relation: unknown")
    return lines


def build_behavior_context(
    scene: dict[str, Any],
    record: dict[str, Any],
    event: dict[str, Any],
    participant_ids: list[str],
    motion: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    subject = as_agent_id(record.get("subject_agent_id"))
    reference = as_agent_id(record.get("reference_agent_id"))
    relation = record.get("interaction_relation", {})
    if not isinstance(relation, dict):
        relation = {}
    before = relation.get("before", {}) if isinstance(relation.get("before"), dict) else {}
    during = relation.get("during", {}) if isinstance(relation.get("during"), dict) else {}
    after = relation.get("after", {}) if isinstance(relation.get("after"), dict) else {}
    subject_motion = motion.get(subject or "", {})
    reference_motion = motion.get(reference or "", {})
    if reference is None:
        # No reference means this remains an agent-only caption.  Do not invent
        # a relation from an unrelated vehicle.
        reference_motion = {}

    from_lane = as_agent_id(event.get("from_lane_id")) or subject_motion.get("lane_before")
    to_lane = as_agent_id(event.get("to_lane_id")) or subject_motion.get("lane_during")
    reference_lane_during = (
        reference_motion.get("lane_during")
        or reference_motion.get("lane_before")
        or reference_motion.get("lane_after")
    )
    target_matches_reference_lane = None
    if to_lane is not None and reference_lane_during is not None:
        target_matches_reference_lane = str(to_lane) == str(reference_lane_during)
    behavior_type = record.get("behavior_type") or event.get("type")
    subtype = record.get("behavior_subtype") or event.get("subtype")
    distance = distance_context(
        scene,
        record,
        normalize_window(record.get("caption_window"))[0],
        normalize_window(record.get("caption_window"))[1],
    )
    record_speed_relation = record.get("speed_relation", {})
    if not isinstance(record_speed_relation, dict):
        record_speed_relation = {}
    speed_relation = {
        "subject": subject_motion.get("speed_relation")
        or record_speed_relation.get("subject")
        or "unknown",
        "reference": reference_motion.get("speed_relation")
        or record_speed_relation.get("reference")
        or "unknown",
    }
    context = {
        "behavior": {
            "type": behavior_type,
            "subtype": subtype,
        },
        "subject_agent_id": subject,
        "reference_agent_id": reference,
        "before": {
            "subject_lane": subject_motion.get("lane_before") or from_lane,
            "reference_lane": reference_motion.get("lane_before"),
            "relative_position": before.get("relative_position"),
            "lane_relation": before.get("lane_relation"),
        },
        "during": {
            "relation": during.get("relation") or ("lane_crossing" if behavior_type == "lane_change" else behavior_type),
            "subject_from_lane": from_lane,
            "subject_to_lane": to_lane,
            "reference_lane": reference_lane_during,
            "target_matches_reference_lane": target_matches_reference_lane,
            "transition_frame": event.get("transition_frame"),
        },
        "after": {
            "subject_lane": subject_motion.get("lane_after") or to_lane,
            "reference_lane": reference_motion.get("lane_after"),
            "relative_position": after.get("relative_position"),
            "lane_relation": after.get("lane_relation"),
        },
        "distance": distance,
        "interaction_strength": {
            "distance_change": distance.get("distance_trend"),
            "interaction_intensity": distance.get("interaction_intensity"),
            "minimum_distance_m": distance.get("minimum_distance_m"),
            "pet_s": distance.get("pet_s"),
            "minimum_ttc_s": distance.get("minimum_ttc_s"),
            "distance_change_alone_is_not_approach": True,
        },
        "motion": {
            "subject": {
                "speed_before_mps": subject_motion.get("speed_before_mps")
                or get_path(record, "speed_before_mps", "subject"),
                "speed_during_mps": subject_motion.get("speed_during_mps")
                or get_path(record, "speed_during_mps", "subject"),
                "speed_relation": speed_relation["subject"],
            },
            "reference": {
                "speed_before_mps": reference_motion.get("speed_before_mps")
                or get_path(record, "speed_before_mps", "reference"),
                "speed_during_mps": reference_motion.get("speed_during_mps")
                or get_path(record, "speed_during_mps", "reference"),
                "speed_relation": speed_relation["reference"],
            },
        },
        "evidence": record.get("evidence", {}) if isinstance(record.get("evidence", {}), dict) else {},
        "candidate_pair_agent_ids": key_agent_ids(scene, record),
        "caption_participant_ids": participant_ids,
    }
    context["interaction_type"] = _interaction_type(context)
    context["interaction_summary"] = _interaction_summary(
        context, context["interaction_type"]
    )
    return context


def _speed_phrase(agent_id: str | None, relation: str, before: Any, during: Any) -> str:
    if agent_id is None:
        return ""
    speed_trend = relation if relation in {"accelerating", "decelerating", "stable"} else "unknown"
    return f"agent: {agent_id}\n  speed_trend: {speed_trend}"


def _interaction_type(context: dict[str, Any]) -> str:
    """Derive a conservative second-level interaction label for the LLM.

    The original InterHub ``interaction_type`` is retained as source evidence,
    but labels such as ``lane_change_with_neighbor`` are too coarse for text
    generation.  This function only promotes a lane change to merge/overtake/
    avoid when the behavior type or explicit evidence supports it.  A regular
    lane change with a reference vehicle is therefore ``lane_change_parallel``.
    """
    behavior = context.get("behavior", {})
    behavior_name = str(behavior.get("type") or "unknown")
    reference = context.get("reference_agent_id")
    during = context.get("during", {})
    after = context.get("after", {})
    evidence = context.get("evidence", {})
    if not isinstance(evidence, dict):
        evidence = {}

    if behavior_name == "overtake":
        return "lane_change_overtake"
    if behavior_name == "merge":
        return "lane_change_merge"
    if behavior_name in {"avoid", "evasive_lane_change"}:
        return "lane_change_avoid"
    if behavior_name == "cut_in":
        return "lane_change_cut_in"
    if behavior_name == "pass":
        return "pass"
    if behavior_name == "follow_stop":
        subtype = context.get("behavior", {}).get("subtype")
        if subtype == "stop_behind_lead":
            return "stop_behind_lead"
        distance = context.get("distance", {})
        subject_speed = get_path(context, "motion", "subject", "speed_relation")
        if distance.get("distance_trend") in {"decreasing", "decreasing_then_increasing"}:
            return "follow_approaching"
        if subject_speed == "decelerating":
            return "follow_decelerating"
        return "follow_stable"
    if behavior_name != "lane_change":
        return behavior_name
    if reference is None:
        return "lane_change"

    explicit_overtake = bool(
        evidence.get("order_flip")
        or evidence.get("longitudinal_order_changed")
        or evidence.get("pass_event_id")
    )
    if explicit_overtake:
        return "lane_change_overtake"

    explicit_avoid = bool(
        evidence.get("avoidance")
        or evidence.get("avoidance_maneuver")
        or evidence.get("avoid")
    )
    if explicit_avoid:
        return "lane_change_avoid"

    # A confirmed lane convergence is stronger than merely having a nearby
    # candidate.  The build stage already uses ``merge`` for topology-
    # confirmed merges; these lane checks support compatible legacy records.
    target_matches_reference = during.get("target_matches_reference_lane") is True
    ends_same_lane = after.get("lane_relation") == "same_lane"
    if target_matches_reference or ends_same_lane:
        return "lane_change_merge"
    return "lane_change_parallel"


def _interaction_summary(context: dict[str, Any], interaction_type: str) -> list[str]:
    """Create grounded fact lines, not caption-like sentences."""
    subject = context.get("subject_agent_id")
    reference = context.get("reference_agent_id")
    if subject is None:
        return []
    direction = context.get("behavior", {}).get("subtype")
    during = context.get("during", {})
    after = context.get("after", {})

    if reference is None:
        if interaction_type == "lane_change_overtake":
            return [
                "maneuver:\n  subject_action: lane_change\n  purpose: overtake",
            ]
        direction_line = f"\n  direction: {direction}" if direction in {"left", "right"} else ""
        return [f"maneuver:\n  subject_action: lane_change{direction_line}"]

    if interaction_type == "lane_change_overtake":
        return [
            "maneuver:\n  subject_action: lane_change\n  purpose: overtake",
            f"reference_agent: {reference}",
            "relative_outcome: subject_passed_reference",
        ]
    if interaction_type == "lane_change_avoid":
        return [
            "maneuver:\n  subject_action: lane_change\n  purpose: avoid_reference",
            f"reference_agent: {reference}",
        ]
    if interaction_type == "lane_change_cut_in":
        return [
            "maneuver:\n  subject_action: lane_change\n  subtype: cut_in",
            f"reference_agent: {reference}",
            "destination_lane_relation: reference_lane",
        ]
    if interaction_type == "lane_change_merge":
        direction_line = f"\n  direction: {direction}" if direction in {"left", "right"} else ""
        return [
            f"maneuver:\n  subject_action: lane_change{direction_line}\n  subtype: merge",
            f"reference_agent: {reference}",
            "destination_lane_relation: reference_lane",
            "final_lane_relation: same_lane_or_lane_convergence",
        ]
    if interaction_type == "lane_change_parallel":
        direction_line = f"\n  direction: {direction}" if direction in {"left", "right"} else ""
        lines = [
            f"maneuver:\n  subject_action: lane_change{direction_line}",
            "reference_response: maintains_original_lane",
        ]
        if after.get("lane_relation") in {"different_lane", "adjacent", "adjacent_lane"}:
            lines.append("final_lane_relation: separate_lanes")
            lines.append("lane_overlap: none")
        return lines
    if interaction_type == "follow_approaching":
        return [
            "subject_action: follow",
            f"subject_agent: {subject}",
            f"reference_agent: {reference}",
            "lane_relation: same_lane",
            "distance_trend: decreasing_gap",
        ]
    if interaction_type == "follow_decelerating":
        return [
            "subject_action: follow",
            f"subject_agent: {subject}",
            f"reference_agent: {reference}",
            "lane_relation: same_lane",
            "subject_response: decelerating",
            "following_purpose: maintain_following",
        ]
    if interaction_type == "follow_stable":
        return [
            "subject_action: follow",
            f"subject_agent: {subject}",
            f"reference_agent: {reference}",
            "lane_relation: same_lane",
            "distance_trend: stable_following_distance",
        ]
    if interaction_type == "pass":
        return [
            "maneuver:\n  subject_action: pass",
            f"reference_agent: {reference}",
            "relative_outcome: subject_passed_reference",
        ]
    return [
        f"subject_agent: {subject}",
        f"reference_agent: {reference}",
        f"relation_label: {interaction_type}",
    ]


def build_context_text(context: dict[str, Any]) -> str:
    behavior = context.get("behavior", {})
    subject = context.get("subject_agent_id")
    reference = context.get("reference_agent_id")
    before = context.get("before", {})
    during = context.get("during", {})
    after = context.get("after", {})
    motion = context.get("motion", {})
    subject_motion = motion.get("subject", {})
    reference_motion = motion.get("reference", {})
    behavior_name = str(behavior.get("type") or "unknown")
    subtype = behavior.get("subtype")
    behavior_line = f"Behavior: {behavior_name}{' ' + str(subtype) if subtype else ''}"
    lines = [behavior_line, f"Subject: Vehicle {subject}"]
    if reference is not None:
        lines.append(f"Reference: Vehicle {reference}")
    interaction_type = context.get("interaction_type") or _interaction_type(context)
    lines.append("Interaction type: " + interaction_type)
    lines.append("Interaction facts:")
    lines.extend(f"- {fact}" for fact in _interaction_summary(context, interaction_type))
    # In different lanes, longitudinal order is weak contextual evidence and
    # should not be exposed as "ahead" or "behind" to the LLM.  Apply this
    # suppression at the final text-rendering boundary as well as inside the
    # relation helper, so stale or externally assembled context cannot leak a
    # misleading following-style relation into context_text.
    before_for_text = dict(before) if isinstance(before, dict) else {}
    if before_for_text.get("lane_relation") == "different_lane":
        before_for_text["relative_position"] = None
    lines.extend(_relation_fact_lines("Before", subject, reference, before_for_text))
    lines.append("During facts:")
    if (
        behavior_name == "lane_change"
        and during.get("subject_from_lane") is not None
        and during.get("subject_to_lane") is not None
    ):
        lines.append("- subject lane transition:")
        lines.append(f"  from lane {during['subject_from_lane']}")
        lines.append(f"  to lane {during['subject_to_lane']}")
    else:
        relation = str(during.get("relation") or behavior_name).replace("_", " ")
        lines.append(f"- subject relation: {relation}")
    if reference is not None and during.get("reference_lane") is not None:
        lines.append(f"- reference lane: {during['reference_lane']}")
    if during.get("transition_frame") is not None:
        lines.append(f"- transition frame: {during['transition_frame']}")
    after_for_text = dict(after) if isinstance(after, dict) else {}
    if after_for_text.get("lane_relation") == "different_lane":
        after_for_text["relative_position"] = None
    lines.extend(_relation_fact_lines("After", subject, reference, after_for_text))
    motion_parts = [
        _speed_phrase(
            subject,
            subject_motion.get("speed_relation", "unknown"),
            subject_motion.get("speed_before_mps"),
            subject_motion.get("speed_during_mps"),
        )
    ]
    if reference is not None:
        motion_parts.append(
            _speed_phrase(
                reference,
                reference_motion.get("speed_relation", "unknown"),
                reference_motion.get("speed_before_mps"),
                reference_motion.get("speed_during_mps"),
            )
        )
    lines.append("Detailed motion facts (speed for detailed description only):")
    lines.extend(f"- {part}" for part in motion_parts if part)
    distance = context.get("distance", {})
    if distance.get("minimum_distance_m") is not None or distance.get("distance_trend") not in (None, "unknown"):
        lines.append("Distance facts:")
        if distance.get("minimum_distance_m") is not None:
            lines.append(f"- minimum distance: {distance['minimum_distance_m']} m")
        if distance.get("distance_trend") not in (None, "unknown"):
            lines.append(f"- trend: {distance['distance_trend']}")
    strength = context.get("interaction_strength", {})
    if strength:
        lines.append("Interaction strength facts:")
        lines.append(
            f"- level: {strength.get('interaction_intensity', 'unknown')}"
        )
        if strength.get("pet_s") is not None:
            lines.append(f"- PET: {strength['pet_s']} s")
        if strength.get("minimum_ttc_s") is not None:
            lines.append(f"- TTC: {strength['minimum_ttc_s']} s")
    return "\n".join(lines)


def scene_source(scene: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    source = scene.get("source", {})
    record_source = record.get("source", {})
    if not isinstance(source, dict):
        source = {}
    if not isinstance(record_source, dict):
        record_source = {}
    merged = {**source, **record_source}
    return {
        key: merged.get(key)
        for key in ("dataset", "split", "dataset_key", "scene_id", "scenario_index")
    }


def extract_behavior_record(scene: dict[str, Any], record: dict[str, Any], index: int) -> dict[str, Any]:
    event = find_behavior_event(scene, record)
    subject = as_agent_id(record.get("subject_agent_id") or event.get("agent_id"))
    reference, reference_resolution = resolve_reference_agent(scene, record, subject)
    behavior_type = record.get("behavior_type") or event.get("type")
    caption_scope = record.get("caption_scope") or "agent_only"
    # A recovered InterHub reference upgrades a stale lane-change record to a
    # pair caption.  This is the compatibility path for old behavior bundles.
    if reference is not None and behavior_type == "lane_change":
        caption_scope = "pair"
    context_record = {
        **record,
        "subject_agent_id": subject,
        "reference_agent_id": reference,
        "behavior_type": behavior_type,
        "caption_scope": caption_scope,
    }
    participant_ids = caption_participant_ids(scene, context_record)
    start, end = normalize_window(record.get("caption_window"))
    if start is None or end is None:
        start, end = normalize_window(record.get("behavior_window"))
    transition = event.get("transition_frame")
    if not isinstance(transition, int):
        transition = get_path(record, "evidence", "transition_frame")
    motion = motion_summary(scene, context_record, participant_ids, start, end, transition)
    context = build_behavior_context(scene, context_record, event, participant_ids, motion)
    semantic_graph = build_semantic_behavior_graph({
        **context_record,
        "behavior_type": behavior_type,
        "behavior_subtype": context_record.get("behavior_subtype") or event.get("subtype"),
        "interaction_type": context.get("interaction_type"),
        "evidence": record.get("evidence", {}),
        "before": context.get("before", {}),
        "during": context.get("during", {}),
        "after": context.get("after", {}),
        "motion": context.get("motion", {}),
    })
    semantic_behavior = {
        "agents": semantic_graph["agents"],
        "interaction": semantic_graph["interaction"],
    }
    selected_facts = select_grounded_facts(semantic_graph)
    fact_policy = build_fact_policy_audit(semantic_graph, selected_facts)
    grounded_caption = render_grounded_caption(semantic_graph, selected_facts)
    behavior = {
        "type": behavior_type,
        "subtype": record.get("behavior_subtype") or event.get("subtype"),
        "subject_agent_id": subject,
        "reference_agent_id": reference,
        # Keep the original coarse label for audit, but expose the grounded
        # second-level label to downstream generation and viewers.
        "interaction_type": context.get("interaction_type") or record.get("interaction_type"),
        "base_interaction_type": record.get("interaction_type"),
    }
    participants = [
        {
            "agent_id": agent_id,
            "role": "subject" if agent_id == subject else "reference" if agent_id == reference else "participant",
            "control_type": get_agents_by_id(scene).get(agent_id, {}).get("control_type"),
        }
        for agent_id in participant_ids
    ]
    eligible = record.get("caption_eligible") is not False
    output = {
        "schema_version": LLM_INPUT_SCHEMA_VERSION,
        "source_schema_version": scene.get("schema_version"),
        "generation_input": {
            "extractor_version": EXTRACTOR_VERSION,
            "granularity": "one_behavior_record_one_caption_candidate",
            "record_index_in_bundle": index,
            "instruction": "Paraphrase the grounded interaction caption into natural language without adding unsupported behaviors or changing the interaction semantics.",
            "reference_resolution": reference_resolution,
        },
        "behavior_record_id": record.get("record_id"),
        "interaction_id": record.get("interaction_id") or scene.get("interaction_id"),
        "source": scene_source(scene, record),
        "interaction": {
            "participant_ids": [as_agent_id(value) for value in get_path(scene, "interaction", "participant_ids", default=[]) if as_agent_id(value) is not None],
            "key_agent_ids": key_agent_ids(scene, record),
            "reference_resolution": reference_resolution,
            "caption_participant_ids": participant_ids,
            "caption_scope": caption_scope,
        },
        "participants": participants,
        "behavior": behavior,
        "semantic_behavior": semantic_behavior,
        "behavior_event": {
            "event_id": event.get("event_id"),
            "start_frame": event.get("start_frame"),
            "transition_frame": transition,
            "end_frame": event.get("end_frame"),
            "from_lane_id": as_agent_id(event.get("from_lane_id")),
            "to_lane_id": as_agent_id(event.get("to_lane_id")),
        },
        "generation_context": {
            "behavior": behavior,
            "interaction_type": context.get("interaction_type") or record.get("interaction_type"),
            "base_interaction_type": record.get("interaction_type"),
            "interaction_summary": context.get("interaction_summary", []),
            "caption_scope": caption_scope,
            "subject_agent_id": subject,
            "reference_agent_id": reference,
            "caption_window": [start, end],
            "behavior_context": context,
            "semantic_behavior_graph": semantic_graph,
            "semantic_behavior": semantic_behavior,
            "fact_policy": fact_policy,
            "selected_facts": selected_facts,
            "grounded_caption": grounded_caption,
            "audit_context_text": build_context_text(context),
            "agent_motion_summary": motion,
            "interaction_relation": record.get("interaction_relation", {}),
            "minimum_distance_m": context.get("distance", {}).get("minimum_distance_m"),
            "distance_trend": context.get("distance", {}).get("distance_trend"),
            "interaction_strength": context.get("interaction_strength", {}),
            "speed_before_mps": {
                "subject": get_path(context, "motion", "subject", "speed_before_mps"),
                "reference": get_path(context, "motion", "reference", "speed_before_mps"),
            },
            "speed_during_mps": {
                "subject": get_path(context, "motion", "subject", "speed_during_mps"),
                "reference": get_path(context, "motion", "reference", "speed_during_mps"),
            },
            "speed_relation": {
                "subject": get_path(context, "motion", "subject", "speed_relation"),
                "reference": get_path(context, "motion", "reference", "speed_relation"),
            },
            "evidence": record.get("evidence", {}),
            "key_agent_ids": key_agent_ids(scene, record),
        },
        "audit_context": {
            "behavior_record": context_record,
            "key_agent_ids": key_agent_ids(scene, record),
            "interaction_participant_ids": get_path(scene, "interaction", "participant_ids", default=[]),
        },
        "quality": {
            "caption_eligible": record.get("caption_eligible"),
            "eligible_for_text_generation": eligible,
            "caption_scope": caption_scope,
            "has_subject": subject is not None,
            "has_reference": reference is not None,
            "reference_resolution": reference_resolution,
            "key_agent_pair_preserved": len(key_agent_ids(scene, record)) >= 2,
            "review_reasons": ["reference_agent_missing"] if caption_scope == "pair" and reference is None else [],
        },
    }
    return output


def safe_filename(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "record"))
    return text.strip("._") or "record"


def find_scene(record: dict[str, Any], scene_paths: list[Path], by_interaction: dict[str, Path]) -> Path | None:
    interaction_id = record.get("interaction_id")
    if interaction_id is not None and str(interaction_id) in by_interaction:
        return by_interaction[str(interaction_id)]
    source = record.get("source", {})
    if not isinstance(source, dict):
        source = {}
    matches = []
    for path in scene_paths:
        try:
            scene = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        scene_source = scene.get("source", {})
        if not isinstance(scene_source, dict):
            continue
        if all(source.get(key) is None or scene_source.get(key) == source.get(key) for key in ("dataset_key", "scene_id", "scenario_index")):
            expected_pair = {as_agent_id(value) for value in record.get("candidate_pair_agent_ids", [])}
            actual_pair = {as_agent_id(value) for value in get_path(scene, "interaction", "participant_ids", default=[])}
            if not expected_pair or expected_pair.issubset(actual_pair):
                matches.append(path)
    return matches[0] if len(matches) == 1 else None


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--behavior-records-dir", type=Path)
    parser.add_argument("--include-ineligible", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    scene_paths = sorted(args.input_dir.glob("row_*_v3_classified.json"))
    if not scene_paths:
        raise SystemExit(f"No classified JSON files found in {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scene_cache: dict[Path, dict[str, Any]] = {}
    by_interaction: dict[str, Path] = {}
    failed: list[dict[str, str]] = []
    for path in scene_paths:
        try:
            scene = json.loads(path.read_text(encoding="utf-8"))
            scene_cache[path] = scene
            if scene.get("interaction_id") is not None:
                by_interaction[str(scene["interaction_id"])] = path
        except Exception as exc:
            failed.append({"input_file": str(path), "output_file": "", "error": str(exc), "error_type": type(exc).__name__})

    success = skipped = ineligible = 0
    if args.behavior_records_dir is None:
        for scene_path, scene in scene_cache.items():
            record = get_path(scene, "interaction", "behavior", default={})
            if not isinstance(record, dict):
                record = {}
            record = {
                **record,
                "record_id": scene.get("interaction_id"),
                "interaction_id": scene.get("interaction_id"),
                "behavior_type": record.get("type"),
                "behavior_subtype": record.get("subtype"),
                "caption_scope": "pair" if record.get("reference_agent_id") is not None else "agent_only",
                "caption_window": get_path(scene, "temporal", "interaction_start_frame", default=None),
            }
            start = get_path(scene, "temporal", "interaction_start_frame")
            end = get_path(scene, "temporal", "interaction_end_frame")
            record["caption_window"] = {"start_frame": start, "end_frame": end}
            output_path = args.output_dir / scene_path.name.replace("_v3_classified.json", "_v3_llm_input.json")
            if args.skip_existing and output_path.exists():
                skipped += 1
                continue
            try:
                write_json(output_path, extract_behavior_record(scene, record, 0))
                success += 1
            except Exception as exc:
                failed.append({"input_file": str(scene_path), "output_file": str(output_path), "error": str(exc), "error_type": type(exc).__name__})
    else:
        bundle_paths = sorted(args.behavior_records_dir.glob("row_*_behavior_records.json"))
        if not bundle_paths:
            raise SystemExit(f"No behavior-record bundles found in {args.behavior_records_dir}")
        for bundle_path in bundle_paths:
            try:
                bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
                records = bundle.get("behavior_records", [])
                if not isinstance(records, list):
                    raise ValueError("behavior_records must be a list")
                for index, record in enumerate(records):
                    if not isinstance(record, dict):
                        raise ValueError("behavior_records contains a non-object item")
                    scene_path = find_scene(record, scene_paths, by_interaction)
                    if scene_path is None:
                        raise FileNotFoundError(f"no classified scene matches {record.get('interaction_id')}")
                    if not args.include_ineligible and record.get("caption_eligible") is False:
                        ineligible += 1
                        continue
                    scene = scene_cache[scene_path]
                    output = extract_behavior_record(scene, record, index)
                    stem = safe_filename(bundle_path.stem.replace("_behavior_records", ""))
                    record_token = safe_filename(record.get("record_id") or f"record_{index:04d}")
                    output_path = args.output_dir / f"{stem}_{record_token}_{index:04d}_llm_input.json"
                    if args.skip_existing and output_path.exists():
                        skipped += 1
                        continue
                    write_json(output_path, output)
                    success += 1
            except Exception as exc:
                failed.append({"input_file": str(bundle_path), "output_file": str(args.output_dir), "error": str(exc), "error_type": type(exc).__name__})

    failed_csv = args.output_dir / "failed_files.csv"
    with failed_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["input_file", "output_file", "error_type", "error"])
        writer.writeheader()
        writer.writerows(failed)
    print(f"success: {success}")
    print(f"skipped: {skipped}")
    print(f"ineligible_skipped: {ineligible}")
    print(f"failed: {len(failed)}")
    print(f"failed_files_csv: {failed_csv}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
