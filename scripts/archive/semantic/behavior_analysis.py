"""Rule-based behavior and TTC analysis for scene_motion_v3.

This module consumes the already matched per-state lane records.  It does not
change map matching or source labels, and it keeps ``unknown`` separate from
``pending`` and ``insufficient_evidence``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class BehaviorConfig:
    min_stable_frames: int = 3
    min_lane_match_confidence: float = 0.5
    min_valid_match_ratio: float = 0.5
    min_lateral_delta_m: float = 0.05
    evidence_window_frames: int = 4
    classification_padding_frames: int = 10
    lane_group_max_hops: int = 5
    min_same_lane_frames: int = 3
    min_same_lane_ratio: float = 0.5
    same_lane_distance_threshold_m: float = 30.0
    cut_in_gap_threshold_m: float = 15.0
    deceleration_threshold_mps: float = 1.0
    gap_decrease_threshold_m: float = 1.0


@dataclass(frozen=True)
class TTCConfig:
    epsilon_m: float = 1e-3
    closing_speed_epsilon_mps: float = 1e-3
    default_safety_radius_m: float = 2.0
    lateral_not_applicable_ratio: float = 2.0


def _finite(value: Any) -> Optional[float]:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _lane_id(lane: Any) -> str:
    return str(getattr(lane, "id", lane))


def _lane_items(lane_graph: Any) -> List[Any]:
    if lane_graph is None:
        return []
    lanes = getattr(lane_graph, "lanes", lane_graph)
    if isinstance(lanes, Mapping):
        return list(lanes.values())
    return list(lanes)


def _lane_index(lane_graph: Any) -> Dict[str, Any]:
    return {_lane_id(lane): lane for lane in _lane_items(lane_graph)}


def _ids(lane: Any, attribute: str) -> set:
    values = getattr(lane, attribute, set()) or set()
    return {_lane_id(value) for value in values}


def _is_connected(from_id: str, to_id: str, lanes: Mapping[str, Any]) -> bool:
    source = lanes.get(from_id)
    target = lanes.get(to_id)
    if source is None or target is None:
        return False
    return to_id in _ids(source, "next_lanes") or from_id in _ids(target, "prev_lanes")


def _adjacent_direction(
    from_id: str, to_id: str, lanes: Mapping[str, Any]
) -> Optional[str]:
    source = lanes.get(from_id)
    target = lanes.get(to_id)
    if source is None or target is None:
        return None
    if to_id in _ids(source, "adj_lanes_left") or from_id in _ids(target, "adj_lanes_right"):
        return "left"
    if to_id in _ids(source, "adj_lanes_right") or from_id in _ids(target, "adj_lanes_left"):
        return "right"
    return None


def _matched_state(state: Mapping[str, Any], config: BehaviorConfig) -> bool:
    lane = state.get("map", {})
    confidence = _finite(lane.get("lane_match_confidence"))
    return bool(
        state.get("valid")
        and lane.get("lane_id") is not None
        and lane.get("match_status") == "matched"
        and confidence is not None
        and confidence >= config.min_lane_match_confidence
    )


def _make_runs(states: Sequence[Mapping[str, Any]], config: BehaviorConfig) -> List[List[Mapping[str, Any]]]:
    runs: List[List[Mapping[str, Any]]] = []
    for state in states:
        if not _matched_state(state, config):
            continue
        lane_id = state["map"]["lane_id"]
        if (
            runs
            and runs[-1][-1]["frame"] + 1 == state["frame"]
            and runs[-1][-1]["map"]["lane_id"] == lane_id
        ):
            runs[-1].append(state)
        else:
            runs.append([state])
    return runs


def _trend(states: Sequence[Mapping[str, Any]], field: str, absolute: bool = False) -> Optional[float]:
    values = [_finite(state.get("map", {}).get(field)) for state in states]
    values = [abs(value) if absolute else value for value in values if value is not None]
    if len(values) < 2:
        return None
    return values[-1] - values[0]


def _lateral_support(
    previous: Sequence[Mapping[str, Any]],
    current: Sequence[Mapping[str, Any]],
    config: BehaviorConfig,
) -> Tuple[bool, float]:
    old_tail = list(previous[-config.evidence_window_frames :])
    new_head = list(current[: config.evidence_window_frames])
    before_trends = [
        _trend(old_tail, "centerline_distance_m"),
        _trend(old_tail, "lateral_offset_m", absolute=True),
    ]
    after_trends = [
        _trend(new_head, "centerline_distance_m"),
        _trend(new_head, "lateral_offset_m", absolute=True),
    ]
    before_supported = any(
        trend is not None and trend >= config.min_lateral_delta_m
        for trend in before_trends
    )
    after_supported = any(
        trend is not None and trend <= -config.min_lateral_delta_m
        for trend in after_trends
    )
    support = before_supported and after_supported
    score = (float(before_supported) + float(after_supported)) / 2.0
    return support, score


def detect_lane_change_events(
    agent_states: Sequence[Mapping[str, Any]],
    lane_graph: Any,
    config: BehaviorConfig = BehaviorConfig(),
) -> Dict[str, Any]:
    """Detect stable lateral transfers between adjacent lanes."""
    valid_states = [state for state in agent_states if state.get("valid")]
    matched_states = [state for state in valid_states if _matched_state(state, config)]
    if len(valid_states) < config.min_stable_frames * 2:
        return {"lane_change_detected": None, "lane_change_events": [], "confidence": None}
    if len(matched_states) / float(len(valid_states)) < config.min_valid_match_ratio:
        return {"lane_change_detected": None, "lane_change_events": [], "confidence": None}

    lanes = _lane_index(lane_graph)
    runs = [run for run in _make_runs(agent_states, config) if len(run) >= config.min_stable_frames]
    events: List[Dict[str, Any]] = []
    for previous, current in zip(runs, runs[1:]):
        if previous[-1]["frame"] + 1 != current[0]["frame"]:
            continue
        from_id = previous[-1]["map"]["lane_id"]
        to_id = current[0]["map"]["lane_id"]
        if from_id == to_id or _is_connected(from_id, to_id, lanes):
            continue
        direction = _adjacent_direction(from_id, to_id, lanes)
        if direction is None:
            continue
        lateral_supported, lateral_score = _lateral_support(previous, current, config)
        if not lateral_supported:
            continue

        evidence_states = list(previous[-config.evidence_window_frames :]) + list(
            current[: config.evidence_window_frames]
        )
        map_confidences = [
            _finite(state["map"].get("lane_match_confidence"))
            for state in evidence_states
        ]
        map_confidences = [value for value in map_confidences if value is not None]
        map_score = sum(map_confidences) / len(map_confidences) if map_confidences else 0.0
        stability_score = min(
            1.0,
            min(len(previous), len(current)) / float(config.min_stable_frames + 2),
        )
        confidence = round(
            0.4 * map_score + 0.2 * stability_score + 0.2 * lateral_score + 0.2,
            6,
        )
        start_offset = min(config.evidence_window_frames, len(previous))
        end_offset = min(config.evidence_window_frames, len(current))
        events.append(
            {
                "from_lane_id": from_id,
                "to_lane_id": to_id,
                "start_frame": previous[-start_offset]["frame"],
                "transition_frame": current[0]["frame"],
                "end_frame": current[end_offset - 1]["frame"],
                "direction": direction,
                "confidence": confidence,
            }
        )

    if events:
        confidence = max(event["confidence"] for event in events)
        return {"lane_change_detected": True, "lane_change_events": events, "confidence": confidence}
    return {"lane_change_detected": False, "lane_change_events": [], "confidence": None}


def build_agent_behaviors(
    agents: Sequence[Mapping[str, Any]],
    lane_graph: Any,
    config: BehaviorConfig = BehaviorConfig(),
) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for agent in agents:
        if lane_graph is None:
            detection = {"lane_change_detected": None, "lane_change_events": [], "confidence": None}
        else:
            detection = detect_lane_change_events(agent["states"], lane_graph, config)
        detected = detection["lane_change_detected"]
        maneuver = "lane_change" if detected is True else "lane_keep" if detected is False else "unknown"
        agent_id = str(agent["agent_id"])
        behavior_events = []
        for event_index, event in enumerate(detection["lane_change_events"], start=1):
            event_record = dict(event)
            event_record.update(
                {
                    "event_id": f"event_{agent_id}_{event_index:03d}",
                    "agent_id": agent_id,
                    "type": "lane_change",
                    "subtype": event.get("direction"),
                    "source": "trajectory_map_rule",
                }
            )
            behavior_events.append(event_record)
        result[str(agent["agent_id"])] = {
            "maneuver": maneuver,
            **detection,
            "behavior_events": behavior_events,
        }
    return result


def collect_behavior_events(
    agent_behaviors: Mapping[str, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Return deterministic agent-level behavior events.

    Events intentionally contain only the acting agent.  A reference agent is
    assigned later by pair-association logic, because a lane change does not
    inherently belong to one particular neighboring vehicle.
    """
    events: List[Dict[str, Any]] = []
    for agent_id in sorted(agent_behaviors, key=str):
        for event in agent_behaviors[agent_id].get("behavior_events", []):
            events.append(dict(event))
    return events


def _reachable(start_id: str, lanes: Mapping[str, Any], max_hops: int) -> set:
    visited = {start_id}
    frontier = {start_id}
    for _ in range(max_hops):
        next_frontier = set()
        for lane_id in frontier:
            lane = lanes.get(lane_id)
            if lane is not None:
                next_frontier.update(_ids(lane, "next_lanes"))
        next_frontier -= visited
        visited.update(next_frontier)
        frontier = next_frontier
        if not frontier:
            break
    return visited


def _same_lane_group(
    lane_a: Optional[str], lane_b: Optional[str], lanes: Mapping[str, Any], max_hops: int
) -> bool:
    if lane_a is None or lane_b is None:
        return False
    if lane_a == lane_b:
        return True
    return lane_b in _reachable(lane_a, lanes, max_hops) or lane_a in _reachable(
        lane_b, lanes, max_hops
    )


def _state_at(states: Sequence[Mapping[str, Any]], frame: int) -> Optional[Mapping[str, Any]]:
    return min(states, key=lambda state: abs(state["frame"] - frame), default=None)


def _longitudinal_relation(
    subject: Mapping[str, Any], reference: Mapping[str, Any]
) -> Optional[str]:
    ps, pr = subject.get("position", {}), reference.get("position", {})
    sx, sy, rx, ry = (_finite(ps.get("x")), _finite(ps.get("y")), _finite(pr.get("x")), _finite(pr.get("y")))
    if None in (sx, sy, rx, ry):
        return None
    heading = _finite(reference.get("heading_rad"))
    if heading is None:
        return None
    longitudinal = (sx - rx) * math.cos(heading) + (sy - ry) * math.sin(heading)
    if longitudinal > 0.5:
        return "ahead"
    if longitudinal < -0.5:
        return "behind"
    return "alongside"


def _speed(state: Optional[Mapping[str, Any]]) -> Optional[float]:
    return _finite((state or {}).get("velocity", {}).get("speed"))


def _pair_state(pairwise: Mapping[str, Any], frame: int) -> Optional[Mapping[str, Any]]:
    states = pairwise.get("states", [])
    return min(states, key=lambda state: abs(state["frame"] - frame), default=None)


def _event_overlap_frames(event: Mapping[str, Any], start: int, end: int) -> int:
    event_start = int(event.get("start_frame", event.get("transition_frame", start)))
    event_end = int(event.get("end_frame", event.get("transition_frame", end)))
    return max(0, min(event_end, end) - max(event_start, start) + 1)


def _select_interaction_lane_change_event(
    events: Sequence[Mapping[str, Any]],
    interaction_start: int,
    interaction_end: int,
) -> Optional[Mapping[str, Any]]:
    """Select only an event that overlaps the InterHub interaction window."""
    overlapping_events = [
        event
        for event in events
        if _event_overlap_frames(event, interaction_start, interaction_end) > 0
    ]
    if not overlapping_events:
        return None
    midpoint = (interaction_start + interaction_end) / 2.0
    return min(
        overlapping_events,
        key=lambda event: abs(
            float(event.get("transition_frame", midpoint)) - midpoint
        ),
    )


def _longitudinal_gap(
    subject: Optional[Mapping[str, Any]],
    reference: Optional[Mapping[str, Any]],
) -> Optional[float]:
    """Return absolute longitudinal separation in the reference heading frame."""
    if subject is None or reference is None:
        return None
    ps, pr = subject.get("position", {}), reference.get("position", {})
    sx, sy, rx, ry = (
        _finite(ps.get("x")),
        _finite(ps.get("y")),
        _finite(pr.get("x")),
        _finite(pr.get("y")),
    )
    heading = _finite(reference.get("heading_rad"))
    if None in (sx, sy, rx, ry, heading):
        return None
    return abs((sx - rx) * math.cos(heading) + (sy - ry) * math.sin(heading))


def _euclidean_distance(
    first: Optional[Mapping[str, Any]],
    second: Optional[Mapping[str, Any]],
) -> Optional[float]:
    if first is None or second is None:
        return None
    first_position = first.get("position", {})
    second_position = second.get("position", {})
    values = [
        _finite(first_position.get("x")),
        _finite(first_position.get("y")),
        _finite(second_position.get("x")),
        _finite(second_position.get("y")),
    ]
    if any(value is None for value in values):
        return None
    return math.hypot(values[0] - values[2], values[1] - values[3])


def _association_evidence(
    event: Mapping[str, Any],
    subject_states: Sequence[Mapping[str, Any]],
    reference_states: Sequence[Mapping[str, Any]],
    pairwise_features: Optional[Mapping[str, Any]],
    lanes: Mapping[str, Any],
    config: BehaviorConfig,
) -> Dict[str, Any]:
    """Collect evidence without assigning a causal relationship by time alone."""
    transition = int(event.get("transition_frame", event.get("start_frame", 0)))
    subject_before = _state_at(subject_states, transition - 1)
    subject_after = _state_at(
        subject_states, transition + config.min_stable_frames
    )
    reference_before = _state_at(reference_states, transition - 1)
    reference_after = _state_at(
        reference_states, transition + config.min_stable_frames
    )

    relation_before = (
        _longitudinal_relation(subject_before, reference_before)
        if subject_before and reference_before
        else None
    )
    relation_after = (
        _longitudinal_relation(subject_after, reference_after)
        if subject_after and reference_after
        else None
    )

    pair_before = (
        _pair_state(pairwise_features, transition - 1)
        if pairwise_features is not None
        else None
    )
    pair_after = (
        _pair_state(
            pairwise_features, transition + config.min_stable_frames
        )
        if pairwise_features is not None
        else None
    )
    distance_before = _finite((pair_before or {}).get("distance_m"))
    distance_after = _finite((pair_after or {}).get("distance_m"))
    if distance_before is None:
        distance_before = _euclidean_distance(subject_before, reference_before)
    if distance_after is None:
        distance_after = _euclidean_distance(subject_after, reference_after)

    distance_decreased = (
        distance_before is not None
        and distance_after is not None
        and distance_before - distance_after >= config.gap_decrease_threshold_m
    )
    longitudinal_order_changed = (
        relation_before is not None
        and relation_after is not None
        and relation_before != relation_after
    )
    subject_lane_after = (subject_after or {}).get("map", {}).get("lane_id")
    reference_lane_after = (reference_after or {}).get("map", {}).get("lane_id")
    entered_reference_lane = (
        subject_lane_after is not None
        and reference_lane_after is not None
        and _same_lane_group(
            subject_lane_after,
            reference_lane_after,
            lanes,
            config.lane_group_max_hops,
        )
    )
    reference_speed_before = _speed(reference_before)
    reference_speed_after = _speed(reference_after)
    reference_decelerated = (
        reference_speed_before is not None
        and reference_speed_after is not None
        and reference_speed_before - reference_speed_after
        >= config.deceleration_threshold_mps
    )
    strong_evidence_count = sum(
        [
            bool(distance_decreased),
            bool(longitudinal_order_changed),
            bool(entered_reference_lane),
            bool(reference_decelerated),
        ]
    )
    return {
        "distance_before_m": distance_before,
        "distance_after_m": distance_after,
        "distance_decreased": distance_decreased,
        "longitudinal_relation_before": relation_before,
        "longitudinal_relation_after": relation_after,
        "longitudinal_order_changed": longitudinal_order_changed,
        "entered_reference_lane": entered_reference_lane,
        "reference_decelerated": reference_decelerated,
        "strong_evidence_count": strong_evidence_count,
    }


def associate_behavior_event_with_pair(
    event: Mapping[str, Any],
    reference_agent_id: str,
    agents: Sequence[Mapping[str, Any]],
    key_agent_ids: Sequence[str],
    interaction_start: int,
    interaction_end: int,
    pairwise_features: Optional[Mapping[str, Any]],
    lane_graph: Any,
    config: BehaviorConfig = BehaviorConfig(),
) -> Dict[str, Any]:
    """Associate an agent event with a pair using evidence, not proximity alone."""
    subject_agent_id = str(event.get("agent_id"))
    reference_agent_id = str(reference_agent_id)
    agent_by_id = {str(agent["agent_id"]): agent for agent in agents}
    subject = agent_by_id.get(subject_agent_id, {})
    reference = agent_by_id.get(reference_agent_id, {})
    subject_states = subject.get("states", [])
    reference_states = reference.get("states", [])
    event_start = int(event.get("start_frame", event.get("transition_frame", 0)))
    event_end = int(event.get("end_frame", event.get("transition_frame", event_start)))
    overlap_frames = max(
        0,
        min(event_end, int(interaction_end))
        - max(event_start, int(interaction_start))
        + 1,
    )
    evidence = _association_evidence(
        event,
        subject_states,
        reference_states,
        pairwise_features,
        _lane_index(lane_graph),
        config,
    )
    same_interhub_pair = set(key_agent_ids) == {
        subject_agent_id,
        reference_agent_id,
    }
    if not same_interhub_pair:
        relation = "unrelated"
    elif overlap_frames > 0:
        relation = "overlap"
    elif evidence["strong_evidence_count"] >= 2:
        relation = "associated_but_outside"
    else:
        relation = "nearby_unverified"
    return {
        "subject_agent_id": subject_agent_id,
        "reference_agent_id": reference_agent_id,
        "behavior_event_id": event.get("event_id"),
        "relation": relation,
        "overlap_frames": overlap_frames,
        "temporal_gap_frames": (
            0
            if overlap_frames > 0
            else min(
                abs(event_start - int(interaction_end)),
                abs(int(interaction_start) - event_end),
            )
        ),
        "evidence": evidence,
    }


def associate_behavior_events(
    events: Sequence[Mapping[str, Any]],
    agents: Sequence[Mapping[str, Any]],
    key_agent_ids: Sequence[str],
    interaction_start: int,
    interaction_end: int,
    pairwise_features: Optional[Mapping[str, Any]],
    lane_graph: Any,
    config: BehaviorConfig = BehaviorConfig(),
) -> List[Dict[str, Any]]:
    """Associate key-agent behavior events with the other key agent."""
    key_agent_ids = [str(agent_id) for agent_id in key_agent_ids]
    if len(key_agent_ids) != 2:
        return []
    associations: List[Dict[str, Any]] = []
    for event in events:
        subject_agent_id = str(event.get("agent_id"))
        if subject_agent_id not in key_agent_ids:
            continue
        reference_agent_id = next(
            agent_id for agent_id in key_agent_ids if agent_id != subject_agent_id
        )
        associations.append(
            associate_behavior_event_with_pair(
                event,
                reference_agent_id,
                agents,
                key_agent_ids,
                interaction_start,
                interaction_end,
                pairwise_features,
                lane_graph,
                config,
            )
        )
    return associations


def select_caption_behavior_event(
    events: Sequence[Mapping[str, Any]],
    associations: Sequence[Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Select one event for captioning without upgrading an unverified pair link."""
    event_by_id = {
        event.get("event_id"): dict(event)
        for event in events
        if event.get("event_id") is not None
    }
    relation_rank = {
        "overlap": 0,
        "associated_but_outside": 1,
        "nearby_unverified": 2,
    }
    candidates = [
        association
        for association in associations
        if association.get("relation") in relation_rank
        and association.get("behavior_event_id") in event_by_id
    ]
    if not candidates:
        return None
    selected_association = min(
        candidates,
        key=lambda association: (
            relation_rank[association["relation"]],
            -int(association.get("overlap_frames", 0)),
            int(
                event_by_id[association["behavior_event_id"]].get(
                    "start_frame", 0
                )
            ),
        ),
    )
    event = event_by_id[selected_association["behavior_event_id"]]
    relation = selected_association["relation"]
    return {
        "event": event,
        "association": dict(selected_association),
        "caption_scope": "pair" if relation in {"overlap", "associated_but_outside"} else "agent_only",
    }


def _behavior_record(
    behavior_type: str,
    subject_id: Optional[str],
    reference_id: Optional[str],
    confidence: Optional[float],
    status: str,
    evidence: Mapping[str, Any],
    subtype: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "type": behavior_type,
        "subtype": subtype,
        "subject_agent_id": subject_id,
        "reference_agent_id": reference_id,
        "status": status,
        "source": "trajectory_map_rule",
        "rule_version": "v2",
        "confidence": confidence,
        "evidence": dict(evidence),
    }


def _merge_evidence(
    agent_states: Mapping[str, Sequence[Mapping[str, Any]]],
    lanes: Mapping[str, Any],
    config: BehaviorConfig,
) -> Optional[Dict[str, Any]]:
    ids = list(agent_states)
    if len(ids) != 2:
        return None
    first, second = agent_states[ids[0]], agent_states[ids[1]]
    valid_first = [state for state in first if _matched_state(state, config)]
    valid_second = [state for state in second if _matched_state(state, config)]
    if len(valid_first) < config.min_stable_frames or len(valid_second) < config.min_stable_frames:
        return None
    initial_first = valid_first[0]["map"]["lane_id"]
    initial_second = valid_second[0]["map"]["lane_id"]
    final_first = valid_first[-1]["map"]["lane_id"]
    final_second = valid_second[-1]["map"]["lane_id"]
    if initial_first == initial_second:
        return None
    common = _reachable(initial_first, lanes, config.lane_group_max_hops) & _reachable(
        initial_second, lanes, config.lane_group_max_hops
    )
    if not common:
        return None
    converged = final_first == final_second or _same_lane_group(
        final_first, final_second, lanes, config.lane_group_max_hops
    )
    if not converged:
        return None
    return {
        "initial_lane_ids": [initial_first, initial_second],
        "final_lane_ids": [final_first, final_second],
        "common_successor_candidates": sorted(common),
    }


def classify_interaction_behavior(
    interaction: Mapping[str, Any],
    agents: Sequence[Mapping[str, Any]],
    agent_behaviors: Mapping[str, Mapping[str, Any]],
    pairwise_features: Optional[Mapping[str, Any]],
    lane_graph: Any,
    config: BehaviorConfig = BehaviorConfig(),
) -> Dict[str, Any]:
    """Classify pairwise behavior without using InterHub source labels as truth."""
    if pairwise_features is None or len(interaction.get("key_agent_ids", [])) != 2:
        return {
            "behavior": _behavior_record(
                "unknown", None, None, None, "insufficient_evidence", {}
            ),
            "processing_status": "insufficient_evidence",
        }
    agent_by_id = {str(agent["agent_id"]): agent for agent in agents}
    key_ids = [str(value) for value in interaction["key_agent_ids"]]
    if any(agent_id not in agent_by_id for agent_id in key_ids):
        return {
            "behavior": _behavior_record(
                "unknown", None, None, None, "insufficient_evidence", {}
            ),
            "processing_status": "insufficient_evidence",
        }
    start = int(interaction["start"]) - config.classification_padding_frames
    end = int(interaction["end"]) + config.classification_padding_frames
    states_by_id = {
        agent_id: [
            state
            for state in agent_by_id[agent_id]["states"]
            if start <= state["frame"] <= end
        ]
        for agent_id in key_ids
    }
    lanes = _lane_index(lane_graph)
    candidates = [
        agent_id
        for agent_id in key_ids
        if agent_behaviors.get(agent_id, {}).get("lane_change_detected") is True
    ]
    if len(candidates) > 1:
        return {
            "behavior": _behavior_record(
                "unknown", None, None, None, "classified", {"multiple_lane_change_subjects": candidates}
            ),
            "processing_status": "completed",
        }

    merge = _merge_evidence(states_by_id, lanes, config)
    if merge is not None and not candidates:
        return {
            "behavior": _behavior_record(
                "merge", key_ids[0], key_ids[1], 0.75, "labeled", merge
            ),
            "processing_status": "completed",
        }

    if candidates:
        subject_id = candidates[0]
        reference_id = key_ids[1] if key_ids[0] == subject_id else key_ids[0]
        interaction_start = int(interaction["start"])
        interaction_end = int(interaction["end"])
        event = _select_interaction_lane_change_event(
            agent_behaviors[subject_id]["lane_change_events"],
            interaction_start,
            interaction_end,
        )
        if event is None:
            return {
                "behavior": _behavior_record(
                    "unknown", subject_id, reference_id, None, "insufficient_evidence", {}
                ),
                "processing_status": "insufficient_evidence",
            }
        subject_states = states_by_id[subject_id]
        reference_states = states_by_id[reference_id]
        before = _state_at(subject_states, event["transition_frame"] - 1)
        after = _state_at(subject_states, event["transition_frame"] + config.min_stable_frames)
        reference_before = _state_at(reference_states, event["transition_frame"] - 1)
        reference_after = _state_at(reference_states, event["transition_frame"] + config.min_stable_frames)
        relation_before = _longitudinal_relation(before, reference_before) if before and reference_before else None
        relation_after = _longitudinal_relation(after, reference_after) if after and reference_after else None
        pair_before = _pair_state(pairwise_features, event["transition_frame"] - 1)
        pair_after = _pair_state(pairwise_features, event["transition_frame"] + config.min_stable_frames)
        gap_before = _finite((pair_before or {}).get("distance_m"))
        gap_after = _finite((pair_after or {}).get("distance_m"))
        longitudinal_gap_before = _longitudinal_gap(before, reference_before)
        longitudinal_gap_after = _longitudinal_gap(after, reference_after)
        subject_lane_after = (after or {}).get("map", {}).get("lane_id")
        reference_lane_after = (reference_after or {}).get("map", {}).get("lane_id")
        entered_reference_lane = (
            subject_lane_after is not None
            and reference_lane_after is not None
            and _same_lane_group(subject_lane_after, reference_lane_after, lanes, config.lane_group_max_hops)
        )
        ref_speed_before = _speed(reference_before)
        ref_speed_after = _speed(reference_after)
        reference_decelerated = (
            ref_speed_before is not None
            and ref_speed_after is not None
            and ref_speed_before - ref_speed_after >= config.deceleration_threshold_mps
        )
        gap_decreased = (
            gap_before is not None
            and gap_after is not None
            and gap_before - gap_after >= config.gap_decrease_threshold_m
        )
        small_longitudinal_gap_after = (
            longitudinal_gap_after is not None
            and longitudinal_gap_after <= config.cut_in_gap_threshold_m
        )
        longitudinal_gap_decreased = (
            longitudinal_gap_before is not None
            and longitudinal_gap_after is not None
            and longitudinal_gap_before - longitudinal_gap_after
            >= config.gap_decrease_threshold_m
        )
        relative_position_evidence = relation_before is not None or relation_after is not None
        evidence = {
            "subject_lane_changed": True,
            "entered_reference_lane": entered_reference_lane,
            "longitudinal_relation_before": relation_before,
            "longitudinal_relation_after": relation_after,
            "gap_before_m": gap_before,
            "gap_after_m": gap_after,
            "longitudinal_gap_before_m": longitudinal_gap_before,
            "longitudinal_gap_after_m": longitudinal_gap_after,
            "small_longitudinal_gap_after": small_longitudinal_gap_after,
            "gap_decreased": gap_decreased,
            "longitudinal_gap_decreased": longitudinal_gap_decreased,
            "reference_decelerated": reference_decelerated,
            "relative_position_evidence": relative_position_evidence,
            "event_overlap_frames": _event_overlap_frames(
                event, interaction_start, interaction_end
            ),
            "event_distance_to_interaction_midpoint_frames": abs(
                float(event.get("transition_frame", 0))
                - (interaction_start + interaction_end) / 2.0
            ),
            "lane_change_event": event,
        }
        cut_in_support = (
            entered_reference_lane
            and relative_position_evidence
            and (
                small_longitudinal_gap_after
                or longitudinal_gap_decreased
                or reference_decelerated
            )
        )
        if cut_in_support:
            evidence_score = sum(
                [
                    float(entered_reference_lane),
                    float(relative_position_evidence),
                    float(small_longitudinal_gap_after),
                    float(longitudinal_gap_decreased),
                    float(reference_decelerated),
                ]
            ) / 5.0
            confidence = round(agent_behaviors[subject_id]["confidence"] * evidence_score, 6)
            return {
                "behavior": _behavior_record(
                    "cut_in", subject_id, reference_id, confidence, "labeled", evidence
                ),
                "processing_status": "completed",
            }
        return {
            "behavior": _behavior_record(
                "lane_change",
                subject_id,
                reference_id,
                agent_behaviors[subject_id]["confidence"],
                "labeled",
                evidence,
                subtype=event.get("direction"),
            ),
            "processing_status": "completed",
        }

    pair_states = [
        state
        for state in pairwise_features.get("states", [])
        if start <= state["frame"] <= end
    ]
    same_lane_states = [state for state in pair_states if state.get("same_lane") is True]
    valid_pair_states = [state for state in pair_states if state.get("same_lane") is not None]
    same_ratio = len(same_lane_states) / float(len(valid_pair_states)) if valid_pair_states else 0.0
    meaningful = any(
        (_finite(state.get("distance_m")) or 0.0) <= config.same_lane_distance_threshold_m
        or abs(_finite(state.get("speed_difference_mps")) or 0.0) >= 1.0
        or state.get("ttc_status") == "valid"
        for state in same_lane_states
    )
    if len(same_lane_states) >= config.min_same_lane_frames and same_ratio >= config.min_same_lane_ratio and meaningful:
        confidence = round(min(1.0, 0.5 * same_ratio + 0.5 * min(1.0, len(same_lane_states) / 10.0)), 6)
        return {
            "behavior": _behavior_record(
                "same_lane_interaction", key_ids[0], key_ids[1], confidence, "labeled", {
                    "same_lane_frame_count": len(same_lane_states),
                    "valid_pair_frame_count": len(valid_pair_states),
                    "same_lane_ratio": round(same_ratio, 6),
                }
            ),
            "processing_status": "completed",
        }
    if len(valid_pair_states) < config.min_same_lane_frames:
        status = "insufficient_evidence"
    else:
        status = "classified"
    return {
        "behavior": _behavior_record("unknown", key_ids[0], key_ids[1], None, status, {}),
        "processing_status": "insufficient_evidence" if status == "insufficient_evidence" else "completed",
    }


def _state_safety_radius(state: Mapping[str, Any], config: TTCConfig) -> float:
    dimensions = state.get("dimensions", {})
    length = _finite(dimensions.get("length_m")) if isinstance(dimensions, Mapping) else None
    width = _finite(dimensions.get("width_m")) if isinstance(dimensions, Mapping) else None
    if length is not None and width is not None and length > 0 and width > 0:
        return 0.5 * math.sqrt(length * length + width * width)
    return config.default_safety_radius_m


def compute_ttc_metrics(
    state_i: Mapping[str, Any],
    state_j: Mapping[str, Any],
    config: TTCConfig = TTCConfig(),
) -> Dict[str, Any]:
    """Compute point-relative TTC with explicit status for non-applicable cases."""
    if not state_i.get("valid") or not state_j.get("valid"):
        return {"closing_speed_mps": None, "ttc_seconds": None, "ttc_status": "insufficient_data"}
    pi, pj = state_i.get("position", {}), state_j.get("position", {})
    vi, vj = state_i.get("velocity", {}), state_j.get("velocity", {})
    values = [_finite(pi.get("x")), _finite(pi.get("y")), _finite(pj.get("x")), _finite(pj.get("y")), _finite(vi.get("vx")), _finite(vi.get("vy")), _finite(vj.get("vx")), _finite(vj.get("vy"))]
    if any(value is None for value in values):
        return {"closing_speed_mps": None, "ttc_seconds": None, "ttc_status": "insufficient_data"}
    rx, ry = values[2] - values[0], values[3] - values[1]
    distance = math.hypot(rx, ry)
    if distance <= config.epsilon_m:
        return {"closing_speed_mps": None, "ttc_seconds": 0.0, "ttc_status": "overlapping"}
    rvx, rvy = values[6] - values[4], values[7] - values[5]
    parallel = (rx * rvx + ry * rvy) / distance
    closing_speed = -parallel
    relative_speed_norm = math.hypot(rvx, rvy)
    lateral_speed = math.sqrt(max(0.0, relative_speed_norm * relative_speed_norm - parallel * parallel))
    if closing_speed <= config.closing_speed_epsilon_mps:
        status = (
            "not_applicable"
            if relative_speed_norm > config.closing_speed_epsilon_mps
            and lateral_speed >= config.lateral_not_applicable_ratio * max(abs(closing_speed), config.closing_speed_epsilon_mps)
            else "not_closing"
        )
        return {"closing_speed_mps": round(max(0.0, closing_speed), 6), "ttc_seconds": None, "ttc_status": status}
    effective_distance = distance - _state_safety_radius(state_i, config) - _state_safety_radius(state_j, config)
    if effective_distance <= 0:
        return {"closing_speed_mps": round(closing_speed, 6), "ttc_seconds": 0.0, "ttc_status": "overlapping"}
    return {
        "closing_speed_mps": round(closing_speed, 6),
        "ttc_seconds": round(effective_distance / closing_speed, 6),
        "ttc_status": "valid",
    }
