"""Build an auditable before/during/after timeline for one vehicle pair.

The input is a ``scene_motion_v3`` record produced by ``export_scene_json``.
InterHub contributes only the candidate scene and the two agent ids; this
module reports trajectory facts and does not assign an interaction type.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class TimelineConfig:
    min_stable_frames: int = 3
    context_frames: int = 20
    speed_window_frames: int = 10
    deceleration_threshold_mps: float = 0.5
    longitudinal_epsilon_m: float = 0.5


def _finite(value: Any) -> Optional[float]:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _agent_map(record: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    return {
        str(agent.get("agent_id")): agent
        for agent in record.get("agents", [])
        if isinstance(agent, Mapping) and agent.get("agent_id") is not None
    }


def _state_map(agent: Mapping[str, Any]) -> Dict[int, Mapping[str, Any]]:
    return {
        int(state["frame"]): state
        for state in agent.get("states", [])
        if isinstance(state, Mapping)
        and isinstance(state.get("frame"), int)
    }


def _valid_lane(state: Optional[Mapping[str, Any]]) -> bool:
    if not isinstance(state, Mapping) or not state.get("valid"):
        return False
    lane = state.get("map", {})
    return isinstance(lane, Mapping) and lane.get("lane_id") is not None


def _lane_id(state: Optional[Mapping[str, Any]]) -> Optional[str]:
    if not _valid_lane(state):
        return None
    return str(state["map"]["lane_id"])


def _contiguous_lane_runs(
    states: Mapping[int, Mapping[str, Any]], config: TimelineConfig
) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    for frame in sorted(states):
        lane = _lane_id(states[frame])
        if lane is None:
            continue
        if (
            runs
            and runs[-1]["end_frame"] + 1 == frame
            and runs[-1]["lane_id"] == lane
        ):
            runs[-1]["end_frame"] = frame
            runs[-1]["frames"].append(frame)
        else:
            runs.append({"lane_id": lane, "start_frame": frame, "end_frame": frame, "frames": [frame]})
    return [run for run in runs if len(run["frames"]) >= config.min_stable_frames]


def _find_lane_changes(
    states: Mapping[int, Mapping[str, Any]], config: TimelineConfig
) -> List[Dict[str, Any]]:
    """Find every stable lane transition from the complete trajectory."""
    runs = _contiguous_lane_runs(states, config)
    candidates: List[Dict[str, Any]] = []
    for before, after in zip(runs, runs[1:]):
        if before["lane_id"] == after["lane_id"]:
            continue
        transition = int(after["start_frame"])
        event = {
            "from_lane_id": before["lane_id"],
            "to_lane_id": after["lane_id"],
            "transition_frame": transition,
            "start_frame": before["start_frame"],
            "end_frame": after["end_frame"],
            "before_run": before,
            "after_run": after,
        }
        candidates.append(event)
    return candidates


def _select_pair_event(
    lane_events_a: Sequence[Mapping[str, Any]],
    lane_events_b: Sequence[Mapping[str, Any]],
    states_a: Mapping[int, Mapping[str, Any]],
    states_b: Mapping[int, Mapping[str, Any]],
    config: TimelineConfig,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Select the pair-relevant transition using trajectory evidence only.

    InterHub windows are deliberately absent from this function. A lane
    transition is pair-relevant only when it changes the pair relation
    (same/different lane or longitudinal order). A transition that changes
    only one agent's lane while leaving the pair relation unchanged remains
    in the audit list but is not promoted to the pair event.
    """
    candidates: List[Tuple[Tuple[int, int, int, int], str, Dict[str, Any]]] = []
    for agent_id, events in (("agent_A", lane_events_a), ("agent_B", lane_events_b)):
        for event in events:
            transition = int(event["transition_frame"])
            stable_frame = transition + config.min_stable_frames - 1
            a_state = states_a.get(stable_frame)
            b_state = states_b.get(stable_frame)
            same_after = int(_lane_id(a_state) is not None and _lane_id(a_state) == _lane_id(b_state))
            previous_a_state = states_a.get(transition - 1)
            previous_b_state = states_b.get(transition - 1)
            same_before = int(
                _lane_id(previous_a_state) is not None
                and _lane_id(previous_a_state) == _lane_id(previous_b_state)
            )
            relation_before = _state_relation(
                previous_a_state, previous_b_state, config.longitudinal_epsilon_m
            )
            relation_after = _state_relation(
                states_a.get(stable_frame), states_b.get(stable_frame), config.longitudinal_epsilon_m
            )
            order_flip = int(relation_before is not None and relation_after is not None and relation_before != relation_after)
            relation_changed = int(same_before != same_after or order_flip)
            # Earlier transitions win only after pair-level evidence ties.
            score = (relation_changed, same_after, order_flip, -transition)
            candidates.append((score, agent_id, dict(event)))
    # A single-agent lane change is not automatically a pair event.  Require
    # a pair-level transition before creating before/during/after semantics.
    pair_candidates = [item for item in candidates if item[0][0]]
    if not pair_candidates:
        return None, None
    _, agent_id, event = max(pair_candidates, key=lambda item: item[0])
    return event, agent_id


def _pair_distance(
    first: Optional[Mapping[str, Any]], second: Optional[Mapping[str, Any]]
) -> Optional[float]:
    if not first or not second or not first.get("valid") or not second.get("valid"):
        return None
    first_position = first.get("position", {})
    second_position = second.get("position", {})
    values = [
        _finite(first_position.get("x")) if isinstance(first_position, Mapping) else None,
        _finite(first_position.get("y")) if isinstance(first_position, Mapping) else None,
        _finite(second_position.get("x")) if isinstance(second_position, Mapping) else None,
        _finite(second_position.get("y")) if isinstance(second_position, Mapping) else None,
    ]
    if any(value is None for value in values):
        return None
    return math.hypot(values[0] - values[2], values[1] - values[3])


def _closest_pair_frame(
    states_a: Mapping[int, Mapping[str, Any]],
    states_b: Mapping[int, Mapping[str, Any]],
    common_frames: Sequence[int],
) -> int:
    return min(
        common_frames,
        key=lambda frame: (_pair_distance(states_a.get(frame), states_b.get(frame)) is None,
                           _pair_distance(states_a.get(frame), states_b.get(frame)) or float("inf"),
                           frame),
    )


def _state_relation(subject: Optional[Mapping[str, Any]], reference: Optional[Mapping[str, Any]], epsilon: float) -> Optional[str]:
    if not subject or not reference or not subject.get("valid") or not reference.get("valid"):
        return None
    sp, rp = subject.get("position", {}), reference.get("position", {})
    heading = _finite(reference.get("heading_rad"))
    values = [
        _finite(sp.get("x")) if isinstance(sp, Mapping) else None,
        _finite(sp.get("y")) if isinstance(sp, Mapping) else None,
        _finite(rp.get("x")) if isinstance(rp, Mapping) else None,
        _finite(rp.get("y")) if isinstance(rp, Mapping) else None,
        heading,
    ]
    if any(value is None for value in values):
        return None
    longitudinal = (values[0] - values[2]) * math.cos(values[4]) + (values[1] - values[3]) * math.sin(values[4])
    if longitudinal > epsilon:
        return "ahead"
    if longitudinal < -epsilon:
        return "behind"
    return "alongside"


def _mode(values: Iterable[Optional[str]]) -> Optional[str]:
    values = [value for value in values if value is not None]
    return Counter(values).most_common(1)[0][0] if values else None


def _phase_relation(
    subject_states: Mapping[int, Mapping[str, Any]],
    reference_states: Mapping[int, Mapping[str, Any]],
    frames: Sequence[int],
    epsilon: float,
) -> Dict[str, Any]:
    common = [frame for frame in frames if frame in subject_states and frame in reference_states]
    lane_relations = []
    longitudinal = []
    for frame in common:
        subject, reference = subject_states[frame], reference_states[frame]
        subject_lane, reference_lane = _lane_id(subject), _lane_id(reference)
        lane_relations.append(
            "same_lane" if subject_lane is not None and subject_lane == reference_lane
            else "different_lane" if subject_lane is not None and reference_lane is not None
            else "unknown"
        )
        longitudinal.append(_state_relation(subject, reference, epsilon))
    return {
        "frame_start": min(common) if common else None,
        "frame_end": max(common) if common else None,
        "frame_count": len(common),
        "lane_relation": _mode(lane_relations),
        "lane_relation_counts": dict(Counter(lane_relations)),
        "relative_position": _mode(longitudinal),
        "relative_position_counts": dict(Counter(value for value in longitudinal if value is not None)),
        "agent_A_lane_ids": sorted({value for value in (_lane_id(subject_states[f]) for f in common) if value is not None}),
        "agent_B_lane_ids": sorted({value for value in (_lane_id(reference_states[f]) for f in common) if value is not None}),
    }


def _speed(state: Optional[Mapping[str, Any]]) -> Optional[float]:
    velocity = state.get("velocity", {}) if isinstance(state, Mapping) else {}
    return _finite(velocity.get("speed")) if isinstance(velocity, Mapping) else None


def _mean_speed(states: Mapping[int, Mapping[str, Any]], frames: Sequence[int]) -> Optional[float]:
    values = [speed for frame in frames for speed in [_speed(states.get(frame))] if speed is not None]
    return sum(values) / len(values) if values else None


def _direction(event: Optional[Mapping[str, Any]], agent: Mapping[str, Any]) -> Optional[str]:
    if event is not None:
        direction = event.get("direction")
        if direction in {"left", "right"}:
            return direction
    for event in agent.get("behavior_events", []) if isinstance(agent.get("behavior_events"), list) else []:
        if event.get("type") == "lane_change" and event.get("direction") in {"left", "right"}:
            return event["direction"]
    return None


def _full_state(state: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    return dict(state) if isinstance(state, Mapping) else None


def build_pair_behavior_timeline(
    record: Mapping[str, Any],
    agent_a_id: str,
    agent_b_id: str,
    config: TimelineConfig = TimelineConfig(),
) -> Dict[str, Any]:
    """Return a pair timeline with explicit evidence and final fact labels."""
    agent_a_id, agent_b_id = str(agent_a_id), str(agent_b_id)
    agents = _agent_map(record)
    if agent_a_id not in agents or agent_b_id not in agents:
        raise ValueError(f"Both agents must be present: {agent_a_id}, {agent_b_id}")
    states_a, states_b = _state_map(agents[agent_a_id]), _state_map(agents[agent_b_id])
    common_frames = sorted(set(states_a) & set(states_b))
    if not common_frames:
        raise ValueError("The two agents have no common frames")

    lane_events_a = _find_lane_changes(states_a, config)
    lane_events_b = _find_lane_changes(states_b, config)
    action_event, action_agent = _select_pair_event(
        lane_events_a, lane_events_b, states_a, states_b, config
    )
    lane_event_a = (
        next(
            (event for event in lane_events_a
             if event["transition_frame"] == action_event["transition_frame"]),
            None,
        )
        if action_event is not None
        else None
    )
    lane_event_b = (
        next(
            (event for event in lane_events_b
             if event["transition_frame"] == action_event["transition_frame"]),
            None,
        )
        if action_event is not None
        else None
    )
    # The previous stable run is context, not the action itself.  The action
    # begins at the first frame of the new lane and ends after the new lane is
    # stable.  This keeps the before phase genuinely pre-action.
    # With no pair transition, anchor only the factual observation at the
    # closest approach in the complete trajectory.
    action_start = action_event["transition_frame"] if action_event else _closest_pair_frame(states_a, states_b, common_frames)
    action_end = (
        action_event["transition_frame"] + config.min_stable_frames - 1
        if action_event
        else action_start
    )
    before_frames = [frame for frame in common_frames if frame < action_start][-config.context_frames:]
    during_frames = [frame for frame in common_frames if action_start <= frame <= action_end]
    after_frames = [frame for frame in common_frames if frame > action_end][:config.context_frames]
    if not before_frames:
        before_frames = [frame for frame in common_frames if frame < action_start]
    if not after_frames:
        after_frames = [frame for frame in common_frames if frame > action_end]

    before = _phase_relation(states_a, states_b, before_frames, config.longitudinal_epsilon_m)
    during = _phase_relation(states_a, states_b, during_frames, config.longitudinal_epsilon_m)
    after = _phase_relation(states_a, states_b, after_frames, config.longitudinal_epsilon_m)

    a_before_lane = _lane_id(states_a.get(before["frame_end"]))
    a_after_lane = _lane_id(states_a.get(after["frame_start"]))
    b_before_lane = _lane_id(states_b.get(before["frame_end"]))
    b_after_lane = _lane_id(states_b.get(after["frame_start"]))
    pair_transition_found = action_event is not None
    a_lane_changed = (
        bool(lane_event_a and lane_event_a["from_lane_id"] != lane_event_a["to_lane_id"])
        if pair_transition_found
        else None
    )
    b_maintained_lane = (
        bool(b_before_lane is not None and b_before_lane == b_after_lane)
        if pair_transition_found
        else None
    )
    entered_b_lane = (
        bool(a_after_lane is not None and a_after_lane == b_after_lane and a_before_lane != a_after_lane)
        if pair_transition_found
        else None
    )
    moved_behind_to_ahead = (
        before["relative_position"] == "behind" and after["relative_position"] == "ahead"
        if pair_transition_found
        else None
    )

    speed_before_frames = [frame for frame in common_frames if frame < action_start][-config.speed_window_frames:]
    speed_during_frames = [frame for frame in common_frames if action_start <= frame <= action_end]
    b_speed_before = _mean_speed(states_b, speed_before_frames)
    b_speed_during = _mean_speed(states_b, speed_during_frames)
    b_during_speeds = [_speed(states_b[frame]) for frame in speed_during_frames]
    b_during_speeds = [value for value in b_during_speeds if value is not None]
    b_min_during = min(b_during_speeds) if b_during_speeds else None
    b_reduced_speed = (
        bool(
            b_speed_before is not None
            and b_min_during is not None
            and b_speed_before - b_min_during >= config.deceleration_threshold_mps
        )
        if pair_transition_found
        else None
    )
    b_deceleration_onset = next(
        (frame for frame in speed_during_frames if _speed(states_b[frame]) is not None and b_speed_before is not None and _speed(states_b[frame]) <= b_speed_before - config.deceleration_threshold_mps),
        None,
    )

    direction = _direction(lane_event_a, agents[agent_a_id])
    agent_a_facts = []
    if a_lane_changed:
        agent_a_facts.append(f"changed_lane_{direction}" if direction else "changed_lane")
    if moved_behind_to_ahead:
        agent_a_facts.append("moved_from_behind_to_ahead")
    if entered_b_lane:
        agent_a_facts.append("entered_agent_B_lane")
    agent_b_facts = []
    if b_maintained_lane:
        agent_b_facts.append("maintained_lane")
    if b_reduced_speed:
        agent_b_facts.append("reduced_speed")

    frame_timeline = [
        {
            "frame": frame,
            "phase": "before" if frame in before_frames else "during" if frame in during_frames else "after",
            "agent_A": _full_state(states_a[frame]),
            "agent_B": _full_state(states_b[frame]),
            "relative_position": _state_relation(states_a[frame], states_b[frame], config.longitudinal_epsilon_m),
            "lane_relation": (
                "same_lane" if _lane_id(states_a[frame]) is not None and _lane_id(states_a[frame]) == _lane_id(states_b[frame])
                else "different_lane" if _lane_id(states_a[frame]) is not None and _lane_id(states_b[frame]) is not None
                else "unknown"
            ),
        }
        for frame in common_frames
    ]
    return {
        "schema_version": "vehicle_pair_behavior_timeline_v1",
        "scene_id": record.get("scene_id") or record.get("source", {}).get("scene_id"),
        "source": record.get("source", {}),
        "candidate": {"agent_A": agent_a_id, "agent_B": agent_b_id},
        "analysis_window": {"start_frame": action_start if action_event else None, "end_frame": action_end if action_event else None, "transition_frame": action_event.get("transition_frame") if action_event else None, "transition_source_agent": action_agent, "status": "pair_transition" if action_event else "no_pair_transition_closest_approach_anchor"},
        "agents": {
            "agent_A": {"agent_id": agent_a_id, "facts": agent_a_facts, "before": before, "during": during, "after": after, "lane_change_event": lane_event_a, "lane_change_events": lane_events_a, "states": [dict(states_a[frame]) for frame in sorted(states_a)]},
            "agent_B": {"agent_id": agent_b_id, "facts": agent_b_facts, "before": _phase_relation(states_b, states_a, before_frames, config.longitudinal_epsilon_m), "during": _phase_relation(states_b, states_a, during_frames, config.longitudinal_epsilon_m), "after": _phase_relation(states_b, states_a, after_frames, config.longitudinal_epsilon_m), "lane_change_event": lane_event_b, "lane_change_events": lane_events_b, "speed": {"before_mean_mps": b_speed_before, "during_mean_mps": b_speed_during, "during_min_mps": b_min_during, "delta_to_during_min_mps": b_speed_before - b_min_during if b_speed_before is not None and b_min_during is not None else None, "deceleration_onset_frame": b_deceleration_onset}, "states": [dict(states_b[frame]) for frame in sorted(states_b)]},
        },
        "pair_facts": {
            "A_changed_lane": a_lane_changed,
            "B_maintained_lane": b_maintained_lane,
            "A_entered_B_lane": entered_b_lane,
            "A_behind_to_ahead": moved_behind_to_ahead,
            "B_speed_reduced_during_event": b_reduced_speed,
        },
        "frame_timeline": frame_timeline,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--agent-a", required=True)
    parser.add_argument("--agent-b", required=True)
    args = parser.parse_args()
    record = json.loads(args.input.read_text(encoding="utf-8"))
    result = build_pair_behavior_timeline(record, args.agent_a, args.agent_b)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
