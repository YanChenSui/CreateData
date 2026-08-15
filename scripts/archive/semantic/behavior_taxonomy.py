"""Behavior-mode taxonomy and conservative trajectory rules.

This module is intentionally independent from InterHub candidate extraction.
It consumes a classified ``scene_motion_v3`` record and returns observable
behavior events.  Pair association is kept explicit so an agent maneuver is
not automatically attributed to one neighboring vehicle.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


BEHAVIOR_TYPES = (
    "lane_change",
    "merge",
    "merge_candidate",
    "follow_stop",
    "pass",
    "overtake",
)

ASSOCIATION_RELATIONS = (
    "overlap",
    "associated_but_outside",
    "nearby_unverified",
    "unrelated",
)


@dataclass(frozen=True)
class BehaviorThresholds:
    min_frames: int = 3
    stable_order_frames: int = 3
    max_follow_distance_m: float = 30.0
    max_follow_speed_difference_mps: float = 3.0
    stop_speed_mps: float = 1.0
    min_order_margin_m: float = 0.5
    max_heading_difference_rad: float = 0.6
    max_event_gap_frames: int = 5


def finite(value: Any) -> Optional[float]:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def as_agent_id(value: Any) -> Optional[str]:
    return None if value is None else str(value)


def state_by_frame(agent: Mapping[str, Any]) -> Dict[int, Mapping[str, Any]]:
    return {
        int(state["frame"]): state
        for state in agent.get("states", [])
        if isinstance(state, Mapping) and isinstance(state.get("frame"), int)
    }


def pair_states(record: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    pairwise = record.get("pairwise", {})
    states = pairwise.get("states", []) if isinstance(pairwise, Mapping) else []
    return [state for state in states if isinstance(state, Mapping)]


def position(state: Optional[Mapping[str, Any]]) -> Optional[Tuple[float, float]]:
    if state is None:
        return None
    values = [
        finite(state.get("position", {}).get("x")),
        finite(state.get("position", {}).get("y")),
    ]
    if any(value is None for value in values):
        return None
    return values[0], values[1]


def longitudinal_relation(
    subject: Optional[Mapping[str, Any]],
    reference: Optional[Mapping[str, Any]],
    margin_m: float = 0.5,
) -> Optional[str]:
    subject_position = position(subject)
    reference_position = position(reference)
    if subject_position is None or reference_position is None:
        return None
    heading = finite((reference or {}).get("heading_rad"))
    if heading is None:
        return None
    longitudinal = (
        (subject_position[0] - reference_position[0]) * math.cos(heading)
        + (subject_position[1] - reference_position[1]) * math.sin(heading)
    )
    if longitudinal > margin_m:
        return "ahead"
    if longitudinal < -margin_m:
        return "behind"
    return "alongside"


def euclidean_distance(
    first: Optional[Mapping[str, Any]],
    second: Optional[Mapping[str, Any]],
) -> Optional[float]:
    first_position = position(first)
    second_position = position(second)
    if first_position is None or second_position is None:
        return None
    return math.hypot(
        first_position[0] - second_position[0],
        first_position[1] - second_position[1],
    )


def _agent_map(record: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    return {
        str(agent["agent_id"]): agent
        for agent in record.get("agents", [])
        if isinstance(agent, Mapping) and agent.get("agent_id") is not None
    }


def _key_pair(record: Mapping[str, Any]) -> Optional[Tuple[str, str]]:
    interaction = record.get("interaction", {})
    key_ids = interaction.get("key_agent_ids", []) if isinstance(interaction, Mapping) else []
    if not isinstance(key_ids, list) or len(key_ids) != 2:
        return None
    return str(key_ids[0]), str(key_ids[1])


def find_neighbor_vehicle(
    subject_agent: Any,
    candidate_agents: Sequence[Any],
) -> Optional[str]:
    """Return the other vehicle from an already selected InterHub pair.

    This deliberately does not search all scene agents.  ``candidate_agents``
    must come from the InterHub candidate pair, so the returned reference keeps
    the original interaction membership instead of inventing a neighbor.
    """
    subject_id = as_agent_id(subject_agent)
    if subject_id is None:
        return None
    for candidate in candidate_agents:
        candidate_id = (
            as_agent_id(candidate.get("agent_id"))
            if isinstance(candidate, Mapping)
            else as_agent_id(candidate)
        )
        if candidate_id is not None and candidate_id != subject_id:
            return candidate_id
    return None


def _window(record: Mapping[str, Any], name: str = "interaction") -> Tuple[Optional[int], Optional[int]]:
    temporal = record.get("temporal", {})
    if not isinstance(temporal, Mapping):
        return None, None
    start = temporal.get(f"{name}_start_frame")
    end = temporal.get(f"{name}_end_frame")
    if isinstance(start, int) and isinstance(end, int) and end >= start:
        return start, end
    return None, None


def _interval_event(
    event_id: str,
    behavior_type: str,
    subject_id: Optional[str],
    reference_id: Optional[str],
    start_frame: int,
    end_frame: int,
    evidence: Mapping[str, Any],
    subtype: Optional[str] = None,
    confidence: Optional[float] = None,
    source: str = "behavior_taxonomy_v1",
) -> Dict[str, Any]:
    return {
        "event_id": event_id,
        "agent_id": subject_id,
        "type": behavior_type,
        "subtype": subtype,
        "subject_agent_id": subject_id,
        "reference_agent_id": reference_id,
        "start_frame": int(start_frame),
        "transition_frame": int((start_frame + end_frame) // 2),
        "end_frame": int(end_frame),
        "confidence": confidence,
        "source": source,
        "evidence": dict(evidence),
    }


def _contiguous_runs(frames: Iterable[int]) -> List[Tuple[int, int]]:
    ordered = sorted(set(int(frame) for frame in frames))
    if not ordered:
        return []
    runs: List[Tuple[int, int]] = []
    start = previous = ordered[0]
    for frame in ordered[1:]:
        if frame != previous + 1:
            runs.append((start, previous))
            start = frame
        previous = frame
    runs.append((start, previous))
    return runs


def interval_gap(
    first_start: int,
    first_end: int,
    second_start: int,
    second_end: int,
) -> int:
    """Return the distance between two inclusive frame intervals."""
    if max(first_start, second_start) <= min(first_end, second_end):
        return 0
    if first_end < second_start:
        return second_start - first_end
    return first_start - second_end


def _stable_relation_runs(
    features: Sequence[Mapping[str, Any]],
    min_frames: int,
) -> List[Tuple[str, List[Mapping[str, Any]]]]:
    """Return consecutive, frame-contiguous ahead/behind runs."""
    ordered = sorted(
        (
            item
            for item in features
            if item.get("relation") in {"ahead", "behind"}
            and isinstance(item.get("frame"), int)
        ),
        key=lambda item: int(item["frame"]),
    )
    runs: List[Tuple[str, List[Mapping[str, Any]]]] = []
    current_relation: Optional[str] = None
    current_items: List[Mapping[str, Any]] = []
    previous_frame: Optional[int] = None

    def flush() -> None:
        if current_relation is not None and len(current_items) >= min_frames:
            runs.append((current_relation, list(current_items)))

    for item in ordered:
        relation = str(item["relation"])
        frame = int(item["frame"])
        is_contiguous = previous_frame is not None and frame == previous_frame + 1
        if relation != current_relation or not is_contiguous:
            flush()
            current_relation = relation
            current_items = [item]
        else:
            current_items.append(item)
        previous_frame = frame
    flush()
    return runs


def _stable_flag_runs(
    features: Sequence[Mapping[str, Any]],
    field: str,
    value: bool,
    min_frames: int,
) -> List[List[Mapping[str, Any]]]:
    """Return frame-contiguous runs whose boolean field has ``value``."""
    ordered = sorted(
        (
            item
            for item in features
            if item.get("pair", {}).get(field) in {True, False}
            and isinstance(item.get("frame"), int)
        ),
        key=lambda item: int(item["frame"]),
    )
    runs: List[List[Mapping[str, Any]]] = []
    current: List[Mapping[str, Any]] = []
    previous_frame: Optional[int] = None
    for item in ordered:
        frame = int(item["frame"])
        is_contiguous = previous_frame is not None and frame == previous_frame + 1
        if (
            not current
            or item["pair"].get(field) != value
            or not is_contiguous
        ):
            if current and len(current) >= min_frames:
                runs.append(current)
            current = [item] if item["pair"].get(field) == value else []
        else:
            current.append(item)
        previous_frame = frame
    if current and len(current) >= min_frames:
        runs.append(current)
    return runs


def _agent_event_fallback(record: Mapping[str, Any]) -> List[Dict[str, Any]]:
    events = record.get("behavior_events", [])
    if isinstance(events, list) and events:
        return [dict(event) for event in events if isinstance(event, Mapping)]
    events = []
    for agent_id, behavior in (record.get("agent_behaviors", {}) or {}).items():
        if not isinstance(behavior, Mapping):
            continue
        for index, event in enumerate(behavior.get("lane_change_events", []), start=1):
            if not isinstance(event, Mapping):
                continue
            normalized = dict(event)
            normalized.setdefault("event_id", f"event_{agent_id}_{index:03d}")
            normalized.setdefault("agent_id", str(agent_id))
            normalized.setdefault("type", "lane_change")
            normalized.setdefault("subtype", event.get("direction"))
            normalized.setdefault("source", "trajectory_map_rule")
            events.append(normalized)
    return events


def detect_lane_change_events(record: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Return existing agent-level lane-change events without rebinding them."""
    result: List[Dict[str, Any]] = []
    candidate_pair = _key_pair(record)
    for source_event in _agent_event_fallback(record):
        if (
            source_event.get("type") != "lane_change"
            or not isinstance(source_event.get("agent_id"), (str, int))
            or not isinstance(source_event.get("start_frame"), int)
            or not isinstance(source_event.get("end_frame"), int)
        ):
            continue
        event = dict(source_event)
        subject_id = str(event["agent_id"])
        event.setdefault("subject_agent_id", subject_id)
        evidence = event.get("evidence")
        if not isinstance(evidence, Mapping):
            evidence = {
                key: event.get(key)
                for key in ("from_lane_id", "to_lane_id", "direction", "transition_frame")
                if event.get(key) is not None
            }
        evidence = dict(evidence)
        if candidate_pair is not None and subject_id in candidate_pair:
            reference_id = find_neighbor_vehicle(subject_id, candidate_pair)
            if reference_id is not None:
                event["reference_agent_id"] = reference_id
                event["candidate_pair_agent_ids"] = list(candidate_pair)
                evidence["candidate_pair_agent_ids"] = list(candidate_pair)
                event["interaction_type"] = "lane_change_with_neighbor"
            else:
                event["interaction_type"] = "lane_change"
        else:
            event["interaction_type"] = "lane_change"
        event["evidence"] = evidence
        result.append(event)
    return result


def _pair_frame_features(
    record: Mapping[str, Any],
    subject_id: str,
    reference_id: str,
) -> List[Dict[str, Any]]:
    agents = _agent_map(record)
    subject_states = state_by_frame(agents.get(subject_id, {}))
    reference_states = state_by_frame(agents.get(reference_id, {}))
    result: List[Dict[str, Any]] = []
    for pair_state in pair_states(record):
        frame = pair_state.get("frame")
        if not isinstance(frame, int):
            continue
        subject = subject_states.get(frame)
        reference = reference_states.get(frame)
        distance_m = finite(pair_state.get("distance_m"))
        if distance_m is None:
            distance_m = euclidean_distance(subject, reference)
        result.append(
            {
                "frame": frame,
                "pair": pair_state,
                "subject": subject,
                "reference": reference,
                "relation": longitudinal_relation(subject, reference),
                "distance_m": distance_m,
            }
        )
    return result


def _same_lane_count(features: Sequence[Mapping[str, Any]]) -> int:
    return sum(1 for item in features if item["pair"].get("same_lane") is True)


def detect_follow_stop(
    record: Mapping[str, Any],
    thresholds: BehaviorThresholds = BehaviorThresholds(),
) -> List[Dict[str, Any]]:
    """Detect conservative same-lane following or stopping-behind events."""
    pair = _key_pair(record)
    if pair is None:
        return []
    candidates: List[Tuple[str, str, List[int]]] = []
    for subject_id, reference_id in (pair, pair[::-1]):
        # ``relation`` is directional: recompute it after swapping the
        # subject/reference agents instead of reusing pair[0] vs pair[1].
        features = _pair_frame_features(record, subject_id, reference_id)
        frames = []
        for item in features:
            pair_state = item["pair"]
            speed_difference = finite(pair_state.get("speed_difference_mps"))
            if (
                pair_state.get("same_lane") is True
                and item["relation"] == "behind"
                and item["distance_m"] is not None
                and item["distance_m"] <= thresholds.max_follow_distance_m
                and (
                    speed_difference is None
                    or abs(speed_difference) <= thresholds.max_follow_speed_difference_mps
                )
            ):
                frames.append(item["frame"])
        for start, end in _contiguous_runs(frames):
            if end - start + 1 >= thresholds.min_frames:
                candidates.append((subject_id, reference_id, list(range(start, end + 1))))
    if not candidates:
        return []
    subject_id, reference_id, frames = max(candidates, key=lambda item: len(item[2]))
    agent_map = _agent_map(record)
    subject_states = state_by_frame(agent_map.get(subject_id, {}))
    stop_frames = [
        frame
        for frame in frames
        if finite(subject_states.get(frame, {}).get("velocity", {}).get("speed")) is not None
        and finite(subject_states[frame]["velocity"]["speed"]) <= thresholds.stop_speed_mps
    ]
    subtype = "stop_behind_lead" if len(stop_frames) >= thresholds.min_frames else "follow_lead"
    evidence = {
        "same_lane_frame_count": len(frames),
        "stop_frame_count": len(stop_frames),
        "subject_relation": "behind",
        "max_follow_distance_m": thresholds.max_follow_distance_m,
    }
    return [
        _interval_event(
            "follow_stop_%s_%s_%d" % (subject_id, reference_id, frames[0]),
            "follow_stop",
            subject_id,
            reference_id,
            frames[0],
            frames[-1],
            evidence,
            subtype=subtype,
            confidence=min(1.0, len(frames) / 10.0),
        )
    ]


def detect_pass(
    record: Mapping[str, Any],
    thresholds: BehaviorThresholds = BehaviorThresholds(),
) -> List[Dict[str, Any]]:
    """Detect an observable longitudinal order flip."""
    pair = _key_pair(record)
    if pair is None:
        return []
    features = _pair_frame_features(record, pair[0], pair[1])
    valid = [item for item in features if item["relation"] in {"ahead", "behind"}]
    stable_frames = max(thresholds.min_frames, thresholds.stable_order_frames)
    stable_runs = _stable_relation_runs(features, stable_frames)
    if len(valid) < stable_frames * 2 or len(stable_runs) < 2:
        return []

    transition = None
    for before_index, (before_relation, before_items) in enumerate(stable_runs[:-1]):
        for after_relation, after_items in stable_runs[before_index + 1 :]:
            if before_relation == after_relation:
                continue
            transition = (before_relation, before_items, after_relation, after_items)
            break
        if transition is not None:
            break
    if transition is None:
        return []
    before_relation, before_items, after_relation, after_items = transition
    first = before_items[0]
    last = after_items[-1]
    if before_relation == "behind" and after_relation == "ahead":
        subject_id, reference_id = pair
    elif before_relation == "ahead" and after_relation == "behind":
        subject_id, reference_id = pair[::-1]
    else:
        return []
    aligned = [
        item
        for item in valid
        if finite(item["pair"].get("relative_heading_rad")) is not None
        and abs(finite(item["pair"].get("relative_heading_rad")))
        <= thresholds.max_heading_difference_rad
    ]
    if len(aligned) < thresholds.min_frames:
        return []
    same_lane = _same_lane_count(valid)
    if same_lane < thresholds.min_frames:
        return []
    evidence = {
        "relation_before": before_relation,
        "relation_after": after_relation,
        "stable_before_frame_count": len(before_items),
        "stable_after_frame_count": len(after_items),
        "stable_order_frames_required": stable_frames,
        "transition_frame": int(after_items[0]["frame"]),
        "same_lane_frame_count": same_lane,
        "aligned_frame_count": len(aligned),
        "order_flip": True,
    }
    event = _interval_event(
        "pass_%s_%s_%d" % (subject_id, reference_id, first["frame"]),
        "pass",
        subject_id,
        reference_id,
        first["frame"],
        last["frame"],
        evidence,
        confidence=min(1.0, 0.5 + 0.5 * min(1.0, len(aligned) / 10.0)),
    )
    event["transition_frame"] = int(after_items[0]["frame"])
    return [event]


def detect_overtake(
    record: Mapping[str, Any],
    lane_change_events: Sequence[Mapping[str, Any]],
    pass_events: Sequence[Mapping[str, Any]],
    thresholds: BehaviorThresholds = BehaviorThresholds(),
) -> List[Dict[str, Any]]:
    """Upgrade a pass only when the same agent also has a nearby lane change."""
    result: List[Dict[str, Any]] = []
    for pass_event in pass_events:
        subject_id = str(pass_event.get("agent_id"))
        for lane_event in lane_change_events:
            if str(lane_event.get("agent_id")) != subject_id:
                continue
            gap = interval_gap(
                int(lane_event.get("start_frame", 0)),
                int(lane_event.get("end_frame", 0)),
                int(pass_event.get("start_frame", 0)),
                int(pass_event.get("end_frame", 0)),
            )
            if gap > thresholds.max_event_gap_frames:
                continue
            evidence = dict(pass_event.get("evidence", {}))
            evidence.update(
                {
                    "lane_change_event_id": lane_event.get("event_id"),
                    "pass_event_id": pass_event.get("event_id"),
                    "interval_gap_frames": gap,
                    "order_flip": True,
                }
            )
            event = _interval_event(
                "overtake_%s_%s_%d"
                % (
                    subject_id,
                    pass_event.get("reference_agent_id"),
                    int(pass_event.get("start_frame", 0)),
                ),
                "overtake",
                subject_id,
                as_agent_id(pass_event.get("reference_agent_id")),
                min(int(pass_event["start_frame"]), int(lane_event["start_frame"])),
                max(int(pass_event["end_frame"]), int(lane_event["end_frame"])),
                evidence,
                subtype="with_lane_change",
                confidence=min(
                    finite(pass_event.get("confidence")) or 0.0,
                    finite(lane_event.get("confidence")) or 0.0,
                ),
            )
            event["derived_from"] = [
                pass_event.get("event_id"),
                lane_event.get("event_id"),
            ]
            result.append(event)
    return result


def detect_merge_candidate(
    record: Mapping[str, Any],
    thresholds: BehaviorThresholds = BehaviorThresholds(),
) -> List[Dict[str, Any]]:
    """Detect a low-confidence merge candidate from raw trajectory evidence.

    This detector intentionally does not consume the legacy pairwise behavior
    classifier.  A merge candidate requires a stable, frame-contiguous
    ``same_lane=False -> same_lane=True`` transition.  If raw lane IDs do not
    identify a unique lane-changing agent, the result remains pair-level and
    must be reviewed before caption generation.

    The candidate subtype is semantic: ``path_convergence`` means that the
    path topology supports a merge interpretation (``P-M`` or ``M-P``),
    whereas ``lane_convergence`` means only that the vehicles enter the same
    lane and does not distinguish an ordinary lane change from a merge.
    """
    pair = _key_pair(record)
    if pair is None:
        return []
    features = _pair_frame_features(record, pair[0], pair[1])
    stable_frames = max(thresholds.min_frames, thresholds.stable_order_frames)
    if len(features) < stable_frames * 2:
        return []
    path_relation = str(
        record.get("interaction", {})
        .get("source_labels", {})
        .get("path_relation", "")
    )
    non_same_lane_runs = _stable_flag_runs(
        features, "same_lane", False, stable_frames
    )
    same_lane_runs = _stable_flag_runs(
        features, "same_lane", True, stable_frames
    )
    transition = None
    for before_run in non_same_lane_runs:
        after_runs = [
            run for run in same_lane_runs if run[0]["frame"] > before_run[-1]["frame"]
        ]
        if after_runs:
            transition = (before_run, after_runs[0])
            break
    if transition is None:
        return []
    before_run, after_run = transition
    start_frame = int(before_run[0]["frame"])
    transition_frame = int(after_run[0]["frame"])
    end_frame = int(after_run[-1]["frame"])

    lane_history: Dict[str, Dict[str, Optional[str]]] = {}
    agents = _agent_map(record)
    for agent_id in pair:
        states = state_by_frame(agents.get(agent_id, {}))
        before_candidates = [
            (frame, state) for frame, state in states.items()
            if frame <= int(before_run[-1]["frame"])
            and state.get("map", {}).get("lane_id") is not None
        ]
        after_candidates = [
            (frame, state) for frame, state in states.items()
            if frame >= transition_frame
            and state.get("map", {}).get("lane_id") is not None
        ]
        before_state = (
            max(before_candidates, key=lambda item: item[0])[1]
            if before_candidates
            else None
        )
        after_state = (
            min(after_candidates, key=lambda item: item[0])[1]
            if after_candidates
            else None
        )
        lane_history[agent_id] = {
            "before_lane_id": (
                str(before_state["map"]["lane_id"])
                if before_state is not None
                else None
            ),
            "after_lane_id": (
                str(after_state["map"]["lane_id"])
                if after_state is not None
                else None
            ),
        }

    changed_agents = [
        agent_id
        for agent_id, history in lane_history.items()
        if history["before_lane_id"] is not None
        and history["after_lane_id"] is not None
        and history["before_lane_id"] != history["after_lane_id"]
    ]
    subject_id: Optional[str] = changed_agents[0] if len(changed_agents) == 1 else None
    reference_id: Optional[str] = (
        next(agent_id for agent_id in pair if agent_id != subject_id)
        if subject_id is not None
        else None
    )
    same_lane_frames = [item["frame"] for item in after_run]
    non_same_lane_frames = [item["frame"] for item in before_run]
    evidence = {
        "path_relation": path_relation,
        "candidate_pair_agent_ids": list(pair),
        "same_lane_transition": True,
        "transition_frame": transition_frame,
        "stable_order_frames_required": stable_frames,
        "same_lane_frame_count": len(same_lane_frames),
        "non_same_lane_frame_count": len(non_same_lane_frames),
        "lane_history": lane_history,
        "subject_inference": "unique_lane_id_change" if subject_id else "ambiguous",
    }
    event_id = "merge_candidate_%s_%s_%d" % (pair[0], pair[1], start_frame)
    candidate_subtype = (
        "path_convergence"
        if path_relation in {"P-M", "M-P"}
        else "lane_convergence"
    )
    event = _interval_event(
        event_id,
        "merge_candidate",
        subject_id,
        reference_id,
        start_frame,
        end_frame,
        evidence,
        subtype=candidate_subtype,
        confidence=0.5 if subject_id is not None else 0.35,
    )
    event["transition_frame"] = transition_frame
    return [event]


def _promote_confirmed_merge(
    event: Mapping[str, Any],
    thresholds: BehaviorThresholds,
) -> Dict[str, Any]:
    """Promote a candidate only when trajectory and topology support merge.

    A confirmed merge requires:

    1. stable ``same_lane=False -> same_lane=True`` convergence;
    2. a uniquely inferred subject;
    3. the subject entering a lane kept by the reference agent; and
    4. source path topology supporting merging, such as ``P-M`` or ``M-P``.

    ``unique_lane_id_change`` identifies who changed lanes; it does not by
    itself distinguish a road merge from an ordinary lane change.  In
    particular, ``P-P`` is never promoted.  Otherwise the event remains a
    ``merge_candidate``.
    """

    if event.get("type") != "merge_candidate":
        return dict(event)
    evidence = event.get("evidence", {})
    if not isinstance(evidence, Mapping):
        return dict(event)
    if evidence.get("subject_inference") != "unique_lane_id_change":
        return dict(event)
    path_relation = evidence.get("path_relation")
    if path_relation not in {"P-M", "M-P"}:
        return dict(event)
    subject_id = event.get("agent_id")
    reference_id = event.get("reference_agent_id")
    lane_history = evidence.get("lane_history", {})
    if (
        subject_id is None
        or reference_id is None
        or not isinstance(lane_history, Mapping)
    ):
        return dict(event)
    subject_history = lane_history.get(str(subject_id), {})
    reference_history = lane_history.get(str(reference_id), {})
    if not isinstance(subject_history, Mapping) or not isinstance(reference_history, Mapping):
        return dict(event)
    subject_before = subject_history.get("before_lane_id")
    subject_after = subject_history.get("after_lane_id")
    reference_before = reference_history.get("before_lane_id")
    reference_after = reference_history.get("after_lane_id")
    stable_required = max(thresholds.min_frames, thresholds.stable_order_frames)
    if not all(value is not None for value in (subject_before, subject_after, reference_before, reference_after)):
        return dict(event)
    if subject_before == subject_after:
        return dict(event)
    if subject_after != reference_after or reference_before != reference_after:
        return dict(event)
    if int(evidence.get("same_lane_frame_count", 0) or 0) < stable_required:
        return dict(event)
    if int(evidence.get("non_same_lane_frame_count", 0) or 0) < stable_required:
        return dict(event)

    promoted = dict(event)
    candidate_event_id = str(event.get("event_id"))
    promoted["event_id"] = candidate_event_id.replace(
        "merge_candidate_", "merge_", 1
    )
    promoted["type"] = "merge"
    promoted["subtype"] = "path_convergence"
    promoted["source"] = "trajectory_map_rule_confirmed_merge"
    promoted["confidence"] = max(float(event.get("confidence") or 0.0), 0.75)
    promoted["derived_from"] = [candidate_event_id]
    promoted_evidence = dict(evidence)
    promoted_evidence["merge_confirmation"] = (
        "path_topology_and_unique_subject_lane_convergence"
    )
    promoted_evidence["confirmed_subject_agent_id"] = str(subject_id)
    promoted_evidence["confirmed_reference_agent_id"] = str(reference_id)
    promoted["evidence"] = promoted_evidence
    return promoted


def detect_behavior_events(
    record: Mapping[str, Any],
    thresholds: BehaviorThresholds = BehaviorThresholds(),
) -> List[Dict[str, Any]]:
    """Run the phase-2 behavior mode and return deterministic event records."""
    lane_change_events = detect_lane_change_events(record)
    merge_events = [
        _promote_confirmed_merge(event, thresholds)
        for event in detect_merge_candidate(record, thresholds)
    ]
    follow_events = detect_follow_stop(record, thresholds)
    pass_events = detect_pass(record, thresholds)
    overtake_events = detect_overtake(
        record, lane_change_events, pass_events, thresholds
    )
    events = lane_change_events + merge_events + follow_events + pass_events + overtake_events
    return sorted(
        events,
        key=lambda event: (
            int(event.get("start_frame", 0)),
            str(event.get("agent_id")),
            str(event.get("type")),
        ),
    )
