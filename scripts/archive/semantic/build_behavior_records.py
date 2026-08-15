"""Build phase-2 behavior records from classified scene_motion_v3 JSON files.

The input records already contain Waymo states, map matches, pairwise
features, and the InterHub candidate window.  This script does not rerun
InterHub extraction.  It detects behavior events, keeps agent-level events
separate from pair associations, and writes one behavior-record bundle per
classified input plus a JSONL index.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

try:
    from .behavior_taxonomy import (
        ASSOCIATION_RELATIONS,
        BehaviorThresholds,
        detect_behavior_events,
    )
except ImportError:  # pragma: no cover - supports direct script execution.
    from behavior_taxonomy import (
        ASSOCIATION_RELATIONS,
        BehaviorThresholds,
        detect_behavior_events,
    )


BEHAVIOR_RECORD_SCHEMA_VERSION = "behavior_records_v1"


def _behavior_evidence_sufficient(
    event: Mapping[str, Any],
    thresholds: BehaviorThresholds,
) -> bool:
    """Judge whether the event itself is describable without InterHub relation."""
    event_type = event.get("type")
    evidence = event.get("evidence", {})
    if not isinstance(evidence, Mapping):
        evidence = {}
    if event.get("agent_id") is None:
        return False
    if not (
        isinstance(event.get("start_frame"), int)
        and isinstance(event.get("end_frame"), int)
        and event["end_frame"] >= event["start_frame"]
    ):
        return False
    if event_type == "lane_change":
        lane_from = event.get("from_lane_id", evidence.get("from_lane_id"))
        lane_to = event.get("to_lane_id", evidence.get("to_lane_id"))
        return lane_from is not None and lane_to is not None
    if event_type == "follow_stop":
        return (
            evidence.get("subject_relation") == "behind"
            and int(evidence.get("same_lane_frame_count", 0) or 0)
            >= thresholds.min_frames
        )
    if event_type in {"pass", "overtake"}:
        stable_required = max(
            thresholds.min_frames, thresholds.stable_order_frames
        )
        return (
            evidence.get("order_flip") is True
            and int(evidence.get("stable_before_frame_count", 0) or 0)
            >= stable_required
            and int(evidence.get("stable_after_frame_count", 0) or 0)
            >= stable_required
        )
    if event_type == "merge":
        return evidence.get("merge_confirmation") == (
            "path_topology_and_unique_subject_lane_convergence"
        )
    return False


def _derived_event_ids(event: Mapping[str, Any]) -> List[str]:
    derived = event.get("derived_from", [])
    if isinstance(derived, list):
        return [str(event_id) for event_id in derived if event_id is not None]
    return []


def _write_jsonl_records(handle: Any, bundle: Mapping[str, Any]) -> None:
    records = bundle.get("behavior_records", [])
    if not isinstance(records, list):
        return
    for behavior_record in records:
        handle.write(
            json.dumps(behavior_record, ensure_ascii=False, allow_nan=False)
            + "\n"
        )


def _finite_int(value: Any) -> Optional[int]:
    return int(value) if isinstance(value, int) else None


def _window(record: Mapping[str, Any], name: str) -> Tuple[Optional[int], Optional[int]]:
    temporal = record.get("temporal", {})
    if not isinstance(temporal, Mapping):
        return None, None
    start = _finite_int(temporal.get(f"{name}_start_frame"))
    end = _finite_int(temporal.get(f"{name}_end_frame"))
    if start is None or end is None or end < start:
        return None, None
    return start, end


def _overlap_frames(
    first_start: Optional[int],
    first_end: Optional[int],
    second_start: Optional[int],
    second_end: Optional[int],
) -> int:
    if None in (first_start, first_end, second_start, second_end):
        return 0
    return max(0, min(first_end, second_end) - max(first_start, second_start) + 1)


def _key_pair(record: Mapping[str, Any]) -> Tuple[str, ...]:
    interaction = record.get("interaction", {})
    key_ids = interaction.get("key_agent_ids", []) if isinstance(interaction, Mapping) else []
    return tuple(str(agent_id) for agent_id in key_ids if agent_id is not None)


def _finite_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _agent_map(record: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    return {
        str(agent.get("agent_id")): agent
        for agent in record.get("agents", [])
        if isinstance(agent, Mapping) and agent.get("agent_id") is not None
    }


def _state_map(agent: Optional[Mapping[str, Any]]) -> Dict[int, Mapping[str, Any]]:
    if not isinstance(agent, Mapping):
        return {}
    return {
        int(state["frame"]): state
        for state in agent.get("states", [])
        if isinstance(state, Mapping) and isinstance(state.get("frame"), int)
    }


def _nearest_state(
    states: Mapping[int, Mapping[str, Any]],
    frame: int,
    before: bool,
) -> Optional[Mapping[str, Any]]:
    frames = [item for item in states if item < frame] if before else [item for item in states if item >= frame]
    if not frames:
        return None
    selected = max(frames) if before else min(frames)
    return states[selected]


def _speed(state: Optional[Mapping[str, Any]]) -> Optional[float]:
    if not isinstance(state, Mapping):
        return None
    velocity = state.get("velocity", {})
    if not isinstance(velocity, Mapping):
        return None
    return _finite_float(velocity.get("speed"))


def _mean_speed(
    states: Mapping[int, Mapping[str, Any]],
    start_frame: int,
    end_frame: int,
) -> Optional[float]:
    values = [
        speed
        for frame, state in states.items()
        if start_frame <= frame <= end_frame
        for speed in [_speed(state)]
        if speed is not None
    ]
    return sum(values) / len(values) if values else None


def _mean_speed_before_caption(
    states: Mapping[int, Mapping[str, Any]],
    caption_start_frame: int,
    dt_seconds: Optional[float],
) -> Optional[float]:
    """Use the same one-second pre-caption mean for every speed relation."""
    if dt_seconds is None or dt_seconds <= 0:
        offset_frames = 10
    else:
        offset_frames = max(1, int(round(1.0 / dt_seconds)))
    return _mean_speed(
        states,
        caption_start_frame - offset_frames,
        caption_start_frame - 1,
    )


def _pairwise_states(record: Mapping[str, Any]) -> Dict[int, Mapping[str, Any]]:
    pairwise = record.get("pairwise", {})
    states = pairwise.get("states", []) if isinstance(pairwise, Mapping) else []
    return {
        int(state["frame"]): state
        for state in states
        if isinstance(state, Mapping) and isinstance(state.get("frame"), int)
    }


def _behavior_evidence(
    record: Mapping[str, Any],
    event: Mapping[str, Any],
) -> Dict[str, Any]:
    evidence: Dict[str, Any] = {}
    interaction = record.get("interaction", {})
    behavior = interaction.get("behavior", {}) if isinstance(interaction, Mapping) else {}
    if isinstance(behavior, Mapping) and isinstance(behavior.get("evidence"), Mapping):
        evidence.update(behavior["evidence"])
    if isinstance(event.get("evidence"), Mapping):
        evidence.update(event["evidence"])
    return evidence


def _lane_relation_at(
    pair_states: Mapping[int, Mapping[str, Any]],
    subject_states: Mapping[int, Mapping[str, Any]],
    reference_states: Mapping[int, Mapping[str, Any]],
    frame: int,
) -> Optional[str]:
    pair = pair_states.get(frame)
    if isinstance(pair, Mapping) and pair.get("same_lane") is True:
        return "same_lane"
    if isinstance(pair, Mapping) and pair.get("same_lane") is False:
        return "different_lane"
    subject = subject_states.get(frame, {})
    reference = reference_states.get(frame, {})
    subject_lane = subject.get("map", {}).get("lane_id") if isinstance(subject, Mapping) else None
    reference_lane = reference.get("map", {}).get("lane_id") if isinstance(reference, Mapping) else None
    if subject_lane is None or reference_lane is None:
        return None
    return "same_lane" if str(subject_lane) == str(reference_lane) else "different_lane"


def _relative_position_at(
    subject_states: Mapping[int, Mapping[str, Any]],
    reference_states: Mapping[int, Mapping[str, Any]],
    frame: int,
) -> Optional[str]:
    subject = subject_states.get(frame)
    reference = reference_states.get(frame)
    if not isinstance(subject, Mapping) or not isinstance(reference, Mapping):
        return None
    subject_position = subject.get("position", {})
    reference_position = reference.get("position", {})
    heading = _finite_float(reference.get("heading_rad"))
    if not isinstance(subject_position, Mapping) or not isinstance(reference_position, Mapping) or heading is None:
        return None
    sx, sy = _finite_float(subject_position.get("x")), _finite_float(subject_position.get("y"))
    rx, ry = _finite_float(reference_position.get("x")), _finite_float(reference_position.get("y"))
    if None in (sx, sy, rx, ry):
        return None
    longitudinal = (sx - rx) * math.cos(heading) + (sy - ry) * math.sin(heading)
    if longitudinal > 0.5:
        return "ahead"
    if longitudinal < -0.5:
        return "behind"
    return "alongside"


def _distance_trend(
    record: Mapping[str, Any],
    start_frame: int,
    end_frame: int,
) -> Tuple[Optional[float], str]:
    pair_states = _pairwise_states(record)
    distances = {
        frame: _finite_float(state.get("distance_m"))
        for frame, state in pair_states.items()
    }
    distances = {frame: value for frame, value in distances.items() if value is not None}
    summary = record.get("pairwise", {}).get("summary", {})
    minimum = _finite_float(summary.get("minimum_distance_m")) if isinstance(summary, Mapping) else None
    before = [value for frame, value in distances.items() if frame < start_frame]
    during = [value for frame, value in distances.items() if start_frame <= frame <= end_frame]
    after = [value for frame, value in distances.items() if frame > end_frame]
    if minimum is None and during:
        minimum = min(during)
    if not before or not during or not after:
        return minimum, "unknown"
    before_value = before[-1]
    during_minimum = min(during)
    after_value = after[0]
    epsilon = 0.25
    if before_value > during_minimum + epsilon and after_value > during_minimum + epsilon:
        return minimum, "decreasing_then_increasing"
    if before_value - after_value > epsilon:
        return minimum, "decreasing"
    if after_value - before_value > epsilon:
        return minimum, "increasing"
    return minimum, "stable"


def _speed_context(
    record: Mapping[str, Any],
    subject_id: Optional[str],
    reference_id: Optional[str],
    start_frame: int,
    end_frame: int,
) -> Tuple[Dict[str, Optional[float]], Dict[str, Optional[float]], Dict[str, str]]:
    agents = _agent_map(record)
    subject_states = _state_map(agents.get(str(subject_id)))
    reference_states = _state_map(agents.get(str(reference_id)))
    temporal = record.get("temporal", {})
    dt_seconds = (
        _finite_float(temporal.get("dt_seconds"))
        if isinstance(temporal, Mapping)
        else None
    )
    values_before = {
        "subject": _mean_speed_before_caption(subject_states, start_frame, dt_seconds),
        "reference": _mean_speed_before_caption(reference_states, start_frame, dt_seconds),
    }
    values_during = {
        "subject": _mean_speed(subject_states, start_frame, end_frame),
        "reference": _mean_speed(reference_states, start_frame, end_frame),
    }

    def trend(before: Optional[float], during: Optional[float]) -> str:
        if before is None or during is None:
            return "unknown"
        delta = during - before
        if delta >= 0.3:
            return "accelerating"
        if delta <= -0.3:
            return "decelerating"
        return "stable"

    return values_before, values_during, {
        "subject": trend(values_before["subject"], values_during["subject"]),
        "reference": trend(values_before["reference"], values_during["reference"]),
    }


def _interaction_relation(
    record: Mapping[str, Any],
    event: Mapping[str, Any],
    subject_id: Optional[str],
    reference_id: Optional[str],
    start_frame: int,
    end_frame: int,
) -> Dict[str, Any]:
    evidence = _behavior_evidence(record, event)
    agents = _agent_map(record)
    subject_states = _state_map(agents.get(str(subject_id)))
    reference_states = _state_map(agents.get(str(reference_id)))
    pair_states = _pairwise_states(record)
    before_frame = max([frame for frame in pair_states if frame < start_frame], default=start_frame - 1)
    after_frame = min([frame for frame in pair_states if frame > end_frame], default=end_frame + 1)
    before_position = evidence.get("longitudinal_relation_before") or _relative_position_at(subject_states, reference_states, before_frame)
    after_position = evidence.get("longitudinal_relation_after") or _relative_position_at(subject_states, reference_states, after_frame)
    before_lane = evidence.get("lane_relation_before") or _lane_relation_at(pair_states, subject_states, reference_states, before_frame)
    after_lane = evidence.get("lane_relation_after") or _lane_relation_at(pair_states, subject_states, reference_states, after_frame)
    behavior_type = event.get("type")
    during_relation = {
        "lane_change": "lane_crossing",
        "merge": "lane_convergence",
        "merge_candidate": "possible_lane_convergence",
        "follow_stop": "following",
        "pass": "longitudinal_passing",
        "overtake": "longitudinal_passing_with_lane_change",
    }.get(behavior_type, behavior_type or "unknown")
    return {
        "before": {
            "relative_position": before_position,
            "lane_relation": before_lane,
        },
        "during": {"relation": during_relation},
        "after": {
            "relative_position": after_position,
            "lane_relation": after_lane,
        },
    }


def _existing_associations(record: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    associations = record.get("behavior_associations", [])
    if not isinstance(associations, list):
        return {}
    return {
        str(association.get("behavior_event_id")): association
        for association in associations
        if isinstance(association, Mapping)
        and association.get("behavior_event_id") is not None
    }


def _same_pair(event: Mapping[str, Any], key_pair: Tuple[str, ...]) -> bool:
    subject = event.get("agent_id")
    reference = event.get("reference_agent_id")
    evidence = event.get("evidence", {})
    candidate_pair = (
        evidence.get("candidate_pair_agent_ids", [])
        if isinstance(evidence, Mapping)
        else []
    )
    if subject is None and isinstance(candidate_pair, list):
        return {str(agent_id) for agent_id in candidate_pair} == set(key_pair)
    if reference is not None:
        return {str(subject), str(reference)} == set(key_pair)
    return str(subject) in key_pair and len(key_pair) == 2


def _derive_relation(
    event: Mapping[str, Any],
    record: Mapping[str, Any],
    existing: Mapping[str, Any],
    key_pair: Tuple[str, ...],
) -> Tuple[str, Optional[str], str]:
    """Return relation, reference ID, and caption scope.

    For lane changes, trust the first-phase association result.  For pair
    behaviors detected in this phase, an outside-window association requires
    behavior evidence (order flip, same-lane run, or lane convergence); time
    proximity alone is never sufficient.
    """
    event_start = _finite_int(event.get("start_frame"))
    event_end = _finite_int(event.get("end_frame"))
    interhub_start, interhub_end = _window(record, "interaction")
    overlap = _overlap_frames(event_start, event_end, interhub_start, interhub_end)
    subject_id = event.get("agent_id")
    reference = event.get("reference_agent_id")
    reference_id = str(reference) if reference is not None else None

    if existing:
        relation = str(existing.get("relation", "nearby_unverified"))
        if relation not in ASSOCIATION_RELATIONS:
            relation = "nearby_unverified"
        if relation in {"overlap", "associated_but_outside"}:
            reference_id = (
                str(existing.get("reference_agent_id"))
                if existing.get("reference_agent_id") is not None
                else reference_id
            )
        return relation, reference_id, "pair" if relation in {"overlap", "associated_but_outside"} else "agent_only"

    if not _same_pair(event, key_pair):
        return "unrelated", None, "agent_only"
    if (
        event.get("type") == "lane_change"
        and reference_id is not None
        and event.get("interaction_type") == "lane_change_with_neighbor"
    ):
        return (
            "overlap" if overlap > 0 else "associated_but_outside",
            reference_id,
            "pair",
        )
    if overlap > 0:
        if reference_id is None and subject_id is not None and len(key_pair) == 2:
            reference_id = next(
                agent_id for agent_id in key_pair if agent_id != str(subject_id)
            )
        return "overlap", reference_id, "pair" if len(key_pair) == 2 else "agent_only"

    evidence = event.get("evidence", {})
    if not isinstance(evidence, Mapping):
        evidence = {}
    strong_evidence = any(
        bool(evidence.get(name))
        for name in (
            "order_flip",
            "same_lane_transition",
            "longitudinal_order_changed",
        )
    ) or int(evidence.get("same_lane_frame_count", 0) or 0) >= 3
    if strong_evidence:
        if reference_id is None and subject_id is not None and len(key_pair) == 2:
            reference_id = next(
                agent_id for agent_id in key_pair if agent_id != str(subject_id)
            )
        return (
            "associated_but_outside",
            reference_id,
            "pair" if len(key_pair) == 2 else "agent_only",
        )
    return "nearby_unverified", None, "agent_only"


def build_behavior_records(
    record: Mapping[str, Any],
    thresholds: BehaviorThresholds = BehaviorThresholds(),
) -> Dict[str, Any]:
    events = detect_behavior_events(record, thresholds)
    key_pair = _key_pair(record)
    existing_associations = _existing_associations(record)
    interhub_start, interhub_end = _window(record, "interaction")
    suppressed_by: Dict[str, str] = {}
    for event in events:
        if event.get("type") != "overtake":
            continue
        overtake_id = str(event.get("event_id"))
        for derived_id in _derived_event_ids(event):
            suppressed_by[derived_id] = overtake_id

    # Keep agent-level lane_change events in the behavior event list, but do
    # not generate a duplicate caption when the same subject has an
    # overlapping, topology-confirmed merge.  The merge is the higher-level
    # caption semantics; suppression applies only to caption eligibility.
    for merge_event in events:
        if merge_event.get("type") != "merge":
            continue
        merge_subject = merge_event.get("agent_id")
        merge_start = _finite_int(merge_event.get("start_frame"))
        merge_end = _finite_int(merge_event.get("end_frame"))
        if merge_subject is None or merge_start is None or merge_end is None:
            continue
        for lane_event in events:
            if lane_event.get("type") != "lane_change":
                continue
            if str(lane_event.get("agent_id")) != str(merge_subject):
                continue
            lane_start = _finite_int(lane_event.get("start_frame"))
            lane_end = _finite_int(lane_event.get("end_frame"))
            if _overlap_frames(merge_start, merge_end, lane_start, lane_end) <= 0:
                continue
            lane_event_id = str(lane_event.get("event_id"))
            suppressed_by.setdefault(lane_event_id, str(merge_event.get("event_id")))
    behavior_records: List[Dict[str, Any]] = []

    for event in events:
        event_id = str(event.get("event_id"))
        relation, reference_id, caption_scope = _derive_relation(
            event,
            record,
            existing_associations.get(event_id, {}),
            key_pair,
        )
        start_frame = _finite_int(event.get("start_frame"))
        end_frame = _finite_int(event.get("end_frame"))
        if start_frame is None or end_frame is None or end_frame < start_frame:
            continue
        evidence_sufficient = _behavior_evidence_sufficient(event, thresholds)
        suppressed_by_event = suppressed_by.get(event_id)
        behavior_type = event.get("type")
        if behavior_type == "merge_candidate":
            caption_eligible = False
        elif suppressed_by_event is not None:
            caption_eligible = False
        elif behavior_type == "lane_change":
            caption_eligible = evidence_sufficient
        elif behavior_type in {"follow_stop", "pass", "overtake", "merge"}:
            caption_eligible = (
                evidence_sufficient and reference_id is not None
            )
        else:
            caption_eligible = False
        interaction_relation = _interaction_relation(
            record,
            event,
            str(event["agent_id"]) if event.get("agent_id") is not None else None,
            reference_id,
            start_frame,
            end_frame,
        )
        minimum_distance_m, distance_trend = _distance_trend(
            record,
            start_frame,
            end_frame,
        )
        speed_before_mps, speed_during_mps, speed_relation = _speed_context(
            record,
            str(event["agent_id"]) if event.get("agent_id") is not None else None,
            reference_id,
            start_frame,
            end_frame,
        )
        behavior_record = {
            "record_id": f"{record.get('interaction_id')}::{event_id}",
            "interaction_id": record.get("interaction_id"),
            "source": record.get("source", {}),
            "subject_agent_id": (
                str(event["agent_id"]) if event.get("agent_id") is not None else None
            ),
            "reference_agent_id": reference_id,
            "candidate_pair_agent_ids": event.get("evidence", {}).get(
                "candidate_pair_agent_ids", []
            ) if isinstance(event.get("evidence", {}), Mapping) else [],
            "behavior_event_id": event_id,
            "behavior_type": behavior_type,
            "behavior_subtype": event.get("subtype"),
            "interaction_type": event.get(
                "interaction_type",
                "lane_change_with_neighbor"
                if behavior_type == "lane_change" and reference_id is not None
                else behavior_type,
            ),
            "interaction_relation": interaction_relation,
            "minimum_distance_m": minimum_distance_m,
            "distance_trend": distance_trend,
            "speed_before_mps": speed_before_mps,
            "speed_during_mps": speed_during_mps,
            "speed_relation": speed_relation,
            "behavior_window": {
                "start_frame": start_frame,
                "end_frame": end_frame,
            },
            # Phase 2 initially keeps caption window equal to behavior window.
            "caption_window": {
                "start_frame": start_frame,
                "end_frame": end_frame,
            },
            "interhub_window": {
                "start_frame": interhub_start,
                "end_frame": interhub_end,
            },
            "window_relation": relation,
            "caption_scope": caption_scope,
            "confidence": event.get("confidence"),
            "evidence": event.get("evidence", {}),
            "behavior_evidence_sufficient": evidence_sufficient,
            "derived_from": event.get("derived_from", []),
            "caption_suppressed_by": suppressed_by_event,
            "caption_eligible": caption_eligible,
            "needs_review": (
                relation == "nearby_unverified"
                or behavior_type == "merge_candidate"
                or not evidence_sufficient
            ),
        }
        behavior_records.append(behavior_record)

    return {
        "schema_version": BEHAVIOR_RECORD_SCHEMA_VERSION,
        "interaction_id": record.get("interaction_id"),
        "source": record.get("source", {}),
        "interhub_window": {
            "start_frame": interhub_start,
            "end_frame": interhub_end,
        },
        "key_agent_ids": list(key_pair),
        "behavior_events": events,
        "behavior_records": behavior_records,
        "quality": {
            "behavior_event_count": len(events),
            "caption_eligible_count": sum(
                1 for item in behavior_records if item["caption_eligible"]
            ),
            "needs_review_count": sum(
                1 for item in behavior_records if item["needs_review"]
            ),
        },
    }


def iter_input_paths(input_dir: Path) -> Iterable[Path]:
    yield from sorted(input_dir.glob("row_*_v3_classified.json"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--min-frames", type=int, default=3)
    parser.add_argument("--stable-order-frames", type=int, default=3)
    parser.add_argument("--max-follow-distance-m", type=float, default=30.0)
    parser.add_argument("--stop-speed-mps", type=float, default=1.0)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    input_paths = list(iter_input_paths(args.input_dir))
    if not input_paths:
        raise SystemExit(f"No classified JSON files found in {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = BehaviorThresholds(
        min_frames=max(1, args.min_frames),
        stable_order_frames=max(1, args.stable_order_frames),
        max_follow_distance_m=max(0.0, args.max_follow_distance_m),
        stop_speed_mps=max(0.0, args.stop_speed_mps),
    )
    jsonl_path = args.output_dir / "behavior_records.jsonl"
    failed: List[Dict[str, str]] = []
    success = skipped = 0
    with jsonl_path.open("w", encoding="utf-8", newline="\n") as jsonl:
        for input_path in input_paths:
            output_name = input_path.name.replace(
                "_v3_classified.json", "_behavior_records.json"
            )
            output_path = args.output_dir / output_name
            try:
                if args.skip_existing and output_path.exists():
                    existing_bundle = json.loads(
                        output_path.read_text(encoding="utf-8")
                    )
                    _write_jsonl_records(jsonl, existing_bundle)
                    skipped += 1
                    continue
                record = json.loads(input_path.read_text(encoding="utf-8"))
                bundle = build_behavior_records(record, thresholds)
                output_path.write_text(
                    json.dumps(bundle, ensure_ascii=False, indent=2, allow_nan=False)
                    + "\n",
                    encoding="utf-8",
                )
                _write_jsonl_records(jsonl, bundle)
                success += 1
            except Exception as exc:  # keep batch processing after one bad file
                failed.append(
                    {
                        "input_file": str(input_path),
                        "output_file": str(output_path),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )

    failed_csv = args.output_dir / "failed_files.csv"
    with failed_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["input_file", "output_file", "error_type", "error"],
        )
        writer.writeheader()
        writer.writerows(failed)
    print(f"success: {success}")
    print(f"skipped: {skipped}")
    print(f"failed: {len(failed)}")
    print(f"behavior_records_jsonl: {jsonl_path}")
    print(f"failed_files_csv: {failed_csv}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
