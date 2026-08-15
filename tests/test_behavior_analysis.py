from types import SimpleNamespace

from scripts.archive.semantic.behavior_analysis import (
    BehaviorConfig,
    TTCConfig,
    classify_interaction_behavior,
    compute_ttc_metrics,
    detect_lane_change_events,
)


def _lane(lane_id, next_lanes=(), left=(), right=()):
    return SimpleNamespace(
        id=lane_id,
        next_lanes=set(next_lanes),
        prev_lanes=set(),
        adj_lanes_left=set(left),
        adj_lanes_right=set(right),
    )


def _state(frame, lane_id, x, y, distance, offset, speed=5.0, valid=True):
    return {
        "frame": frame,
        "valid": valid,
        "position": {"x": x, "y": y, "z": 0.0},
        "velocity": {"vx": speed, "vy": 0.0, "speed": speed},
        "heading_rad": 0.0,
        "map": {
            "lane_id": lane_id,
            "lane_match_confidence": 0.9 if valid else None,
            "match_status": "matched" if valid else "unmatched",
            "centerline_distance_m": distance if valid else None,
            "lateral_offset_m": offset if valid else None,
        },
    }


def _lane_change_states(target_lane="B"):
    states = []
    for frame in range(5):
        states.append(_state(frame, "A", frame, 0.0, 0.4 + 0.15 * frame, 0.4 + 0.15 * frame))
    for frame in range(5, 10):
        step = frame - 5
        states.append(_state(frame, target_lane, frame, 2.0, 1.0 - 0.2 * step, 1.0 - 0.2 * step))
    return states


def test_lane_change_event_requires_adjacent_topology_and_lateral_support():
    lanes = [_lane("A", left=("B",)), _lane("B", right=("A",)), _lane("C")]
    result = detect_lane_change_events(_lane_change_states(), SimpleNamespace(lanes=lanes))
    assert result["lane_change_detected"] is True
    assert result["lane_change_events"][0]["from_lane_id"] == "A"
    assert result["lane_change_events"][0]["to_lane_id"] == "B"
    assert result["lane_change_events"][0]["direction"] == "left"
    assert result["lane_change_events"][0]["confidence"] is not None


def test_right_lane_change_direction_is_reported():
    lanes = [_lane("A", right=("C",)), _lane("C", left=("A",))]
    result = detect_lane_change_events(
        _lane_change_states(target_lane="C"), SimpleNamespace(lanes=lanes)
    )
    assert result["lane_change_detected"] is True
    assert result["lane_change_events"][0]["direction"] == "right"


def test_lane_change_does_not_mark_normal_next_lane_or_one_frame_jump():
    connected = [_lane("A", next_lanes=("B",)), _lane("B")]
    states = _lane_change_states()
    assert detect_lane_change_events(states, SimpleNamespace(lanes=connected))["lane_change_detected"] is False

    adjacent = [_lane("A", left=("B",)), _lane("B", right=("A",))]
    short_jump = [
        _state(frame, "A" if frame != 5 else "B", frame, 0.0 if frame != 5 else 2.0, 0.5, 0.5)
        for frame in range(11)
    ]
    assert detect_lane_change_events(short_jump, SimpleNamespace(lanes=adjacent))["lane_change_detected"] is False


def test_lane_change_returns_null_when_evidence_is_insufficient():
    lanes = [_lane("A", left=("B",)), _lane("B", right=("A",))]
    states = [_state(0, "A", 0.0, 0.0, 0.5, 0.5), _state(1, "B", 1.0, 2.0, 0.5, 0.5)]
    result = detect_lane_change_events(states, SimpleNamespace(lanes=lanes))
    assert result == {"lane_change_detected": None, "lane_change_events": [], "confidence": None}


def test_ttc_statuses_distinguish_approach_recede_lateral_and_overlap():
    config = TTCConfig(default_safety_radius_m=2.0)
    approaching_i = _state(0, "A", 0.0, 0.0, 0.0, 0.0, speed=10.0)
    approaching_j = _state(0, "A", 20.0, 0.0, 0.0, 0.0, speed=0.0)
    result = compute_ttc_metrics(approaching_i, approaching_j, config)
    assert result["ttc_status"] == "valid"
    assert abs(result["closing_speed_mps"] - 10.0) < 1e-6
    assert abs(result["ttc_seconds"] - 1.6) < 1e-6

    receding_j = _state(0, "A", 20.0, 0.0, 0.0, 0.0, speed=10.0)
    assert compute_ttc_metrics(approaching_i, receding_j, config)["ttc_status"] == "not_closing"

    lateral_j = _state(0, "A", 0.0, 20.0, 0.0, 0.0, speed=0.0)
    lateral_i = _state(0, "A", 0.0, 0.0, 0.0, 0.0, speed=10.0)
    assert compute_ttc_metrics(lateral_i, lateral_j, config)["ttc_status"] == "not_applicable"

    overlap_j = _state(0, "A", 0.0, 0.0, 0.0, 0.0, speed=0.0)
    overlap = compute_ttc_metrics(lateral_i, overlap_j, config)
    assert overlap["ttc_status"] == "overlapping"
    assert overlap["ttc_seconds"] == 0.0


def test_ttc_uses_state_dimensions_when_available():
    config = TTCConfig(default_safety_radius_m=2.0)
    approaching_i = _state(0, "A", 0.0, 0.0, 0.0, 0.0, speed=10.0)
    approaching_j = _state(0, "A", 20.0, 0.0, 0.0, 0.0, speed=0.0)
    approaching_i["dimensions"] = {"length_m": 4.0, "width_m": 2.0}
    approaching_j["dimensions"] = {"length_m": 4.0, "width_m": 2.0}

    result = compute_ttc_metrics(approaching_i, approaching_j, config)

    expected_radius = 0.5 * (4.0**2 + 2.0**2) ** 0.5
    assert abs(result["ttc_seconds"] - (20.0 - 2 * expected_radius) / 10.0) < 1e-6


def test_classifier_uses_two_key_agents_with_extra_participant():
    states = [_state(frame, "B", frame, 0.0, 0.2, 0.0) for frame in range(10)]
    agents = [
        {"agent_id": "subject", "states": states},
        {"agent_id": "reference", "states": [_state(frame, "B", frame + 10.0, 0.0, 0.2, 0.0) for frame in range(10)]},
        {"agent_id": "other", "states": [_state(frame, "C", frame, 20.0, 0.2, 0.0) for frame in range(10)]},
    ]
    agent_behaviors = {
        agent["agent_id"]: {
            "lane_change_detected": False,
            "lane_change_events": [],
            "confidence": None,
        }
        for agent in agents
    }
    pairwise = {
        "states": [
            {
                "frame": frame,
                "same_lane": True,
                "distance_m": 10.0,
                "speed_difference_mps": 2.0,
                "ttc_status": "not_closing",
            }
            for frame in range(10)
        ]
    }

    result = classify_interaction_behavior(
        {
            "start": 2,
            "end": 7,
            "participant_ids": ["subject", "reference", "other"],
            "key_agent_ids": ["subject", "reference"],
        },
        agents,
        agent_behaviors,
        pairwise,
        SimpleNamespace(lanes=[_lane("B"), _lane("C")]),
    )

    assert result["processing_status"] == "completed"
    assert result["behavior"]["type"] == "same_lane_interaction"
    assert result["behavior"]["subject_agent_id"] == "subject"
    assert result["behavior"]["reference_agent_id"] == "reference"


def test_classifier_tolerates_middle_invalid_state():
    subject = [_state(frame, "B", frame, 0.0, 0.2, 0.0) for frame in range(10)]
    subject[5] = _state(5, "B", None, None, 0.0, 0.0, valid=False)
    reference = [_state(frame, "B", frame + 10.0, 0.0, 0.2, 0.0) for frame in range(10)]
    agents = [
        {"agent_id": "subject", "states": subject},
        {"agent_id": "reference", "states": reference},
    ]
    agent_behaviors = {
        "subject": {"lane_change_detected": False, "lane_change_events": [], "confidence": None},
        "reference": {"lane_change_detected": False, "lane_change_events": [], "confidence": None},
    }
    pairwise = {
        "states": [
            {
                "frame": frame,
                "same_lane": None if frame == 5 else True,
                "distance_m": None if frame == 5 else 10.0,
                "speed_difference_mps": None if frame == 5 else 2.0,
                "ttc_status": "insufficient_data" if frame == 5 else "not_closing",
            }
            for frame in range(10)
        ]
    }

    result = classify_interaction_behavior(
        {"start": 2, "end": 7, "key_agent_ids": ["subject", "reference"]},
        agents,
        agent_behaviors,
        pairwise,
        SimpleNamespace(lanes=[_lane("B")]),
    )

    assert result["processing_status"] == "completed"
    assert result["behavior"]["type"] == "same_lane_interaction"


def test_behavior_classifier_prioritizes_cut_in_over_lane_change():
    lanes = [_lane("A", left=("B",)), _lane("B", right=("A",))]
    subject = _lane_change_states()
    reference = [
        _state(frame, "B", 0.0, 2.0, 0.2, 0.0, speed=5.0 if frame < 5 else 3.0)
        for frame in range(10)
    ]
    for frame in range(5):
        subject[frame]["position"]["x"] = -5.0 + frame * 0.1
    for frame in range(5, 10):
        subject[frame]["position"]["x"] = 3.0 + (frame - 5) * 0.1
    agents = [{"agent_id": "subject", "states": subject}, {"agent_id": "reference", "states": reference}]
    agent_behaviors = {
        "subject": {
            "lane_change_detected": True,
            "confidence": 0.9,
            "lane_change_events": [
                {
                    "start_frame": 0,
                    "transition_frame": 0,
                    "end_frame": 1,
                    "direction": "left",
                    "confidence": 0.8,
                },
                {
                    "start_frame": 4,
                    "transition_frame": 5,
                    "end_frame": 8,
                    "direction": "left",
                    "confidence": 0.9,
                },
            ],
        },
        "reference": {"lane_change_detected": False, "lane_change_events": [], "confidence": None},
    }
    pairwise = {
        "states": [
            {"frame": frame, "same_lane": frame >= 5, "distance_m": 6.0 if frame < 5 else 3.0, "speed_difference_mps": 0.0, "ttc_status": "not_closing"}
            for frame in range(10)
        ]
    }
    result = classify_interaction_behavior(
        {"start": 2, "end": 7, "key_agent_ids": ["subject", "reference"]},
        agents,
        agent_behaviors,
        pairwise,
        SimpleNamespace(lanes=lanes),
    )
    assert result["processing_status"] == "completed"
    assert result["behavior"]["type"] == "cut_in"
    assert result["behavior"]["evidence"]["lane_change_event"]["transition_frame"] == 5


def test_behavior_classifier_accepts_cut_in_without_longitudinal_order_flip():
    lanes = [_lane("A", left=("B",)), _lane("B", right=("A",))]
    subject = [
        _state(frame, "A" if frame < 5 else "B", 1.0 + 0.1 * frame, 0.0 if frame < 5 else 2.0, 0.4, 0.4)
        for frame in range(10)
    ]
    reference = [_state(frame, "B", 0.0, 2.0, 0.2, 0.0) for frame in range(10)]
    agents = [{"agent_id": "subject", "states": subject}, {"agent_id": "reference", "states": reference}]
    agent_behaviors = {
        "subject": {
            "lane_change_detected": True,
            "confidence": 0.9,
            "lane_change_events": [
                {"start_frame": 4, "transition_frame": 5, "end_frame": 8, "direction": "left"}
            ],
        },
        "reference": {"lane_change_detected": False, "lane_change_events": [], "confidence": None},
    }
    pairwise = {
        "states": [
            {
                "frame": frame,
                "same_lane": frame >= 5,
                "distance_m": 2.0,
                "speed_difference_mps": 0.0,
                "ttc_status": "not_closing",
            }
            for frame in range(10)
        ]
    }

    result = classify_interaction_behavior(
        {"start": 2, "end": 7, "key_agent_ids": ["subject", "reference"]},
        agents,
        agent_behaviors,
        pairwise,
        SimpleNamespace(lanes=lanes),
    )

    assert result["behavior"]["type"] == "cut_in"
    assert result["behavior"]["evidence"]["longitudinal_relation_before"] == "ahead"
    assert result["behavior"]["evidence"]["longitudinal_relation_after"] == "ahead"
    assert result["behavior"]["evidence"]["small_longitudinal_gap_after"] is True


def test_behavior_classifier_distinguishes_merge_and_same_lane_interaction():
    merge_lanes = [
        _lane("A", next_lanes=("B",)),
        _lane("C", next_lanes=("B",)),
        _lane("B"),
    ]
    first = [_state(frame, "A" if frame < 5 else "B", frame, 0.0, 0.5, 0.5) for frame in range(10)]
    second = [_state(frame, "C" if frame < 5 else "B", frame, 3.0, 0.5, 0.5) for frame in range(10)]
    agents = [{"agent_id": "a", "states": first}, {"agent_id": "b", "states": second}]
    agent_behaviors = {
        "a": {"lane_change_detected": False, "lane_change_events": [], "confidence": None},
        "b": {"lane_change_detected": False, "lane_change_events": [], "confidence": None},
    }
    pairwise = {"states": [{"frame": frame, "same_lane": frame >= 5, "distance_m": 3.0, "speed_difference_mps": 0.0, "ttc_status": "not_closing"} for frame in range(10)]}
    result = classify_interaction_behavior(
        {"start": 2, "end": 7, "key_agent_ids": ["a", "b"]},
        agents,
        agent_behaviors,
        pairwise,
        SimpleNamespace(lanes=merge_lanes),
    )
    assert result["behavior"]["type"] == "merge"

    same_agents = [
        {"agent_id": "a", "states": [_state(frame, "B", frame, 0.0, 0.2, 0.0) for frame in range(10)]},
        {"agent_id": "b", "states": [_state(frame, "B", frame + 10.0, 0.0, 0.2, 0.0) for frame in range(10)]},
    ]
    same_behaviors = {
        "a": {"lane_change_detected": False, "lane_change_events": [], "confidence": None},
        "b": {"lane_change_detected": False, "lane_change_events": [], "confidence": None},
    }
    same_pairwise = {"states": [{"frame": frame, "same_lane": True, "distance_m": 10.0, "speed_difference_mps": 2.0, "ttc_status": "not_closing"} for frame in range(10)]}
    same_result = classify_interaction_behavior(
        {"start": 2, "end": 7, "key_agent_ids": ["a", "b"]},
        same_agents,
        same_behaviors,
        same_pairwise,
        SimpleNamespace(lanes=[_lane("B")]),
    )
    assert same_result["behavior"]["type"] == "same_lane_interaction"
