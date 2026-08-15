from scripts.archive.semantic.classify_interaction import classify_interaction
from scripts.archive.compat.pair_behavior_timeline import build_pair_behavior_timeline


def _state(frame, lane_id, x, speed, y=0.0, heading=0.0, valid=True):
    return {
        "frame": frame,
        "time_s": frame * 0.1,
        "relative_time_s": frame * 0.1,
        "position": {"x": float(x), "y": float(y), "z": 0.0},
        "velocity": {"vx": float(speed), "vy": 0.0, "speed": float(speed)},
        "acceleration": {"ax": 0.0, "ay": 0.0},
        "heading_rad": heading,
        "map": {
            "lane_id": lane_id,
            "lane_match_confidence": 0.95 if valid else None,
            "match_status": "matched" if valid else "unmatched",
            "centerline_distance_m": 0.1 if valid else None,
            "heading_error_rad": 0.0 if valid else None,
            "lane_s_m": float(x) if valid else None,
            "lateral_offset_m": 0.0 if valid else None,
        },
        "valid": valid,
    }


def _record(b_speed_during=8.0):
    a_states = []
    b_states = []
    for frame in range(40):
        # A starts behind B and crosses to ahead after entering B's lane.
        a_x = frame - 8.0
        if frame >= 30:
            a_x += 24.0
        a_states.append(_state(frame, "lane_A" if frame < 10 else "lane_B", a_x, 10.0))
        b_states.append(_state(frame, "lane_B", 0.0, 10.0 if frame < 10 else b_speed_during))
    return {
        "scene_id": "scene_21",
        "source": {"dataset_key": "waymo_train", "scene_id": "scene_21"},
        "temporal": {"interaction_start_frame": 8, "interaction_end_frame": 15, "dt_seconds": 0.1},
        "agents": [
            {"agent_id": "396", "states": a_states},
            {"agent_id": "421", "states": b_states},
        ],
    }


def test_pair_timeline_confirms_merge_and_yield_for_fixed_agent_roles():
    result = build_pair_behavior_timeline(_record(), "396", "421")

    assert result["scene_id"] == "scene_21"
    assert result["candidate"]["agent_A"] == "396"
    assert result["candidate"]["agent_B"] == "421"
    assert "interaction" not in result
    assert result["pair_facts"] == {
        "A_changed_lane": True,
        "B_maintained_lane": True,
        "A_entered_B_lane": True,
        "A_behind_to_ahead": True,
        "B_speed_reduced_during_event": True,
    }
    assert result["agents"]["agent_A"]["facts"] == [
        "changed_lane",
        "moved_from_behind_to_ahead",
        "entered_agent_B_lane",
    ]
    assert result["agents"]["agent_B"]["facts"] == ["maintained_lane", "reduced_speed"]
    assert result["agents"]["agent_A"]["before"]["relative_position"] == "behind"
    assert result["agents"]["agent_A"]["after"]["relative_position"] == "ahead"
    assert result["agents"]["agent_A"]["before"]["lane_relation"] == "different_lane"
    assert result["agents"]["agent_A"]["after"]["lane_relation"] == "same_lane"
    assert len(result["agents"]["agent_A"]["states"]) == 40
    assert len(result["agents"]["agent_B"]["states"]) == 40
    classified = classify_interaction(result)
    assert classified["interaction_type"] == "overtake"
    assert classified["modifiers"] == ["with_lane_change", "with_yielding"]


def test_pair_timeline_does_not_confirm_yield_without_speed_reduction():
    result = build_pair_behavior_timeline(_record(b_speed_during=10.0), "396", "421")

    assert "interaction" not in result
    assert result["pair_facts"]["B_speed_reduced_during_event"] is False
    assert result["agents"]["agent_B"]["facts"] == ["maintained_lane"]
    classified = classify_interaction(result)
    assert classified["interaction_type"] == "overtake"
    assert classified["modifiers"] == ["with_lane_change"]


def test_interhub_temporal_metadata_does_not_affect_timeline():
    record_without_temporal = _record()
    record_with_different_temporal = _record()
    record_with_different_temporal["temporal"] = {
        "interaction_start_frame": 35,
        "interaction_end_frame": 39,
    }

    result_without = build_pair_behavior_timeline(record_without_temporal, "396", "421")
    result_with = build_pair_behavior_timeline(record_with_different_temporal, "396", "421")

    assert result_without == result_with
    assert set(result_with["candidate"]) == {"agent_A", "agent_B"}


def test_single_agent_lane_change_without_pair_relation_change_is_not_promoted():
    record = _record()
    for state in record["agents"][1]["states"]:
        state["map"]["lane_id"] = "lane_C"

    timeline = build_pair_behavior_timeline(record, "396", "421")

    assert timeline["analysis_window"]["status"] == "no_pair_transition_closest_approach_anchor"
    assert timeline["analysis_window"]["transition_frame"] is None
    assert timeline["pair_facts"] == {
        "A_changed_lane": None,
        "B_maintained_lane": None,
        "A_entered_B_lane": None,
        "A_behind_to_ahead": None,
        "B_speed_reduced_during_event": None,
    }
    assert len(timeline["agents"]["agent_A"]["lane_change_events"]) == 1
    assert classify_interaction(timeline)["interaction_type"] == "unknown"


def test_classifier_separates_lane_change_with_yielding_from_overtake():
    record = _record()
    for agent in record["agents"]:
        if agent["agent_id"] != "396":
            continue
        for state in agent["states"]:
            state["position"]["x"] = -10.0

    timeline = build_pair_behavior_timeline(record, "396", "421")
    classified = classify_interaction(timeline)

    assert timeline["pair_facts"]["A_behind_to_ahead"] is False
    assert timeline["pair_facts"]["B_speed_reduced_during_event"] is True
    assert classified["interaction_type"] == "lane_change_with_yielding"
