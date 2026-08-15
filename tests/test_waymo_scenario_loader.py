from types import SimpleNamespace

from scripts.archive.compat.pair_behavior_timeline import build_pair_behavior_timeline
from scripts.archive.legacy.waymo_scenario_loader import scenario_to_motion_record


def _point(x, y):
    return SimpleNamespace(x=x, y=y, z=0.0)


def _state(x, y, heading=0.0, valid=True, speed=5.0):
    return SimpleNamespace(
        valid=valid,
        center_x=x,
        center_y=y,
        center_z=0.0,
        velocity_x=speed,
        velocity_y=0.0,
        heading=heading,
    )


def test_scenario_proto_shape_is_normalized_and_lane_matched():
    scenario = SimpleNamespace(
        scenario_id="scene_proto",
        sdc_track_index=-1,
        tracks_to_predict=[],
        map_features=[
            SimpleNamespace(
                id=490,
                lane=SimpleNamespace(
                    polyline=[_point(0, 0), _point(10, 0), _point(20, 0)]
                ),
            ),
            SimpleNamespace(
                id=637,
                lane=SimpleNamespace(
                    polyline=[_point(0, 4), _point(10, 4), _point(20, 4)]
                ),
            ),
        ],
        tracks=[
            SimpleNamespace(
                id=396,
                object_type=1,
                states=[_state(1, 0), _state(5, 4), _state(10, 4)],
            ),
            SimpleNamespace(
                id=421,
                object_type=1,
                states=[_state(2, 4), _state(6, 4), _state(11, 4)],
            ),
        ],
    )

    record = scenario_to_motion_record(scenario)

    assert record["scene_id"] == "scene_proto"
    agent_a_lanes = [state["map"]["lane_id"] for state in record["agents"][0]["states"]]
    agent_b_lanes = [state["map"]["lane_id"] for state in record["agents"][1]["states"]]
    assert agent_a_lanes[0] != agent_a_lanes[1] == agent_a_lanes[2]
    assert agent_b_lanes[0] == agent_a_lanes[1]
    assert record["agents"][0]["states"][0]["velocity"]["speed"] == 5.0


def test_scenario_record_feeds_pair_timeline_facts():
    scenario = SimpleNamespace(
        scenario_id="scene_proto_pair",
        sdc_track_index=-1,
        tracks_to_predict=[],
        map_features=[
            SimpleNamespace(
                id=490,
                lane=SimpleNamespace(
                    polyline=[_point(0, 0), _point(10, 0), _point(20, 0)]
                ),
            ),
            SimpleNamespace(
                id=637,
                lane=SimpleNamespace(
                    polyline=[_point(0, 4), _point(10, 4), _point(20, 4)]
                ),
            ),
        ],
        tracks=[
            SimpleNamespace(
                id=396,
                object_type=1,
                states=[_state(1, 0), _state(3, 0), _state(5, 0)]
                + [_state(7, 4), _state(9, 4), _state(11, 4)]
                + [_state(13, 4), _state(15, 4), _state(17, 4)],
            ),
            SimpleNamespace(
                id=421,
                object_type=1,
                states=[_state(2, 4), _state(4, 4), _state(6, 4)]
                + [_state(8, 4), _state(10, 4), _state(12, 4)]
                + [_state(14, 4), _state(16, 4), _state(18, 4)],
            ),
        ],
    )

    record = scenario_to_motion_record(scenario)
    timeline = build_pair_behavior_timeline(record, "396", "421")

    assert timeline["analysis_window"]["status"] == "pair_transition"
    assert timeline["pair_facts"]["A_changed_lane"] is True
    assert timeline["pair_facts"]["A_entered_B_lane"] is True
    assert timeline["pair_facts"]["B_maintained_lane"] is True
