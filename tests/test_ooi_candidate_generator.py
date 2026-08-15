from types import SimpleNamespace

import numpy as np

from scripts.facts.build_pair_timeline import (
    LaneNeighborSegment,
    LaneFrameMatch,
    LaneMap,
    LaneSegment,
    LaneTimeline,
    _short_lane_run_confidence_is_lower,
    _lane_segment_runs,
    build_physical_lane_chain,
    _compute_route_transition,
    extract_heading_motion_facts,
    extract_pair_geometry_facts,
)
from scripts.facts.ooi_candidate_generator import (
    candidate_pairs,
    unique_ooi_ids,
    validate_physical_facts,
    vehicle_ooi_ids,
)


def _track(track_id, object_type):
    return SimpleNamespace(id=track_id, object_type=object_type)


def test_ooi_ids_are_deduplicated_and_vehicle_filter_uses_track_id():
    scenario = SimpleNamespace(
        objects_of_interest=[20, 10, 20, 30, 99],
        tracks=[
            _track(10, 1),
            _track(20, 2),
            _track(30, 1),
        ],
    )

    assert unique_ooi_ids(scenario) == [20, 10, 30, 99]
    assert vehicle_ooi_ids(scenario) == ([10, 30], [99])


def test_candidate_pairs_are_unordered_and_deterministic():
    assert candidate_pairs([30, 10, 20, 10]) == [(10, 20), (10, 30), (20, 30)]


def test_physical_facts_validator_accepts_complete_result():
    def agent(agent_id):
        return {
            "id": agent_id,
            "object_type": 1,
            "is_object_of_interest": True,
        }

    result = {
        "schema_version": "pair_physical_facts_v1",
        "pair": {"agent_A": agent(10), "agent_B": agent(20)},
        "physical_facts": {
            "agent_A": {
                "physical_lateral_maneuver": {},
                "speed_change": {},
                "heading_motion": {
                    "status": "uncertain",
                    "heading_start_deg": 0.0,
                    "heading_end_deg": 0.0,
                    "net_heading_change_deg": 0.0,
                    "heading_change_direction": "roughly_stable",
                    "turning_episode": {"status": "no_clear_heading_change_episode"},
                    "start_frame": 0,
                    "end_frame": 1,
                },
                "route_transition": {
                    "route_transition": False,
                    "transitions": [],
                    "primary_transition": None,
                    "lane_before": None,
                    "lane_after": None,
                    "incoming_lane_heading_deg": None,
                    "outgoing_lane_heading_deg": None,
                    "lane_heading_change_deg": None,
                    "connected_in_lane_graph": None,
                },
            },
            "agent_B": {
                "physical_lateral_maneuver": {},
                "speed_change": {},
                "heading_motion": {
                    "status": "uncertain",
                    "heading_start_deg": 0.0,
                    "heading_end_deg": 0.0,
                    "net_heading_change_deg": 0.0,
                    "heading_change_direction": "roughly_stable",
                    "turning_episode": {"status": "no_clear_heading_change_episode"},
                    "start_frame": 0,
                    "end_frame": 1,
                },
                "route_transition": {
                    "route_transition": False,
                    "transitions": [],
                    "primary_transition": None,
                    "lane_before": None,
                    "lane_after": None,
                    "incoming_lane_heading_deg": None,
                    "outgoing_lane_heading_deg": None,
                    "lane_heading_change_deg": None,
                    "connected_in_lane_graph": None,
                },
            },
            "pair_relation": {
                "common_valid_frame_count": 1,
                "travel_channel": {},
                "longitudinal": {},
                "per_frame": [{
                    "frame": 0,
                    "distance_m": 3.0,
                    "travel_channel_relation": "same_travel_channel",
                    "longitudinal_relation": "agent_B_ahead_of_agent_A",
                }],
            },
            "pair_geometry": {
                "heading_relation": {
                    "status": "uncertain",
                    "dominant": "uncertain",
                    "frame_counts": {"uncertain": 1},
                    "runs": [{"start_frame": 0, "end_frame": 0, "duration_frames": 1, "value": "aligned"}],
                    "heading_difference_summary_deg": {"median": 0.0, "min": 0.0, "max": 0.0},
                },
                "closest_approach": {"status": "unavailable"},
                "distance_evolution": {
                    "status": "insufficient_data",
                    "overall": "uncertain",
                },
                "path_geometry": {
                    "status": "insufficient_data",
                    "min_path_distance_m": None,
                    "closest_path_points": None,
                    "spatial_overlap": "uncertain",
                    "convergence_status": "uncertain",
                },
            },
            "map_context": {
                "near_intersection": None,
                "intersection_evidence": {
                    "lane_graph_branching": None,
                    "nearby_crosswalk": None,
                    "nearby_traffic_signal": None,
                    "nearby_stop_sign": None,
                },
                "nearby_crosswalk": None,
                "nearby_stop_sign": None,
                "nearby_traffic_signal": None,
            },
        },
    }

    audit = validate_physical_facts(result, 10, 20)
    assert audit["valid"] is True
    assert audit["errors"] == []


def test_physical_facts_validator_rejects_wrong_ooi_identity():
    result = {
        "schema_version": "pair_physical_facts_v1",
        "pair": {
            "agent_A": {"id": 10, "object_type": 1, "is_object_of_interest": False},
            "agent_B": {"id": 20, "object_type": 1, "is_object_of_interest": True},
        },
        "physical_facts": {},
    }

    audit = validate_physical_facts(result, 10, 20)
    assert audit["valid"] is False
    assert any("is_object_of_interest" in error for error in audit["errors"])


def _geometry_track(yaws, valid=None):
    if valid is None:
        valid = [True] * len(yaws)
    return SimpleNamespace(
        T=len(yaws),
        yaw=np.asarray(yaws, dtype=float),
        xy=np.stack([np.arange(len(yaws), dtype=float), np.zeros(len(yaws))], axis=-1),
        valid=np.asarray(valid, dtype=bool),
        timestamps=np.arange(len(yaws), dtype=float) * 0.1,
    )


def _geometry_relation(distances):
    return {
        "per_frame": [
            {
                "frame": i,
                "distance_m": float(distance),
                "travel_channel_relation": "uncertain",
                "longitudinal_relation": "uncertain",
            }
            for i, distance in enumerate(distances)
        ]
    }


def _lane_segment(lane_id, y=0.0, entry=None, exit=None, left=None, right=None):
    points = np.array([[0.0, y, 0.0], [5.0, y, 0.0]], dtype=float)
    return LaneSegment(
        lane_id=lane_id,
        lane_type=1,
        polyline=points,
        tangent_xy=np.array([[1.0, 0.0], [1.0, 0.0]], dtype=float),
        entry_lanes=list(entry or []),
        exit_lanes=list(exit or []),
        left_neighbors=list(left or []),
        right_neighbors=list(right or []),
    )


def test_physical_lane_chain_normalizes_exit_segments_to_lane_keep():
    track = _geometry_track([0.0] * 9)
    matches = [
        LaneFrameMatch(i, True, lane_id, group_id, 1.0, 0.0, [])
        for i, (lane_id, group_id) in enumerate(
            [(490, 490)] * 3 + [(637, 637)] * 3 + [(812, 812)] * 3
        )
    ]
    timeline = LaneTimeline(
        matches,
        [490] * 3 + [637] * 3 + [812] * 3,
        [490] * 3 + [637] * 3 + [812] * 3,
        [490] * 3 + [637] * 3 + [812] * 3,
    )
    lane_map = LaneMap(
        segments={
            490: _lane_segment(490, exit=[637]),
            637: _lane_segment(637, entry=[490], exit=[812]),
            812: _lane_segment(812, entry=[637]),
        },
        segment_to_group={490: 490, 637: 637, 812: 812},
        group_to_segments={490: [490], 637: [637], 812: [812]},
        group_neighbors={490: [], 637: [], 812: []},
        group_left_neighbors={490: [], 637: [], 812: []},
        group_right_neighbors={490: [], 637: [], 812: []},
        continuation_edges=[(490, 637), (637, 812)],
        point_xy=np.zeros((1, 2)),
        point_z=np.zeros(1),
        point_dir=np.ones((1, 2)),
        point_lane_id=np.array([490]),
    )

    result = build_physical_lane_chain(track, timeline, lane_map)

    assert result["lane_chain"] == [490, 637, 812]
    assert [item["relation"] for item in result["topological_relations"]] == [
        "continuation",
        "continuation",
    ]
    assert result["behavior"] == "lane_keep"


def test_physical_lane_chain_requires_neighbor_and_lateral_motion_for_lane_change():
    track = _geometry_track([0.0] * 6)
    matches = [
        LaneFrameMatch(i, True, 490 if i < 3 else 701, 490 if i < 3 else 701, 1.0, 0.0, [])
        for i in range(6)
    ]
    timeline = LaneTimeline(
        matches,
        [490] * 3 + [701] * 3,
        [490] * 3 + [701] * 3,
        [490] * 3 + [701] * 3,
    )
    source = _lane_segment(490, left=[701])
    source.left_neighbor_segments = [LaneNeighborSegment(701, 0, 1, 0, 1)]
    lane_map = LaneMap(
        segments={490: source, 701: _lane_segment(701, y=3.0)},
        segment_to_group={490: 490, 701: 701},
        group_to_segments={490: [490], 701: [701]},
        group_neighbors={490: [701], 701: [490]},
        group_left_neighbors={490: [701], 701: []},
        group_right_neighbors={490: [], 701: []},
        continuation_edges=[],
        point_xy=np.zeros((1, 2)),
        point_z=np.zeros(1),
        point_dir=np.ones((1, 2)),
        point_lane_id=np.array([490]),
    )

    result = build_physical_lane_chain(
        track,
        timeline,
        lane_map,
        {"episodes": [{
            "start_frame": 2,
            "end_frame": 4,
            "direction_relative_to_local_lane": "left",
        }]},
    )

    transition = result["topological_relations"][0]
    assert transition["relation"] == "left_neighbor"
    assert transition["physical_lane_change_supported"] is True
    assert result["behavior"] == "left_lane_change"


def test_segment_flicker_is_removed_only_with_low_confidence_and_no_forward_topology():
    track = _geometry_track([0.0] * 7)
    matches = [
        LaneFrameMatch(
            i,
            True,
            637 if i == 3 else 490,
            637 if i == 3 else 490,
            0.20 if i == 3 else 0.90,
            0.0,
            [],
        )
        for i in range(7)
    ]
    timeline = LaneTimeline(
        matches,
        [490, 490, 490, 637, 490, 490, 490],
        [490, 490, 490, 637, 490, 490, 490],
        [490, 490, 490, 637, 490, 490, 490],
    )
    lane_map = LaneMap(
        segments={490: _lane_segment(490), 637: _lane_segment(637)},
        segment_to_group={490: 490, 637: 637},
        group_to_segments={490: [490], 637: [637]},
        group_neighbors={490: [], 637: []},
        group_left_neighbors={490: [], 637: []},
        group_right_neighbors={490: [], 637: []},
        continuation_edges=[],
        point_xy=np.zeros((1, 2)),
        point_z=np.zeros(1),
        point_dir=np.ones((1, 2)),
        point_lane_id=np.array([490]),
    )

    raw, runs, filtered, rejected, unavailable = _lane_segment_runs(track, timeline)

    assert raw == [490, 490, 490, 637, 490, 490, 490]
    assert [run["lane_id"] for run in runs] == [490, 490]
    assert filtered == []
    assert unavailable == []
    assert rejected[0]["lane_id"] == 637
    assert rejected[0]["reason"] == "below_lane_confidence_min"


def test_flicker_confidence_must_be_lower_than_both_neighbor_runs():
    confidences = [0.90, 0.90, 0.50, 0.55, 0.55]

    assert _short_lane_run_confidence_is_lower(
        confidences,
        start=2,
        end=2,
        before_start=0,
        before_end=1,
        after_start=3,
        after_end=4,
    ) is False


def test_short_segment_is_preserved_when_it_forms_forward_topology_chain():
    track = _geometry_track([0.0] * 6)
    matches = [
        LaneFrameMatch(
            i,
            True,
            lane_id,
            lane_id,
            0.90,
            0.0,
            [],
        )
        for i, lane_id in enumerate([490, 490, 637, 637, 812, 812])
    ]
    timeline = LaneTimeline(
        matches,
        [490, 490, 637, 637, 812, 812],
        [490, 490, 637, 637, 812, 812],
        [490, 490, 637, 637, 812, 812],
    )
    lane_map = LaneMap(
        segments={
            490: _lane_segment(490, exit=[637]),
            637: _lane_segment(637, entry=[490], exit=[812]),
            812: _lane_segment(812, entry=[637]),
        },
        segment_to_group={490: 490, 637: 637, 812: 812},
        group_to_segments={490: [490], 637: [637], 812: [812]},
        group_neighbors={490: [], 637: [], 812: []},
        group_left_neighbors={490: [], 637: [], 812: []},
        group_right_neighbors={490: [], 637: [], 812: []},
        continuation_edges=[(490, 637), (637, 812)],
        point_xy=np.zeros((1, 2)),
        point_z=np.zeros(1),
        point_dir=np.ones((1, 2)),
        point_lane_id=np.array([490]),
    )

    _, runs, filtered, rejected, unavailable = _lane_segment_runs(track, timeline)

    assert [run["lane_id"] for run in runs] == [490, 637, 812]
    assert filtered == []
    assert rejected == []
    assert unavailable == []


def test_long_low_confidence_segment_is_rejected_before_topology():
    lane_ids = [490, 490, 490, 637, 637, 637, 637, 812, 812, 812]
    track = _geometry_track([0.0] * len(lane_ids))
    matches = [
        LaneFrameMatch(
            frame,
            True,
            lane_id,
            lane_id,
            0.20 if lane_id == 637 else 0.90,
            0.0,
            [],
        )
        for frame, lane_id in enumerate(lane_ids)
    ]
    timeline = LaneTimeline(matches, lane_ids, lane_ids, lane_ids)
    lane_map = LaneMap(
        segments={lane_id: _lane_segment(lane_id) for lane_id in [490, 637, 812]},
        segment_to_group={lane_id: lane_id for lane_id in [490, 637, 812]},
        group_to_segments={lane_id: [lane_id] for lane_id in [490, 637, 812]},
        group_neighbors={lane_id: [] for lane_id in [490, 637, 812]},
        group_left_neighbors={lane_id: [] for lane_id in [490, 637, 812]},
        group_right_neighbors={lane_id: [] for lane_id in [490, 637, 812]},
        continuation_edges=[],
        point_xy=np.zeros((1, 2)),
        point_z=np.zeros(1),
        point_dir=np.ones((1, 2)),
        point_lane_id=np.array([490]),
    )

    raw, runs, filtered, rejected, unavailable = _lane_segment_runs(track, timeline)

    assert raw[3:7] == [637, 637, 637, 637]
    assert 637 not in [run["lane_id"] for run in runs]
    assert filtered == []
    assert rejected == [{
        "start_frame": 3,
        "end_frame": 6,
        "lane_id": 637,
        "confidence_median": 0.2,
        "reason": "below_lane_confidence_min",
    }]
    assert unavailable == []
    chain = build_physical_lane_chain(track, timeline, lane_map)
    assert chain["behavior"] == "unknown"
    assert chain["topological_relations"] == []
    assert chain["unresolved_confidence_gaps"][0]["before_lane_id"] == 490
    assert chain["unresolved_confidence_gaps"][0]["after_lane_id"] == 812


def test_short_low_confidence_a_to_b_to_c_is_preserved():
    lane_ids = [490, 490, 490, 637, 812, 812, 812]
    track = _geometry_track([0.0] * len(lane_ids))
    matches = [
        LaneFrameMatch(
            frame,
            True,
            lane_id,
            lane_id,
            0.35 if lane_id == 637 else 0.90,
            0.0,
            [],
        )
        for frame, lane_id in enumerate(lane_ids)
    ]
    timeline = LaneTimeline(matches, lane_ids, lane_ids, lane_ids)

    _, runs, filtered, rejected, unavailable = _lane_segment_runs(track, timeline)

    assert [run["lane_id"] for run in runs] == [490, 637, 812]
    assert filtered == []
    assert rejected == []
    assert unavailable == []


def test_unavailable_match_gap_sets_behavior_unknown_without_cross_gap_transition():
    lane_ids = [490, 490, 490, None, None, None, 701, 701, 701]
    track = _geometry_track([0.0] * len(lane_ids))
    matches = []
    for frame, lane_id in enumerate(lane_ids):
        if lane_id is None:
            matches.append(LaneFrameMatch(frame, False, None, None, None, None, []))
        else:
            matches.append(LaneFrameMatch(frame, True, lane_id, lane_id, 0.90, 0.0, []))
    timeline = LaneTimeline(
        matches,
        lane_ids,
        lane_ids,
        lane_ids,
    )
    lane_map = LaneMap(
        segments={490: _lane_segment(490), 701: _lane_segment(701)},
        segment_to_group={490: 490, 701: 701},
        group_to_segments={490: [490], 701: [701]},
        group_neighbors={490: [], 701: []},
        group_left_neighbors={490: [], 701: []},
        group_right_neighbors={490: [], 701: []},
        continuation_edges=[],
        point_xy=np.zeros((1, 2)),
        point_z=np.zeros(1),
        point_dir=np.ones((1, 2)),
        point_lane_id=np.array([490]),
    )

    _, runs, filtered, rejected, unavailable = _lane_segment_runs(track, timeline)
    chain = build_physical_lane_chain(track, timeline, lane_map)

    assert [run["lane_id"] for run in runs] == [490, 701]
    assert filtered == []
    assert rejected == []
    assert unavailable[0]["start_frame"] == 3
    assert unavailable[0]["end_frame"] == 5
    assert chain["behavior"] == "unknown"
    assert chain["topological_relations"] == []
    assert chain["unresolved_lane_gaps"][0]["gap_types"] == ["match_unavailable"]


def test_pair_geometry_same_direction_and_closest_approach():
    distances = [12, 11, 10, 9, 8, 7, 6, 5, 6, 7, 8, 9, 10, 11, 12]
    result = extract_pair_geometry_facts(
        _geometry_track([0.0] * len(distances)),
        _geometry_track([0.05] * len(distances)),
        _geometry_relation(distances),
    )
    assert result["heading_relation"]["dominant"] == "aligned"
    assert result["closest_approach"]["frame"] == 7
    assert result["closest_approach"]["distance_m"] == 5.0
    assert result["distance_evolution"]["overall"] == "approaching_then_separating"
    assert result["path_geometry"]["status"] == "available"


def test_closest_approach_uses_exact_argmin_without_tolerance_shift():
    distances = [3.00, 2.96, 3.00]
    result = extract_pair_geometry_facts(
        _geometry_track([0.0] * len(distances)),
        _geometry_track([0.0] * len(distances)),
        _geometry_relation(distances),
    )
    assert result["closest_approach"]["frame"] == 1
    assert result["closest_approach"]["minimum_distance_m"] == 2.96


def test_path_geometry_detects_spatial_overlap_and_convergence():
    a = _geometry_track([0.0] * 6)
    b = _geometry_track([0.0] * 6)
    a.xy = np.stack([np.arange(6, dtype=float), np.zeros(6)], axis=-1)
    b.xy = np.stack([np.arange(6, dtype=float), np.linspace(3.0, 0.0, 6)], axis=-1)
    distances = [float(np.linalg.norm(a.xy[i] - b.xy[i])) for i in range(6)]
    result = extract_pair_geometry_facts(a, b, _geometry_relation(distances))
    assert result["path_geometry"]["spatial_overlap"] == "paths_have_spatial_overlap"
    assert result["path_geometry"]["convergence_status"] == "paths_converge"


def test_route_transition_uses_continuous_lane_groups_and_graph_connection():
    track = _geometry_track([0.0] * 4)
    track.xy = np.stack([np.arange(4, dtype=float), np.zeros(4)], axis=-1)
    matches = [
        LaneFrameMatch(i, True, 10 if i < 2 else 20, 100 if i < 2 else 200, 1.0, 0.0, [])
        for i in range(4)
    ]
    timeline = LaneTimeline(matches, [100, 100, 200, 200], [100, 100, 200, 200], [100, 100, 200, 200])
    seg_a = LaneSegment(10, 1, np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), np.array([[1.0, 0.0], [1.0, 0.0]]), [], [20], [], [])
    seg_b = LaneSegment(20, 1, np.array([[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]), np.array([[1.0, 0.0], [1.0, 0.0]]), [10], [], [], [])
    lane_map = LaneMap(
        segments={10: seg_a, 20: seg_b},
        segment_to_group={10: 100, 20: 200},
        group_to_segments={100: [10], 200: [20]},
        group_neighbors={100: [], 200: []},
        group_left_neighbors={100: [], 200: []},
        group_right_neighbors={100: [], 200: []},
        continuation_edges=[],
        point_xy=np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]),
        point_z=np.zeros(4),
        point_dir=np.tile(np.array([[1.0, 0.0]]), (4, 1)),
        point_lane_id=np.array([10, 10, 20, 20]),
    )
    result = _compute_route_transition(track, timeline, lane_map)
    assert result["route_transition"] is True
    assert result["lane_before"] == 100
    assert result["lane_after"] == 200
    assert result["connected_in_lane_graph"] is True


def test_route_transition_keeps_all_transitions_and_selects_primary_by_heading_change():
    track = _geometry_track([0.0] * 6)
    track.xy = np.stack([np.arange(6, dtype=float), np.zeros(6)], axis=-1)
    matches = [
        LaneFrameMatch(i, True, 10 if i < 2 else 20 if i < 4 else 30, 100 if i < 2 else 200 if i < 4 else 300, 1.0, 0.0, [])
        for i in range(6)
    ]
    timeline = LaneTimeline(matches, [100, 100, 200, 200, 300, 300], [100, 100, 200, 200, 300, 300], [100, 100, 200, 200, 300, 300])
    seg_a = LaneSegment(10, 1, np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), np.array([[1.0, 0.0], [1.0, 0.0]]), [], [20], [], [])
    seg_b = LaneSegment(20, 1, np.array([[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]), np.array([[1.0, 0.0], [1.0, 0.0]]), [10], [30], [], [])
    seg_c = LaneSegment(30, 1, np.array([[4.0, 0.0, 0.0], [5.0, 0.0, 0.0]]), np.array([[0.0, 1.0], [0.0, 1.0]]), [20], [], [], [])
    lane_map = LaneMap(
        segments={10: seg_a, 20: seg_b, 30: seg_c},
        segment_to_group={10: 100, 20: 200, 30: 300},
        group_to_segments={100: [10], 200: [20], 300: [30]},
        group_neighbors={100: [], 200: [], 300: []},
        group_left_neighbors={100: [], 200: [], 300: []},
        group_right_neighbors={100: [], 200: [], 300: []},
        continuation_edges=[],
        point_xy=np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0]]),
        point_z=np.zeros(6),
        point_dir=np.tile(np.array([[1.0, 0.0]]), (6, 1)),
        point_lane_id=np.array([10, 10, 20, 20, 30, 30]),
    )
    result = _compute_route_transition(track, timeline, lane_map)
    assert len(result["transitions"]) == 2
    assert result["primary_transition"]["transition_frame"] == 4
    assert result["primary_transition"]["lane_heading_change_deg"] == 90.0


def test_route_transition_heading_uses_stable_run_windows():
    track = _geometry_track([0.0] * 10)
    track.xy = np.stack([np.arange(10, dtype=float), np.zeros(10)], axis=-1)
    matches = [
        LaneFrameMatch(i, True, 10 if i < 5 else 20, 100 if i < 5 else 200, 1.0, 0.0, [])
        for i in range(10)
    ]
    timeline = LaneTimeline(
        matches,
        [100] * 5 + [200] * 5,
        [100] * 5 + [200] * 5,
        [100] * 5 + [200] * 5,
    )
    points_a = np.stack([np.arange(5, dtype=float), np.zeros(5), np.zeros(5)], axis=-1)
    points_b = np.stack([np.arange(5, 10, dtype=float), np.zeros(5), np.zeros(5)], axis=-1)
    angles_a = np.radians([0.0, 10.0, 20.0, 30.0, 40.0])
    angles_b = np.radians([50.0, 60.0, 70.0, 80.0, 90.0])
    tangents_a = np.stack([np.cos(angles_a), np.sin(angles_a)], axis=-1)
    tangents_b = np.stack([np.cos(angles_b), np.sin(angles_b)], axis=-1)
    seg_a = LaneSegment(10, 1, points_a, tangents_a, [], [20], [], [])
    seg_b = LaneSegment(20, 1, points_b, tangents_b, [10], [], [], [])
    lane_map = LaneMap(
        segments={10: seg_a, 20: seg_b},
        segment_to_group={10: 100, 20: 200},
        group_to_segments={100: [10], 200: [20]},
        group_neighbors={100: [], 200: []},
        group_left_neighbors={100: [], 200: []},
        group_right_neighbors={100: [], 200: []},
        continuation_edges=[],
        point_xy=np.concatenate([points_a[:, :2], points_b[:, :2]], axis=0),
        point_z=np.zeros(10),
        point_dir=np.concatenate([tangents_a, tangents_b], axis=0),
        point_lane_id=np.array([10] * 5 + [20] * 5),
    )
    result = _compute_route_transition(track, timeline, lane_map)
    transition = result["primary_transition"]
    assert transition["incoming_lane_heading_deg"] == 20.0
    assert transition["outgoing_lane_heading_deg"] == 70.0
    assert transition["lane_heading_change_deg"] == 50.0


def test_pair_geometry_cross_direction_keeps_orthogonal_heading_fact():
    distances = [12, 11, 10, 9, 8, 7, 6, 5, 6, 7, 8, 9, 10, 11, 12]
    result = extract_pair_geometry_facts(
        _geometry_track([0.0] * len(distances)),
        _geometry_track([np.pi / 2.0] * len(distances)),
        _geometry_relation(distances),
    )
    assert result["heading_relation"]["dominant"] == "cross_direction"
    assert result["closest_approach"]["frame"] == 7
    assert result["distance_evolution"]["overall"] == "approaching_then_separating"


def test_pair_geometry_opposing_and_single_agent_heading_motion():
    distances = [12] * 15
    result = extract_pair_geometry_facts(
        _geometry_track([0.0] * len(distances)),
        _geometry_track([np.pi] * len(distances)),
        _geometry_relation(distances),
    )
    assert result["heading_relation"]["dominant"] == "opposing"
    motion = extract_heading_motion_facts(
        _geometry_track(np.linspace(0.0, np.pi / 2.0, 31))
    )
    assert motion["net_heading_change_deg"] == 90.0
    assert motion["heading_change_direction"] == "increasing_ccw"
    assert motion["turning_episode"]["status"] == "available"


def test_pair_geometry_insufficient_data_is_non_crashing_and_uncertain():
    result = extract_pair_geometry_facts(
        _geometry_track([0.0, 0.0], valid=[True, False]),
        _geometry_track([0.0, 0.0], valid=[True, False]),
        {"per_frame": [{"frame": 0, "distance_m": 3.0}]},
    )
    assert result["heading_relation"]["status"] in {"uncertain", "insufficient_data"}
    assert result["heading_relation"]["dominant"] is None
    assert result["closest_approach"]["status"] == "available"
    assert result["distance_evolution"]["overall"] == "uncertain"
