import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from scripts.archive.export.export_scene_json import (
    export_scene_v2,
    export_scene_v3,
    smooth_acceleration,
    validate_scene_motion_v3,
)
from scripts.archive.export.map_matching import MapMatchingConfig, match_trajectory


def _write_fixture(
    root: Path,
    agent_ids=("2255", "2270"),
    frames=range(91),
    drop_frames=None,
    multi=False,
    interaction_start=34,
    interaction_end=53,
    key_agents=None,
):
    scene_dir = root / "scene_16"
    scene_dir.mkdir()
    drop_frames = drop_frames or {}
    rows = []
    for agent_index, agent_id in enumerate(agent_ids):
        for frame in frames:
            if frame in set(drop_frames.get(agent_id, ())):
                continue
            rows.append(
                {
                    "agent_id": agent_id,
                    "scene_ts": frame,
                    "x": 100.0 + frame + agent_index,
                    "y": 200.0 + 0.5 * frame,
                    "z": 0.0,
                    "vx": 10.0 + agent_index,
                    "vy": 0.5,
                    "ax": -28.85 + 0.1 * frame,
                    "ay": 16.99 - 0.05 * frame,
                    "heading": 0.1 * agent_index,
                    "length": 4.5 + agent_index,
                    "width": 1.8 + 0.1 * agent_index,
                    "height": 1.6,
                }
            )
    pd.DataFrame(rows).to_feather(scene_dir / "agent_data_dt0.10.feather")

    vehicle_types = ["HV", "HV", "AV"] if multi else ["HV"] * len(agent_ids)
    key_agents = list(key_agents or agent_ids[:2])
    record = {
        "dataset": "waymo_train",
        "folder": "waymo_train",
        "scenario_idx": 16,
        "track_id": ";".join(agent_ids),
        "start": interaction_start,
        "end": interaction_end,
        "intensity": 0.872940138,
        "PET": 0.272076373,
        "two/multi": "multi" if multi else "two",
        "vehicle_type": str(vehicle_types),
        "AV_included": "AV" if multi else "all_HV",
        "key_agents": ";".join(key_agents),
        "path_category": "MP",
        "path_relation": "P-M",
        "turn_label": "S-S",
        "priority_label": agent_ids[1],
        "original_scene_id": "scene_16",
        "original_track_id": ";".join(agent_ids),
    }
    csv_path = root / "results.csv"
    pd.DataFrame([record]).to_csv(csv_path, index=False)
    return scene_dir, csv_path


def test_current_sample_v3(tmp_path):
    scene_dir, csv_path = _write_fixture(tmp_path)
    output = tmp_path / "scene_16_interaction_row0_context3s_v3.json"

    payload = export_scene_v3(
        scene_dir,
        output,
        0.1,
        csv_path,
        context_before=30,
        context_after=30,
    )

    validate_scene_motion_v3(payload)
    assert payload["schema_version"] == "scene_motion_v3"
    assert payload["interaction_id"] == "waymo_train_scene_16_2255_2270_34_53"
    assert payload["temporal"]["export_start_frame"] == 4
    assert payload["temporal"]["export_end_frame"] == 83
    assert payload["temporal"]["num_export_frames"] == 80
    assert payload["temporal"]["interaction_start_time_s"] == 3.4
    assert payload["interaction"]["participant_ids"] == ["2255", "2270"]
    assert payload["interaction"]["key_agent_ids"] == ["2255", "2270"]
    assert payload["interaction"]["source_labels"]["priority_agent_id"] == "2270"
    assert payload["interaction"]["scene_scale"] == "pairwise"
    assert payload["interaction"]["analysis_scale"] == "pairwise"
    assert payload["interaction"]["behavior"]["type"] is None
    assert payload["processing_status"]["map_matching"] == "pending"
    assert payload["pairwise"] is not None
    behavior_config = payload["provenance"]["behavior_ttc_config"]
    assert behavior_config["behavior_rule_version"] == "v2"
    assert behavior_config["min_valid_match_ratio"] == 0.5
    assert behavior_config["min_lateral_delta_m"] == 0.05
    assert behavior_config["evidence_window_frames"] == 4
    assert behavior_config["lane_group_max_hops"] == 5
    assert behavior_config["min_same_lane_frames"] == 3
    assert behavior_config["min_same_lane_ratio"] == 0.5
    assert behavior_config["same_lane_distance_threshold_m"] == 30.0
    assert behavior_config["cut_in_gap_threshold_m"] == 15.0
    assert behavior_config["deceleration_threshold_mps"] == 1.0
    assert behavior_config["gap_decrease_threshold_m"] == 1.0
    assert behavior_config["ttc_safety_radius_source"] == "state_dimensions"
    state = next(state for state in payload["agents"][0]["states"] if state["frame"] == 34)
    assert state["dimensions"] == {"length_m": 4.5, "width_m": 1.8}
    assert next(
        state for state in payload["agents"][0]["states"] if state["frame"] == 34
    )["relative_time_s"] == 0.0
    assert json.loads(output.read_text(encoding="utf-8"))["schema_version"] == "scene_motion_v3"


def test_key_agent_order_is_preserved_and_deduplicated(tmp_path):
    scene_dir, csv_path = _write_fixture(
        tmp_path,
        key_agents=("2270", "2255", "2270"),
    )
    payload = export_scene_v3(
        scene_dir,
        tmp_path / "ordered_keys.json",
        0.1,
        csv_path,
        context_before=0,
        context_after=0,
    )

    assert payload["interaction"]["key_agent_ids"] == ["2270", "2255"]
    assert payload["pairwise"]["agent_i"] == "2270"
    assert payload["pairwise"]["agent_j"] == "2255"
    assert payload["interaction_id"] == "waymo_train_scene_16_2270_2255_34_53"


def test_multi_agent_interaction_keeps_key_agents_distinct(tmp_path):
    scene_dir, csv_path = _write_fixture(
        tmp_path,
        agent_ids=("101", "102", "103"),
        multi=True,
    )
    payload = export_scene_v3(
        scene_dir,
        tmp_path / "multi.json",
        0.1,
        csv_path,
        context_before=0,
        context_after=0,
    )

    assert payload["interaction"]["participant_ids"] == ["101", "102", "103"]
    assert payload["interaction"]["key_agent_ids"] == ["101", "102"]
    assert payload["interaction"]["scene_scale"] == "multi_agent"
    assert payload["interaction"]["analysis_scale"] == "pairwise"
    assert payload["interaction"]["interaction_scale"] == "multi_agent"
    assert payload["pairwise"] is not None
    assert payload["pairwise"]["agent_i"] == "101"
    assert payload["pairwise"]["agent_j"] == "102"
    assert payload["provenance"]["behavior_ttc_config"]["ttc_safety_radius_source"] == "state_dimensions"


def test_missing_frames_are_explicit_invalid_states(tmp_path):
    scene_dir, csv_path = _write_fixture(
        tmp_path,
        frames=range(10),
        drop_frames={"2255": [3]},
        interaction_start=2,
        interaction_end=5,
    )
    payload = export_scene_v3(
        scene_dir,
        tmp_path / "missing.json",
        0.1,
        csv_path,
        context_before=0,
        context_after=0,
    )
    agent = payload["agents"][0]
    missing = next(state for state in agent["states"] if state["frame"] == 3)
    assert agent["state_count"] == 4
    assert agent["observed_state_count"] == 3
    assert missing["valid"] is False
    assert missing["position"]["x"] is None
    assert missing["acceleration"]["source"] == "insufficient_data"
    assert missing["relative_time_s"] == 0.1


def test_raw_acceleration_is_preserved_and_json_is_finite(tmp_path):
    scene_dir, csv_path = _write_fixture(
        tmp_path,
        frames=range(20),
        interaction_start=2,
        interaction_end=5,
    )
    payload = export_scene_v3(
        scene_dir,
        tmp_path / "acceleration.json",
        0.1,
        csv_path,
        context_before=0,
        context_after=0,
    )
    state = payload["agents"][0]["states"][0]
    assert abs(state["acceleration"]["ax_raw"] - (-28.65)) < 1e-6
    assert abs(state["acceleration"]["ay_raw"] - 16.89) < 1e-6
    assert payload["kinematics_processing"]["smoothing_enabled"] is True
    validate_scene_motion_v3(payload)


def test_acceleration_falls_back_to_velocity_difference(tmp_path):
    scene_dir, csv_path = _write_fixture(
        tmp_path,
        frames=range(8),
        interaction_start=2,
        interaction_end=5,
    )
    feather_path = scene_dir / "agent_data_dt0.10.feather"
    frame_df = pd.read_feather(feather_path)
    frame_df["ax"] = float("nan")
    frame_df["ay"] = float("nan")
    mask = frame_df["agent_id"] == "2255"
    frame_df.loc[mask, "vx"] = frame_df.loc[mask, "scene_ts"].astype(float)
    frame_df.to_feather(feather_path)

    payload = export_scene_v3(
        scene_dir,
        tmp_path / "derived_acceleration.json",
        0.1,
        csv_path,
        context_before=0,
        context_after=0,
    )
    state = next(
        state for state in payload["agents"][0]["states"] if state["frame"] == 3
    )

    assert state["acceleration"]["ax_raw"] is None
    assert abs(state["acceleration"]["ax_effective"] - 10.0) < 1e-6
    assert state["acceleration"]["source"] == "derived_from_velocity"
    assert (
        payload["kinematics_processing"]["acceleration_source"]
        == "source_or_derived_from_velocity"
    )


def test_acceleration_fallback_accepts_missing_acceleration_columns(tmp_path):
    scene_dir, csv_path = _write_fixture(
        tmp_path,
        frames=range(8),
        interaction_start=2,
        interaction_end=5,
    )
    feather_path = scene_dir / "agent_data_dt0.10.feather"
    frame_df = pd.read_feather(feather_path)
    frame_df = frame_df.drop(columns=["ax", "ay"])
    mask = frame_df["agent_id"] == "2255"
    frame_df.loc[mask, "vx"] = frame_df.loc[mask, "scene_ts"].astype(float)
    frame_df.to_feather(feather_path)

    payload = export_scene_v3(
        scene_dir,
        tmp_path / "missing_acceleration_columns.json",
        0.1,
        csv_path,
        context_before=0,
        context_after=0,
    )
    state = next(
        state for state in payload["agents"][0]["states"] if state["frame"] == 3
    )

    assert state["acceleration"]["ax_raw"] is None
    assert abs(state["acceleration"]["ax_effective"] - 10.0) < 1e-6
    assert state["acceleration"]["source"] == "derived_from_velocity"


def test_acceleration_smoothing_skips_only_initial_missing_value():
    values = [None] + [10.0] * 7

    smoothed = smooth_acceleration(list(range(8)), values)

    assert smoothed[0] is None
    assert all(value is not None for value in smoothed[1:])


def test_v2_export_remains_available(tmp_path):
    scene_dir, csv_path = _write_fixture(tmp_path)
    payload = export_scene_v2(
        scene_dir,
        tmp_path / "legacy_v2.json",
        0.1,
        csv_path,
        context_before=0,
        context_after=0,
    )
    assert payload["schema_version"] == "scene_motion_v2"
    assert payload["interaction"]["track_id"] == "2255;2270"


def test_viterbi_uses_topology_and_suppresses_short_lane_jump():
    def lane(lane_id, points, next_lanes=(), adj_lanes_left=()):
        return SimpleNamespace(
            id=lane_id,
            center=SimpleNamespace(points=points),
            next_lanes=set(next_lanes),
            prev_lanes=set(),
            adj_lanes_left=set(adj_lanes_left),
            adj_lanes_right=set(),
        )

    lane_a = lane("A", [[0.0, 0.0], [10.0, 0.0]], next_lanes=("B",))
    lane_b = lane("B", [[10.0, 0.0], [20.0, 0.0]])
    lane_c = lane("C", [[0.0, 2.0], [20.0, 2.0]], adj_lanes_left=("A",))
    vector_map = SimpleNamespace(lanes=[lane_a, lane_b, lane_c])

    observations = []
    for frame in range(12):
        y = 2.0 if frame == 5 else 0.0
        observations.append(
            {
                "frame": frame,
                "valid": True,
                "x": float(frame + 1),
                "y": y,
                "z": 0.0,
                "heading": 0.0,
            }
        )
    observations[8]["x"] = 100.0

    records = match_trajectory(
        vector_map,
        observations,
        MapMatchingConfig(distance_threshold_m=3.0, top_k=5),
    )
    assert [record["lane_id"] for record in records[:4]] == ["A"] * 4
    assert records[5]["lane_id"] == "A"
    assert records[8]["lane_id"] is None
    assert records[8]["lane_match_confidence"] is None
    assert records[8]["match_status"] == "unmatched"


def test_v3_export_populates_map_fields_only_when_map_is_supplied(tmp_path):
    scene_dir, csv_path = _write_fixture(tmp_path)
    lane = SimpleNamespace(
        id="L1",
        center=SimpleNamespace(points=[[95.0, 197.5], [200.0, 250.0]]),
        next_lanes=set(),
        prev_lanes=set(),
        adj_lanes_left=set(),
        adj_lanes_right=set(),
    )
    vector_map = SimpleNamespace(lanes=[lane])
    payload = export_scene_v3(
        scene_dir,
        tmp_path / "with_map.json",
        0.1,
        csv_path,
        context_before=0,
        context_after=0,
        vector_map=vector_map,
        map_matching_config=MapMatchingConfig(heading_threshold_rad=0.6),
    )
    validate_scene_motion_v3(payload)
    assert payload["processing_status"]["map_matching"] == "completed"
    assert all(
        state["map"]["lane_id"] == "L1"
        for state in payload["agents"][0]["states"]
    )
    assert all(state["same_lane"] is True for state in payload["pairwise"]["states"])
