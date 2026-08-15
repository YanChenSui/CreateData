import json

from scripts.llm.build_full_llm_facts import (
    build_from_jsonl,
    build_full_llm_facts,
    pair_json_path,
    remove_implementation_noise,
)


def _agent(agent_id):
    return {
        "id": agent_id,
        "physical_lateral_maneuver": {"status": "uncertain", "policy": {"x": 1}},
        "speed_change": {"status": "available"},
        "heading_motion": {"status": "uncertain", "turning_episode": {"status": "insufficient_data"}},
        "route_transition": {
            "route_transition": False,
            "transitions": [],
            "primary_transition": None,
            "policy": {"x": 1},
        },
    }


def _row(valid=True):
    return {
        "scenario_id": "scene-1",
        "agent_A": 741,
        "agent_B": 3341,
        "pair_ordinal": 1,
        "source_file": "training.tfrecords",
        "record_index": 3,
        "validation": {"valid": valid, "errors": []},
        "result": {
            "schema_version": "pair_physical_facts_v1",
            "implementation_revision": "facts_only_canonical_pair_geometry_route_context",
            "scene_id": "scene-1",
            "timeline": {"num_frames": 91, "current_time_index": 10, "timestamps_seconds": [0.0, 0.1]},
            "pair": {
                "agent_A": {"id": 741, "valid_frame_count": 90, "first_valid_frame": 0, "last_valid_frame": 90, "is_sdc": False, "track_index": 1, "track_to_predict": True, "is_object_of_interest": True},
                "agent_B": {"id": 3341, "valid_frame_count": 91, "first_valid_frame": 0, "last_valid_frame": 90, "is_sdc": True, "track_index": 2, "track_to_predict": True, "is_object_of_interest": True},
            },
            "physical_facts": {
                "agent_A": _agent(741),
                "agent_B": _agent(3341),
                "pair_relation": {
                    "common_valid_frame_count": 1,
                    "travel_channel": {"dominant": "uncertain", "policy": {"x": 1}},
                    "longitudinal": {"dominant": "uncertain"},
                    "per_frame": [{"frame": 0, "distance_m": 4.0}],
                },
                "pair_geometry": {
                    "heading_relation": {"status": "uncertain", "dominant": "uncertain"},
                    "closest_approach": {"status": "available", "frame": 0, "distance_m": 4.0},
                    "distance_evolution": {"status": "uncertain", "overall": "uncertain"},
                    "path_geometry": {"status": "available", "spatial_overlap": "uncertain"},
                },
                "map_context": {
                    "near_intersection": True,
                    "intersection_evidence": {"nearby_crosswalk": True},
                    "policy": {"x": 1},
                },
            },
            "map_matching_audit_only": {"internal": True},
        },
    }


def test_full_llm_facts_projects_ids_and_removes_implementation_noise():
    result = build_full_llm_facts(_row())
    assert result["schema_version"] == "full_llm_facts_v1"
    assert result["scene_id"] == "scene-1"
    assert result["agent_A_id"] == 741
    assert result["agent_B_id"] == 3341
    assert result["timeline"]["num_frames"] == 91
    assert result["timeline"]["current_time_index"] == 10
    assert result["pair"]["common_valid_frame_count"] == 1
    assert result["agents"]["agent_A"]["valid_frame_count"] == 90
    assert result["agents"]["agent_A"]["first_valid_frame"] == 0
    assert result["agents"]["agent_A"]["last_valid_frame"] == 90
    assert result["agents"]["agent_A"]["is_sdc"] is False
    assert "track_index" not in result["agents"]["agent_A"]
    assert "track_to_predict" not in result["agents"]["agent_A"]
    assert "is_object_of_interest" not in result["agents"]["agent_A"]
    assert result["pair"]["per_frame"] == [{"frame": 0, "distance_m": 4.0}]
    assert result["map_context"]["near_intersection"] is True
    dumped = json.dumps(result, ensure_ascii=False)
    assert "policy" not in dumped
    assert "map_matching_audit_only" not in dumped
    assert "implementation_revision" not in dumped
    assert "source_file" not in dumped
    assert "record_index" not in dumped
    assert "pair_ordinal" not in dumped


def test_uncertain_is_preserved():
    result = build_full_llm_facts(_row())
    assert result["agents"]["agent_A"]["heading_motion"]["status"] == "uncertain"
    assert result["pair"]["geometry"]["distance_evolution"]["overall"] == "uncertain"


def test_invalid_validation_is_skipped_to_errors_jsonl(tmp_path):
    input_path = tmp_path / "pair_facts.jsonl"
    input_path.write_text(
        json.dumps(_row()) + "\n" + json.dumps(_row(valid=False)) + "\n",
        encoding="utf-8",
    )
    counts = build_from_jsonl(input_path, tmp_path / "out")
    assert counts == {"input_rows": 2, "written_rows": 1, "skipped_rows": 1}
    output_rows = (tmp_path / "out" / "full_llm_facts.jsonl").read_text(encoding="utf-8").splitlines()
    error_rows = (tmp_path / "out" / "full_llm_facts_errors.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(output_rows) == 1
    assert len(error_rows) == 1
    assert json.loads(error_rows[0])["error_type"] == "invalid_validation"


def test_noise_cleaner_removes_nested_policies_and_audits_only():
    cleaned = remove_implementation_noise({
        "status": "uncertain",
        "policy": {"internal": True},
        "map_matching_audit_only": {"internal": True},
        "nested": [{"value": 1, "policy": {"internal": True}}],
    })
    assert cleaned == {"status": "uncertain", "nested": [{"value": 1}]}


def test_pair_json_is_written_directly_under_output_dir(tmp_path):
    path = pair_json_path(tmp_path, build_full_llm_facts(_row()))
    assert path.parent == tmp_path
    assert path.name == "scene-1__A_741__B_3341_full_llm_facts.json"
