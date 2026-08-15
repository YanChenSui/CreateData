import json

from scripts.llm.build_full_vehicle_facts import (
    build_from_jsonl,
    build_full_vehicle_facts,
    remove_implementation_noise,
    vehicle_json_path,
)


def _row():
    return {
        "schema_version": "vehicle_physical_facts_v1",
        "vehicle_record_id": "scene-1::vehicle_1148",
        "scene_id": "scene-1",
        "vehicle_id": 1148,
        "status": "facts_extracted",
        "source_file": "training.tfrecord",
        "record_index": 3,
        "implementation_revision": "internal",
        "metadata": {
            "is_sdc": False,
            "is_object_of_interest": False,
            "track_to_predict": False,
            "valid_frame_count": 91,
            "first_valid_frame": 0,
            "last_valid_frame": 90,
        },
        "timeline": {
            "num_frames": 91,
            "current_time_index": 10,
            "timestamps_seconds": [0.0, 0.1],
            "full_scenario_used": True,
        },
        "track_motion_summary": {
            "valid_frame_count": 91,
            "max_speed_mps": 0.0,
            "p95_speed_mps": 0.0,
            "endpoint_displacement_m": 0.0,
            "spatial_extent_m": 0.0,
            "path_length_m": 0.0,
        },
        "trajectory_quality": {
            "status": "suspicious",
            "issues": [{
                "type": "heading_jump",
                "frame": 69,
                "delta_deg": 41.1,
            }],
            "policy": {"repairs_applied": False},
        },
        "physical_facts": {
            "physical_lateral_maneuver": {
                "status": "none_detected",
                "policy": {"internal": True},
            },
            "speed_change": {"status": "available"},
            "heading_motion": {"status": "insufficient_data"},
            "turn_maneuver": {"status": "turn_uncertain"},
            "route_transition": {
                "status": "no_route_transition",
                "map_matching_audit_only": {"internal": True},
            },
            "map_context": {
                "near_intersection": True,
                "nearby_crosswalk": False,
                "policy": {"internal": True},
            },
        },
    }


def test_projects_vehicle_schema_and_removes_noise():
    result = build_full_vehicle_facts(_row())
    assert result["schema_version"] == "full_vehicle_facts_v1"
    assert result["vehicle_record_id"] == "scene-1::vehicle_1148"
    assert result["scene_id"] == "scene-1"
    assert result["vehicle_id"] == 1148
    assert result["timeline"]["num_frames"] == 91
    assert result["track_motion_summary"]["max_speed_mps"] == 0.0
    assert result["trajectory_quality"] == {
        "status": "suspicious",
        "issues": [{
            "type": "heading_jump",
            "frame": 69,
            "delta_deg": 41.1,
        }],
    }
    assert result["vehicle"]["id"] == 1148
    assert result["vehicle"]["is_sdc"] is False
    assert result["vehicle"]["is_object_of_interest"] is False
    assert result["vehicle"]["track_to_predict"] is False
    assert result["vehicle"]["valid_frame_count"] == 91
    assert result["vehicle"]["first_valid_frame"] == 0
    assert result["vehicle"]["last_valid_frame"] == 90
    assert result["vehicle"]["speed_change"] == {"status": "available"}
    assert result["vehicle"]["heading_motion"] == {"status": "insufficient_data"}
    assert result["vehicle"]["turn_maneuver"] == {"status": "turn_uncertain"}
    assert "physical_lateral_maneuver" not in result["vehicle"]
    assert "lateral_motion_evidence" not in result["vehicle"]
    assert result["map_context"]["near_intersection"] is True
    assert set(result) == {
        "schema_version",
        "vehicle_record_id",
        "scene_id",
        "vehicle_id",
        "timeline",
        "trajectory_quality",
        "track_motion_summary",
        "vehicle",
        "map_context",
    }
    dumped = json.dumps(result, ensure_ascii=False)
    for noise in ("policy", "map_matching_audit_only", "implementation_revision", "source_file", "record_index"):
        assert noise not in dumped


def test_noise_cleaner_is_recursive():
    cleaned = remove_implementation_noise({
        "value": 1,
        "policy": {"internal": True},
        "nested": [{"record_index": 2, "value": 3}],
    })
    assert cleaned == {"value": 1, "nested": [{"value": 3}]}


def test_vehicle_json_is_written_under_vehicle_subdir(tmp_path):
    path = vehicle_json_path(tmp_path, build_full_vehicle_facts(_row()))
    assert path.parent == tmp_path / "vehicles"
    assert path.name == "scene-1__vehicle_1148_full_vehicle_facts.json"


def test_jsonl_conversion(tmp_path):
    input_path = tmp_path / "vehicle_facts.jsonl"
    input_path.write_text(json.dumps(_row()) + "\n", encoding="utf-8")
    counts = build_from_jsonl(input_path, tmp_path / "out")
    assert counts == {"input_rows": 1, "written_rows": 1, "skipped_rows": 0}
    output_rows = (tmp_path / "out" / "full_vehicle_facts.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(output_rows) == 1
    assert json.loads(output_rows[0])["schema_version"] == "full_vehicle_facts_v1"
    assert (tmp_path / "out" / "vehicles" / "scene-1__vehicle_1148_full_vehicle_facts.json").is_file()
