import json

from qwen_generate_interaction_llm_descriptions import (
    EXPECTED_INPUT_SCHEMA_VERSION,
    OUTPUT_SCHEMA_VERSION,
    build_output,
    build_messages,
    build_prompt,
    prompt_input,
    validate_caption_result,
)

PROMPT_TEMPLATE = "Describe only physical facts.\nFULL PHYSICAL FACTS:\n{{FULL_LLM_FACTS_JSON}}"


def _facts():
    return {
        "schema_version": EXPECTED_INPUT_SCHEMA_VERSION,
        "scene_id": "scene-1",
        "agent_A_id": 741,
        "agent_B_id": 3341,
        "timeline": {"num_frames": 91, "current_time_index": 10},
        "agents": {"agent_A": {}, "agent_B": {}},
        "pair": {},
        "map_context": {},
    }


def test_prompt_input_preserves_full_facts_without_second_compression():
    data = _facts()
    data["pair"]["per_frame"] = [{"frame": 0, "distance_m": 4.0}]
    assert prompt_input(data) == data
    prompt = build_prompt(data, PROMPT_TEMPLATE)
    assert "FULL PHYSICAL FACTS:" in prompt
    assert '"distance_m": 4.0' in prompt


def test_prompt_template_requires_external_placeholder(tmp_path):
    from qwen_generate_interaction_llm_descriptions import load_prompt
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text(PROMPT_TEMPLATE, encoding="utf-8")
    assert load_prompt(prompt_file) == PROMPT_TEMPLATE


def test_qwen_messages_include_required_user_query():
    messages = build_messages("external prompt")
    assert messages[0] == {"role": "system", "content": "external prompt"}
    assert messages[1]["role"] == "user"
    assert messages[1]["content"]


def test_prompt_is_copied_to_output_dir(tmp_path):
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text(PROMPT_TEMPLATE, encoding="utf-8")
    output_dir = tmp_path / "qwen_descriptions"
    output_dir.mkdir()
    copied = output_dir / prompt_file.name
    copied.write_text(prompt_file.read_text(encoding="utf-8"), encoding="utf-8")
    assert copied.read_text(encoding="utf-8") == PROMPT_TEMPLATE


def test_prompt_input_rejects_old_schema():
    try:
        prompt_input({"schema_version": "scene_motion_llm_input_v8"})
    except ValueError as exc:
        assert EXPECTED_INPUT_SCHEMA_VERSION in str(exc)
    else:
        raise AssertionError("old schema should be rejected")


def test_caption_validation_checks_only_four_fields_and_frame_bounds():
    result = validate_caption_result({
        "description_short": "Vehicles approach.",
        "description_detailed": "The distance decreases over the observed frames.",
        "supporting_frame_ranges": [{
            "start_frame": 7,
            "end_frame": 31,
            "description": "Distance decreases.",
        }],
        "uncertainty_notes": "No causal relation is established.",
    }, 91)
    assert result["supporting_frame_ranges"][0]["start_frame"] == 7
    assert result["uncertainty_notes"]


def test_caption_validation_rejects_out_of_range_frame():
    try:
        validate_caption_result({
            "description_short": "x",
            "description_detailed": "y",
            "supporting_frame_ranges": [{"start_frame": 0, "end_frame": 91, "description": "x"}],
            "uncertainty_notes": "",
        }, 91)
    except ValueError as exc:
        assert "outside timeline" in str(exc)
    else:
        raise AssertionError("out-of-range frame should be rejected")


def test_output_schema_has_only_new_pair_fields():
    output = build_output(
        _facts(),
        {
            "description_short": "x",
            "description_detailed": "y",
            "supporting_frame_ranges": [],
            "uncertainty_notes": "uncertain",
        },
        "built_in_full_physical_facts_prompt",
        "abc",
    )
    assert output["schema_version"] == OUTPUT_SCHEMA_VERSION
    assert output["scene_id"] == "scene-1"
    assert output["agent_A_id"] == 741
    assert output["agent_B_id"] == 3341
    assert set(output) == {
        "schema_version", "scene_id", "agent_A_id", "agent_B_id",
        "description_short", "description_detailed",
        "supporting_frame_ranges", "uncertainty_notes", "generation",
    }
