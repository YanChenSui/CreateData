import json

from scripts.archive.semantic.semantic_behavior_graph import (
    build_fact_policy_audit,
    build_semantic_behavior_graph,
    render_grounded_caption,
    render_semantic_text,
    select_grounded_facts,
)
from scripts.archive.llm.extract_llm_inputs import _interaction_type


def test_v6_graph_keeps_only_normalized_sayable_facts():
    graph = build_semantic_behavior_graph({
        "subject_agent_id": 1638,
        "reference_agent_id": 1646,
        "behavior_type": "lane_change",
        "behavior_subtype": "left",
        "interaction_type": "lane_change_merge",
        "before": {
            "lane_relation": "different_lane",
            "relative_position": "behind",
            "subject_lane": "308",
        },
        "during": {
            "target_matches_reference_lane": True,
            "transition_frame": 67,
        },
        "after": {
            "lane_relation": "same_lane",
            "relative_position": "ahead",
            "subject_lane": "309",
        },
        "motion": {
            "subject": {"speed_relation": "accelerating", "speed_during_mps": 9.4},
            "reference": {"speed_relation": "stable", "speed_during_mps": 8.8},
        },
        "evidence": {"pet_s": 0.4, "minimum_ttc_s": 1.2},
    })

    assert graph == {
        "schema_version": "semantic_behavior_graph_v6",
        "agents": {"A1638": "subject", "A1646": "reference"},
        "interaction": {
            "behavior_type": "lane_change",
            "behavior_subtype": "left",
            "interaction_type": "lane_change_merge",
            "initiator": "A1638",
            "reference": "A1646",
            "direction": "left",
            "relation_before": {"lane_relation": "different_lanes"},
            "relation_during": {"target_matches_reference_lane": True},
            "relation_after": {
                "lane_relation": "same_lane",
                "initiator_position": "ahead",
            },
            "initiator_motion": {"speed_trend": "accelerating"},
            "reference_motion": {"speed_trend": "stable"},
        },
    }
    serialized = json.dumps(graph)
    assert all(token not in serialized for token in ("308", "309", "67", "pet_s", "minimum_ttc_s", "9.4", "8.8"))


def test_extractor_interaction_type_is_not_reclassified_by_graph():
    graph = build_semantic_behavior_graph({
        "subject_agent_id": 396,
        "reference_agent_id": 260,
        "behavior_type": "lane_change",
        "interaction_type": "lane_change_parallel",
        "before": {"lane_relation": "same_lane", "relative_position": "behind"},
        "after": {"lane_relation": "same_lane", "relative_position": "ahead"},
    })
    interaction = graph["interaction"]
    assert interaction["interaction_type"] == "lane_change_parallel"
    assert interaction["order_flip"] is True


def test_position_facts_require_same_lane_and_ahead_or_behind():
    for lane_relation in ("different_lane", "adjacent_lane"):
        graph = build_semantic_behavior_graph({
            "subject_agent_id": 396,
            "reference_agent_id": 260,
            "behavior_type": "lane_change",
            "interaction_type": "lane_change_parallel",
            "after": {
                "lane_relation": lane_relation,
                "relative_position": "ahead",
            },
        })
        assert "initiator_position" not in graph["interaction"]["relation_after"]

    alongside = build_semantic_behavior_graph({
        "subject_agent_id": 396,
        "reference_agent_id": 260,
        "behavior_type": "follow_stop",
        "interaction_type": "follow_stable",
        "after": {
            "lane_relation": "same_lane",
            "relative_position": "alongside",
        },
    })
    assert "initiator_position" not in alongside["interaction"]["relation_after"]


def test_overtake_caption_requires_order_flip_fact():
    with_flip = build_semantic_behavior_graph({
        "subject_agent_id": 396,
        "reference_agent_id": 260,
        "behavior_type": "lane_change",
        "behavior_subtype": "left",
        "interaction_type": "lane_change_overtake",
        "evidence": {"order_flip": True},
    })
    assert render_grounded_caption(with_flip) == (
        "A396 overtakes A260 by changing to the left lane."
    )

    without_flip = build_semantic_behavior_graph({
        "subject_agent_id": 396,
        "reference_agent_id": 260,
        "behavior_type": "lane_change",
        "behavior_subtype": "left",
        "interaction_type": "lane_change_overtake",
    })
    assert "overtakes" not in render_grounded_caption(without_flip)


def test_grounded_caption_is_composed_from_selected_facts():
    merge = build_semantic_behavior_graph({
        "subject_agent_id": 1409,
        "reference_agent_id": 1443,
        "behavior_type": "merge",
        "interaction_type": "lane_change_merge",
        "during": {"target_matches_reference_lane": True},
    })
    assert merge["interaction"]["behavior_type"] == "merge"
    assert merge["interaction"]["interaction_type"] == "lane_change_merge"
    assert render_grounded_caption(merge) == (
        "A1409 merges into the lane occupied by A1443."
    )

    parallel = build_semantic_behavior_graph({
        "subject_agent_id": 396,
        "reference_agent_id": 260,
        "behavior_type": "lane_change",
        "behavior_subtype": "left",
        "interaction_type": "lane_change_parallel",
        "after": {"lane_relation": "adjacent_lane"},
    })
    assert parallel["interaction"]["behavior_type"] == "lane_change"
    assert parallel["interaction"]["interaction_type"] == "lane_change_parallel"
    assert render_grounded_caption(parallel) == (
        "A396 changes to the left lane while A260 remains in the adjacent lane."
    )
    assert render_semantic_text(parallel) == render_grounded_caption(parallel)


def test_agent_only_lane_change_remains_supported():
    graph = build_semantic_behavior_graph({
        "subject_agent_id": "396",
        "behavior_type": "lane_change",
        "behavior_subtype": "right",
        "interaction_type": "lane_change",
    })
    assert graph["agents"] == {"A396": "subject"}
    assert graph["interaction"]["reference"] is None
    assert render_grounded_caption(graph) == "A396 changes to the right lane."


def test_selected_facts_are_ordered_and_auditable():
    graph = build_semantic_behavior_graph({
        "subject_agent_id": 1638,
        "reference_agent_id": 1646,
        "behavior_type": "merge",
        "interaction_type": "lane_change_merge",
        "during": {"target_matches_reference_lane": True},
        "after": {
            "lane_relation": "same_lane",
            "relative_position": "ahead",
        },
        "motion": {
            "reference": {"speed_relation": "stable"},
        },
    })

    assert select_grounded_facts(graph) == [
        {
            "key": "merge",
            "type": "primary_behavior",
            "value": "merge",
            "priority": 100,
            "qualifiers": {
                "target_lane_relation": "enters_reference_lane",
            },
        },
        {
            "key": "final_position",
            "type": "final_longitudinal_relation",
            "value": "subject_ahead",
            "priority": 70,
        },
        {
            "key": "reference_stable",
            "type": "reference_motion",
            "value": "stable",
            "priority": 35,
        },
    ]


def test_selected_facts_use_behavior_specific_relation_priority_and_fact_budget():
    graph = build_semantic_behavior_graph({
        "subject_agent_id": 396,
        "reference_agent_id": 260,
        "behavior_type": "lane_change",
        "behavior_subtype": "left",
        "interaction_type": "lane_change_overtake",
        "during": {"target_matches_reference_lane": True},
        "before": {"lane_relation": "same_lane", "relative_position": "behind"},
        "after": {"lane_relation": "same_lane", "relative_position": "ahead"},
        "motion": {
            "subject": {"speed_relation": "accelerating"},
            "reference": {"speed_relation": "decelerating"},
        },
    })

    selected = select_grounded_facts(graph)
    assert selected == [
        {
            "key": "lane_change",
            "type": "primary_behavior",
            "value": "lane_change",
            "priority": 100,
            "qualifiers": {"direction": "left"},
        },
        {
            "key": "behind_to_ahead",
            "type": "longitudinal_order_change",
            "value": "behind_to_ahead",
            "priority": 90,
        },
        {
            "key": "reference_decelerating",
            "type": "reference_motion",
            "value": "decelerating",
            "priority": 65,
        },
    ]
    assert len(selected) == 3
    assert render_grounded_caption(graph, selected).startswith(
        "A396 overtakes A260 by changing to the left lane."
    )


def test_selected_facts_block_unselected_overtake_claim():
    graph = build_semantic_behavior_graph({
        "subject_agent_id": 396,
        "reference_agent_id": 260,
        "behavior_type": "lane_change",
        "behavior_subtype": "left",
        "interaction_type": "lane_change_overtake",
        "evidence": {"order_flip": True},
    })
    selected_facts = [
        {"type": "primary_behavior", "value": "lane_change"},
        {"type": "direction", "value": "left"},
    ]

    caption = render_grounded_caption(graph, selected_facts)
    assert caption == "A396 changes to the left lane."
    assert "overtake" not in caption


def test_interaction_policy_filters_ineligible_relation_facts():
    graph = build_semantic_behavior_graph({
        "subject_agent_id": 396,
        "reference_agent_id": 260,
        "behavior_type": "lane_change",
        "behavior_subtype": "left",
        "interaction_type": "lane_change_parallel",
        "during": {"target_matches_reference_lane": True},
        "after": {
            "lane_relation": "same_lane",
            "relative_position": "ahead",
        },
        "motion": {"reference": {"speed_relation": "stable"}},
    })

    selected = select_grounded_facts(graph)
    keys = [fact["key"] for fact in selected]
    assert keys == ["lane_change", "reference_stable"]
    assert "enters_reference_lane" not in keys
    assert "final_same_lane" not in keys
    assert "final_position" not in keys


def test_parallel_caption_does_not_generalize_different_lane_to_adjacent():
    adjacent = build_semantic_behavior_graph({
        "subject_agent_id": 396,
        "reference_agent_id": 260,
        "behavior_type": "lane_change",
        "behavior_subtype": "left",
        "interaction_type": "lane_change_parallel",
        "after": {"lane_relation": "adjacent_lane"},
    })
    different = build_semantic_behavior_graph({
        "subject_agent_id": 396,
        "reference_agent_id": 260,
        "behavior_type": "lane_change",
        "behavior_subtype": "left",
        "interaction_type": "lane_change_parallel",
        "after": {"lane_relation": "different_lane"},
    })
    unknown = build_semantic_behavior_graph({
        "subject_agent_id": 396,
        "reference_agent_id": 260,
        "behavior_type": "lane_change",
        "behavior_subtype": "left",
        "interaction_type": "lane_change_parallel",
    })

    assert [fact["key"] for fact in select_grounded_facts(adjacent)] == [
        "lane_change",
        "final_adjacent_lanes",
    ]
    assert "adjacent lane" in render_grounded_caption(adjacent)

    assert [fact["key"] for fact in select_grounded_facts(different)] == [
        "lane_change",
        "final_separate_lanes",
    ]
    assert "separate lane" in render_grounded_caption(different)
    assert "adjacent lane" not in render_grounded_caption(different)

    assert [fact["key"] for fact in select_grounded_facts(unknown)] == [
        "lane_change",
    ]
    assert "adjacent lane" not in render_grounded_caption(unknown)


def test_lane_change_merge_policy_allows_grounded_final_position():
    graph = build_semantic_behavior_graph({
        "subject_agent_id": 7,
        "reference_agent_id": 5,
        "behavior_type": "lane_change",
        "behavior_subtype": "left",
        "interaction_type": "lane_change_merge",
        "during": {"target_matches_reference_lane": True},
        "after": {
            "lane_relation": "same_lane",
            "relative_position": "ahead",
        },
        "motion": {"reference": {"speed_relation": "stable"}},
    })

    selected = select_grounded_facts(graph)
    assert [fact["key"] for fact in selected] == [
        "lane_change",
        "final_position",
        "reference_stable",
    ]
    assert render_grounded_caption(graph, selected) == (
        "A7 moves left into A5's lane and ends ahead of A5.\n"
        "A5 remains in the lane and maintains a relatively steady speed."
    )


def test_clause_composer_adds_grounded_reference_deceleration():
    graph = build_semantic_behavior_graph({
        "subject_agent_id": 7,
        "reference_agent_id": 5,
        "behavior_type": "lane_change",
        "behavior_subtype": "right",
        "interaction_type": "lane_change_merge",
        "during": {"target_matches_reference_lane": True},
        "motion": {"reference": {"speed_relation": "decelerating"}},
    })

    assert render_grounded_caption(graph) == (
        "A7 moves right into A5's lane.\n"
        "A5 decelerates during the maneuver."
    )


def test_required_fact_policy_reports_missing_evidence_without_inference():
    graph = build_semantic_behavior_graph({
        "subject_agent_id": 396,
        "reference_agent_id": 260,
        "behavior_type": "lane_change",
        "interaction_type": "lane_change_overtake",
    })
    selected = select_grounded_facts(graph)
    audit = build_fact_policy_audit(graph, selected)

    assert audit["required"] == ["lane_change", "behind_to_ahead"]
    assert audit["missing_required"] == ["behind_to_ahead"]
    assert audit["requirements_satisfied"] is False
    assert "overtake" not in render_grounded_caption(graph, selected)


def test_stop_behind_lead_requires_following_and_stopping_relation():
    graph = build_semantic_behavior_graph({
        "subject_agent_id": 11,
        "reference_agent_id": 12,
        "behavior_type": "follow_stop",
        "interaction_type": "stop_behind_lead",
        "after": {
            "lane_relation": "same_lane",
            "relative_position": "behind",
        },
    })
    selected = select_grounded_facts(graph)
    audit = build_fact_policy_audit(graph, selected)

    assert [fact["key"] for fact in selected] == [
        "following",
        "stop_behind_reference",
    ]
    assert audit["requirements_satisfied"] is True
    assert render_grounded_caption(graph, selected) == "A11 stops behind A12."


def test_follow_stable_behind_does_not_infer_stopping_fact():
    graph = build_semantic_behavior_graph({
        "subject_agent_id": 0,
        "reference_agent_id": 55,
        "behavior_type": "follow_stop",
        "behavior_subtype": "follow_lead",
        "interaction_type": "follow_stable",
        "after": {
            "lane_relation": "same_lane",
            "relative_position": "behind",
        },
    })
    selected = select_grounded_facts(graph)
    caption = render_grounded_caption(graph, selected)

    assert graph["interaction"]["behavior_subtype"] == "follow_lead"
    assert "stop_behind_reference" not in [fact["key"] for fact in selected]
    assert caption == "A0 follows behind A55 in the same lane."
    assert "stop" not in caption.lower()


def test_extractor_prioritizes_explicit_stop_behind_lead_subtype():
    context = {
        "behavior": {
            "type": "follow_stop",
            "subtype": "stop_behind_lead",
        },
        "distance": {"distance_trend": "decreasing"},
        "motion": {"subject": {"speed_relation": "decelerating"}},
    }

    assert _interaction_type(context) == "stop_behind_lead"


def test_pass_uses_order_change_without_lane_change_or_overtake_wording():
    graph = build_semantic_behavior_graph({
        "subject_agent_id": 1646,
        "reference_agent_id": 1638,
        "behavior_type": "pass",
        "interaction_type": "pass",
        "before": {
            "lane_relation": "same_lane",
            "relative_position": "behind",
        },
        "after": {
            "lane_relation": "same_lane",
            "relative_position": "ahead",
        },
        "evidence": {"order_flip": True},
    })
    selected = select_grounded_facts(graph)
    audit = build_fact_policy_audit(graph, selected)
    caption = render_grounded_caption(graph, selected)

    assert [fact["key"] for fact in selected] == [
        "pass",
        "behind_to_ahead",
    ]
    assert audit["required"] == ["behind_to_ahead"]
    assert audit["requirements_satisfied"] is True
    assert caption == "A1646 passes A1638, moving from behind to ahead."
    assert "lane" not in caption.lower()
    assert "overtake" not in caption.lower()
