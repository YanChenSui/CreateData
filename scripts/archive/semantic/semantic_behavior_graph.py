"""Convert grounded behavior records into a small semantic behavior graph.

The graph is deliberately deterministic and auditable.  It is an intermediate
representation between behavior extraction and optional LLM paraphrasing; it
does not infer a maneuver from prose or change the original InterHub labels.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence


FACT_PRIORITY = {
    "primary_behavior": 100,
    "enters_reference_lane": 90,
    "behind_to_ahead": 90,
    "stop_behind_reference": 90,
    "final_same_lane": 70,
    "final_adjacent_lanes": 70,
    "final_separate_lanes": 70,
    "final_position": 70,
    "reference_decelerating": 65,
    "reference_accelerating": 55,
    "reference_stable": 35,
    "subject_speed_trend": 30,
    "distance_change": 20,
    "pet_ttc": 0,
    "raw_lane_id": 0,
}

MAX_SELECTED_FACTS = 3


INTERACTION_FACT_POLICY = {
    "lane_change_merge": {
        "allowed": (
            "lane_change",
            "merge",
            "direction",
            "enters_reference_lane",
            "final_same_lane",
            "final_position",
            "reference_motion",
            "subject_speed_trend",
        ),
        "required": (),
    },
    "lane_change_parallel": {
        "allowed": (
            "lane_change",
            "direction",
            "final_adjacent_lanes",
            "final_separate_lanes",
            "reference_motion",
        ),
        "required": (),
    },
    "lane_change_overtake": {
        "allowed": (
            "lane_change",
            "direction",
            "behind_to_ahead",
            "final_same_lane",
            "reference_motion",
        ),
        "required": ("lane_change", "behind_to_ahead"),
    },
    "overtake": {
        "allowed": (
            "lane_change",
            "direction",
            "behind_to_ahead",
            "final_same_lane",
            "reference_motion",
        ),
        "required": ("lane_change", "behind_to_ahead"),
    },
    "pass": {
        "allowed": (
            "behind_to_ahead",
            "final_same_lane",
            "reference_motion",
        ),
        "required": ("behind_to_ahead",),
    },
    "stop_behind_lead": {
        "allowed": (
            "following",
            "stop_behind_reference",
            "final_same_lane",
            "reference_motion",
        ),
        "required": ("following", "stop_behind_reference"),
    },
    "follow_approaching": {
        "allowed": (
            "following",
            "final_same_lane",
            "stop_behind_reference",
            "reference_motion",
            "subject_speed_trend",
            "distance_change",
        ),
        "required": ("following",),
    },
    "follow_decelerating": {
        "allowed": (
            "following",
            "final_same_lane",
            "stop_behind_reference",
            "reference_motion",
            "subject_speed_trend",
        ),
        "required": ("following",),
    },
    "follow_stable": {
        "allowed": (
            "following",
            "final_same_lane",
            "stop_behind_reference",
            "reference_motion",
            "subject_speed_trend",
        ),
        "required": ("following",),
    },
    "lane_change": {
        "allowed": ("lane_change", "direction", "reference_motion"),
        "required": ("lane_change",),
    },
}

DEFAULT_FACT_POLICY = {
    "allowed": (),
    "required": (),
}


def _agent_label(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text.startswith("A") else f"A{text}"


def _safe_direction(record: Mapping[str, Any]) -> str | None:
    evidence = record.get("evidence", {})
    if not isinstance(evidence, Mapping):
        evidence = {}
    direction = record.get("behavior_subtype") or evidence.get("direction")
    return str(direction) if direction in {"left", "right"} else None


def _safe_lane_relation(value: Any) -> str | None:
    if isinstance(value, Mapping):
        value = value.get("lane_relation")
    return {
        "same_lane": "same_lane",
        "different_lane": "different_lanes",
        "different_lanes": "different_lanes",
        "adjacent": "adjacent_lanes",
        "adjacent_lane": "adjacent_lanes",
        "adjacent_lanes": "adjacent_lanes",
    }.get(value)


def _safe_after_position(after: Any) -> str | None:
    if not isinstance(after, Mapping):
        return None
    if _safe_lane_relation(after) != "same_lane":
        return None
    position = after.get("relative_position")
    return str(position) if position in {"ahead", "behind"} else None


def _safe_before_position(before: Any) -> str | None:
    # The same conservative same-lane requirement applies before a maneuver.
    return _safe_after_position(before)


def _safe_speed_trend(value: Any) -> str | None:
    if not isinstance(value, Mapping):
        return None
    speed_trend = value.get("speed_relation") or value.get("speed_trend")
    return str(speed_trend) if speed_trend in {
        "accelerating",
        "decelerating",
        "stable",
    } else None


def _relation_snapshot(
    value: Any,
    position_reader: Callable[[Any], str | None],
) -> dict[str, Any]:
    """Keep only relation facts that are safe to verbalize."""
    if not isinstance(value, Mapping):
        return {}
    lane_relation = _safe_lane_relation(value)
    result: dict[str, Any] = {}
    if lane_relation is not None:
        result["lane_relation"] = lane_relation
    position = position_reader(value)
    if position is not None:
        result["initiator_position"] = position
    return result


def _during_snapshot(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, Any] = {}
    target_matches = value.get("target_matches_reference_lane")
    if isinstance(target_matches, bool):
        result["target_matches_reference_lane"] = target_matches
    return result


def _motion_snapshot(value: Any) -> dict[str, str]:
    speed_trend = _safe_speed_trend(value)
    return {"speed_trend": speed_trend} if speed_trend is not None else {}


def build_semantic_behavior_graph(record: Mapping[str, Any]) -> dict[str, Any]:
    """Build the v6 graph from already normalized extractor-level facts.

    This function does not reclassify the interaction.  The extractor's
    ``interaction_type`` is authoritative; the behavior type is only a
    fallback for direct callers and legacy records.
    """
    subject_value = record.get("subject_agent_id")
    if subject_value is None:
        subject_value = record.get("agent_id")
    subject = _agent_label(subject_value)
    reference = _agent_label(record.get("reference_agent_id"))
    behavior_type = str(record.get("behavior_type") or record.get("type") or "unknown")
    behavior_subtype_value = record.get("behavior_subtype")
    behavior_subtype = (
        str(behavior_subtype_value)
        if behavior_subtype_value not in {None, ""}
        else None
    )
    interaction_type = str(record.get("interaction_type") or behavior_type)
    direction = _safe_direction(record)
    agents: dict[str, str] = {}
    if subject is not None:
        agents[subject] = "subject"
    if reference is not None:
        agents[reference] = "reference"

    interaction_node: dict[str, Any] = {
        "behavior_type": behavior_type,
        "interaction_type": interaction_type,
        "initiator": subject,
        "reference": reference,
    }
    if behavior_subtype is not None:
        interaction_node["behavior_subtype"] = behavior_subtype
    if direction is not None:
        interaction_node["direction"] = direction
    before = _relation_snapshot(record.get("before"), _safe_before_position)
    during = _during_snapshot(record.get("during"))
    after = _relation_snapshot(record.get("after"), _safe_after_position)
    if before:
        interaction_node["relation_before"] = before
    if during:
        interaction_node["relation_during"] = during
    if after:
        interaction_node["relation_after"] = after
    evidence = record.get("evidence", {})
    explicit_order_flip = (
        isinstance(evidence, Mapping) and evidence.get("order_flip") is True
    )
    relation_order_flip = (
        before.get("initiator_position") == "behind"
        and after.get("initiator_position") == "ahead"
    )
    if explicit_order_flip or relation_order_flip:
        interaction_node["order_flip"] = True

    motion = record.get("motion", {})
    if not isinstance(motion, Mapping):
        motion = {}
    initiator_motion = _motion_snapshot(motion.get("subject", {}))
    reference_motion = _motion_snapshot(motion.get("reference", {}))
    if initiator_motion:
        interaction_node["initiator_motion"] = initiator_motion
    if reference_motion:
        interaction_node["reference_motion"] = reference_motion

    return {
        "schema_version": "semantic_behavior_graph_v6",
        "agents": agents,
        "interaction": interaction_node,
    }


def _graph_interaction(graph: Mapping[str, Any]) -> Mapping[str, Any]:
    semantic_behavior = graph.get("semantic_behavior")
    if isinstance(semantic_behavior, Mapping):
        interaction = semantic_behavior.get("interaction", {})
    else:
        interaction = graph.get("interaction", {})
    return interaction if isinstance(interaction, Mapping) else {}


def _primary_fact_key(behavior_type: str, interaction_type: str) -> str:
    if interaction_type == "stop_behind_lead" or interaction_type.startswith("follow_"):
        return "following"
    if interaction_type in {"lane_change_overtake", "overtake"}:
        return "lane_change"
    if behavior_type in {"lane_change", "overtake", "avoid", "evasive_lane_change", "cut_in"}:
        return "lane_change"
    if behavior_type == "follow_stop":
        return "following"
    return behavior_type


def _fact_policy(interaction_type: str, primary_key: str) -> dict[str, tuple[str, ...]]:
    configured = INTERACTION_FACT_POLICY.get(interaction_type, DEFAULT_FACT_POLICY)
    allowed = tuple(dict.fromkeys((primary_key, *configured["allowed"])))
    return {
        "allowed": allowed,
        "required": tuple(configured["required"]),
    }


def _fact_is_allowed(fact: Mapping[str, Any], allowed: set[str]) -> bool:
    key = fact.get("key")
    fact_type = fact.get("type")
    return isinstance(key, str) and (
        key in allowed
        or fact_type in allowed
        or (fact_type == "reference_motion" and "reference_motion" in allowed)
    )


def select_grounded_facts(graph: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Select eligible facts under the interaction policy and fact budget."""
    interaction = _graph_interaction(graph)
    behavior_type = str(interaction.get("behavior_type") or "unknown")
    interaction_type = str(
        interaction.get("interaction_type")
        or interaction.get("type")
        or behavior_type
    )
    primary_key = _primary_fact_key(behavior_type, interaction_type)
    policy = _fact_policy(interaction_type, primary_key)
    allowed = set(policy["allowed"])

    primary_fact: dict[str, Any] = {
        "key": primary_key,
        "type": "primary_behavior",
        "value": primary_key,
        "priority": FACT_PRIORITY["primary_behavior"],
    }
    direction = interaction.get("direction")
    if direction in {"left", "right"} and "direction" in allowed:
        primary_fact["qualifiers"] = {"direction": direction}

    relation_candidates: list[dict[str, Any]] = []

    def relation_fact(key: str, fact_type: str, value: str) -> None:
        priority_key = key if key in FACT_PRIORITY else fact_type
        fact = {
            "key": key,
            "type": fact_type,
            "value": value,
            "priority": FACT_PRIORITY.get(priority_key, 0),
        }
        if _fact_is_allowed(fact, allowed):
            relation_candidates.append(fact)

    during = interaction.get("relation_during", {})
    if (
        isinstance(during, Mapping)
        and during.get("target_matches_reference_lane") is True
        and "enters_reference_lane" in allowed
    ):
        primary_fact.setdefault("qualifiers", {})[
            "target_lane_relation"
        ] = "enters_reference_lane"

    after = interaction.get("relation_after", {})
    if isinstance(after, Mapping):
        lane_relation = after.get("lane_relation")
        final_position = after.get("initiator_position")
        if final_position in {"ahead", "behind"}:
            relation_fact(
                "final_position",
                "final_longitudinal_relation",
                f"subject_{final_position}",
            )
        if lane_relation == "same_lane":
            relation_fact("final_same_lane", "final_lane_relation", "same_lane")
        elif lane_relation == "adjacent_lanes":
            relation_fact(
                "final_adjacent_lanes",
                "final_lane_relation",
                "adjacent_lanes",
            )
        elif lane_relation == "different_lanes":
            relation_fact(
                "final_separate_lanes",
                "final_lane_relation",
                "separate_lanes",
            )
        if (
            interaction_type == "stop_behind_lead"
            and lane_relation == "same_lane"
            and final_position == "behind"
        ):
            relation_fact(
                "stop_behind_reference",
                "stopping_relation",
                "stop_behind_reference",
            )

    if interaction.get("order_flip") is True:
        relation_fact(
            "behind_to_ahead",
            "longitudinal_order_change",
            "behind_to_ahead",
        )

    motion_candidates: list[dict[str, Any]] = []
    initiator_motion = interaction.get("initiator_motion", {})
    if isinstance(initiator_motion, Mapping):
        speed_trend = initiator_motion.get("speed_trend")
        fact = {
            "key": "subject_speed_trend",
            "type": "subject_motion",
            "value": speed_trend,
            "priority": FACT_PRIORITY["subject_speed_trend"],
        }
        if (
            speed_trend in {"accelerating", "decelerating", "stable"}
            and _fact_is_allowed(fact, allowed)
        ):
            motion_candidates.append(fact)
    reference_motion = interaction.get("reference_motion", {})
    if isinstance(reference_motion, Mapping):
        speed_trend = reference_motion.get("speed_trend")
        priority_key = f"reference_{speed_trend}"
        fact = {
            "key": priority_key,
            "type": "reference_motion",
            "value": speed_trend,
            "priority": FACT_PRIORITY.get(priority_key, 0),
        }
        if priority_key in FACT_PRIORITY and _fact_is_allowed(fact, allowed):
            motion_candidates.append(fact)

    selected: list[dict[str, Any]] = [primary_fact]
    selected_keys = {primary_key}
    candidate_by_key = {
        fact["key"]: fact
        for fact in (*relation_candidates, *motion_candidates)
    }

    # Required facts take precedence over optional facts and are never inferred.
    for required_key in policy["required"]:
        if required_key in selected_keys or len(selected) >= MAX_SELECTED_FACTS:
            continue
        required_fact = candidate_by_key.get(required_key)
        if required_fact is not None:
            selected.append(required_fact)
            selected_keys.add(required_key)

    relation_types = {
        "target_lane_relation",
        "final_lane_relation",
        "final_longitudinal_relation",
        "longitudinal_order_change",
        "stopping_relation",
    }
    if (
        len(selected) < MAX_SELECTED_FACTS
        and not any(fact["type"] in relation_types for fact in selected)
        and relation_candidates
    ):
        best_relation = max(relation_candidates, key=lambda fact: fact["priority"])
        selected.append(best_relation)
        selected_keys.add(best_relation["key"])

    if (
        len(selected) < MAX_SELECTED_FACTS
        and not any(fact["type"] in {"subject_motion", "reference_motion"} for fact in selected)
        and motion_candidates
    ):
        best_motion = max(motion_candidates, key=lambda fact: fact["priority"])
        selected.append(best_motion)
    return selected[:MAX_SELECTED_FACTS]


def build_fact_policy_audit(
    graph: Mapping[str, Any],
    selected_facts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Describe which fact policy was applied and whether requirements passed."""
    interaction = _graph_interaction(graph)
    behavior_type = str(interaction.get("behavior_type") or "unknown")
    interaction_type = str(
        interaction.get("interaction_type")
        or interaction.get("type")
        or behavior_type
    )
    primary_key = _primary_fact_key(behavior_type, interaction_type)
    policy = _fact_policy(interaction_type, primary_key)
    selected_keys = {
        fact.get("key")
        for fact in selected_facts
        if isinstance(fact.get("key"), str)
    }
    for fact in selected_facts:
        qualifiers = fact.get("qualifiers", {})
        if not isinstance(qualifiers, Mapping):
            continue
        if qualifiers.get("direction") in {"left", "right"}:
            selected_keys.add("direction")
        if qualifiers.get("target_lane_relation") == "enters_reference_lane":
            selected_keys.add("enters_reference_lane")
    missing_required = [
        key for key in policy["required"] if key not in selected_keys
    ]
    return {
        "interaction_type": interaction_type,
        "allowed": list(policy["allowed"]),
        "required": list(policy["required"]),
        "missing_required": missing_required,
        "requirements_satisfied": not missing_required,
    }


def _selected_fact_map(
    selected_facts: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    result: dict[str, str] = {}
    for fact in selected_facts:
        fact_key = fact.get("key")
        fact_type = fact.get("type")
        value = fact.get("value")
        if isinstance(fact_type, str) and isinstance(value, str):
            result[fact_type] = value
        if isinstance(fact_key, str) and isinstance(value, str):
            result[fact_key] = value
        if fact_type == "primary_behavior":
            qualifiers = fact.get("qualifiers", {})
            if isinstance(qualifiers, Mapping):
                direction = qualifiers.get("direction")
                if direction in {"left", "right"}:
                    result["direction"] = direction
                target_lane_relation = qualifiers.get("target_lane_relation")
                if target_lane_relation == "enters_reference_lane":
                    result["enters_reference_lane"] = target_lane_relation
    return result


def render_grounded_caption(
    graph: Mapping[str, Any],
    selected_facts: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    """Compose deterministic clauses from authorized selected facts.

    When ``selected_facts`` is omitted, they are derived deterministically
    from the graph.  No interaction-type sentence template is consulted; for
    example, ``overtake`` wording requires the selected combination of a lane
    change and a longitudinal order-change fact.
    """
    interaction_node = _graph_interaction(graph)
    subject = interaction_node.get("initiator") or "The subject vehicle"
    reference = interaction_node.get("reference") or interaction_node.get("affected")
    if selected_facts is None:
        selected_facts = select_grounded_facts(graph)
    facts = _selected_fact_map(selected_facts)
    behavior = facts.get("primary_behavior", "unknown")
    direction = facts.get("direction")
    enters_reference_lane = "enters_reference_lane" in facts
    behind_to_ahead = "behind_to_ahead" in facts
    final_same_lane = "final_same_lane" in facts
    final_adjacent_lanes = "final_adjacent_lanes" in facts
    final_separate_lanes = "final_separate_lanes" in facts
    final_position = facts.get("final_position")
    stop_behind_reference = "stop_behind_reference" in facts
    reference_motion = facts.get("reference_motion")
    subject_motion = facts.get("subject_motion")

    clauses: list[str] = []
    if behavior == "lane_change":
        if reference and behind_to_ahead:
            lane_change_phrase = (
                f"changing to the {direction} lane" if direction else "changing lanes"
            )
            clauses.append(f"{subject} overtakes {reference} by {lane_change_phrase}")
        elif reference and enters_reference_lane:
            movement = f"moves {direction}" if direction else "moves laterally"
            clauses.append(f"{subject} {movement} into {reference}'s lane")
        else:
            destination = f"to the {direction} lane" if direction else "lanes"
            clauses.append(f"{subject} changes {destination}")
    elif behavior == "merge":
        direction_phrase = f" {direction}" if direction else ""
        if reference and enters_reference_lane:
            clauses.append(
                f"{subject} merges{direction_phrase} into the lane occupied by {reference}"
            )
        else:
            clauses.append(f"{subject} merges{direction_phrase}")
    elif behavior == "following":
        if reference and stop_behind_reference:
            clauses.append(f"{subject} stops behind {reference}")
        elif reference and final_same_lane:
            clauses.append(f"{subject} follows behind {reference} in the same lane")
        else:
            clauses.append(f"{subject} follows {reference or 'the lead vehicle'}")
    elif behavior == "pass":
        if reference and behind_to_ahead:
            clauses.append(
                f"{subject} passes {reference}, moving from behind to ahead"
            )
        elif reference:
            clauses.append(f"{subject} passes {reference}")
        else:
            clauses.append(f"{subject} passes another vehicle")
    else:
        clauses.append(f"{subject} performs {str(behavior).replace('_', ' ')}")

    if reference and not behind_to_ahead:
        if final_position == "subject_ahead":
            clauses[-1] += f" and ends ahead of {reference}"
        elif final_position == "subject_behind":
            clauses[-1] += f" and ends behind {reference}"
        elif (
            final_same_lane
            and not enters_reference_lane
            and behavior != "following"
        ):
            clauses[-1] += f" and ends in the same lane as {reference}"
        elif final_adjacent_lanes:
            clauses[-1] += f" while {reference} remains in the adjacent lane"
        elif final_separate_lanes:
            clauses[-1] += f" while {reference} remains in a separate lane"

    first_sentence = clauses[0] + "."

    motion_sentence = None
    if reference and reference_motion == "decelerating":
        motion_sentence = f"{reference} decelerates during the maneuver."
    elif reference and reference_motion == "accelerating":
        motion_sentence = f"{reference} accelerates during the maneuver."
    elif reference and reference_motion == "stable":
        if enters_reference_lane or final_same_lane or final_position is not None:
            motion_sentence = (
                f"{reference} remains in the lane and maintains a relatively "
                "steady speed."
            )
        else:
            motion_sentence = f"{reference} maintains a relatively steady speed."
    elif subject_motion == "decelerating":
        motion_sentence = f"{subject} decelerates during the maneuver."
    elif subject_motion == "accelerating":
        motion_sentence = f"{subject} accelerates during the maneuver."
    elif subject_motion == "stable":
        motion_sentence = f"{subject} maintains a relatively steady speed."
    return (
        f"{first_sentence}\n{motion_sentence}"
        if motion_sentence
        else first_sentence
    )


def render_semantic_text(graph: Mapping[str, Any]) -> str:
    """Backward-compatible alias for :func:`render_grounded_caption`."""
    return render_grounded_caption(graph)
