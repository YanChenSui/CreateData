"""Classify an interaction from factual pair-timeline evidence.

This module intentionally does not read InterHub records or raw trajectories.
It consumes only the factual output of ``pair_behavior_timeline``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping


def classify_interaction(timeline: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a semantic interaction label without changing the facts."""
    facts = timeline.get("pair_facts", {})
    if not isinstance(facts, Mapping):
        facts = {}
    checks = {
        "A_changed_lane": facts.get("A_changed_lane") is True,
        "B_maintained_lane": facts.get("B_maintained_lane") is True,
        "A_entered_B_lane": facts.get("A_entered_B_lane") is True,
        "A_behind_to_ahead": facts.get("A_behind_to_ahead") is True,
        "B_speed_reduced_during_event": facts.get("B_speed_reduced_during_event") is True,
    }

    if checks["A_entered_B_lane"] and checks["A_behind_to_ahead"]:
        interaction_type = "overtake"
        modifiers = ["with_lane_change"]
        if checks["B_speed_reduced_during_event"]:
            modifiers.append("with_yielding")
        status = "classified"
    elif checks["A_entered_B_lane"] and checks["B_speed_reduced_during_event"]:
        interaction_type = "lane_change_with_yielding"
        modifiers = []
        status = "classified"
    elif checks["A_entered_B_lane"]:
        interaction_type = "merge"
        modifiers = []
        status = "classified"
    elif checks["A_changed_lane"]:
        interaction_type = "lane_change"
        modifiers = []
        status = "classified"
    else:
        interaction_type = "unknown"
        modifiers = []
        status = "insufficient_evidence"

    return {
        "schema_version": "vehicle_pair_interaction_classification_v1",
        "scene_id": timeline.get("scene_id"),
        "agent_A": timeline.get("candidate", {}).get("agent_A"),
        "agent_B": timeline.get("candidate", {}).get("agent_B"),
        "interaction_type": interaction_type,
        "modifiers": modifiers,
        "status": status,
        "facts_used": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    timeline = json.loads(args.input.read_text(encoding="utf-8"))
    result = classify_interaction(timeline)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
