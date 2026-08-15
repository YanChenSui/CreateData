#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summarize v3 event types and compare them with InterHub source labels.

The comparison is descriptive only.  ``source_behavior=unknown`` is not
treated as a negative label, and the InterHub window is retained as an audit
field rather than used to select an event.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


CSV_FIELDS = [
    "scene",
    "pair",
    "source_behavior",
    "interhub_input_status",
    "v3_status",
    "v3_event_family",
    "v3_primary_kind",
    "event_score",
    "event_frame",
    "event_scope",
    "within_interhub_window",
    "distance_to_interhub_window",
    "lane_event_type",
    "lane_event_pair_level",
    "longitudinal_event_type",
    "pair_relation_event_type",
    "speed_event_types",
    "transition_agent",
    "reference_agent",
    "entered_reference_lane",
    "reference_maintained_lane",
    "speed_response",
    "behind_to_ahead",
    "num_lane_events",
    "num_pair_level_lane_events",
    "num_individual_lane_events",
    "num_longitudinal_events",
    "num_pair_relation_events",
    "num_speed_response_events",
    "classified_interaction",
]


def get_selected(result, channel):
    return (result.get("event_channels", {}).get(channel, {}) or {}).get(
        "selected_event"
    )


def scope_of(event):
    if not event:
        return None
    return event.get("event_scope_match") or {}


def first_event(*events):
    for event in events:
        if event:
            return event
    return None


def event_family(result):
    analysis = result.get("analysis", {}) or {}
    status = analysis.get("status")
    primary = analysis.get("primary_event") or {}
    primary_type = primary.get("event_type")
    if primary_type:
        return primary_type
    if status == "no_supported_event":
        return "no_supported_event"

    lane = get_selected(result, "lane_event")
    longitudinal = get_selected(result, "longitudinal_event")
    relation = get_selected(result, "pair_relation_event")
    speed_events = (result.get("event_channels", {}).get("speed_event", {}) or {}).get(
        "events", []
    )
    fallback = first_event(lane, longitudinal, relation)
    if fallback and fallback.get("event_type"):
        return fallback["event_type"]
    if speed_events:
        return "speed_response_only"
    return "other_supported_event"


def role_facts(result):
    facts = result.get("role_centric_facts", {}) or {}
    return {
        "transition_agent": facts.get("transition_agent_id"),
        "reference_agent": facts.get("reference_agent_id"),
        "entered_reference_lane": facts.get("transition_agent_entered_reference_lane"),
        "reference_maintained_lane": facts.get("reference_agent_maintained_lane"),
        "speed_response": facts.get("reference_speed_reduced_during_event"),
        "behind_to_ahead": facts.get("transition_agent_behind_to_ahead"),
    }


def to_text(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def build_row(item):
    inp = item.get("input", {}) or {}
    result = item.get("result", {}) or {}
    analysis = result.get("analysis", {}) or {}
    primary = analysis.get("primary_event") or {}
    lane = get_selected(result, "lane_event")
    longitudinal = get_selected(result, "longitudinal_event")
    relation = get_selected(result, "pair_relation_event")
    speed_channel = result.get("event_channels", {}).get("speed_event", {}) or {}
    speed_events = speed_channel.get("events", []) or []
    anchor = first_event(primary, lane, longitudinal, relation, speed_events[0] if speed_events else None)
    scope = scope_of(anchor)
    summary = analysis.get("event_channel_summary", {}) or {}
    roles = role_facts(result)

    row = {
        "scene": inp.get("scene_id"),
        "pair": "%s-%s" % (inp.get("agent_A"), inp.get("agent_B")),
        "source_behavior": inp.get("source_behavior", "unknown"),
        "interhub_input_status": (inp.get("analysis_window") or {}).get("status"),
        "v3_status": analysis.get("status"),
        "v3_event_family": event_family(result),
        "v3_primary_kind": primary.get("kind"),
        "event_score": primary.get("score"),
        "event_frame": primary.get("frame"),
        "event_scope": scope.get("category") if scope else None,
        "within_interhub_window": scope.get("within_window") if scope else None,
        "distance_to_interhub_window": scope.get("distance_frames") if scope else None,
        "lane_event_type": lane.get("event_type") if lane else None,
        "lane_event_pair_level": lane.get("pair_level") if lane else None,
        "longitudinal_event_type": longitudinal.get("event_type") if longitudinal else None,
        "pair_relation_event_type": relation.get("event_type") if relation else None,
        "speed_event_types": [x.get("event_type") for x in speed_events],
        "transition_agent": roles["transition_agent"],
        "reference_agent": roles["reference_agent"],
        "entered_reference_lane": roles["entered_reference_lane"],
        "reference_maintained_lane": roles["reference_maintained_lane"],
        "speed_response": roles["speed_response"],
        "behind_to_ahead": roles["behind_to_ahead"],
        "num_lane_events": summary.get("num_lane_events", 0),
        "num_pair_level_lane_events": summary.get("num_pair_level_lane_events", 0),
        "num_individual_lane_events": summary.get("num_individual_lane_events", 0),
        "num_longitudinal_events": summary.get("num_longitudinal_events", 0),
        "num_pair_relation_events": summary.get("num_pair_relation_events", 0),
        "num_speed_response_events": summary.get("num_speed_response_events", 0),
        "classified_interaction": inp.get("classified_interaction"),
    }
    return row


def counter_dict(values):
    return dict(Counter(values))


def nested_counter(rows, key1, key2):
    table = defaultdict(Counter)
    for row in rows:
        table[str(row.get(key1))][str(row.get(key2))] += 1
    return {key: dict(value) for key, value in sorted(table.items())}


def summarize(rows, source):
    event_rows = [row for row in rows if row["v3_status"] != "no_supported_event"]
    pair_level_rows = [
        row
        for row in rows
        if row["v3_status"] == "pair_event_detected"
    ]
    role_available = [row for row in rows if row["transition_agent"] not in (None, "")]
    scope_counts = Counter(
        row["event_scope"] if row["event_scope"] else "no_selected_event"
        for row in rows
    )
    within_counts = Counter(
        str(row["within_interhub_window"])
        if row["within_interhub_window"] is not None
        else "no_selected_event"
        for row in rows
    )

    role_fields = [
        "entered_reference_lane",
        "reference_maintained_lane",
        "speed_response",
        "behind_to_ahead",
    ]
    role_counts = {
        field: counter_dict(row[field] for row in rows)
        for field in role_fields
    }

    channel_selected = {
        "lane_event": sum(bool(row["lane_event_type"]) for row in rows),
        "longitudinal_event": sum(bool(row["longitudinal_event_type"]) for row in rows),
        "pair_relation_event": sum(bool(row["pair_relation_event_type"]) for row in rows),
        "speed_response_event": sum(bool(row["speed_event_types"]) for row in rows),
    }

    by_source = {}
    for label in sorted(set(row["source_behavior"] for row in rows)):
        subset = [row for row in rows if row["source_behavior"] == label]
        by_source[label] = {
            "count": len(subset),
            "v3_status": counter_dict(row["v3_status"] for row in subset),
            "v3_event_family": counter_dict(row["v3_event_family"] for row in subset),
            "event_rows": sum(row["v3_status"] != "no_supported_event" for row in subset),
            "pair_event_rows": sum(row["v3_status"] == "pair_event_detected" for row in subset),
            "role_fact_available": sum(row["transition_agent"] not in (None, "") for row in subset),
            "within_interhub_window": sum(row["within_interhub_window"] is True for row in subset),
        }

    return {
        "schema_version": "v3_interhub_event_type_analysis",
        "input_file": source,
        "num_rows": len(rows),
        "v3_status_counts": counter_dict(row["v3_status"] for row in rows),
        "v3_event_family_counts": counter_dict(row["v3_event_family"] for row in rows),
        "v3_primary_kind_counts": counter_dict(row["v3_primary_kind"] for row in rows),
        "selected_channel_counts": channel_selected,
        "lane_event_type_counts": counter_dict(row["lane_event_type"] or "none" for row in rows),
        "longitudinal_event_type_counts": counter_dict(row["longitudinal_event_type"] or "none" for row in rows),
        "pair_relation_event_type_counts": counter_dict(row["pair_relation_event_type"] or "none" for row in rows),
        "speed_event_type_counts": counter_dict(
            event_type
            for row in rows
            for event_type in (row["speed_event_types"] or ["none"])
        ),
        "event_scope_counts": dict(scope_counts),
        "within_interhub_window_counts": dict(within_counts),
        "role_fact_counts": role_counts,
        "role_fact_available": len(role_available),
        "source_behavior_counts": counter_dict(row["source_behavior"] for row in rows),
        "source_behavior_x_v3_status": nested_counter(rows, "source_behavior", "v3_status"),
        "source_behavior_x_v3_event_family": nested_counter(rows, "source_behavior", "v3_event_family"),
        "source_behavior_summary": by_source,
        "v3_status_x_event_scope": nested_counter(rows, "v3_status", "event_scope"),
        "interpretation_notes": [
            "source_behavior=unknown is retained as unknown and is not treated as a negative label.",
            "The comparison is descriptive overlap analysis, not precision/recall or label accuracy.",
            "InterHub windows are reported as audit metadata; v3 event selection used the full trajectory.",
            "individual_maneuver_only means a lane maneuver was accepted but pair-level relation evidence was not accepted.",
            "no_supported_event does not prove that the full scene contains no interaction.",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/2_extracted_results/remote_pair_timeline_v3_batch_500.json",
    )
    parser.add_argument(
        "--csv-output",
        default="data/2_extracted_results/v3_interhub_event_type_analysis_500.csv",
    )
    parser.add_argument(
        "--json-output",
        default="data/2_extracted_results/v3_interhub_event_type_analysis_500.json",
    )
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    rows = [build_row(item) for item in payload.get("results", [])]
    summary = summarize(rows, args.input)

    csv_path = Path(args.csv_output)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: to_text(row.get(field)) for field in CSV_FIELDS})

    json_path = Path(args.json_output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(
            {"summary": summary, "rows": rows},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("CSV:", csv_path)
    print("JSON:", json_path)


if __name__ == "__main__":
    main()
