#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate WOMD ``objects_of_interest`` vehicle pairs and validate pair facts.

This is deliberately the first OOI-only stage.  It does not import or read
InterHub records, and it does not call any semantic classifier or LLM code.
For every scenario it reports the OOI count, the vehicle-only OOI count, and
the number of unordered vehicle pairs.  When pair-fact extraction is enabled,
each pair is passed to ``build_pair_timeline_v3`` and the complete
``physical_facts`` block is written to JSONL.

The OOI field is a candidate-group definition, not a behavior label.  A pair
with no supported physical evidence remains a valid pair-facts result; it is
not silently discarded at this stage.
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    from . import build_pair_timeline as facts_builder
except ImportError:  # pragma: no cover - direct script execution
    import build_pair_timeline as facts_builder


VEHICLE_OBJECT_TYPE = 1
FACTS_SCHEMA_VERSION = "pair_physical_facts_v1"


def _as_int(value: Any) -> int:
    return int(value)


def _matches_int(value: Any, expected: int) -> bool:
    try:
        return _as_int(value) == int(expected)
    except (TypeError, ValueError):
        return False


def unique_ooi_ids(scenario: Any) -> list[int]:
    """Return stable, de-duplicated OOI IDs exactly as scenario Track IDs."""
    seen: set[int] = set()
    result: list[int] = []
    for value in getattr(scenario, "objects_of_interest", []):
        agent_id = _as_int(value)
        if agent_id not in seen:
            seen.add(agent_id)
            result.append(agent_id)
    return result


def tracks_by_id(scenario: Any) -> dict[int, Any]:
    """Index tracks by Track.id and reject duplicate IDs."""
    result: dict[int, Any] = {}
    for track in getattr(scenario, "tracks", []):
        agent_id = _as_int(track.id)
        if agent_id in result:
            raise ValueError(f"duplicate Track.id={agent_id}")
        result[agent_id] = track
    return result


def vehicle_ooi_ids(scenario: Any) -> tuple[list[int], list[int]]:
    """Return ``(vehicle_ids, missing_ids)`` for one WOMD scenario."""
    index = tracks_by_id(scenario)
    vehicle_ids: list[int] = []
    missing_ids: list[int] = []
    for agent_id in unique_ooi_ids(scenario):
        track = index.get(agent_id)
        if track is None:
            missing_ids.append(agent_id)
        elif _as_int(track.object_type) == VEHICLE_OBJECT_TYPE:
            vehicle_ids.append(agent_id)
    return vehicle_ids, missing_ids


def candidate_pairs(agent_ids: Sequence[int]) -> list[tuple[int, int]]:
    """Return deterministic unordered pairs from vehicle OOI IDs."""
    return [tuple(pair) for pair in itertools.combinations(sorted(set(agent_ids)), 2)]


def _contains_path(payload: Mapping[str, Any], path: Sequence[str]) -> bool:
    current: Any = payload
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return False
        current = current[key]
    return True


def validate_physical_facts(
    result: Mapping[str, Any],
    agent_a: int,
    agent_b: int,
) -> dict[str, Any]:
    """Validate structural completeness without judging behavior semantics."""
    errors: list[str] = []
    if result.get("schema_version") != FACTS_SCHEMA_VERSION:
        errors.append(f"schema_version={result.get('schema_version')!r}")

    pair = result.get("pair")
    physical = result.get("physical_facts")
    if not isinstance(pair, Mapping):
        errors.append("missing pair")
        pair = {}
    if not isinstance(physical, Mapping):
        errors.append("missing physical_facts")
        physical = {}

    for role, expected_id in (("agent_A", agent_a), ("agent_B", agent_b)):
        metadata = pair.get(role)
        if not isinstance(metadata, Mapping):
            errors.append(f"missing pair.{role}")
        else:
            if not _matches_int(metadata.get("id"), expected_id):
                errors.append(f"pair.{role}.id mismatch")
            if metadata.get("is_object_of_interest") is not True:
                errors.append(f"pair.{role}.is_object_of_interest is not true")
            if not _matches_int(metadata.get("object_type"), VEHICLE_OBJECT_TYPE):
                errors.append(f"pair.{role}.object_type is not vehicle")

        has_lateral_evidence = _contains_path(
            physical, (role, "lateral_motion_evidence")
        ) or _contains_path(
            physical, (role, "physical_lateral_motion_evidence")
        ) or _contains_path(physical, (role, "physical_lateral_maneuver"))
        if not has_lateral_evidence:
            errors.append(
                f"missing physical_facts.{role}.lateral_motion_evidence"
            )
        if not _contains_path(physical, (role, "speed_change")):
            errors.append(f"missing physical_facts.{role}.speed_change")
        role_facts = physical.get(role)
        heading_motion = role_facts.get("heading_motion") if isinstance(role_facts, Mapping) else None
        if not isinstance(heading_motion, Mapping):
            errors.append(f"missing physical_facts.{role}.heading_motion")
        else:
            for key in (
                "status",
                "heading_start_deg",
                "heading_end_deg",
                "net_heading_change_deg",
                "heading_change_direction",
                "turning_episode",
                "start_frame",
                "end_frame",
            ):
                if key not in heading_motion:
                    errors.append(f"missing physical_facts.{role}.heading_motion.{key}")
        route_transition = role_facts.get("route_transition") if isinstance(role_facts, Mapping) else None
        if not isinstance(route_transition, Mapping):
            errors.append(f"missing physical_facts.{role}.route_transition")
        else:
            for key in (
                "route_transition",
                "transitions",
                "primary_transition",
                "lane_before",
                "lane_after",
                "incoming_lane_heading_deg",
                "outgoing_lane_heading_deg",
                "lane_heading_change_deg",
                "connected_in_lane_graph",
            ):
                if key not in route_transition:
                    errors.append(f"missing physical_facts.{role}.route_transition.{key}")
            if not isinstance(route_transition.get("transitions"), list):
                errors.append(f"physical_facts.{role}.route_transition.transitions is not a list")
            primary_transition = route_transition.get("primary_transition")
            if primary_transition is not None and not isinstance(primary_transition, Mapping):
                errors.append(f"physical_facts.{role}.route_transition.primary_transition is not an object or null")
            for index, transition in enumerate(route_transition.get("transitions", [])):
                if not isinstance(transition, Mapping):
                    errors.append(f"physical_facts.{role}.route_transition.transitions[{index}] is not an object")
                    continue
                for key in (
                    "transition_frame",
                    "lane_before",
                    "lane_after",
                    "incoming_lane_heading_deg",
                    "outgoing_lane_heading_deg",
                    "lane_heading_change_deg",
                    "connected_in_lane_graph",
                ):
                    if key not in transition:
                        errors.append(
                            f"missing physical_facts.{role}.route_transition.transitions[{index}].{key}"
                        )

    relation = physical.get("pair_relation")
    if not isinstance(relation, Mapping):
        errors.append("missing physical_facts.pair_relation")
        relation = {}
    for key in ("common_valid_frame_count", "travel_channel", "longitudinal", "per_frame"):
        if key not in relation:
            errors.append(f"missing physical_facts.pair_relation.{key}")
    if not isinstance(relation.get("per_frame"), list):
        errors.append("physical_facts.pair_relation.per_frame is not a list")
    elif any(
        not isinstance(frame, Mapping)
        or any(key not in frame for key in ("frame", "distance_m", "travel_channel_relation", "longitudinal_relation"))
        for frame in relation["per_frame"]
    ):
        errors.append("incomplete pair_relation.per_frame entry")

    geometry = physical.get("pair_geometry")
    if not isinstance(geometry, Mapping):
        errors.append("missing physical_facts.pair_geometry")
        geometry = {}
    for key in ("heading_relation", "closest_approach", "distance_evolution", "path_geometry"):
        if key not in geometry:
            errors.append(f"missing physical_facts.pair_geometry.{key}")
        elif not isinstance(geometry.get(key), Mapping):
            errors.append(f"missing physical_facts.pair_geometry.{key}")

    heading = geometry.get("heading_relation", {})
    if isinstance(heading, Mapping):
        for key in ("status", "dominant", "frame_counts", "runs"):
            if key not in heading:
                errors.append(f"missing pair_geometry.heading_relation.{key}")
        if not isinstance(heading.get("runs"), list):
            errors.append("pair_geometry.heading_relation.runs is not a list")
        if not isinstance(heading.get("frame_counts"), Mapping):
            errors.append("pair_geometry.heading_relation.frame_counts is not an object")
        summary = heading.get("heading_difference_summary_deg")
        if not isinstance(summary, Mapping) or any(
            key not in summary for key in ("median", "min", "max")
        ):
            errors.append("incomplete pair_geometry.heading_relation.heading_difference_summary_deg")

    closest = geometry.get("closest_approach", {})
    if isinstance(closest, Mapping):
        if "status" not in closest:
            errors.append("missing pair_geometry.closest_approach.status")
        elif closest.get("status") == "available":
            for key in ("frame", "distance_m"):
                if key not in closest:
                    errors.append(f"missing pair_geometry.closest_approach.{key}")

    evolution = geometry.get("distance_evolution", {})
    if isinstance(evolution, Mapping):
        for key in ("status", "overall"):
            if key not in evolution:
                errors.append(f"missing pair_geometry.distance_evolution.{key}")

    path_geometry = geometry.get("path_geometry", {})
    if isinstance(path_geometry, Mapping):
        for key in (
            "status",
            "min_path_distance_m",
            "closest_path_points",
            "spatial_overlap",
            "convergence_status",
        ):
            if key not in path_geometry:
                errors.append(f"missing pair_geometry.path_geometry.{key}")

    map_context = physical.get("map_context")
    if not isinstance(map_context, Mapping):
        errors.append("missing physical_facts.map_context")
    else:
        for key in (
            "near_intersection",
            "intersection_evidence",
            "nearby_crosswalk",
            "nearby_stop_sign",
            "nearby_traffic_signal",
        ):
            if key not in map_context:
                errors.append(f"missing physical_facts.map_context.{key}")
        intersection_evidence = map_context.get("intersection_evidence")
        if not isinstance(intersection_evidence, Mapping):
            errors.append("physical_facts.map_context.intersection_evidence is not an object")
        else:
            for key in (
                "lane_graph_branching",
                "nearby_crosswalk",
                "nearby_traffic_signal",
                "nearby_stop_sign",
            ):
                if key not in intersection_evidence:
                    errors.append(f"missing physical_facts.map_context.intersection_evidence.{key}")

    try:
        json.dumps(result, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        errors.append(f"not_json_safe: {exc}")

    return {
        "valid": not errors,
        "error_count": len(errors),
        "errors": errors,
        "common_valid_frame_count": relation.get("common_valid_frame_count"),
    }


def _json_write(handle: Any, payload: Any) -> None:
    handle.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")


def _scenario_summary(scenario: Any, source_file: str, record_index: int) -> dict[str, Any]:
    ooi_ids = unique_ooi_ids(scenario)
    index = tracks_by_id(scenario)
    vehicle_ids, missing_ids = vehicle_ooi_ids(scenario)
    nonvehicle_ids = [
        agent_id
        for agent_id in ooi_ids
        if agent_id in index and _as_int(index[agent_id].object_type) != VEHICLE_OBJECT_TYPE
    ]
    pairs = candidate_pairs(vehicle_ids)
    return {
        "scenario_id": str(getattr(scenario, "scenario_id", "")),
        "source_file": str(source_file),
        "record_index": int(record_index),
        "ooi_ids": ooi_ids,
        "ooi_count": len(ooi_ids),
        "vehicle_ooi_ids": vehicle_ids,
        "vehicle_ooi_count": len(vehicle_ids),
        "nonvehicle_ooi_ids": nonvehicle_ids,
        "nonvehicle_ooi_count": len(nonvehicle_ids),
        "missing_ooi_ids": missing_ids,
        "missing_ooi_count": len(missing_ids),
        "candidate_pair_count": len(pairs),
        "candidate_pairs": [list(pair) for pair in pairs],
    }


def _iter_scenarios(
    tfrecord_spec: str,
    compression_type: str,
    limit: int,
) -> Iterable[tuple[Any, str, int]]:
    paths = facts_builder.resolve_tfrecord_paths(tfrecord_spec)
    facts_builder._require_runtime_deps()
    yielded = 0
    for path in paths:
        options = None
        if compression_type:
            options = facts_builder.tf.io.TFRecordOptions(compression_type=compression_type)
        records = facts_builder.tf.compat.v1.io.tf_record_iterator(path, options=options)
        for record_index, raw_record in enumerate(records):
            if limit > 0 and yielded >= limit:
                return
            scenario = facts_builder._parse_scenario(bytes(raw_record))
            yield scenario, path, record_index
            yielded += 1


def _distribution(values: Iterable[int]) -> dict[str, int]:
    counts = collections.Counter(int(value) for value in values)
    return {str(key): int(counts[key]) for key in sorted(counts)}


def run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scenario_path = output_dir / "scenario_summary.jsonl"
    pair_path = output_dir / "pair_candidates.jsonl"
    facts_path = output_dir / "pair_facts.jsonl"
    errors_path = output_dir / "pair_fact_errors.jsonl"

    scenario_count = 0
    pair_count = 0
    facts_success = 0
    facts_failed = 0
    ooi_counts: list[int] = []
    vehicle_ooi_counts: list[int] = []
    pair_counts: list[int] = []
    object_type_counts: collections.Counter[str] = collections.Counter()
    started = time.time()

    with scenario_path.open("w", encoding="utf-8") as scenario_handle:
        with pair_path.open("w", encoding="utf-8") as pair_handle:
            with facts_path.open("w", encoding="utf-8") as facts_handle:
                with errors_path.open("w", encoding="utf-8") as errors_handle:
                    for scenario, source_file, record_index in _iter_scenarios(
                        args.tfrecord, args.compression_type, args.limit
                    ):
                        summary = _scenario_summary(scenario, source_file, record_index)
                        pairs = [tuple(pair) for pair in summary["candidate_pairs"]]
                        if args.max_pairs_per_scenario > 0:
                            pairs = pairs[: args.max_pairs_per_scenario]
                        summary["pairs_truncated"] = len(pairs) != summary["candidate_pair_count"]
                        summary["processed_pair_count"] = len(pairs)
                        _json_write(scenario_handle, summary)

                        for agent_id in summary["ooi_ids"]:
                            track = tracks_by_id(scenario).get(agent_id)
                            if track is not None:
                                object_type_counts[str(_as_int(track.object_type))] += 1

                        scenario_count += 1
                        ooi_counts.append(summary["ooi_count"])
                        vehicle_ooi_counts.append(summary["vehicle_ooi_count"])
                        pair_counts.append(summary["candidate_pair_count"])

                        for pair_ordinal, (agent_a, agent_b) in enumerate(pairs, start=1):
                            pair_count += 1
                            candidate = {
                                "scenario_id": summary["scenario_id"],
                                "source_file": source_file,
                                "record_index": record_index,
                                "pair_ordinal": pair_ordinal,
                                "agent_A": agent_a,
                                "agent_B": agent_b,
                            }
                            _json_write(pair_handle, candidate)
                            try:
                                result = facts_builder.build_pair_timeline_v3(
                                    scenario=scenario,
                                    agent_a_id=agent_a,
                                    agent_b_id=agent_b,
                                )
                                audit = validate_physical_facts(result, agent_a, agent_b)
                                row = {**candidate, "validation": audit, "result": result}
                                if audit["valid"]:
                                    facts_success += 1
                                else:
                                    facts_failed += 1
                                    _json_write(errors_handle, row)
                                _json_write(facts_handle, row)
                            except Exception as exc:  # retain one failed pair for audit
                                facts_failed += 1
                                _json_write(errors_handle, {
                                    **candidate,
                                    "validation": {"valid": False, "error_count": 1, "errors": [f"{type(exc).__name__}: {exc}"]},
                                })

                        if scenario_count % 10 == 0:
                            print(
                                f"[PROGRESS] scenarios={scenario_count} pairs={pair_count} "
                                f"facts_ok={facts_success} facts_failed={facts_failed} "
                                f"elapsed={time.time() - started:.1f}s",
                                flush=True,
                            )

    report = {
        "schema_version": "womd_ooi_candidate_report_v1",
        "input": {
            "tfrecord": args.tfrecord,
            "compression_type": args.compression_type,
            "limit": args.limit,
            "max_pairs_per_scenario": args.max_pairs_per_scenario,
        },
        "scope": {
            "interhub_used": False,
            "llm_called": False,
            "behavior_classifier_called": False,
            "candidate_rule": "all unordered combinations of vehicle OOI IDs",
            "facts_builder": "scripts.facts.build_pair_timeline.build_pair_timeline_v3",
        },
        "counts": {
            "scenarios": scenario_count,
            "total_ooi": sum(ooi_counts),
            "total_vehicle_ooi": sum(vehicle_ooi_counts),
            "total_candidate_pairs": sum(pair_counts),
            "processed_pairs": pair_count,
            "physical_facts_success": facts_success,
            "physical_facts_failed": facts_failed,
        },
        "distributions": {
            "ooi_count_per_scenario": _distribution(ooi_counts),
            "vehicle_ooi_count_per_scenario": _distribution(vehicle_ooi_counts),
            "candidate_pair_count_per_scenario": _distribution(pair_counts),
            "ooi_object_type_counts": dict(sorted(object_type_counts.items())),
        },
        "outputs": {
            "scenario_summary": str(scenario_path),
            "pair_candidates": str(pair_path),
            "pair_facts": str(facts_path),
            "pair_fact_errors": str(errors_path),
        },
        "elapsed_seconds": round(time.time() - started, 3),
    }
    (output_dir / "coverage_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["counts"], ensure_ascii=False, indent=2))
    return 0 if facts_failed == 0 else 2


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tfrecord", required=True, help="WOMD Scenario TFRecord file, directory, or glob")
    parser.add_argument("--output-dir", required=True, help="New output directory for OOI audit artifacts")
    parser.add_argument("--limit", type=int, default=0, help="Maximum number of scenarios; 0 means all")
    parser.add_argument("--max-pairs-per-scenario", type=int, default=0, help="Testing cap; 0 means all pairs")
    parser.add_argument("--compression-type", default="", help="TFRecord compression type")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
