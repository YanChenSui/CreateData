#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a deterministic full LLM-facts view from canonical pair facts.

This adapter performs schema projection and implementation-noise removal only.
It does not infer behavior, causality, interaction type, or turn direction.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple


OUTPUT_SCHEMA_VERSION = "full_llm_facts_v1"


def remove_implementation_noise(obj: Any) -> Any:
    """Recursively remove implementation policy/audit metadata only."""
    if isinstance(obj, Mapping):
        cleaned: Dict[str, Any] = {}
        for key, value in obj.items():
            key_text = str(key)
            if key_text == "policy" or key_text.endswith("_audit_only"):
                continue
            cleaned[key_text] = remove_implementation_noise(value)
        return cleaned
    if isinstance(obj, list):
        return [remove_implementation_noise(value) for value in obj]
    if isinstance(obj, tuple):
        return [remove_implementation_noise(value) for value in obj]
    return obj


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} is not an object")
    return value


def _agent_id(
    pair: Mapping[str, Any],
    role: str,
    row: Mapping[str, Any],
) -> int:
    metadata = pair.get(role)
    if isinstance(metadata, Mapping) and metadata.get("id") is not None:
        return int(metadata["id"])
    fallback = row.get(role)
    if fallback is None:
        raise ValueError(f"missing pair.{role}.id")
    return int(fallback)


def build_full_llm_facts(row: Mapping[str, Any]) -> Dict[str, Any]:
    """Project one valid pair_facts row into ``full_llm_facts_v1``."""
    result = _mapping(row.get("result"), "result")
    physical = _mapping(result.get("physical_facts"), "result.physical_facts")
    pair_metadata = _mapping(result.get("pair"), "result.pair")
    timeline = _mapping(result.get("timeline"), "result.timeline")

    scene_id = result.get("scene_id", row.get("scenario_id"))
    if scene_id is None:
        raise ValueError("missing scene_id")
    agent_a_id = _agent_id(pair_metadata, "agent_A", row)
    agent_b_id = _agent_id(pair_metadata, "agent_B", row)
    metadata_a = _mapping(pair_metadata.get("agent_A"), "result.pair.agent_A")
    metadata_b = _mapping(pair_metadata.get("agent_B"), "result.pair.agent_B")

    agent_a_facts = _mapping(physical.get("agent_A"), "result.physical_facts.agent_A")
    agent_b_facts = _mapping(physical.get("agent_B"), "result.physical_facts.agent_B")
    relation = _mapping(physical.get("pair_relation"), "result.physical_facts.pair_relation")
    geometry = _mapping(physical.get("pair_geometry"), "result.physical_facts.pair_geometry")

    def clean(value: Any) -> Any:
        return remove_implementation_noise(value)

    def lateral_motion_evidence(agent_facts: Mapping[str, Any]) -> Any:
        # Canonical pair facts use the evidence-only field.  Older pair facts
        # remain readable during migration.
        value = agent_facts.get("lateral_motion_evidence")
        if value is None:
            value = agent_facts.get("physical_lateral_motion_evidence")
        if value is None:
            value = agent_facts.get("physical_lateral_maneuver")
        return clean(value)

    return {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "scene_id": scene_id,
        "agent_A_id": agent_a_id,
        "agent_B_id": agent_b_id,
        "timeline": {
            "num_frames": timeline.get("num_frames"),
            "current_time_index": timeline.get("current_time_index"),
            "timestamps_seconds": clean(timeline.get("timestamps_seconds", [])),
        },
        "agents": {
            "agent_A": {
                "id": agent_a_id,
                "valid_frame_count": metadata_a.get("valid_frame_count"),
                "first_valid_frame": metadata_a.get("first_valid_frame"),
                "last_valid_frame": metadata_a.get("last_valid_frame"),
                "is_sdc": metadata_a.get("is_sdc"),
                "lateral_motion_evidence": lateral_motion_evidence(agent_a_facts),
                "physical_lane_chain": clean(agent_a_facts.get("physical_lane_chain")),
                "speed_change": clean(agent_a_facts.get("speed_change")),
                "heading_motion": clean(agent_a_facts.get("heading_motion")),
                "route_transition": clean(agent_a_facts.get("route_transition")),
            },
            "agent_B": {
                "id": agent_b_id,
                "valid_frame_count": metadata_b.get("valid_frame_count"),
                "first_valid_frame": metadata_b.get("first_valid_frame"),
                "last_valid_frame": metadata_b.get("last_valid_frame"),
                "is_sdc": metadata_b.get("is_sdc"),
                "lateral_motion_evidence": lateral_motion_evidence(agent_b_facts),
                "physical_lane_chain": clean(agent_b_facts.get("physical_lane_chain")),
                "speed_change": clean(agent_b_facts.get("speed_change")),
                "heading_motion": clean(agent_b_facts.get("heading_motion")),
                "route_transition": clean(agent_b_facts.get("route_transition")),
            },
        },
        "pair": {
            "common_valid_frame_count": relation.get("common_valid_frame_count"),
            "travel_channel": clean(relation.get("travel_channel")),
            "longitudinal": clean(relation.get("longitudinal")),
            "per_frame": clean(relation.get("per_frame", [])),
            "geometry": {
                "heading_relation": clean(geometry.get("heading_relation")),
                "closest_approach": clean(geometry.get("closest_approach")),
                "distance_evolution": clean(geometry.get("distance_evolution")),
                "path_geometry": clean(geometry.get("path_geometry")),
            },
        },
        "map_context": clean(physical.get("map_context")),
        "source": {
            "pair_facts_schema": result.get("schema_version"),
            "scene_id": scene_id,
            "agent_A_id": agent_a_id,
            "agent_B_id": agent_b_id,
        },
    }


def _safe_component(value: Any) -> str:
    text = str(value)
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._") or "unknown"


def pair_json_path(output_dir: Path, facts: Mapping[str, Any]) -> Path:
    scene = _safe_component(facts["scene_id"])
    name = "{}__A_{}__B_{}_full_llm_facts.json".format(
        scene,
        _safe_component(facts["agent_A_id"]),
        _safe_component(facts["agent_B_id"]),
    )
    return output_dir / name


def _write_json(handle: Any, value: Mapping[str, Any]) -> None:
    handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def build_from_jsonl(input_jsonl: Path, output_dir: Path) -> Dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = output_dir / "full_llm_facts.jsonl"
    errors_jsonl = output_dir / "full_llm_facts_errors.jsonl"
    counts = {"input_rows": 0, "written_rows": 0, "skipped_rows": 0}

    with input_jsonl.open("r", encoding="utf-8") as source:
        with output_jsonl.open("w", encoding="utf-8") as output:
            with errors_jsonl.open("w", encoding="utf-8") as errors:
                for line_number, line in enumerate(source, start=1):
                    if not line.strip():
                        continue
                    counts["input_rows"] += 1
                    try:
                        row = json.loads(line)
                        if not isinstance(row, Mapping):
                            raise ValueError("input row is not an object")
                        validation = row.get("validation")
                        if not isinstance(validation, Mapping) or validation.get("valid") is not True:
                            counts["skipped_rows"] += 1
                            _write_json(errors, {
                                "line_number": line_number,
                                "scene_id": row.get("scenario_id"),
                                "agent_A_id": row.get("agent_A"),
                                "agent_B_id": row.get("agent_B"),
                                "error_type": "invalid_validation",
                                "error": "validation.valid is not true",
                                "validation": validation,
                            })
                            continue

                        facts = build_full_llm_facts(row)
                        _write_json(output, facts)
                        per_pair_path = pair_json_path(output_dir, facts)
                        per_pair_path.parent.mkdir(parents=True, exist_ok=True)
                        per_pair_path.write_text(
                            json.dumps(facts, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                            encoding="utf-8",
                        )
                        counts["written_rows"] += 1
                    except Exception as exc:
                        counts["skipped_rows"] += 1
                        _write_json(errors, {
                            "line_number": line_number,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        })
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", "--input-jsonl", dest="input_jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    counts = build_from_jsonl(args.input_jsonl, args.output_dir)
    print(json.dumps({
        **counts,
        "output_jsonl": str(args.output_dir / "full_llm_facts.jsonl"),
        "errors_jsonl": str(args.output_dir / "full_llm_facts_errors.jsonl"),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
