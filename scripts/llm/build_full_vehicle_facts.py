#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Project vehicle physical facts into the LLM-facing vehicle schema.

This adapter performs projection and implementation-noise removal only.  It
does not infer behavior, interaction, causality, or semantics.  Input records
are ``vehicle_physical_facts_v1`` objects produced by
``scripts/facts/build_vehicle_facts.py``; each output is a
``full_vehicle_facts_v1`` object.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Mapping


INPUT_SCHEMA_VERSION = "vehicle_physical_facts_v1"
OUTPUT_SCHEMA_VERSION = "full_vehicle_facts_v1"
VEHICLE_JSON_SUBDIR = "vehicles"

_NOISE_KEYS = frozenset({
    "implementation_revision",
    "source_file",
    "record_index",
    "record_index_in_shard",
})

_SHIFT_MARKER_RE = re.compile(
    r"(?<![a-z])(?:shift|shifts|shifted|shifting|"
    r"drift|drifts|drifted|drifting)(?![a-z])",
    re.IGNORECASE,
)
_SHIFT_ANNOTATION_FIELDS = frozenset({
    "behavior_segments",
    "supporting_evidence",
    "supporting_frame_ranges",
    "summary",
    "summaries",
})
_SHIFT_TYPE_FIELDS = frozenset({
    "type",
    "behavior",
    "label",
    "action",
    "maneuver",
    "category",
})
_PROTECTED_BEHAVIOR_FIELDS = frozenset({
    "lane_change",
    "lane_change_evidence",
    "physical_lane_chain",
    "route_transition",
    "turn",
    "turning_episode",
    "turning_episodes",
    "speed_change",
    "acceleration_episodes",
    "deceleration_episodes",
})


def remove_implementation_noise(obj: Any) -> Any:
    """Recursively remove policy, audit, and implementation metadata."""
    if isinstance(obj, Mapping):
        cleaned: Dict[str, Any] = {}
        for key, value in obj.items():
            key_text = str(key)
            if (
                key_text == "policy"
                or key_text.endswith("_audit_only")
                or key_text in _NOISE_KEYS
            ):
                continue
            cleaned[key_text] = remove_implementation_noise(value)
        return cleaned
    if isinstance(obj, list):
        return [remove_implementation_noise(value) for value in obj]
    if isinstance(obj, tuple):
        return [remove_implementation_noise(value) for value in obj]
    return obj


def _contains_shift_marker(value: Any) -> bool:
    return isinstance(value, str) and bool(_SHIFT_MARKER_RE.search(value))


def _is_explicit_shift_annotation(value: Any) -> bool:
    """Identify an explicit ordinary shift/drift label, not lateral evidence."""
    if not isinstance(value, Mapping):
        return _contains_shift_marker(value)
    for key in _SHIFT_TYPE_FIELDS:
        if _contains_shift_marker(value.get(key)):
            return True
    return False


def remove_ordinary_shift_annotations(
    obj: Any,
    field_name: str | None = None,
    protected: bool = False,
) -> Any:
    """Remove ordinary shift/drift annotations from the LLM-facing facts.

    This filters explicit ``shift_left``/``shift_right``/``type=shift`` style
    entries from behavior, evidence, and summary containers.  Lane-change,
    turn, acceleration, and deceleration structures are protected and passed
    through unchanged; their lateral or longitudinal evidence is not rewritten.
    """
    if isinstance(obj, Mapping):
        cleaned: Dict[str, Any] = {}
        current_target = field_name in _SHIFT_ANNOTATION_FIELDS
        for key, value in obj.items():
            key_text = str(key)
            key_lower = key_text.lower()
            if not protected and current_target and (
                _contains_shift_marker(key_text)
                or (
                    key_lower in _SHIFT_TYPE_FIELDS
                    and _contains_shift_marker(value)
                )
            ):
                continue
            child_protected = protected or key_lower in _PROTECTED_BEHAVIOR_FIELDS
            cleaned[key_text] = remove_ordinary_shift_annotations(
                value,
                field_name=key_lower,
                protected=child_protected,
            )
        return cleaned
    if isinstance(obj, list):
        cleaned_list = []
        for value in obj:
            if not protected and field_name in _SHIFT_ANNOTATION_FIELDS:
                if _is_explicit_shift_annotation(value):
                    continue
            cleaned_list.append(
                remove_ordinary_shift_annotations(
                    value,
                    field_name=field_name,
                    protected=protected,
                )
            )
        return cleaned_list
    if isinstance(obj, tuple):
        return remove_ordinary_shift_annotations(
            list(obj), field_name=field_name, protected=protected
        )
    return obj


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} is not an object")
    return value


def _vehicle_input(row: Mapping[str, Any]) -> Mapping[str, Any]:
    """Accept direct vehicle records and harmless ``result`` wrappers."""
    candidate = row.get("result", row)
    payload = _mapping(candidate, "vehicle record")
    schema_version = payload.get("schema_version")
    if schema_version != INPUT_SCHEMA_VERSION:
        raise ValueError(
            f"expected schema_version={INPUT_SCHEMA_VERSION!r}, got {schema_version!r}"
        )
    return payload


def build_full_vehicle_facts(row: Mapping[str, Any]) -> Dict[str, Any]:
    """Project one vehicle physical-facts record into the LLM view."""
    vehicle_facts = _vehicle_input(row)
    physical = _mapping(
        vehicle_facts.get("physical_facts"),
        "physical_facts",
    )
    timeline = _mapping(vehicle_facts.get("timeline"), "timeline")
    metadata = vehicle_facts.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}

    scene_id = vehicle_facts.get("scene_id")
    if scene_id is None:
        raise ValueError("missing scene_id")
    vehicle_id = vehicle_facts.get("vehicle_id")
    if vehicle_id is None:
        raise ValueError("missing vehicle_id")
    vehicle_record_id = vehicle_facts.get("vehicle_record_id")
    if vehicle_record_id is None:
        vehicle_record_id = f"{scene_id}::vehicle_{int(vehicle_id)}"

    def clean(value: Any) -> Any:
        return remove_ordinary_shift_annotations(
            remove_implementation_noise(value)
        )
    # ``lateral_motion_evidence`` is intentionally not projected into the LLM
    # view.  It remains available to the upstream lane-chain builder, where it
    # can support a lane change together with a physical lane transition, but
    # it is not an independent language-generation input.
    return {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "vehicle_record_id": str(vehicle_record_id),
        "scene_id": scene_id,
        "vehicle_id": int(vehicle_id),
        "timeline": clean(timeline),
        # Quality is contextual evidence for the next stage.  Do not use it
        # here to erase or downgrade any physical fact sections.
        "trajectory_quality": clean(
            vehicle_facts.get("trajectory_quality", {})
        ),
        "track_motion_summary": clean(
            vehicle_facts.get("track_motion_summary", {})
        ),
        "vehicle": {
            "id": int(vehicle_id),
            "is_sdc": metadata.get("is_sdc"),
            "is_object_of_interest": metadata.get("is_object_of_interest"),
            "track_to_predict": metadata.get("track_to_predict"),
            "valid_frame_count": metadata.get("valid_frame_count"),
            "first_valid_frame": metadata.get("first_valid_frame"),
            "last_valid_frame": metadata.get("last_valid_frame"),
            "speed_change": clean(physical.get("speed_change")),
            "heading_motion": clean(physical.get("heading_motion")),
            "turn_maneuver": clean(physical.get("turn_maneuver")),
            "u_turn_evidence": clean(physical.get("u_turn_evidence")),
            "physical_lane_chain": clean(physical.get("physical_lane_chain")),
            "route_transition": clean(physical.get("route_transition")),
        },
        "map_context": clean(physical.get("map_context")),
    }


def _safe_component(value: Any) -> str:
    text = str(value)
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._") or "unknown"


def vehicle_json_path(output_dir: Path, facts: Mapping[str, Any]) -> Path:
    name = "{}__vehicle_{}_full_vehicle_facts.json".format(
        _safe_component(facts["scene_id"]),
        _safe_component(facts["vehicle_id"]),
    )
    return output_dir / VEHICLE_JSON_SUBDIR / name


def _write_json(handle: Any, value: Mapping[str, Any]) -> None:
    handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def _convert_jsonl(
    input_jsonl: Path,
    output_dir: Path,
    output_jsonl: Path,
    errors_jsonl: Path,
    error_source: str | None = None,
) -> Dict[str, int]:
    """Convert one vehicle-facts JSONL into one full-facts JSONL."""
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {"input_rows": 0, "written_rows": 0, "skipped_rows": 0}

    with input_jsonl.open("r", encoding="utf-8") as source:
        with output_jsonl.open("w", encoding="utf-8") as output:
            with errors_jsonl.open("a", encoding="utf-8") as errors:
                for line_number, line in enumerate(source, start=1):
                    if not line.strip():
                        continue
                    counts["input_rows"] += 1
                    try:
                        row = json.loads(line)
                        if not isinstance(row, Mapping):
                            raise ValueError("input row is not an object")
                        facts = build_full_vehicle_facts(row)
                        _write_json(output, facts)
                        vehicle_path = vehicle_json_path(output_dir, facts)
                        vehicle_path.write_text(
                            json.dumps(facts, ensure_ascii=False, indent=2, allow_nan=False)
                            + "\n",
                            encoding="utf-8",
                        )
                        counts["written_rows"] += 1
                    except Exception as exc:
                        counts["skipped_rows"] += 1
                        _write_json(errors, {
                            "input_file": error_source or str(input_jsonl),
                            "line_number": line_number,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        })
    return counts


def build_from_jsonl(input_jsonl: Path, output_dir: Path) -> Dict[str, int]:
    """Convert one aggregate JSONL and split full facts by scene."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / VEHICLE_JSON_SUBDIR).mkdir(parents=True, exist_ok=True)
    scene_output_dir = output_dir / "scenes"
    scene_output_dir.mkdir(parents=True, exist_ok=True)
    errors_jsonl = output_dir / "full_vehicle_facts_errors.jsonl"
    errors_jsonl.write_text("", encoding="utf-8")
    merged_output = output_dir / "full_vehicle_facts.jsonl"
    counts = {"input_rows": 0, "written_rows": 0, "skipped_rows": 0}
    scene_counts: dict[str, int] = {}
    scene_handles: dict[str, Any] = {}

    try:
        with input_jsonl.open("r", encoding="utf-8") as source:
            with merged_output.open("w", encoding="utf-8") as merged:
                with errors_jsonl.open("a", encoding="utf-8") as errors:
                    for line_number, line in enumerate(source, start=1):
                        if not line.strip():
                            continue
                        counts["input_rows"] += 1
                        try:
                            row = json.loads(line)
                            if not isinstance(row, Mapping):
                                raise ValueError("input row is not an object")
                            facts = build_full_vehicle_facts(row)
                            scene_name = _safe_component(facts["scene_id"])
                            scene_path = scene_output_dir / f"{scene_name}.jsonl"
                            if scene_name not in scene_handles:
                                scene_handles[scene_name] = scene_path.open(
                                    "w", encoding="utf-8"
                                )
                                scene_counts[scene_name] = 0
                            serialized = json.dumps(
                                facts, ensure_ascii=False, allow_nan=False
                            ) + "\n"
                            _write_json(merged, facts)
                            scene_handles[scene_name].write(serialized)
                            vehicle_path = vehicle_json_path(output_dir, facts)
                            vehicle_path.write_text(
                                json.dumps(
                                    facts,
                                    ensure_ascii=False,
                                    indent=2,
                                    allow_nan=False,
                                ) + "\n",
                                encoding="utf-8",
                            )
                            counts["written_rows"] += 1
                            scene_counts[scene_name] += 1
                        except Exception as exc:
                            counts["skipped_rows"] += 1
                            _write_json(errors, {
                                "input_file": str(input_jsonl),
                                "line_number": line_number,
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                            })
    finally:
        for handle in scene_handles.values():
            handle.close()

    summary = {
        **counts,
        "scene_count": len(scene_counts),
        "scene_output_dir": str(scene_output_dir),
        "scene_record_counts": scene_counts,
        "output_jsonl": str(merged_output),
        "errors_jsonl": str(errors_jsonl),
    }
    (output_dir / "full_vehicle_facts_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return counts


def build_from_scene_dir(input_dir: Path, output_dir: Path) -> Dict[str, int]:
    """Convert per-scene JSONLs, then merge them into one Qwen input JSONL."""
    input_files = sorted(
        path for path in input_dir.glob("*.jsonl")
        if path.name not in {"all_vehicle_facts.jsonl", "full_vehicle_facts.jsonl"}
    )
    if not input_files:
        raise ValueError(f"no scene JSONL files found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / VEHICLE_JSON_SUBDIR).mkdir(parents=True, exist_ok=True)
    scene_output_dir = output_dir / "scenes"
    scene_output_dir.mkdir(parents=True, exist_ok=True)
    errors_jsonl = output_dir / "full_vehicle_facts_errors.jsonl"
    errors_jsonl.write_text("", encoding="utf-8")

    totals = {"input_rows": 0, "written_rows": 0, "skipped_rows": 0}
    scene_counts: list[dict[str, Any]] = []
    for input_jsonl in input_files:
        scene_output = scene_output_dir / input_jsonl.name
        counts = _convert_jsonl(
            input_jsonl,
            output_dir,
            scene_output,
            errors_jsonl,
            error_source=str(input_jsonl),
        )
        scene_counts.append({"input": str(input_jsonl), "output": str(scene_output), **counts})
        for key in totals:
            totals[key] += counts[key]

    merged_output = output_dir / "full_vehicle_facts.jsonl"
    with merged_output.open("w", encoding="utf-8") as destination:
        for scene in scene_counts:
            with Path(scene["output"]).open("r", encoding="utf-8") as source:
                for line in source:
                    if line.strip():
                        destination.write(line if line.endswith("\n") else line + "\n")

    summary = {
        **totals,
        "scene_count": len(scene_counts),
        "scenes": scene_counts,
        "input_dir": str(input_dir),
        "scene_output_dir": str(scene_output_dir),
        "output_jsonl": str(merged_output),
        "errors_jsonl": str(errors_jsonl),
    }
    (output_dir / "full_vehicle_facts_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", "--input-jsonl", dest="input_jsonl", required=True, type=Path,
        help="One vehicle-facts JSONL or a directory containing per-scene JSONLs",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.input_jsonl.is_dir():
        counts = build_from_scene_dir(args.input_jsonl, args.output_dir)
    elif args.input_jsonl.is_file():
        counts = build_from_jsonl(args.input_jsonl, args.output_dir)
    else:
        raise SystemExit(f"input path not found: {args.input_jsonl}")
    print(json.dumps(counts, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
