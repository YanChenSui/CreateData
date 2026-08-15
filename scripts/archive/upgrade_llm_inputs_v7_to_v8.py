#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Adapt v7 event inputs to the caption-safe v8 Qwen input contract.

The supplied extractor is v7 while the supplied Qwen runner requires v8.
This adapter preserves observable boolean/relational facts and removes raw
lane IDs, frame numbers, relative geometry, and InterHub audit context from
the model-facing payload.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


DROP_KEYS = {
    "continuous_lane_id",
    "transition_lane_before",
    "transition_lane_after",
    "reference_lane_before",
    "reference_lane_at_event",
    "reference_lane_after",
    "old_continuous_lane_id",
    "new_continuous_lane_id",
}


def sanitize(value):
    if isinstance(value, dict):
        output = {}
        for key, item in value.items():
            if key in DROP_KEYS or "continuous_lane_id" in key:
                continue
            output[key] = sanitize(item)
        return output
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    return value


def sanitize_fact_text(text):
    if not isinstance(text, str):
        return ""
    kept = []
    for line in text.splitlines():
        low = line.lower()
        if low.startswith("event frame:"):
            continue
        if "interhub window relation:" in low:
            continue
        if re.search(r"\blane (before|after)\s*:\s*\d+", low):
            continue
        if re.search(r"\b(transition|reference) lane (before|after)\s*:\s*\d+", low):
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def adapt(data):
    if data.get("schema_version") != "scene_motion_llm_input_v7":
        raise ValueError(
            "expected scene_motion_llm_input_v7, got %r"
            % data.get("schema_version")
        )

    output = dict(data)
    output["schema_version"] = "scene_motion_llm_input_v8"
    output["source_schema_version"] = "scene_motion_llm_input_v7"

    context = dict(data.get("generation_context") or {})
    facts = sanitize(context.get("facts") or {})
    phases = sanitize(context.get("event_aligned_phase_context"))
    speed = sanitize(context.get("associated_speed_responses") or [])

    supported = context.get("supported_semantics")
    if not isinstance(supported, list):
        supported = []
    supported = [
        item
        for item in supported
        if isinstance(item, dict) and isinstance(item.get("label"), str)
    ]

    output["generation_context"] = {
        "event": sanitize(context.get("event") or {}),
        "facts": facts,
        "maneuver": {},
        "supported_semantics": supported,
        "associated_speed_responses": speed,
        "event_aligned_phase_context": phases,
        "fact_text": sanitize_fact_text(context.get("fact_text", "")),
    }
    output["adaptation"] = {
        "from_schema": "scene_motion_llm_input_v7",
        "to_schema": "scene_motion_llm_input_v8",
        "removed_from_model_payload": [
            "raw lane IDs",
            "event frame numbers",
            "relative geometry",
            "InterHub audit context",
        ],
        "supported_semantics_policy": "preserve only explicitly supplied labels",
    }
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(args.input_dir.glob("*_llm_input.json"))
    failed = []
    success = 0
    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            output = adapt(data)
            (args.output_dir / path.name).write_text(
                json.dumps(output, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            success += 1
        except Exception as exc:
            failed.append({"file": path.name, "error": str(exc)})

    (args.output_dir / "failed_files.json").write_text(
        json.dumps(failed, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"input_files": len(paths), "success": success, "failed": len(failed)},
            ensure_ascii=False,
        )
    )
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
