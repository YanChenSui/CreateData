#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate Qwen caption files and audit simple prompt-rule violations."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


PATTERNS = {
    "lane_id_mention": re.compile(r"\blane\s+\d+\b", re.IGNORECASE),
    "frame_mention": re.compile(r"\bframe\s+\d+\b", re.IGNORECASE),
    "implementation_mention": re.compile(
        r"\b(interhub|detector|confidence|score|implementation)\b",
        re.IGNORECASE,
    ),
    "unsupported_semantic_word": re.compile(
        r"\b(cut[- ]?in|merge|overtak|yield|follow|avoid|collision|danger|risk)\b",
        re.IGNORECASE,
    ),
}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--timeline-batch", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    files = sorted(args.input_dir.glob("*_qwen_llm_description.json"))
    records = []
    invalid = []
    violation_counts = Counter()
    violation_examples = {}
    event_types = Counter()
    interaction_ids = set()

    for path in files:
        try:
            data = load_json(path)
            if not isinstance(data, dict):
                raise ValueError("output is not an object")
            text = " ".join(
                str(data.get(key, ""))
                for key in ("description_short", "description_detailed")
            )
            row_violations = []
            for name, pattern in PATTERNS.items():
                if pattern.search(text):
                    row_violations.append(name)
                    violation_counts[name] += 1
                    violation_examples.setdefault(
                        name,
                        {
                            "file": path.name,
                            "description_short": data.get("description_short"),
                            "description_detailed": data.get("description_detailed"),
                        },
                    )
            data["post_generation_audit"] = {
                "simple_prompt_rule_flags": row_violations,
            }
            records.append(data)
            interaction_ids.add(data.get("interaction_id"))
            event_types[data.get("event", {}).get("type")] += 1
        except Exception as exc:
            invalid.append({"file": path.name, "error": str(exc)})

    timeline_payload = load_json(args.timeline_batch)
    timeline_results = timeline_payload.get("results", [])
    timeline_status_counts = Counter(
        (item.get("result", {}).get("analysis", {}) or {}).get("status")
        for item in timeline_results
    )
    described_ids = {x for x in interaction_ids if x}

    aggregate = {
        "schema_version": "scene_motion_qwen_description_v3_batch",
        "timeline_batch": {
            "num_pair_timelines": len(timeline_results),
            "num_selected": timeline_payload.get("num_selected"),
            "num_success": timeline_payload.get("num_success"),
            "num_errors": timeline_payload.get("num_errors"),
            "status_counts": dict(timeline_status_counts),
        },
        "description_generation": {
            "num_event_inputs": len(files),
            "num_descriptions_loaded": len(records),
            "num_invalid_output_files": len(invalid),
            "num_described_pair_interactions": len(described_ids),
            "event_type_counts": dict(event_types),
        },
        "prompt_rule_audit": {
            "records_with_any_flag": sum(
                1
                for record in records
                if record.get("post_generation_audit", {}).get(
                    "simple_prompt_rule_flags"
                )
            ),
            "violation_counts": dict(violation_counts),
            "examples": violation_examples,
            "note": (
                "These are heuristic post-generation checks, not semantic ground truth. "
                "Lane-ID mentions are directly detectable; unsupported semantic words "
                "may be valid only when explicitly grounded by the input facts."
            ),
        },
        "invalid_outputs": invalid,
        "records": records,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "pair_timelines": len(timeline_results),
                "event_inputs": len(files),
                "descriptions": len(records),
                "described_pairs": len(described_ids),
                "invalid": len(invalid),
                "violation_counts": dict(violation_counts),
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
