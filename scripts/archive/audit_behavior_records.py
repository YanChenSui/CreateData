"""Audit behavior records and merge/caption consistency."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


def norm(value: Any) -> Optional[str]:
    return None if value is None else str(value)


def overlap(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    a = first.get("behavior_window", {})
    b = second.get("behavior_window", {})
    try:
        return max(int(a["start_frame"]), int(b["start_frame"])) <= min(
            int(a["end_frame"]), int(b["end_frame"])
        )
    except (KeyError, TypeError, ValueError):
        return False


def iter_rows(
    behavior_records_dir: Path,
    classified_dir: Path,
) -> Iterable[Dict[str, Any]]:
    for bundle_path in sorted(behavior_records_dir.glob("row_*_behavior_records.json")):
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        classified_name = bundle_path.name.replace(
            "_behavior_records.json", "_v3_classified.json"
        )
        classified_path = classified_dir / classified_name
        classified = (
            json.loads(classified_path.read_text(encoding="utf-8"))
            if classified_path.exists()
            else {}
        )
        interaction = classified.get("interaction", {})
        labels = (
            interaction.get("source_labels", {})
            if isinstance(interaction, Mapping)
            else {}
        )
        priority_agent_id = norm(labels.get("priority_agent_id"))
        source = bundle.get("source", {})
        scene_id = source.get("scene_id") if isinstance(source, Mapping) else None

        for record in bundle.get("behavior_records", []):
            evidence = record.get("evidence", {})
            if not isinstance(evidence, Mapping):
                evidence = {}
            lane_history = evidence.get("lane_history", {})
            if not isinstance(lane_history, Mapping):
                lane_history = {}
            subject_id = norm(record.get("subject_agent_id"))
            reference_id = norm(record.get("reference_agent_id"))
            subject_history = lane_history.get(subject_id or "", {})
            reference_history = lane_history.get(reference_id or "", {})
            if not isinstance(subject_history, Mapping):
                subject_history = {}
            if not isinstance(reference_history, Mapping):
                reference_history = {}
            yield {
                "scene_id": scene_id,
                "interaction_id": record.get("interaction_id"),
                "record_id": record.get("record_id"),
                "behavior_event_id": record.get("behavior_event_id"),
                "behavior_type": record.get("behavior_type"),
                "behavior_subtype": record.get("behavior_subtype"),
                "subject_agent_id": subject_id,
                "reference_agent_id": reference_id,
                "path_relation": evidence.get("path_relation"),
                "subject_inference": evidence.get("subject_inference"),
                "before_lane_id": subject_history.get("before_lane_id"),
                "after_lane_id": subject_history.get("after_lane_id"),
                "reference_before_lane_id": reference_history.get("before_lane_id"),
                "reference_after_lane_id": reference_history.get("after_lane_id"),
                "priority_agent_id": priority_agent_id,
                "caption_eligible": bool(record.get("caption_eligible")),
                "caption_suppressed_by": record.get("caption_suppressed_by"),
                "behavior_window": record.get("behavior_window"),
                "evidence": dict(evidence),
            }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--behavior-records-dir", required=True, type=Path)
    parser.add_argument("--classified-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    rows = list(iter_rows(args.behavior_records_dir, args.classified_dir))
    merges = [row for row in rows if row["behavior_type"] == "merge"]
    candidates = [
        row for row in rows if row["behavior_type"] == "merge_candidate"
    ]

    merge_path_bad = [
        row for row in merges if row["path_relation"] not in {"P-M", "M-P"}
    ]
    merge_subject_bad = [
        row
        for row in merges
        if not row["subject_agent_id"]
        or row["before_lane_id"] is None
        or row["after_lane_id"] is None
        or row["before_lane_id"] == row["after_lane_id"]
    ]
    merge_reference_bad = [
        row
        for row in merges
        if not row["reference_agent_id"]
        or row["reference_before_lane_id"] is None
        or row["reference_after_lane_id"] is None
        or row["reference_before_lane_id"] != row["reference_after_lane_id"]
    ]
    p_p_promoted = [row for row in merges if row["path_relation"] == "P-P"]

    duplicate_caption_pairs = []
    for merge in merges:
        for lane_change in rows:
            if lane_change["behavior_type"] != "lane_change":
                continue
            if lane_change["interaction_id"] != merge["interaction_id"]:
                continue
            if lane_change["subject_agent_id"] != merge["subject_agent_id"]:
                continue
            if not lane_change["caption_eligible"] or not merge["caption_eligible"]:
                continue
            if overlap(lane_change, merge):
                duplicate_caption_pairs.append(
                    {
                        "lane_change_record_id": lane_change["record_id"],
                        "merge_record_id": merge["record_id"],
                    }
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scene_id", "interaction_id", "record_id", "behavior_event_id",
        "behavior_type", "behavior_subtype", "subject_agent_id",
        "reference_agent_id", "path_relation", "subject_inference",
        "before_lane_id", "after_lane_id", "reference_before_lane_id",
        "reference_after_lane_id", "priority_agent_id", "caption_eligible",
        "caption_suppressed_by", "behavior_window", "evidence",
    ]
    with (args.output_dir / "behavior_records_audit.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(row[key], ensure_ascii=False)
                if isinstance(row[key], (dict, list))
                else row[key]
                for key in fieldnames
            })

    with (args.output_dir / "merge_records_audit.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in merges + candidates:
            writer.writerow({
                key: json.dumps(row[key], ensure_ascii=False)
                if isinstance(row[key], (dict, list))
                else row[key]
                for key in fieldnames
            })

    summary = {
        "total_behavior_records": len(rows),
        "behavior_type_counts": dict(Counter(row["behavior_type"] for row in rows)),
        "merge_count": len(merges),
        "merge_candidate_count": len(candidates),
        "merge_path_relation_counts": dict(
            Counter(row["path_relation"] for row in merges)
        ),
        "merge_candidate_path_relation_counts": dict(
            Counter(row["path_relation"] for row in candidates)
        ),
        "checks": {
            "all_merge_has_pm_or_mp": {
                "ok": not merge_path_bad,
                "count": len(merge_path_bad),
            },
            "all_merge_subject_changes_lane": {
                "ok": not merge_subject_bad,
                "count": len(merge_subject_bad),
            },
            "all_merge_reference_keeps_lane": {
                "ok": not merge_reference_bad,
                "count": len(merge_reference_bad),
            },
            "p_p_promoted_to_merge": {
                "ok": not p_p_promoted,
                "count": len(p_p_promoted),
            },
            "overlapping_merge_and_lane_change_both_caption_eligible": {
                "ok": not duplicate_caption_pairs,
                "count": len(duplicate_caption_pairs),
            },
        },
        "failures": {
            "merge_path": merge_path_bad,
            "merge_subject": merge_subject_bad,
            "merge_reference": merge_reference_bad,
            "p_p_promoted": p_p_promoted,
            "duplicate_caption_pairs": duplicate_caption_pairs,
        },
    }
    (args.output_dir / "audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary["checks"], ensure_ascii=False, indent=2))
    print(f"total_behavior_records: {len(rows)}")
    print(f"merge_count: {len(merges)}")
    print(f"merge_candidate_count: {len(candidates)}")
    print(f"audit_dir: {args.output_dir}")
    return 1 if any(not item["ok"] for item in summary["checks"].values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
