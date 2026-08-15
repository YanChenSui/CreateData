"""Generate a row-aligned v1/v2 transition comparison table.

The two inputs must describe the same candidate pairs.  Rows are joined by
``scene_id, agent_A, agent_B`` rather than by output order.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


FIELDS = [
    "scene",
    "pair",
    "source_behavior",
    "v1_status",
    "v2_status",
    "transition_agent",
    "reference_agent",
    "event_score",
    "event_frame",
    "entered_reference_lane",
    "reference_maintained_lane",
    "speed_response",
    "behind_to_ahead",
    "lane_before",
    "lane_during",
    "lane_after",
    "interhub_window",
    "distance_to_interhub_window",
]


def _read_rows(path: Path, key: str) -> List[Mapping[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get(key, payload) if isinstance(payload, Mapping) else payload
    if not isinstance(rows, list):
        raise ValueError(f"{path} does not contain a list under {key!r}")
    return rows


def _pair_key(row: Mapping[str, Any]) -> Tuple[str, str, str]:
    return (str(row["scene_id"]), str(row["agent_A"]), str(row["agent_B"]))


def _bool_or_none(value: Any) -> Optional[bool]:
    return value if isinstance(value, bool) else None


def _window_string(start: Any, end: Any) -> Optional[str]:
    if start is None or end is None:
        return None
    return f"{int(start)}-{int(end)}"


def _window_distance(event_frame: Any, start: Any, end: Any) -> Optional[int]:
    if event_frame is None or start is None or end is None:
        return None
    event_frame, start, end = int(event_frame), int(start), int(end)
    if start <= event_frame <= end:
        return 0
    return start - event_frame if event_frame < start else event_frame - end


def _selected_event(analysis: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    event = analysis.get("selected_event")
    return event if isinstance(event, Mapping) else None


def build_comparison_rows(
    v1_rows: Iterable[Mapping[str, Any]],
    v2_rows: Iterable[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    v1_by_key = {_pair_key(row): row for row in v1_rows}
    v2_by_key = {
        _pair_key(row["input"]): row
        for row in v2_rows
        if isinstance(row, Mapping) and isinstance(row.get("input"), Mapping)
    }

    if set(v1_by_key) != set(v2_by_key):
        missing = sorted(set(v1_by_key) - set(v2_by_key))
        extra = sorted(set(v2_by_key) - set(v1_by_key))
        raise ValueError(
            f"v1/v2 pair coverage mismatch: missing={len(missing)}, extra={len(extra)}; "
            f"first_missing={missing[:3]}, first_extra={extra[:3]}"
        )

    rows: List[Dict[str, Any]] = []
    for key in sorted(v1_by_key, key=lambda item: (int(item[0].split("_")[-1]), item[1], item[2])):
        v1 = v1_by_key[key]
        v2 = v2_by_key[key]["result"]
        analysis = v2.get("analysis", {})
        facts = v2.get("role_centric_facts", {})
        event = _selected_event(analysis)
        phase_triplet = v2.get("phase_triplet", {})

        start = v1.get("interhub_window_audit_only")
        end = v1.get("interhub_window_end_audit_only")
        event_frame = event.get("frame") if event is not None else None

        rows.append(
            {
                "scene": key[0],
                "pair": f"{key[1]}-{key[2]}",
                "source_behavior": v1.get("source_behavior"),
                "v1_status": v1.get("analysis_window", {}).get("status"),
                "v2_status": analysis.get("status"),
                "transition_agent": facts.get("transition_agent_id"),
                "reference_agent": facts.get("reference_agent_id"),
                "event_score": event.get("score") if event is not None else None,
                "event_frame": event_frame,
                "entered_reference_lane": _bool_or_none(facts.get("transition_agent_entered_reference_lane")),
                "reference_maintained_lane": _bool_or_none(facts.get("reference_agent_maintained_lane")),
                "speed_response": _bool_or_none(facts.get("reference_speed_reduced_during_event")),
                "behind_to_ahead": _bool_or_none(facts.get("transition_agent_behind_to_ahead")),
                "lane_before": phase_triplet.get("before", {}).get("lane_relation"),
                "lane_during": phase_triplet.get("during", {}).get("lane_relation"),
                "lane_after": phase_triplet.get("after", {}).get("lane_relation"),
                "interhub_window": _window_string(start, end),
                "distance_to_interhub_window": _window_distance(event_frame, start, end),
            }
        )
    return rows


def _csv_value(value: Any) -> Any:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    return value


def write_outputs(rows: List[Mapping[str, Any]], output_csv: Path, output_json: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in FIELDS})
    output_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1", type=Path, required=True)
    parser.add_argument("--v2", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    v1_rows = _read_rows(args.v1, "rows")
    v2_rows = _read_rows(args.v2, "results")
    rows = build_comparison_rows(v1_rows, v2_rows)
    write_outputs(rows, args.output_csv, args.output_json)
    print(f"[OK] rows: {len(rows)}")
    print(f"[OK] csv : {args.output_csv}")
    print(f"[OK] json: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

