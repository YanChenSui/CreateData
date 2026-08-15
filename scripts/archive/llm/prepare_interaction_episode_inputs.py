#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rename extractor records to stable episode ids and emit extraction audit."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def safe(value: Any) -> str:
    text = str(value if value is not None else "unknown")
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("._") or "unknown"


def input_record_id(input_data: dict[str, Any]) -> str:
    source_file = str(input_data.get("source", {}).get("input_file", ""))
    name = Path(source_file).name
    if name.endswith("_pair_timeline.json"):
        return name[: -len("_pair_timeline.json")]
    return safe(input_data.get("interaction_id", "unknown")).replace("::", "_")


def episode_id(input_data: dict[str, Any], ordinal: int) -> str:
    interaction = input_data.get("interaction", {})
    start = interaction.get("episode_start_frame")
    end = interaction.get("episode_end_frame")
    frame = interaction.get("event_frame")
    primary = interaction.get("event_type") or "event"
    start_s = f"{int(start):03d}" if isinstance(start, int) else "xxx"
    end_s = f"{int(end):03d}" if isinstance(end, int) else "xxx"
    frame_s = f"{int(frame):03d}" if isinstance(frame, int) else "xxx"
    return safe(f"episode_{start_s}_{end_s}_{primary}_f{frame_s}_{ordinal:03d}")


def raw_events(timeline: dict[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    channels = timeline.get("event_channels", {})
    if not isinstance(channels, dict):
        return output
    for channel, payload in channels.items():
        if not isinstance(payload, dict):
            continue
        values = payload.get("accepted_events", [])
        if not isinstance(values, list):
            continue
        for index, event in enumerate(values):
            if not isinstance(event, dict):
                continue
            output.append({
                "channel": channel,
                "source_event_index": index,
                "event_type": event.get("event_type"),
                "frame": event.get("frame"),
                "score": event.get("score"),
                "pair_level": event.get("pair_level"),
                "transition_agent_id": event.get("transition_agent_id"),
                "reference_agent_id": event.get("reference_agent_id"),
            })
    return output


def classify_raw_event(event: dict[str, Any], episodes: list[dict[str, Any]]) -> tuple[str, str | None]:
    event_type = event.get("event_type")
    frame = event.get("frame")
    for episode in episodes:
        interaction = episode.get("interaction", {})
        if event_type == interaction.get("event_type") and frame == interaction.get("event_frame"):
            return "primary", episode.get("episode_id")
        episode_types = interaction.get("episode_event_types", [])
        if event_type in episode_types:
            return "absorbed", episode.get("episode_id")
    if event_type == "individual_lane_change_near_pair":
        return "filtered", None
    return "deduplicated_or_filtered", None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-input-dir", type=Path, required=True)
    parser.add_argument("--timeline-dir", type=Path, required=True)
    parser.add_argument("--final-input-dir", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--audit-jsonl", type=Path, required=True)
    parser.add_argument("--episode-map-csv", type=Path, required=True)
    args = parser.parse_args()

    args.final_input_dir.mkdir(parents=True, exist_ok=True)
    raw_inputs = sorted(args.raw_input_dir.glob("*_llm_input.json"))
    episode_records: list[dict[str, Any]] = []
    audit_records: list[dict[str, Any]] = []
    map_rows: list[dict[str, Any]] = []
    by_timeline: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for ordinal, raw_path in enumerate(raw_inputs, start=1):
        data = load(raw_path)
        record_id = input_record_id(data)
        episode = episode_id(data, ordinal)
        data["interhub_record_id"] = record_id
        data["episode_id"] = episode
        source = data.setdefault("source", {})
        source["interhub_record_id"] = record_id
        interaction = data.setdefault("interaction", {})
        interaction["episode_id"] = episode
        final_name = f"{safe(record_id)}_{safe(episode)}_llm_input.json"
        final_path = args.final_input_dir / final_name
        dump(final_path, data)
        episode_records.append(data)
        timeline_path = Path(str(source.get("input_file", "")))
        by_timeline[str(timeline_path)].append(data)
        map_rows.append({
            "interhub_record_id": record_id,
            "episode_id": episode,
            "raw_input_file": str(raw_path),
            "final_input_file": str(final_path),
            "timeline_file": str(timeline_path),
            "event_type": interaction.get("event_type"),
            "event_frame": interaction.get("event_frame"),
            "episode_start_frame": interaction.get("episode_start_frame"),
            "episode_end_frame": interaction.get("episode_end_frame"),
            "episode_member_count": interaction.get("episode_member_count"),
        })

    for timeline_file, episodes in sorted(by_timeline.items()):
        timeline_path = Path(timeline_file)
        timeline = load(timeline_path) if timeline_path.is_file() else {}
        record_id = str(timeline.get("interhub_record_id") or timeline_path.stem.replace("_pair_timeline", ""))
        events = raw_events(timeline)
        for event in events:
            action, target = classify_raw_event(event, episodes)
            audit_records.append({
                "audit_type": "source_event",
                "interhub_record_id": record_id,
                "timeline_file": timeline_file,
                **event,
                "action": action,
                "episode_id": target,
            })
        audit_records.append({
            "audit_type": "record_summary",
            "interhub_record_id": record_id,
            "timeline_file": timeline_file,
            "raw_event_count": len(events),
            "interaction_episode_count": len(episodes),
            "llm_input_count": len(episodes),
            "episode_ids": [item.get("episode_id") for item in episodes],
        })

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for row in episode_records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    args.audit_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.audit_jsonl.open("w", encoding="utf-8") as handle:
        for row in audit_records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    args.episode_map_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in map_rows for key in row})
    with args.episode_map_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(map_rows)
    print(json.dumps({"raw_inputs": len(raw_inputs), "episodes": len(episode_records), "audit_rows": len(audit_records)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
