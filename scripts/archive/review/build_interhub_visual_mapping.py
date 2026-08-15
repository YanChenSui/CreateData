#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Map InterHub record ids to existing visual artifacts by stable row/scene keys."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


ROW_RE = re.compile(r"row_(\d+)")
SCENE_RE = re.compile(r"(?:scene_)?(\d+)")


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def key_from_name(name: str) -> tuple[int | None, int | None]:
    row_match = ROW_RE.search(name)
    row = int(row_match.group(1)) if row_match else None
    scene = None
    scene_match = re.search(r"scene_(\d+)", name)
    if scene_match:
        scene = int(scene_match.group(1))
    elif row_match:
        suffix = name[row_match.end():].lstrip("_-")
        first = re.match(r"(\d+)", suffix)
        if first:
            scene = int(first.group(1))
    return row, scene


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--visual-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    args = parser.parse_args()

    visuals: dict[tuple[int | None, int | None], list[str]] = {}
    for root in args.visual_dir:
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".gif", ".png", ".jpg", ".jpeg"}:
                continue
            key = key_from_name(path.name)
            if key[0] is None:
                continue
            visuals.setdefault(key, []).append(str(path))

    rows: list[dict[str, Any]] = []
    for record in load_jsonl(args.manifest):
        record_id = str(record.get("interhub_record_id", ""))
        row_match = ROW_RE.search(record_id)
        row_number = int(row_match.group(1)) if row_match else None
        scene_value = record.get("interhub_scene_id")
        scene_match = re.search(r"scene_(\d+)", str(scene_value or ""))
        scene_number = int(scene_match.group(1)) if scene_match else None
        paths = visuals.get((row_number, scene_number), [])
        rows.append({
            "interhub_record_id": record_id,
            "interhub_scene_id": scene_value,
            "row_number": row_number,
            "scene_number": scene_number,
            "visual_paths": paths,
            "visual_count": len(paths),
            "visual_mapping_status": "mapped" if paths else "not_found",
        })

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["interhub_record_id", "interhub_scene_id", "row_number", "scene_number", "visual_count", "visual_mapping_status", "visual_paths"]
    with args.output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "visual_paths": ";".join(row["visual_paths"])})
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({
        "records": len(rows),
        "mapped": sum(row["visual_mapping_status"] == "mapped" for row in rows),
        "not_found": sum(row["visual_mapping_status"] == "not_found" for row in rows),
        "visual_files_indexed": sum(len(value) for value in visuals.values()),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
