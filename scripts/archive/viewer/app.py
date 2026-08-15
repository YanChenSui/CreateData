"""Streamlit browser for interaction-level GIF clips and event-level descriptions."""
from __future__ import annotations

import html
import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

VIEWER_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = VIEWER_DIR.parent
DATA_DIR = VIEWER_DIR / "data"
VEHICLE_REVIEW_DIR = DATA_DIR / "vehicle_review"
DEFAULT_MANIFEST_PATH = DATA_DIR / "manifest.csv"
DEFAULT_CLIP_MANIFEST_PATH = DATA_DIR / "clip_manifest.csv"
DEFAULT_VEHICLE_REVIEW_MANIFEST_PATH = VEHICLE_REVIEW_DIR / "vehicle_review_manifest.json"
DEFAULT_VEHICLE_REVIEW_EDITS_PATH = VEHICLE_REVIEW_DIR / "vehicle_review_edits.csv"
DEFAULT_VEHICLE_GIF_DIR = VEHICLE_REVIEW_DIR / "gifs"
EDITS_PATH = DATA_DIR / "description_edits.csv"
CORRECTED_EXPORT_PATH = DATA_DIR / "corrected_descriptions.csv"
EDIT_COLUMNS = (
    "behavior_record_id",
    "original_description",
    "edited_description",
    "status",
    "updated_at",
)
STATUS_OPTIONS = ("unchecked", "correct", "edited", "reject")
VEHICLE_STATUS_OPTIONS = ("unchecked", "correct", "ambiguous", "reject")

LOGGER = logging.getLogger(__name__)


def _parse_runtime_paths() -> argparse.Namespace:
    """Read arguments passed after ``streamlit run app.py --``.

    Streamlit keeps its own command-line arguments in ``sys.argv``.  The
    application-specific arguments are normally placed after ``--``; accepting
    unknown arguments keeps this compatible with different Streamlit versions.
    """
    argv = list(sys.argv[1:])
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--gif-dir",
        type=Path,
        default=None,
        help="Directory containing vehicle/full-scene GIF files.",
    )
    parser.add_argument(
        "--description-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing vehicle_review_manifest.json, manifest.csv, "
            "or scene vehicle-review manifests."
        ),
    )
    parser.add_argument(
        "--description-file",
        type=Path,
        default=None,
        help=(
            "Explicit vehicle_review_manifest.json, interaction manifest.csv, "
            "or a directory of scene vehicle-review manifests."
        ),
    )
    options, _ = parser.parse_known_args(argv)
    if options.description_dir is not None and options.description_file is not None:
        raise ValueError("Use only one of --description-dir and --description-file")
    return options


RUNTIME_PATHS = _parse_runtime_paths()


def _absolute_runtime_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    return path.expanduser().resolve()


GIF_DIR_OVERRIDE = _absolute_runtime_path(RUNTIME_PATHS.gif_dir)
DESCRIPTION_DIR_OVERRIDE = _absolute_runtime_path(RUNTIME_PATHS.description_dir)
DESCRIPTION_FILE_OVERRIDE = _absolute_runtime_path(RUNTIME_PATHS.description_file)


def _description_file_from_override() -> Path | None:
    if DESCRIPTION_FILE_OVERRIDE is not None:
        if DESCRIPTION_FILE_OVERRIDE.exists():
            return DESCRIPTION_FILE_OVERRIDE
        return None
    if DESCRIPTION_DIR_OVERRIDE is None:
        return None
    for name in (
        "vehicle_review_manifest.json",
        "manifest.csv",
        "vehicle_qwen_descriptions_summary.csv",
    ):
        candidate = DESCRIPTION_DIR_OVERRIDE / name
        if candidate.exists():
            return candidate
    if any(DESCRIPTION_DIR_OVERRIDE.glob("*_vehicle_review_manifest.json")):
        return DESCRIPTION_DIR_OVERRIDE
    return None


def _is_vehicle_summary_csv(path: Path | None) -> bool:
    if path is None or not path.is_file() or path.suffix.lower() != ".csv":
        return False
    try:
        columns = set(pd.read_csv(path, nrows=0).columns)
    except (OSError, pd.errors.EmptyDataError, UnicodeDecodeError):
        return False
    return {"scene_id", "vehicle_id", "description_short"}.issubset(columns)


DESCRIPTION_FILE = _description_file_from_override()
VEHICLE_REVIEW_SUMMARY_CSV = (
    DESCRIPTION_FILE if _is_vehicle_summary_csv(DESCRIPTION_FILE) else None
)
MANIFEST_PATH = (
    DESCRIPTION_FILE
    if DESCRIPTION_FILE is not None and DESCRIPTION_FILE.suffix.lower() == ".csv"
    else DEFAULT_MANIFEST_PATH
)
CLIP_MANIFEST_PATH = (
    DESCRIPTION_FILE.parent / "clip_manifest.csv"
    if DESCRIPTION_FILE is not None and DESCRIPTION_FILE.suffix.lower() == ".csv"
    else DEFAULT_CLIP_MANIFEST_PATH
)
VEHICLE_REVIEW_MANIFEST_PATH = (
    DESCRIPTION_FILE
    if DESCRIPTION_FILE is not None and DESCRIPTION_FILE.suffix.lower() == ".json"
    else DEFAULT_VEHICLE_REVIEW_MANIFEST_PATH
)
if DESCRIPTION_FILE is not None and DESCRIPTION_FILE.is_dir():
    VEHICLE_REVIEW_MANIFEST_FILES = tuple(sorted(
        DESCRIPTION_FILE.glob("*_vehicle_review_manifest.json")
    ))
elif VEHICLE_REVIEW_MANIFEST_PATH.exists():
    VEHICLE_REVIEW_MANIFEST_FILES = (VEHICLE_REVIEW_MANIFEST_PATH,)
else:
    VEHICLE_REVIEW_MANIFEST_FILES = ()
VEHICLE_REVIEW_EDITS_PATH = (
    DESCRIPTION_DIR_OVERRIDE / "vehicle_review_edits.csv"
    if DESCRIPTION_DIR_OVERRIDE is not None
    else VEHICLE_REVIEW_SUMMARY_CSV.parent / "vehicle_review_edits.csv"
    if VEHICLE_REVIEW_SUMMARY_CSV is not None
    else DESCRIPTION_FILE / "vehicle_review_edits.csv"
    if DESCRIPTION_FILE is not None and DESCRIPTION_FILE.is_dir()
    else DEFAULT_VEHICLE_REVIEW_EDITS_PATH
)
VEHICLE_GIF_DIR = GIF_DIR_OVERRIDE or DEFAULT_VEHICLE_GIF_DIR


def cell(row: Mapping[str, Any], name: str) -> str:
    value = row.get(name, "")
    return "" if value is None else str(value).strip()


def unique_values(records: Sequence[Mapping[str, Any]], column: str) -> list[str]:
    values: list[str] = []
    for record in records:
        value = cell(record, column)
        if value and value not in values:
            values.append(value)
    return values


def display_scene(raw: str) -> str:
    value = str(raw).strip()
    return value[6:] if value.lower().startswith("scene_") else value


@st.cache_data(show_spinner=False)
def load_manifest(path: str, modified_ns: int) -> pd.DataFrame:
    """Load one manifest once per file version."""
    t0 = time.perf_counter()
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if frame.empty:
        raise ValueError(f"CSV is empty: {path}")
    for column in frame.columns:
        frame[column] = frame[column].fillna("").astype(str)
    LOGGER.debug(
        "manifest loading: path=%s rows=%d elapsed=%.3fs",
        path,
        len(frame),
        time.perf_counter() - t0,
    )
    return frame


@st.cache_data(show_spinner=False)
def load_description_edits(path: str, modified_ns: int) -> pd.DataFrame:
    """Load only the small review state file; cache by modification time."""
    if modified_ns < 0 or not Path(path).exists():
        return pd.DataFrame(columns=EDIT_COLUMNS)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    for column in EDIT_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    return frame[list(EDIT_COLUMNS)].fillna("").astype(str)


def edit_record_id(event: Mapping[str, Any]) -> str:
    return cell(event, "behavior_record_id") or cell(event, "event_id")


def edit_widget_key(record_id: str) -> str:
    return f"description_edit_{record_id}"


def status_widget_key(record_id: str) -> str:
    return f"description_status_{record_id}"


def edits_to_map(frame: pd.DataFrame) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in frame.to_dict(orient="records"):
        record_id = str(row.get("behavior_record_id", "")).strip()
        if record_id:
            result[record_id] = {
                column: str(row.get(column, "")).strip()
                for column in EDIT_COLUMNS
            }
    return result


def persist_description_edit(row: dict[str, str]) -> None:
    if EDITS_PATH.exists():
        frame = pd.read_csv(
            EDITS_PATH, dtype=str, keep_default_na=False
        )
    else:
        frame = pd.DataFrame(columns=EDIT_COLUMNS)
    for column in EDIT_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    existing = frame[list(EDIT_COLUMNS)].fillna("").astype(str)
    existing = existing.drop_duplicates(
        "behavior_record_id", keep="last"
    )
    records = existing.to_dict(orient="records")
    updated = False
    for item in records:
        if item["behavior_record_id"] == row["behavior_record_id"]:
            item.update(row)
            updated = True
            break
    if not updated:
        records.append(row)
    output = pd.DataFrame(records, columns=EDIT_COLUMNS)
    temp_path = EDITS_PATH.with_suffix(".tmp.csv")
    output.to_csv(temp_path, index=False, encoding="utf-8-sig")
    temp_path.replace(EDITS_PATH)


def save_description_edit(
    record_id: str,
    original_description: str,
    next_record_id: str | None = None,
) -> None:
    text = str(
        st.session_state.get(edit_widget_key(record_id), "")
    ).strip()
    status = str(
        st.session_state.get(status_widget_key(record_id), "unchecked")
    ).strip()
    if status not in STATUS_OPTIONS:
        status = "unchecked"
    if status == "unchecked" and text != original_description.strip():
        status = "edited"
        st.session_state[status_widget_key(record_id)] = status

    row = {
        "behavior_record_id": record_id,
        "original_description": original_description,
        "edited_description": text,
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
    }
    persist_description_edit(row)
    st.session_state.setdefault("_description_edit_map", {})[record_id] = row
    st.session_state["_edit_notice"] = (
        f"Saved review for {record_id} ({status})."
    )
    if next_record_id is not None:
        st.session_state["_pending_review_record_id"] = next_record_id


def save_review_status(
    record_id: str,
    original_description: str,
    status: str,
    next_record_id: str | None = None,
) -> None:
    text = str(
        st.session_state.get(edit_widget_key(record_id), "")
    ).strip()
    if status not in STATUS_OPTIONS:
        status = "unchecked"
    row = {
        "behavior_record_id": record_id,
        "original_description": original_description,
        "edited_description": text,
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
    }
    persist_description_edit(row)
    st.session_state.setdefault("_description_edit_map", {})[record_id] = row
    st.session_state[status_widget_key(record_id)] = status
    st.session_state["_edit_notice"] = (
        f"Saved review for {record_id} ({status})."
    )
    if next_record_id is not None:
        st.session_state["_pending_review_record_id"] = next_record_id


def build_corrected_dataset(
    events_by_clip: Mapping[str, Sequence[Mapping[str, Any]]],
    edits_by_record: Mapping[str, Mapping[str, str]],
    keep_unchecked: bool,
) -> bytes:
    rows: list[dict[str, str]] = []
    for events in events_by_clip.values():
        for event in events:
            record_id = edit_record_id(event)
            original = cell(event, "description")
            state = edits_by_record.get(record_id, {})
            status = state.get("status", "unchecked")
            edited = state.get("edited_description", "")
            if not original and not edited:
                continue
            if status == "reject":
                continue
            if status == "correct":
                corrected = original
            elif status == "edited":
                corrected = edited
            elif keep_unchecked:
                corrected = original
            else:
                continue
            rows.append(
                {
                    "behavior_record_id": record_id,
                    "event_id": cell(event, "event_id"),
                    "interaction_id": cell(event, "interaction_id"),
                    "scene_id": cell(event, "scene_id"),
                    "event_frame": cell(event, "event_frame"),
                    "original_description": original,
                    "corrected_description": corrected,
                    "status": status,
                }
            )
    return pd.DataFrame(rows).to_csv(
        index=False, encoding="utf-8-sig"
    ).encode("utf-8-sig")


@st.cache_resource(show_spinner=False)
def prepare_viewer_index(
    manifest_path: str,
    manifest_modified_ns: int,
    clip_manifest_path: str,
    clip_manifest_modified_ns: int,
) -> tuple[
    tuple[dict[str, str], ...],
    dict[str, tuple[dict[str, str], ...]],
    tuple[str, ...],
    dict[str, tuple[int, ...]],
    tuple[tuple[int, str], ...],
    dict[str, int],
    dict[str, tuple[int, dict[str, str]]],
]:
    """Build clip/event records once; navigation only performs dictionary lookups."""
    t0 = time.perf_counter()
    events_frame = load_manifest(manifest_path, manifest_modified_ns)
    if "clip_id" not in events_frame.columns:
        raise ValueError("manifest.csv must contain clip_id")

    if clip_manifest_modified_ns >= 0:
        clips_frame = load_manifest(
            clip_manifest_path, clip_manifest_modified_ns
        )
    else:
        required = ["clip_id", "interaction_id", "scene_id", "gif_path"]
        missing = [column for column in required if column not in events_frame]
        if missing:
            raise ValueError(f"manifest.csv is missing columns: {missing}")
        clips_frame = (
            events_frame[required]
            .drop_duplicates("clip_id", keep="first")
            .copy()
        )
        counts = events_frame.groupby("clip_id", sort=False).size()
        clips_frame["event_count"] = (
            clips_frame["clip_id"].map(counts).fillna(0).astype(int).astype(str)
        )

    clip_records = tuple(
        {str(key): str(value) for key, value in record.items()}
        for record in clips_frame.to_dict(orient="records")
    )

    events_by_clip: dict[str, tuple[dict[str, str], ...]] = {}
    for clip_id, group in events_frame.groupby(
        "clip_id", sort=False, dropna=False
    ):
        group = group.copy()
        if "event_frame" in group.columns:
            group["_frame_sort"] = pd.to_numeric(
                group["event_frame"], errors="coerce"
            )
            sort_columns = ["_frame_sort"]
            if "event_id" in group.columns:
                sort_columns.append("event_id")
            group = group.sort_values(
                sort_columns, na_position="last", kind="stable"
            ).drop(columns="_frame_sort")
        events_by_clip[str(clip_id)] = tuple(
            {str(key): str(value) for key, value in record.items()}
            for record in group.to_dict(orient="records")
        )

    interaction_to_indices: dict[str, list[int]] = {}
    interaction_ids: list[str] = []
    for index, clip in enumerate(clip_records):
        interaction_id = cell(clip, "interaction_id")
        if interaction_id not in interaction_to_indices:
            interaction_to_indices[interaction_id] = []
            interaction_ids.append(interaction_id)
        interaction_to_indices[interaction_id].append(index)

    frozen_interaction_index = {
        key: tuple(indices) for key, indices in interaction_to_indices.items()
    }
    review_sequence: list[tuple[int, str]] = []
    review_positions: dict[str, int] = {}
    event_by_record: dict[str, tuple[int, dict[str, str]]] = {}
    for clip_index, clip in enumerate(clip_records):
        clip_id = cell(clip, "clip_id")
        for event in events_by_clip.get(clip_id, ()):
            record_id = edit_record_id(event)
            if record_id and cell(event, "description"):
                review_positions[record_id] = len(review_sequence)
                review_sequence.append((clip_index, record_id))
                event_by_record[record_id] = (clip_index, event)
    LOGGER.debug(
        "manifest/group preparation: clips=%d events=%d interactions=%d "
        "elapsed=%.3fs",
        len(clip_records),
        sum(len(records) for records in events_by_clip.values()),
        len(interaction_ids),
        time.perf_counter() - t0,
    )
    return (
        clip_records,
        events_by_clip,
        tuple(interaction_ids),
        frozen_interaction_index,
        tuple(review_sequence),
        review_positions,
        event_by_record,
    )


@st.cache_data(show_spinner=False, max_entries=32)
def load_gif_bytes(path: str, modified_ns: int) -> bytes:
    t0 = time.perf_counter()
    with open(path, "rb") as handle:
        payload = handle.read()
    LOGGER.debug(
        "GIF loading: path=%s bytes=%d elapsed=%.3fs",
        path,
        len(payload),
        time.perf_counter() - t0,
    )
    return payload


@st.cache_data(show_spinner=False, max_entries=16)
def load_json_payload(path: str, modified_ns: int) -> Any:
    t0 = time.perf_counter()
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    LOGGER.debug(
        "JSON loading: path=%s elapsed=%.3fs",
        path,
        time.perf_counter() - t0,
    )
    return payload


def resolve_path(raw: str) -> Path:
    path = Path(str(raw).strip())
    if path.is_absolute():
        if path.exists():
            return path
        # A manifest may contain an old absolute path.  When --gif-dir is
        # supplied, prefer the file with the same basename from that root.
        candidates = [
            VEHICLE_GIF_DIR / path.name,
            DESCRIPTION_DIR_OVERRIDE / path.name
            if DESCRIPTION_DIR_OVERRIDE is not None
            else path,
            path,
        ]
        return next((item for item in candidates if item.exists()), path)
    candidates: list[Path] = []
    if GIF_DIR_OVERRIDE is not None:
        candidates.extend((GIF_DIR_OVERRIDE / path, GIF_DIR_OVERRIDE / path.name))
    if DESCRIPTION_DIR_OVERRIDE is not None:
        candidates.extend(
            (DESCRIPTION_DIR_OVERRIDE / path, DESCRIPTION_DIR_OVERRIDE / path.name)
        )
    candidates.extend((PROJECT_ROOT / path, VIEWER_DIR / path, DATA_DIR / path))
    return next((item for item in candidates if item.exists()), candidates[0])


@st.cache_data(show_spinner=False)
def load_vehicle_review_manifests(
    paths: tuple[str, ...],
    signatures: tuple[tuple[str, int], ...],
) -> dict[str, Any]:
    """Load and merge one or more scene manifests for the vehicle reviewer."""
    merged: dict[str, Any] = {
        "schema_version": "vehicle_review_manifest_v1",
        "scene_id": "multiple" if len(paths) > 1 else "",
        "num_frames": 0,
        "items": [],
        "context_policy": "Other vehicles are rendered only as neutral visual context.",
    }
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
            raise ValueError(f"{path} must contain an items list")
        if len(paths) == 1:
            merged["scene_id"] = str(payload.get("scene_id", ""))
            merged["full_scene_gif"] = payload.get("full_scene_gif", "")
        merged["num_frames"] = max(
            int(merged.get("num_frames", 0)),
            int(payload.get("num_frames", 0) or 0),
        )
        merged["items"].extend(
            item for item in payload["items"] if isinstance(item, Mapping)
        )
    return merged


def _json_cell(value: Any, default: Any) -> Any:
    if value is None or str(value).strip() == "":
        return default
    try:
        return json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return default


@st.cache_data(show_spinner=False)
def load_vehicle_summary_csv(
    path: str,
    modified_ns: int,
    gif_dir: str,
) -> dict[str, Any]:
    """Build lightweight review items from vehicle description summary CSV."""
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    items: list[dict[str, Any]] = []
    max_frame = 0
    gif_root = Path(gif_dir)
    for row_index, row in enumerate(frame.to_dict(orient="records")):
        scene_id = str(row.get("scene_id", "")).strip()
        vehicle_id = str(row.get("vehicle_id", "")).strip()
        if not scene_id or not vehicle_id:
            continue
        supporting = _json_cell(row.get("supporting_frame_ranges"), [])
        segments = _json_cell(row.get("behavior_segments"), [])
        if not isinstance(supporting, list):
            supporting = []
        if not isinstance(segments, list):
            segments = []

        windows = [value for value in supporting if isinstance(value, Mapping)]
        if not windows:
            windows = [value for value in segments if isinstance(value, Mapping)]
        if not windows:
            windows = [{
                "start_frame": 0,
                "end_frame": 90,
                "description": "Full observed track",
            }]

        gif_path = gif_root / scene_id / f"{scene_id}__vehicle_{vehicle_id}.gif"
        source_key = str(row.get("output_file", "")).strip() or f"row_{row_index}"
        for range_index, window in enumerate(windows):
            try:
                start = int(window.get("start_frame", 0))
                end = int(window.get("end_frame", start))
            except (TypeError, ValueError):
                continue
            if start < 0 or end < start:
                continue
            max_frame = max(max_frame, end)
            items.append({
                "review_id": f"{scene_id}::vehicle_{vehicle_id}::"
                f"{source_key}::range_{range_index}",
                "scene_id": scene_id,
                "vehicle_id": vehicle_id,
                "gif_path": str(gif_path),
                "frame_range": {
                    "start_frame": start,
                    "end_frame": end,
                    "description": str(window.get("description", "")),
                },
                "description": {
                    "short": str(row.get("description_short", "")),
                    "detailed": str(row.get("description_detailed", "")),
                    "uncertainty_notes": str(row.get("uncertainty_notes", "")),
                },
                "behavior_segments": segments,
                "generation_mode": str(row.get("generation_mode", "")),
                "target_track": {"states": []},
                "context_tracks": [],
                "lane_matching": [],
                "lane_transition": {},
                "physical_facts": {},
            })
    return {
        "schema_version": "vehicle_description_summary_csv_v1",
        "scene_id": "multiple",
        "num_frames": max_frame + 1,
        "items": items,
    }


def load_vehicle_review_edits() -> dict[str, dict[str, str]]:
    if not VEHICLE_REVIEW_EDITS_PATH.exists():
        return {}
    frame = pd.read_csv(VEHICLE_REVIEW_EDITS_PATH, dtype=str, keep_default_na=False)
    return {
        str(row.get("review_id", "")): {str(key): str(value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
        if str(row.get("review_id", "")).strip()
    }


def save_vehicle_review_edit(review_id: str, status: str, note: str) -> None:
    status = status if status in VEHICLE_STATUS_OPTIONS else "unchecked"
    rows = list(load_vehicle_review_edits().values())
    row = {
        "review_id": review_id,
        "status": status,
        "note": note.strip(),
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    rows = [old for old in rows if old.get("review_id") != review_id]
    rows.append(row)
    VEHICLE_REVIEW_EDITS_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=("review_id", "status", "note", "updated_at")).to_csv(
        VEHICLE_REVIEW_EDITS_PATH, index=False, encoding="utf-8-sig"
    )


def _vehicle_state(item: Mapping[str, Any], frame: int) -> Mapping[str, Any] | None:
    states = item.get("target_track", {}).get("states", [])
    if not isinstance(states, list) or frame < 0 or frame >= len(states):
        return None
    state = states[frame]
    return state if isinstance(state, Mapping) and state.get("valid") else None


def render_vehicle_scene(item: Mapping[str, Any], current_frame: int) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    target = item.get("target_track", {})
    target_states = target.get("states", [])
    selected = item.get("frame_range", {})
    start = int(selected.get("start_frame", 0))
    end = int(selected.get("end_frame", start))
    figure, (scene_ax, speed_ax) = plt.subplots(
        1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": (1.35, 1)}
    )
    valid_target = [state for state in target_states if state.get("valid")]
    if valid_target:
        scene_ax.plot(
            [state["x"] for state in valid_target],
            [state["y"] for state in valid_target],
            color="#ef4444", alpha=0.28, linewidth=2, label="target full track",
        )
    range_target = [
        state for state in target_states
        if state.get("valid") and start <= int(state.get("frame", -1)) <= end
    ]
    if range_target:
        scene_ax.plot(
            [state["x"] for state in range_target],
            [state["y"] for state in range_target],
            color="#dc2626", linewidth=3, label=f"supporting range {start}-{end}",
        )
    for context in item.get("context_tracks", []):
        state = _vehicle_state({"target_track": context}, current_frame)
        if state is not None:
            scene_ax.scatter(state["x"], state["y"], color="#9ca3af", s=24, alpha=0.65)
    current = _vehicle_state(item, current_frame)
    if current is not None:
        scene_ax.scatter(current["x"], current["y"], color="#111827", s=85, zorder=5)
        scene_ax.text(current["x"], current["y"], f"  Vehicle {item.get('vehicle_id')}", fontsize=9)
        heading = float(current.get("heading_deg", 0.0)) * np.pi / 180.0
        scene_ax.arrow(current["x"], current["y"], 4 * np.cos(heading), 4 * np.sin(heading),
                       color="#111827", width=0.12, head_width=0.8, length_includes_head=True)
    for state in range_target[::max(1, len(range_target) // 8)]:
        heading = float(state.get("heading_deg", 0.0)) * np.pi / 180.0
        scene_ax.arrow(state["x"], state["y"], 2.5 * np.cos(heading), 2.5 * np.sin(heading),
                       color="#f97316", width=0.06, head_width=0.45, alpha=0.8, length_includes_head=True)
    scene_ax.set_title(f"Vehicle {item.get('vehicle_id')} · frame {current_frame}\nOther vehicles are context only")
    scene_ax.set_xlabel("world x (m)")
    scene_ax.set_ylabel("world y (m)")
    scene_ax.grid(alpha=0.2)
    scene_ax.axis("equal")
    scene_ax.legend(loc="best", fontsize=8)

    frames = [int(state.get("frame", i)) for i, state in enumerate(target_states)]
    speeds = [float(state.get("speed_mps", 0.0)) if state.get("valid") else np.nan for state in target_states]
    speed_ax.plot(frames, speeds, color="#2563eb", linewidth=2)
    speed_ax.axvspan(start, end, color="#f59e0b", alpha=0.2, label=f"supporting range {start}-{end}")
    speed_ax.axvline(current_frame, color="#111827", linestyle="--", linewidth=1)
    speed_ax.set_title("Target speed")
    speed_ax.set_xlabel("frame")
    speed_ax.set_ylabel("m/s")
    speed_ax.grid(alpha=0.2)
    speed_ax.legend(loc="best", fontsize=8)
    figure.tight_layout()
    st.pyplot(figure, use_container_width=True)
    plt.close(figure)


@st.cache_data(show_spinner=False, max_entries=32)
def load_vehicle_gif_bytes(path: str, modified_ns: int) -> bytes:
    return Path(path).read_bytes()


def vehicle_gif_path(item: Mapping[str, Any]) -> Path | None:
    raw_path = str(item.get("gif_path", "")).strip()
    if not raw_path:
        return None
    return resolve_path(raw_path)


def render_vehicle_review() -> None:
    if VEHICLE_REVIEW_SUMMARY_CSV is not None:
        with st.spinner("Loading vehicle description summary..."):
            payload = load_vehicle_summary_csv(
                str(VEHICLE_REVIEW_SUMMARY_CSV),
                VEHICLE_REVIEW_SUMMARY_CSV.stat().st_mtime_ns,
                str(VEHICLE_GIF_DIR),
            )
    elif not VEHICLE_REVIEW_MANIFEST_FILES:
        st.error("No vehicle review manifests were found.")
        return
    else:
        manifest_choices = {
            path.name.replace("_vehicle_review_manifest.json", ""): path
            for path in VEHICLE_REVIEW_MANIFEST_FILES
        }
        if len(manifest_choices) > 1:
            selected_scene = st.sidebar.selectbox(
                "Scene",
                options=tuple(manifest_choices),
                key="vehicle_review_scene_filter",
            )
            selected_manifest_path = manifest_choices[selected_scene]
        else:
            selected_manifest_path = next(iter(manifest_choices.values()))

        manifest_paths = (str(selected_manifest_path),)
        signatures = (
            (str(selected_manifest_path), selected_manifest_path.stat().st_mtime_ns),
        )
        with st.spinner(f"Loading scene {selected_manifest_path.stem}..."):
            payload = load_vehicle_review_manifests(manifest_paths, signatures)
    items = tuple(item for item in payload.get("items", []) if isinstance(item, Mapping))
    edits = load_vehicle_review_edits()
    st.sidebar.header("Vehicle supporting-range review")
    selected_statuses = st.sidebar.multiselect(
        "Review status", options=list(VEHICLE_STATUS_OPTIONS),
        default=list(VEHICLE_STATUS_OPTIONS), key="vehicle_review_status_filter",
    )
    filtered = tuple(item for item in items if edits.get(item.get("review_id", ""), {}).get("status", "unchecked") in selected_statuses)
    st.sidebar.caption(f"Items: {len(items)} · Filtered: {len(filtered)}")
    if not filtered:
        st.info("No vehicle supporting ranges match the current filter.")
        return
    st.title("Vehicle Supporting-Range Review")
    full_scene_raw_path = str(payload.get("full_scene_gif", "")).strip()
    full_scene_path = resolve_path(full_scene_raw_path) if full_scene_raw_path else None
    if full_scene_path is not None and full_scene_path.exists():
        with st.expander("Complete scene overview", expanded=True):
            st.image(
                load_vehicle_gif_bytes(
                    str(full_scene_path), full_scene_path.stat().st_mtime_ns
                ),
                use_container_width=True,
            )
            st.caption(
                "Orange vehicles: speed ≤ 0.1 m/s; blue vehicles: moving; "
                "the title shows valid/stationary/moving counts for each frame."
            )
    options = list(range(len(filtered)))
    current = int(st.session_state.get("vehicle_review_index", 0))
    current = max(0, min(current, len(filtered) - 1))
    selected = st.selectbox(
        "Supporting frame range", options=options, index=current,
        format_func=lambda i: (
            f"{i + 1}: Vehicle {filtered[i].get('vehicle_id')} · "
            f"frames {filtered[i].get('frame_range', {}).get('start_frame')}-"
            f"{filtered[i].get('frame_range', {}).get('end_frame')}"
        ), key=f"vehicle_review_selector_{current}",
    )
    current = int(selected)
    st.session_state.vehicle_review_index = current
    item = filtered[current]
    st.caption(f"Scene: {item.get('scene_id', '')}")
    frame_range = item.get("frame_range", {})
    start = int(frame_range.get("start_frame", 0))
    end = int(frame_range.get("end_frame", start))
    item_num_frames = len(item.get("target_track", {}).get("states", []))
    frame_max = max(end, item_num_frames - 1, 0)
    frame = st.slider(
        "Frame",
        min_value=0,
        max_value=frame_max,
        value=min(start, frame_max),
        key=f"vehicle_frame_{item.get('review_id')}",
    )
    left, right = st.columns([1.6, 1], gap="large")
    with left:
        gif_path = vehicle_gif_path(item)
        if gif_path is not None and gif_path.exists():
            st.image(
                load_vehicle_gif_bytes(str(gif_path), gif_path.stat().st_mtime_ns),
                use_container_width=True,
            )
            st.caption(
                f"Highlighted target: Vehicle {item.get('vehicle_id')} · "
                "other vehicles are visual context only"
            )
        else:
            st.warning(f"Vehicle GIF not found: {gif_path}")
            render_vehicle_scene(item, frame)
    with right:
        st.markdown(f"### Vehicle {item.get('vehicle_id')}")
        st.markdown(f"**Supporting range:** `{start}–{end}`")
        st.markdown(item.get("frame_range", {}).get("description", "") or "—")
        description = item.get("description", {})
        st.markdown("**Description short**")
        st.info(description.get("short", "") or "—")
        st.markdown("**Uncertainty notes**")
        st.write(description.get("uncertainty_notes", "") or "—")
        state = _vehicle_state(item, frame)
        if state:
            st.write({"frame": frame, "heading_deg": state.get("heading_deg"), "speed_mps": state.get("speed_mps")})
        edits_row = edits.get(item.get("review_id", ""), {})
        status = st.selectbox("Review status", options=list(VEHICLE_STATUS_OPTIONS), index=(list(VEHICLE_STATUS_OPTIONS).index(edits_row.get("status", "unchecked")) if edits_row.get("status", "unchecked") in VEHICLE_STATUS_OPTIONS else 0), key=f"vehicle_status_{item.get('review_id')}")
        note = st.text_area("Review note", value=edits_row.get("note", ""), key=f"vehicle_note_{item.get('review_id')}")
        if st.button("Save vehicle review", key=f"save_vehicle_{item.get('review_id')}", use_container_width=True):
            save_vehicle_review_edit(str(item.get("review_id")), status, note)
            st.success("Vehicle review saved.")
    st.subheader("Lane matching / lane transition")
    lane_rows = item.get("lane_matching", [])
    if lane_rows and isinstance(lane_rows[0], Mapping) and "error" not in lane_rows[0]:
        lane_frame = pd.DataFrame(lane_rows)
        st.dataframe(lane_frame[(lane_frame["frame"] >= start) & (lane_frame["frame"] <= end)], use_container_width=True, hide_index=True)
    else:
        st.warning(lane_rows[0].get("error", "No lane matching data") if lane_rows else "No lane matching data")
    st.json(item.get("lane_transition", {}), expanded=False)
    with st.expander("Target-only physical facts", expanded=False):
        st.json(item.get("physical_facts", {}), expanded=False)


def clip_label(row: Mapping[str, Any], index: int) -> str:
    interaction = cell(row, "interaction_id") or cell(row, "clip_id")
    scene = display_scene(cell(row, "scene_id"))
    count = cell(row, "event_count") or "?"
    parts = [part for part in (interaction, scene) if part]
    suffix = " | ".join(parts)
    return f"{index + 1}: {suffix} ({count} events)" if suffix else f"{index + 1}"


def review_status(
    record_id: str,
    edits_by_record: Mapping[str, Mapping[str, str]],
) -> str:
    value = str(edits_by_record.get(record_id, {}).get("status", "")).strip()
    return value if value in STATUS_OPTIONS else "unchecked"


def review_label(
    item: tuple[int, str],
    index: int,
    clip_records: Sequence[Mapping[str, Any]],
    event_by_record: Mapping[str, tuple[int, Mapping[str, Any]]],
    edits_by_record: Mapping[str, Mapping[str, str]],
) -> str:
    clip_index, record_id = item
    event = event_by_record[record_id][1]
    clip = clip_records[clip_index]
    frame = cell(event, "event_frame") or "—"
    behavior = (
        cell(event, "behavior_type")
        or cell(event, "event_type")
        or "event"
    )
    interaction = cell(clip, "interaction_id") or cell(clip, "clip_id")
    return (
        f"{index + 1}: Frame {frame} · {behavior} · "
        f"{interaction} [{review_status(record_id, edits_by_record)}]"
    )


def render_event_descriptions(
    events: Sequence[Mapping[str, Any]],
    edits_by_record: Mapping[str, Mapping[str, str]],
    current_record_id: str,
    next_record_by_id: Mapping[str, str],
) -> None:
    event = next(
        (
            item for item in events
            if edit_record_id(item) == current_record_id
            and cell(item, "description")
        ),
        None,
    )
    if event is None:
        return

    record_id = edit_record_id(event)
    original = cell(event, "description")
    saved = edits_by_record.get(record_id, {})
    text_key = edit_widget_key(record_id)
    status_key = status_widget_key(record_id)
    if text_key not in st.session_state:
        st.session_state[text_key] = (
            saved.get("edited_description", "")
            if saved.get("status") == "edited"
            else original
        )
    if status_key not in st.session_state:
        st.session_state[status_key] = saved.get("status") or "unchecked"

    frame = cell(event, "event_frame") or "—"
    behavior = cell(event, "behavior_type") or cell(event, "event_type") or "event"
    st.markdown("### Description")
    st.markdown(
        f"<div class='event-heading'>Frame {html.escape(frame)} · "
        f"{html.escape(behavior)}</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='original-label'>Original Description</div>"
        f"<div class='original-description'>{html.escape(original)}</div>",
        unsafe_allow_html=True,
    )
    st.text_area(
        "Edited description",
        key=text_key,
        height=140,
        label_visibility="collapsed",
    )
    st.selectbox(
        "Status",
        options=list(STATUS_OPTIONS),
        key=status_key,
        label_visibility="collapsed",
    )
    st.markdown(
        "<div class='shortcut-hint'>"
        "<span>快捷键</span>"
        "<kbd>1</kbd> unchecked"
        "<kbd>2</kbd> correct"
        "<kbd>3</kbd> edited"
        "<kbd>4</kbd> reject"
        "</div>",
        unsafe_allow_html=True,
    )
    status_buttons = st.columns(4)
    for number, status in enumerate(STATUS_OPTIONS, start=1):
        with status_buttons[number - 1]:
            st.button(
                f"{number} {status}",
                key=f"quick_status_{record_id}_{status}",
                use_container_width=True,
                on_click=save_review_status,
                args=(
                    record_id,
                    original,
                    status,
                    next_record_by_id.get(record_id),
                ),
            )
    save_col, next_col = st.columns(2)
    with save_col:
        st.button(
            "Save",
            key=f"save_edit_{record_id}",
            use_container_width=True,
            on_click=save_description_edit,
            args=(record_id, original, None),
        )
    with next_col:
        next_record_id = next_record_by_id.get(record_id)
        st.button(
            "Save & Next",
            key=f"save_next_edit_{record_id}",
            use_container_width=True,
            on_click=save_description_edit,
            args=(record_id, original, next_record_id),
            disabled=next_record_id is None,
        )


def render_scene_summary(events: Sequence[Mapping[str, Any]]) -> None:
    scenes = [
        display_scene(value) for value in unique_values(events, "scene_id")
    ]
    pairs: list[str] = []
    for event in events:
        agent_1 = cell(event, "agent_1")
        agent_2 = cell(event, "agent_2")
        pair = " ↔ ".join(part for part in (agent_1, agent_2) if part)
        if pair and pair not in pairs:
            pairs.append(pair)

    behaviors = unique_values(events, "behavior_type")
    if not behaviors:
        behaviors = unique_values(events, "event_type")
    frames = unique_values(events, "event_frame")

    items = (
        ("Scene", ", ".join(scenes) or "—"),
        ("Agents", " · ".join(pairs) or "—"),
        ("Behavior", " · ".join(behaviors) or "—"),
        ("Event frame", ", ".join(frames) or "—"),
    )
    rows = "".join(
        "<div class='scene-summary-row'>"
        f"<span class='scene-summary-label'>{html.escape(label)}</span>"
        f"<span class='scene-summary-value'>{html.escape(value)}</span>"
        "</div>"
        for label, value in items
    )
    st.markdown(f"<div class='scene-summary'>{rows}</div>", unsafe_allow_html=True)


def render_technical_metadata(
    clip: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
) -> None:
    fields = (
        ("clip_id", "clip_id"),
        ("interaction_id", "interaction_id"),
        ("scene_id", "scene_id"),
        ("gif_path", "gif_path"),
        ("event_count", "event_count"),
    )
    with st.expander("Technical metadata", expanded=False):
        for column, label in fields:
            item = cell(clip, column)
            if item:
                st.markdown(f"**{label}**")
                st.code(item, language=None)

        st.markdown("**Event records**")
        for event in events:
            event_id = cell(event, "event_id") or cell(
                event, "behavior_record_id"
            )
            frame = cell(event, "event_frame") or "—"
            behavior = cell(event, "behavior_type") or cell(
                event, "event_type"
            )
            st.code(
                f"{event_id or '<no event id>'} | frame={frame} | "
                f"behavior={behavior or '<empty>'}",
                language=None,
            )


def render_source_json(
    events: Sequence[Mapping[str, Any]],
    clip_id: str,
) -> None:
    paths = unique_values(events, "source_json_path")
    if not paths:
        return

    with st.expander("Raw / Objective Scene Data", expanded=False):
        st.caption("Raw JSON is loaded only after you click the button.")
        loaded_clip = st.session_state.get("raw_json_loaded_clip")
        if loaded_clip != clip_id:
            if st.button("Load raw data", key=f"load_raw_{clip_id}"):
                st.session_state.raw_json_loaded_clip = clip_id
                loaded_clip = clip_id
        if loaded_clip != clip_id:
            return

        for raw_path in paths:
            path = resolve_path(raw_path)
            st.markdown(f"**{raw_path}**")
            if not path.exists():
                st.warning(f"Source JSON not found: {raw_path}")
                continue
            try:
                payload = load_json_payload(str(path), path.stat().st_mtime_ns)
                st.json(payload)
            except Exception as exc:
                st.warning(f"Could not read source JSON: {raw_path} ({exc})")


def previous_scene() -> None:
    current = int(st.session_state.get("current_review_index", 0))
    st.session_state.current_review_index = max(0, current - 1)


def next_scene(total: int) -> None:
    current = int(st.session_state.get("current_review_index", 0))
    st.session_state.current_review_index = min(total - 1, current + 1)


def inject_keyboard_shortcuts() -> None:
    components.html(
        """
        <script>
        (() => {
          const doc = window.parent.document;
          if (doc.__sceneBrowserKeyboardShortcuts) return;
          doc.__sceneBrowserKeyboardShortcuts = true;
          doc.addEventListener("keydown", (event) => {
            const target = event.target;
            if (target && ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName)) {
              return;
            }
            const key = event.key.toLowerCase();
            const statusLabels = {
              "1": "1 unchecked",
              "2": "2 correct",
              "3": "3 edited",
              "4": "4 reject"
            };
            if (statusLabels[key]) {
              const statusButton = Array.from(
                doc.querySelectorAll("button")
              ).find(
                (item) =>
                  item.innerText.trim() === statusLabels[key] &&
                  !item.disabled
              );
              if (statusButton) {
                event.preventDefault();
                statusButton.click();
              }
              return;
            }
            if (key !== "a" && key !== "d") return;
            const labels = key === "a"
              ? ["← Previous", "Previous"]
              : ["Next →", "Next"];
            const button = Array.from(doc.querySelectorAll("button")).find(
              (item) =>
                labels.includes(item.innerText.trim()) && !item.disabled
            );
            if (button) {
              event.preventDefault();
              button.click();
            }
          });
        })();
        </script>
        """,
        height=0,
        scrolling=False,
    )


def inject_page_styles() -> None:
    st.markdown(
        """
        <style>
        .description-card {
            background: #f7f8fa;
            color: #222;
            border: 1px solid #e4e7eb;
            border-radius: 10px;
            padding: 20px;
            font-size: 19px;
            line-height: 1.65;
            font-weight: 450;
            white-space: normal;
            overflow-wrap: anywhere;
        }
        .original-label {
            margin-top: 0.25rem;
            margin-bottom: 0.35rem;
            color: #6b7280;
            font-size: 0.9rem;
            font-weight: 600;
        }
        .original-description {
            margin-bottom: 0.75rem;
            padding: 12px 14px;
            color: #374151;
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            line-height: 1.55;
            overflow-wrap: anywhere;
        }
        .shortcut-hint {
            display: flex;
            align-items: center;
            gap: 0.42rem;
            margin: 0.15rem 0 0.55rem;
            color: #7a8492;
            font-size: 0.78rem;
            line-height: 1.5;
            white-space: nowrap;
        }
        .shortcut-hint span {
            margin-right: 0.15rem;
            color: #5f6b7a;
            font-weight: 600;
        }
        .shortcut-hint kbd {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 1.25rem;
            height: 1.25rem;
            padding: 0 0.22rem;
            color: #465365;
            background: #eef2f7;
            border: 1px solid #d8e0ea;
            border-radius: 4px;
            font-family: inherit;
            font-size: 0.75rem;
            font-weight: 700;
            box-shadow: inset 0 -1px 0 #cbd5e1;
        }
        .stButton > button {
            min-height: 2.35rem;
            padding: 0.35rem 0.28rem;
            border-radius: 8px;
            border-color: #d7dee8;
            white-space: nowrap;
        }
        .stButton > button p {
            margin: 0;
            white-space: nowrap;
            font-size: 0.84rem;
            line-height: 1.2;
        }
        .event-heading {
            margin-top: 0.75rem;
            margin-bottom: 0.6rem;
            color: #4b5563;
            font-size: 1rem;
            font-weight: 600;
            line-height: 1.45;
            overflow-wrap: anywhere;
        }
        .event-gap {
            height: 1rem;
            border-bottom: 1px solid #eef0f2;
            margin-bottom: 1rem;
        }
        .scene-summary {
            margin-top: 0.25rem;
            margin-bottom: 0.75rem;
        }
        .scene-summary-row {
            display: grid;
            grid-template-columns: 7.5rem minmax(0, 1fr);
            gap: 0.75rem;
            padding: 0.35rem 0;
            border-bottom: 1px solid #eef0f2;
            font-size: 1rem;
            line-height: 1.45;
        }
        .scene-summary-label {
            color: #6b7280;
            font-weight: 500;
        }
        .scene-summary-value {
            color: #30343b;
            overflow-wrap: anywhere;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="Scene Browser", page_icon="🎞️", layout="wide")
    inject_page_styles()
    inject_keyboard_shortcuts()
    if DESCRIPTION_DIR_OVERRIDE is not None and DESCRIPTION_FILE is None:
        st.error(
            "No supported description manifest was found under "
            f"{DESCRIPTION_DIR_OVERRIDE}. Expected vehicle_review_manifest.json "
            "or manifest.csv. Use --description-file to specify one explicitly."
        )
        st.stop()
    if GIF_DIR_OVERRIDE is not None:
        st.sidebar.caption(f"GIF directory: {GIF_DIR_OVERRIDE}")
    if DESCRIPTION_FILE is not None:
        st.sidebar.caption(f"Description manifest: {DESCRIPTION_FILE}")
    if VEHICLE_REVIEW_SUMMARY_CSV is not None or VEHICLE_REVIEW_MANIFEST_FILES:
        if VEHICLE_REVIEW_SUMMARY_CSV is not None:
            st.session_state["review_mode"] = "Vehicle supporting ranges"
        review_mode = st.sidebar.radio(
            "Review mode",
            options=("Interaction / GIF", "Vehicle supporting ranges"),
            index=1 if VEHICLE_REVIEW_SUMMARY_CSV is not None else 0,
            key="review_mode",
        )
        if review_mode == "Vehicle supporting ranges":
            render_vehicle_review()
            return
    render_t0 = time.perf_counter()

    try:
        manifest_modified_ns = MANIFEST_PATH.stat().st_mtime_ns
        clip_manifest_modified_ns = (
            CLIP_MANIFEST_PATH.stat().st_mtime_ns
            if CLIP_MANIFEST_PATH.exists()
            else -1
        )
        (
            clip_records,
            events_by_clip,
            interaction_ids,
            interaction_to_indices,
            review_sequence,
            review_positions,
            event_by_record,
        ) = prepare_viewer_index(
            str(MANIFEST_PATH),
            manifest_modified_ns,
            str(CLIP_MANIFEST_PATH),
            clip_manifest_modified_ns,
        )
    except Exception as exc:
        st.error(f"Could not load viewer data from {DATA_DIR}")
        st.exception(exc)
        st.stop()

    edit_modified_ns = (
        EDITS_PATH.stat().st_mtime_ns if EDITS_PATH.exists() else -1
    )
    edits_by_record = edits_to_map(
        load_description_edits(str(EDITS_PATH), edit_modified_ns)
    )
    edits_by_record.update(
        st.session_state.get("_description_edit_map", {})
    )

    with st.sidebar:
        st.header("Review / Export")
        selected_statuses = st.multiselect(
            "Review Status Filter",
            options=list(STATUS_OPTIONS),
            default=list(STATUS_OPTIONS),
            key="review_status_filter",
        )
        status_counts = {
            status: sum(
                review_status(record_id, edits_by_record) == status
                for _, record_id in review_sequence
            )
            for status in STATUS_OPTIONS
        }
        st.markdown("**Review Status**")
        st.caption("Keyboard: A = Previous · D = Next")
        st.write(f"All: {len(review_sequence)}")
        st.write(f"Unchecked: {status_counts['unchecked']}")
        st.write(f"Correct: {status_counts['correct']}")
        st.write(f"Edited: {status_counts['edited']}")
        st.write(f"Reject: {status_counts['reject']}")

        keep_unchecked = st.checkbox(
            "Keep unchecked descriptions",
            value=True,
            help="If disabled, unchecked records are omitted from the corrected export.",
        )
        if st.button(
            "Export Corrected Dataset",
            use_container_width=True,
        ):
            all_events = tuple(
                event
                for clip_events in events_by_clip.values()
                for event in clip_events
            )
            export_bytes = build_corrected_dataset(
                {"all": all_events},
                edits_by_record,
                keep_unchecked,
            )
            CORRECTED_EXPORT_PATH.write_bytes(export_bytes)
            st.session_state.corrected_export_bytes = export_bytes
            st.success(f"Exported {CORRECTED_EXPORT_PATH.name}")
        export_bytes = st.session_state.get("corrected_export_bytes")
        if export_bytes:
            st.download_button(
                "Download corrected_descriptions.csv",
                data=export_bytes,
                file_name="corrected_descriptions.csv",
                mime="text/csv",
                use_container_width=True,
            )

    selected_statuses = tuple(selected_statuses)
    previous_filter = tuple(
        st.session_state.get(
            "_previous_review_status_filter", STATUS_OPTIONS
        )
    )
    if selected_statuses != previous_filter:
        st.session_state["_previous_review_status_filter"] = selected_statuses
        st.session_state.current_review_index = 0
        st.session_state.pop("_pending_review_record_id", None)

    filtered_sequence = tuple(
        item
        for item in review_sequence
        if review_status(item[1], edits_by_record) in selected_statuses
    )
    st.sidebar.caption(f"Filtered: {len(filtered_sequence)}")

    notice = st.session_state.pop("_edit_notice", None)
    if notice:
        st.success(notice)

    title_col, progress_col = st.columns([5, 1])
    with title_col:
        st.title("Scene Browser")

    if not filtered_sequence:
        with progress_col:
            st.markdown(
                "<div style='text-align:right; padding-top:1.2rem; "
                "font-size:1.1rem'>0 / 0 records</div>",
                unsafe_allow_html=True,
            )
        st.info("No records match the current filter.")
        st.stop()

    pending_record_id = st.session_state.pop(
        "_pending_review_record_id", None
    )
    filtered_positions = {
        record_id: index
        for index, (_, record_id) in enumerate(filtered_sequence)
    }
    if pending_record_id in filtered_positions:
        current = filtered_positions[pending_record_id]
    else:
        current = int(st.session_state.get("current_review_index", 0))
        current = max(0, min(current, len(filtered_sequence) - 1))
    st.session_state.current_review_index = current

    with progress_col:
        st.markdown(
            f"<div style='text-align:right; padding-top:1.2rem; "
            f"font-size:1.1rem'>{current + 1} / "
            f"{len(filtered_sequence)} records</div>",
            unsafe_allow_html=True,
        )

    query = st.text_input(
        "Search interaction_id",
        key="interaction_search",
        placeholder="Enter an interaction_id or part of it",
    )
    previous_query = st.session_state.get("_previous_search", "")
    if query != previous_query:
        st.session_state["_previous_search"] = query
        normalized_query = query.strip().casefold()
        if normalized_query:
            matches = [
                index
                for index, (clip_index, _) in enumerate(filtered_sequence)
                if normalized_query in (
                    cell(clip_records[clip_index], "interaction_id")
                    or cell(clip_records[clip_index], "clip_id")
                ).casefold()
            ]
            if matches:
                current = matches[0]
                st.session_state.current_review_index = current
                st.caption(f"{len(matches)} matching filtered record(s)")
            else:
                st.warning(f"No interaction_id matches: {query}")

    filter_key = "-".join(selected_statuses) or "none"
    selector_key = f"review_record_selector_{filter_key}_{current}"
    selected = st.selectbox(
        "Select filtered event record",
        options=list(range(len(filtered_sequence))),
        index=current,
        format_func=lambda index: review_label(
            filtered_sequence[index],
            index,
            clip_records,
            event_by_record,
            edits_by_record,
        ),
        key=selector_key,
    )
    if selected != current:
        current = int(selected)
        st.session_state.current_review_index = current

    clip_index, current_record_id = filtered_sequence[current]
    clip = clip_records[clip_index]
    clip_id = cell(clip, "clip_id")
    events = events_by_clip.get(clip_id, ())
    next_record_by_id = {
        record_id: filtered_sequence[index + 1][1]
        for index, (_, record_id) in enumerate(filtered_sequence[:-1])
    }

    LOGGER.debug(
        "total render preparation: review_index=%d filtered=%d events=%d elapsed=%.3fs",
        current,
        len(filtered_sequence),
        len(events),
        time.perf_counter() - render_t0,
    )

    left, right = st.columns([1.45, 1], gap="large")
    with left:
        gif_value = cell(clip, "gif_path")
        gif_path = resolve_path(gif_value) if gif_value else None
        if gif_path is None or not gif_path.exists():
            st.warning(f"GIF not found: {gif_value or '<empty path>'}")
        else:
            try:
                gif_bytes = load_gif_bytes(
                    str(gif_path), gif_path.stat().st_mtime_ns
                )
                st.image(gif_bytes, use_container_width=True)
            except OSError:
                st.warning(f"GIF not found: {gif_value or '<empty path>'}")

    with right:
        render_event_descriptions(
            events,
            edits_by_record,
            current_record_id,
            next_record_by_id,
        )
        st.markdown("### Scene Information")
        render_scene_summary(events)
        render_technical_metadata(clip, events)
        render_source_json(events, clip_id)

    st.divider()
    previous_col, next_col = st.columns(2)
    with previous_col:
        st.button(
            "← Previous",
            disabled=current <= 0,
            use_container_width=True,
            on_click=previous_scene,
        )
    with next_col:
        st.button(
            "Next →",
            disabled=current >= len(filtered_sequence) - 1,
            use_container_width=True,
            on_click=next_scene,
            args=(len(filtered_sequence),),
        )


if __name__ == "__main__":
    main()
