"""Compact Streamlit reviewer for WOMD vehicle descriptions."""
from __future__ import annotations

import base64
import argparse
import io
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


VIEWER_DIR = Path(__file__).resolve().parent


def _parse_runtime_paths() -> argparse.Namespace:
    """Read app-specific arguments passed after ``streamlit run ... --``."""
    argv = list(sys.argv[1:])
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gif-dir", type=Path, default=None)
    parser.add_argument("--description-file", type=Path, default=None)
    parser.add_argument("--facts-dir", type=Path, default=None)
    options, _ = parser.parse_known_args(argv)
    return options


RUNTIME_PATHS = _parse_runtime_paths()


def _runtime_path(path: Path | None) -> Path | None:
    return path.expanduser().resolve() if path is not None else None


DEFAULT_GIF_ROOT = VIEWER_DIR / "womd_vehicle_gifs" / "training-00900"
DEFAULT_VALIDATION_ROOT = VIEWER_DIR.parent / "validation" / "vehicle_scene_10_5"
DEFAULT_DESCRIPTION_CSV = (
    DEFAULT_VALIDATION_ROOT / "qwen_descriptions" / "vehicle_qwen_descriptions_summary.csv"
)
DEFAULT_FACTS_JSON_DIR = DEFAULT_VALIDATION_ROOT / "full_vehicle_facts" / "vehicles"
GIF_ROOT = _runtime_path(RUNTIME_PATHS.gif_dir) or DEFAULT_GIF_ROOT
DESCRIPTION_CSV = _runtime_path(RUNTIME_PATHS.description_file) or DEFAULT_DESCRIPTION_CSV
FACTS_JSON_DIR = _runtime_path(RUNTIME_PATHS.facts_dir) or DEFAULT_FACTS_JSON_DIR
REVIEW_PATH = VIEWER_DIR / "data" / "vehicle_reviews.csv"

REVIEW_COLUMNS = ("record_key", "scene_id", "vehicle_id", "status", "error_types", "edited_description", "updated_at")
REVIEW_STATUSES = ("unreviewed", "correct", "wrong", "unsure", "edited")
ERROR_TYPES = ("missed turn", "wrong turn direction", "false turn", "wrong lane change", "wrong speed", "wrong timing", "bad wording", "other")
CAPTION_CATEGORIES = ("Turn", "Lane change", "Accelerate", "Slow", "Forward", "Stationary", "Insufficient observation")
GENERATION_MODES = ("qwen", "stationary_template", "insufficient_observation")


def record_key(scene_id: str, vehicle_id: str) -> str:
    return f"{scene_id}::vehicle_{vehicle_id}"


def parse_json_cell(value: Any, default: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    if isinstance(value, (list, dict)):
        return value
    text = str(value).strip()
    if not text:
        return default
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return default


@st.cache_data(show_spinner=False)
def load_descriptions(path: str, modified_ns: int) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    for column in frame.columns:
        frame[column] = frame[column].fillna("").astype(str)
    frame["record_key"] = [record_key(scene, vehicle) for scene, vehicle in zip(frame["scene_id"], frame["vehicle_id"])]
    return frame.drop_duplicates("record_key", keep="last")


@st.cache_data(show_spinner=False)
def load_reviews(path: str, modified_ns: int) -> pd.DataFrame:
    if modified_ns < 0 or not Path(path).exists():
        return pd.DataFrame(columns=REVIEW_COLUMNS)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    for column in REVIEW_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    return frame[list(REVIEW_COLUMNS)].fillna("").astype(str)


@st.cache_data(show_spinner=False)
def load_facts(path: str, modified_ns: int) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    directory = Path(path)
    if not directory.exists():
        return result
    for file_path in directory.glob("*.json"):
        try:
            value = json.loads(file_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(value, dict) and value.get("scene_id") and value.get("vehicle_id") is not None:
            result[record_key(str(value["scene_id"]), str(value["vehicle_id"]))] = value
    return result


def review_map(frame: pd.DataFrame) -> dict[str, dict[str, str]]:
    return {str(row["record_key"]): {column: str(row.get(column, "")) for column in REVIEW_COLUMNS} for row in frame.to_dict(orient="records") if str(row.get("record_key", "")).strip()}


def save_review(row: dict[str, str]) -> None:
    REVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(REVIEW_PATH, dtype=str, keep_default_na=False) if REVIEW_PATH.exists() else pd.DataFrame(columns=REVIEW_COLUMNS)
    for column in REVIEW_COLUMNS:
        if column not in old.columns:
            old[column] = ""
    rows = old[list(REVIEW_COLUMNS)].fillna("").astype(str).to_dict(orient="records")
    rows = [item for item in rows if item.get("record_key") != row["record_key"]]
    rows.append(row)
    temp = REVIEW_PATH.with_suffix(".tmp.csv")
    pd.DataFrame(rows, columns=REVIEW_COLUMNS).to_csv(temp, index=False, encoding="utf-8-sig")
    temp.replace(REVIEW_PATH)
    load_reviews.clear()


def text_for_filter(row: dict[str, Any]) -> str:
    return " ".join(str(row.get(column, "")) for column in ("description_short", "description_detailed")).lower()


def has_caption_category(text: str, category: str) -> bool:
    patterns = {
        "Turn": r"\bturn(?:s|ed|ing)?\b",
        "Lane change": r"\b(?:change|changes|changing)\s+lanes?\b|\blane\s+change\b",
        "Accelerate": r"\baccelerat(?:e|es|ed|ing)\b",
        "Slow": r"\b(?:slow|slows|slowed|slowing|decelerate|decelerates|deceleration)\b",
        "Forward": r"\bforward\b|\bcontinue(?:s|d|ing)?\s+straight\b",
        "Stationary": r"\bstationary\b",
        "Insufficient observation": r"\binsufficient\s+observation\b",
    }
    return bool(re.search(patterns[category], text, re.IGNORECASE))


def fact_for(row: dict[str, Any], facts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return facts.get(str(row.get("record_key", "")), {})


def heading_facts(fact: dict[str, Any]) -> dict[str, Any]:
    vehicle = fact.get("vehicle", {})
    return vehicle.get("heading_motion", {}) if isinstance(vehicle, dict) else {}


def max_heading_change(fact: dict[str, Any]) -> float:
    heading = heading_facts(fact)
    episodes = heading.get("turning_episodes", [])
    values = [abs(float(item.get("net_heading_change_deg", 0))) for item in episodes if isinstance(item, dict) and item.get("net_heading_change_deg") is not None] if isinstance(episodes, list) else []
    if not values and heading.get("net_heading_change_deg") is not None:
        values.append(abs(float(heading["net_heading_change_deg"])))
    return max(values, default=0.0)


def gif_path(row: dict[str, Any]) -> Path | None:
    scene, vehicle = str(row.get("scene_id", "")), str(row.get("vehicle_id", ""))
    candidates = [GIF_ROOT / scene / f"{scene}__vehicle_{vehicle}.gif", GIF_ROOT / scene / "stationary" / f"{scene}__vehicle_{vehicle}.gif"]
    return next((path for path in candidates if path.exists()), None)


@st.cache_data(show_spinner=False)
def gif_bytes(path: str, start: int | None, end: int | None) -> bytes:
    source = Path(path)
    if start is None or end is None:
        return source.read_bytes()
    try:
        from PIL import Image, ImageSequence
        frames, durations = [], []
        with Image.open(source) as image:
            for index, frame in enumerate(ImageSequence.Iterator(image)):
                if not start <= index <= end:
                    continue
                current = frame.copy()
                frames.append(current if current.mode == "P" else current.convert("P"))
                durations.append(int(frame.info.get("duration", image.info.get("duration", 100))))
        if not frames:
            return source.read_bytes()
        output = io.BytesIO()
        frames[0].save(output, format="GIF", save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=False, disposal=2)
        return output.getvalue()
    except Exception:
        return source.read_bytes()


@st.cache_data(show_spinner=False)
def gif_data_uri(path: str, start: int | None, end: int | None) -> str:
    return base64.b64encode(gif_bytes(path, start, end)).decode("ascii")


@st.cache_resource
def segment_prefetch_executor() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=1, thread_name_prefix="vehicle-gif-prefetch")


@st.cache_resource
def segment_prefetch_keys() -> set[tuple[str, int, int]]:
    return set()


def prefetch_segments(path: Path | None, segments: list[dict[str, Any]]) -> None:
    if path is None:
        return
    pending, executor = segment_prefetch_keys(), segment_prefetch_executor()
    for segment in segments:
        try:
            start, end = int(segment["start_frame"]), int(segment["end_frame"])
        except (KeyError, TypeError, ValueError):
            continue
        key = (str(path), start, end)
        if key not in pending:
            pending.add(key)
            executor.submit(gif_bytes, str(path), start, end)


def render_gif(path: Path | None, start: int | None = None, end: int | None = None) -> None:
    if path is None:
        st.warning("GIF not found for this scene/vehicle.")
        return
    payload = gif_data_uri(str(path), start, end)
    label = "Full trajectory" if start is None else f"Frames {start}–{end}"
    st.markdown(f'<div class="gif-box"><div class="gif-label">{label}</div><img src="data:image/gif;base64,{payload}" alt="vehicle trajectory GIF" /></div>', unsafe_allow_html=True)


def rerun() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    st.experimental_rerun()


def main() -> None:
    st.set_page_config(page_title="Vehicle Caption Review", layout="wide")
    st.markdown("""
    <style>
    .block-container {width:100%; max-width:1900px; min-height:calc(100vh - .5rem); padding:2.25rem .65rem 1rem; margin:0 auto;}
    [data-testid="stAppViewContainer"] {overflow-y:auto !important; overflow-x:hidden;}
    [data-testid="stMain"] {overflow:visible;}
    [data-testid="stExpander"] {margin-top:0; overflow:visible;} [data-testid="stExpander"] summary {min-height:2rem; padding:.35rem .7rem;}
    .top-line {display:flex; justify-content:space-between; align-items:center; gap:1rem; margin-bottom:0;}
    .app-name {font-size:.94rem; font-weight:700; color:#334155;} .record-line {font-size:1.12rem; font-weight:700; color:#0f172a;}
    .status-badge {display:inline-block; padding:.2rem .55rem; border-radius:999px; font-size:.78rem; font-weight:700; background:#e2e8f0; color:#475569;}
    .status-correct {background:#dcfce7; color:#166534;} .status-wrong {background:#fee2e2; color:#991b1b;} .status-unsure {background:#fef3c7; color:#92400e;} .status-edited {background:#dbeafe; color:#1d4ed8;}
    .gif-box {height:min(calc(100vh - 280px),760px); min-height:320px; box-sizing:border-box; background:#111827; border-radius:8px; padding:4px; text-align:center; display:flex; flex-direction:column; justify-content:center;}
    .gif-box img {display:block; width:auto; height:auto; max-width:100%; max-height:min(calc(100vh - 340px),700px); margin:0 auto; object-fit:contain;} .gif-label {color:#cbd5e1; font-size:.72rem; line-height:1; margin-bottom:2px;}
    .caption-card {border:1px solid #e2e8f0; border-radius:8px; padding:.55rem .7rem; background:#fff; margin-bottom:.45rem;} .section-label {font-size:.88rem; font-weight:700; color:#334155; margin:.22rem 0 .2rem;}
    .short-caption {font-size:1.05rem; font-weight:700; line-height:1.25; color:#0f172a;} .detailed-caption {font-size:.84rem; line-height:1.3; color:#64748b; margin-top:.25rem;} .playback-line {font-size:.78rem; color:#64748b; margin:.18rem 0 .25rem;} .fact-line {font-size:.82rem; line-height:1.25; margin:.14rem 0;}
    div[data-testid="column"]:has(.right-column-marker), div[data-testid="stColumn"]:has(.right-column-marker) {max-height:calc(100vh - 150px); overflow-y:auto; overflow-x:hidden; padding-right:.35rem;}
    div[data-testid="column"]:has(.right-column-marker) button, div[data-testid="stColumn"]:has(.right-column-marker) button {white-space:nowrap; font-size:.78rem; padding:.18rem .25rem; min-height:2rem; line-height:1.1;}
    </style>
    """, unsafe_allow_html=True)
    components.html("""
    <script>
    (() => { if (window.parent.__vehicleReviewKeyboardInstalled) return; window.parent.__vehicleReviewKeyboardInstalled = true;
      const handler = (event) => { const tag=(event.target&&event.target.tagName||'').toUpperCase();
        if (event.key === 'Enter' && tag !== 'INPUT' && tag !== 'TEXTAREA') { const save=Array.from(window.parent.document.querySelectorAll('button')).find(x=>(x.innerText||'').includes('Save & Next')); if(save&&!save.disabled){event.preventDefault();save.click();} return; }
        if(tag==='INPUT'||tag==='TEXTAREA'||tag==='SELECT') return; const key=event.key.toLowerCase(); const wanted=event.key==='ArrowLeft'||key==='a'?'Previous':event.key==='ArrowRight'||key==='d'?'Next':event.key==='1'?'Correct':event.key==='2'?'Wrong':event.key==='3'?'Unsure':event.key==='4'?'Edit':null; if(!wanted)return; const button=Array.from(window.parent.document.querySelectorAll('button')).find(x=>(x.innerText||'').includes(wanted)); if(button&&!button.disabled){event.preventDefault();button.click();}
      }; window.parent.addEventListener('keydown',handler); })();
    </script>
    """, height=0, scrolling=False)

    if not DESCRIPTION_CSV.exists():
        st.error(f"Description CSV not found: {DESCRIPTION_CSV}"); return
    descriptions = load_descriptions(str(DESCRIPTION_CSV), DESCRIPTION_CSV.stat().st_mtime_ns)
    reviews = review_map(load_reviews(str(REVIEW_PATH), REVIEW_PATH.stat().st_mtime_ns if REVIEW_PATH.exists() else -1))
    facts = load_facts(str(FACTS_JSON_DIR), FACTS_JSON_DIR.stat().st_mtime_ns if FACTS_JSON_DIR.exists() else -1)
    with st.expander("Filters", expanded=False):
        f1, f2, f3, f4 = st.columns(4)
        with f1: caption_filter=st.selectbox("Caption", ["All", *CAPTION_CATEGORIES]); review_filter=st.selectbox("Review", ["All", "Unreviewed", "Correct", "Wrong", "Unsure", "Edited"])
        with f2: caption_exclude=st.multiselect("Exclude caption", list(CAPTION_CATEGORIES)); generation_exclude=st.multiselect("Exclude generation mode", list(GENERATION_MODES))
        with f3: error_filter=st.multiselect("Error", list(ERROR_TYPES)); missed_turn=st.checkbox("Potential missed turn candidates")
        with f4: heading_threshold=st.slider("Heading threshold (deg)",5,60,15) if missed_turn else 15; st.caption("Filters focus the queue.")

    filtered=[]
    for row in descriptions.to_dict(orient="records"):
        text=text_for_filter(row); review_item=reviews.get(row["record_key"],{}); fact=facts.get(row["record_key"],{})
        if caption_filter!="All" and not has_caption_category(text,caption_filter): continue
        if any(has_caption_category(text,c) for c in caption_exclude): continue
        if str(row.get("generation_mode","")) in generation_exclude: continue
        if review_filter!="All" and review_item.get("status","unreviewed")!=review_filter.lower(): continue
        if error_filter and not any(x in review_item.get("error_types","").split("|") for x in error_filter): continue
        if missed_turn:
            context=fact.get("vehicle",{}).get("map_context",fact.get("map_context",{}))
            if has_caption_category(text,"Turn") or not context.get("near_intersection",False) or max_heading_change(fact)<heading_threshold: continue
        filtered.append(row)
    if not filtered: st.warning("No records match the current filters."); return
    keys=[row["record_key"] for row in filtered]; current_key=st.session_state.get("current_record_key")
    if current_key not in keys: current_key=keys[0]; st.session_state["current_record_key"]=current_key
    index=keys.index(current_key); row=filtered[index]; fact=facts.get(current_key,{}); review=reviews.get(current_key,{})
    status=review.get("status","unreviewed") or "unreviewed"; status_class=f" status-{status}" if status in REVIEW_STATUSES else ""
    st.markdown(f'<div class="top-line"><div><div class="app-name">Vehicle Caption Review</div><div class="record-line">Scene: {row["scene_id"]} · Vehicle: {row["vehicle_id"]} <span style="font-size:.92rem;color:#64748b;font-weight:500">[{index+1} / {len(filtered)}]</span></div></div><span class="status-badge{status_class}">{status.capitalize()}</span></div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:.76rem;color:#94a3b8;margin:.05rem 0 .35rem">A/← Previous · D/→ Next · 1 Correct · 2 Wrong · 3 Unsure · 4 Edit</div>', unsafe_allow_html=True)

    left,right=st.columns([1.5,1.0],gap="large"); segment_choice=st.session_state.get("segment_choice","full"); segments=parse_json_cell(row.get("behavior_segments"),[]); segments=segments if isinstance(segments,list) else []; playback_label="Full trajectory" if segment_choice=="full" else f"Frames {segment_choice[0]}-{segment_choice[1]}"
    with left:
        path=gif_path(row); render_gif(path) if segment_choice=="full" else render_gif(path,int(segment_choice[0]),int(segment_choice[1]));
        if segment_choice=="full": prefetch_segments(path,segments)
    with right:
        st.markdown('<div class="right-column-marker"></div>',unsafe_allow_html=True); st.markdown('<div class="section-label">Caption</div>',unsafe_allow_html=True)
        st.markdown(f'<div class="caption-card"><div class="short-caption">{row.get("description_short","")}</div><div class="detailed-caption">{row.get("description_detailed","")}</div></div>',unsafe_allow_html=True)
        st.markdown('<div class="section-label">Behavior segments</div>',unsafe_allow_html=True)
        if not segments: st.caption("No behavior segments")
        for i,seg in enumerate(segments):
            start,end=seg.get("start_frame"),seg.get("end_frame"); selected=segment_choice!="full" and (start,end)==tuple(segment_choice); label=f"{'▶ ' if selected else ''}[{start}–{end}]  {seg.get('description','')}"
            if st.button(label,key=f"segment_{current_key}_{i}",use_container_width=True): st.session_state["segment_choice"]=(start,end); rerun()
        with st.expander("Facts",expanded=False):
            vehicle=fact.get("vehicle",{}) if isinstance(fact,dict) else {}; heading=heading_facts(fact); episodes=heading.get("turning_episodes",[]) if isinstance(heading.get("turning_episodes",[]),list) else []; speed=vehicle.get("speed_change",{}) if isinstance(vehicle,dict) else {}; speed_eps=speed.get("episodes",[]) if isinstance(speed,dict) else []
            if speed_eps: st.markdown(f'<div class="fact-line"><b>Speed:</b> {len(speed_eps)} episodes</div>',unsafe_allow_html=True)
            if heading: st.markdown(f'<div class="fact-line"><b>Heading:</b> {heading.get("heading_change_direction",heading.get("direction","unknown"))}; net {heading.get("net_heading_change_deg","n/a")}°</div>',unsafe_allow_html=True)
            chain=vehicle.get("physical_lane_chain",{})
            if chain: st.markdown(f'<div class="fact-line"><b>Lane:</b> <code>{chain.get("behavior","unknown")}</code>; evidence {len(chain.get("lane_change_evidence",[])) if isinstance(chain.get("lane_change_evidence",[]),list) else "n/a"}</div>',unsafe_allow_html=True)
            context=vehicle.get("map_context",fact.get("map_context",{})); route=vehicle.get("route_transition",{})
            if context or route: st.markdown(f'<div class="fact-line"><b>Map:</b> near intersection={context.get("near_intersection","unknown")}; route transition={bool(route)}</div>',unsafe_allow_html=True)

    def review_controls() -> None:
        wrong_open=st.session_state.get("wrong_open_for")==current_key; edit_open=st.session_state.get("edit_open_for")==current_key; selected_errors=[]; edited_text=review.get("edited_description","")
        if wrong_open:
            selected_errors=st.multiselect("Error type",list(ERROR_TYPES),default=review.get("error_types","").split("|") if review.get("error_types") else [],key=f"errors_{current_key}")
        if edit_open:
            edited_text=st.text_area("Edited caption",value=review.get("edited_description","") or row.get("description_short",""),key=f"edit_{current_key}",height=70)
        next_key=keys[min(index+1,len(keys)-1)]; prev_key=keys[max(index-1,0)]
        def commit(review_status:str,error_types:list[str]|None=None,edited:str="") -> None:
            save_review({"record_key":current_key,"scene_id":row["scene_id"],"vehicle_id":row["vehicle_id"],"status":review_status,"error_types":"|".join(error_types or []),"edited_description":edited if review_status=="edited" else review.get("edited_description",""),"updated_at":datetime.now(timezone.utc).isoformat(timespec="seconds")}); st.session_state["current_record_key"]=next_key; st.session_state["segment_choice"]="full"; st.session_state.pop("wrong_open_for",None); st.session_state.pop("edit_open_for",None); rerun()
        st.markdown(f'<div class="playback-line">Playing: <b>{playback_label}</b></div>',unsafe_allow_html=True)
        if st.button("Full trajectory",key=f"full_{current_key}"): st.session_state["segment_choice"]="full"; rerun()
        actions=st.columns([1.3,1,1,1,1,1.3])
        if actions[0].button("Previous",use_container_width=True,disabled=index==0,key="review_previous"): st.session_state["current_record_key"]=prev_key; st.session_state["segment_choice"]="full"; rerun()
        if actions[1].button("Correct",use_container_width=True,key="review_correct"): commit("correct")
        if actions[2].button("Wrong",use_container_width=True,key="review_wrong"): st.session_state["wrong_open_for"]=current_key; st.session_state.pop("edit_open_for",None); rerun()
        if actions[3].button("Unsure",use_container_width=True,key="review_unsure"): commit("unsure")
        if actions[4].button("Edit",use_container_width=True,key="review_edit"): st.session_state["edit_open_for"]=current_key; st.session_state.pop("wrong_open_for",None); rerun()
        if actions[5].button("Next",use_container_width=True,disabled=index==len(keys)-1,key="review_next"): st.session_state["current_record_key"]=next_key; st.session_state["segment_choice"]="full"; rerun()
        if wrong_open:
            if st.button("Save & Next",use_container_width=True,key="save_wrong"): commit("wrong",list(selected_errors))
            if st.button("Cancel",key="cancel_wrong"): st.session_state.pop("wrong_open_for",None); rerun()
        if edit_open:
            if st.button("Save edit & next",use_container_width=True,key="save_edit"): commit("edited",[],edited_text)
            if st.button("Cancel",key="cancel_edit"): st.session_state.pop("edit_open_for",None); rerun()
    with right: review_controls()


if __name__ == "__main__":
    main()
