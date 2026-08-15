#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate Qwen descriptions directly from full physical facts.

This runner consumes ``full_llm_facts_v1`` without a second fact compression
step. It describes observable vehicle motion only; it does not infer behavior,
intention, causality, priority, or turn direction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from openai import OpenAI


MODEL_NAME = os.getenv("QWEN_MODEL", "qwen-3.6")
BASE_URL = os.getenv("QWEN_BASE_URL", "http://172.17.0.1:60200/v1")
API_KEY = os.getenv("QWEN_API_KEY", "EMPTY")
TEMPERATURE = 0.2
MAX_TOKENS = 1000
MAX_RETRIES = 3
RETRY_WAIT_SECONDS = 3
EXPECTED_INPUT_SCHEMA_VERSION = "full_llm_facts_v1"
OUTPUT_SCHEMA_VERSION = "pair_qwen_description_v1"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


PROMPT_PLACEHOLDER = "{{FULL_LLM_FACTS_JSON}}"
USER_QUERY = "Return the requested JSON description now."


def default_prompt_path() -> Path:
    return (
        Path(__file__).resolve().parent
        / "prompt"
        / "qwen_generate_interaction_descriptions_prompt.txt"
    )


def load_prompt(path: Path) -> str:
    value = path.read_text(encoding="utf-8").strip()
    if not value:
        raise ValueError(f"Prompt file is empty: {path}")
    if PROMPT_PLACEHOLDER not in value:
        raise ValueError(
            f"Prompt file must contain the placeholder {PROMPT_PLACEHOLDER}: {path}"
        )
    return value


def prompt_hash(template: str) -> str:
    return hashlib.sha256(template.encode("utf-8")).hexdigest()


def prompt_input(data: Mapping[str, Any]) -> Dict[str, Any]:
    if data.get("schema_version") != EXPECTED_INPUT_SCHEMA_VERSION:
        raise ValueError(
            "Expected schema_version={!r}, got {!r}".format(
                EXPECTED_INPUT_SCHEMA_VERSION, data.get("schema_version")
            )
        )
    return dict(data)


def build_prompt(data: Mapping[str, Any], template: str) -> str:
    facts = prompt_input(data)
    return template.replace(
        PROMPT_PLACEHOLDER,
        json.dumps(facts, ensure_ascii=False, indent=2),
    )


def build_messages(prompt: str) -> List[Dict[str, str]]:
    # The Qwen endpoint requires at least one user message even when the
    # complete task prompt is supplied as the system message.
    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": USER_QUERY},
    ]


def clean_json(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_caption_result(result: Mapping[str, Any], num_frames: Any) -> Dict[str, Any]:
    short = result.get("description_short")
    detailed = result.get("description_detailed")
    ranges = result.get("supporting_frame_ranges")
    notes = result.get("uncertainty_notes")
    if not isinstance(short, str) or not short.strip():
        raise ValueError("description_short must be a non-empty string")
    if not isinstance(detailed, str) or not detailed.strip():
        raise ValueError("description_detailed must be a non-empty string")
    if not isinstance(ranges, list):
        raise ValueError("supporting_frame_ranges must be a list")
    if not isinstance(notes, str):
        raise ValueError("uncertainty_notes must be a string")
    if not _is_int(num_frames) or num_frames <= 0:
        raise ValueError("input timeline.num_frames must be a positive integer")

    normalized_ranges = []
    for index, item in enumerate(ranges):
        if not isinstance(item, Mapping):
            raise ValueError(f"supporting_frame_ranges[{index}] must be an object")
        start = item.get("start_frame")
        end = item.get("end_frame")
        description = item.get("description")
        if not _is_int(start) or not _is_int(end):
            raise ValueError(f"supporting_frame_ranges[{index}] frame values must be integers")
        if not (0 <= start <= end < num_frames):
            raise ValueError(f"supporting_frame_ranges[{index}] is outside timeline")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"supporting_frame_ranges[{index}].description must be non-empty")
        normalized_ranges.append({
            "start_frame": start,
            "end_frame": end,
            "description": description.strip(),
        })
    return {
        "description_short": short.strip(),
        "description_detailed": detailed.strip(),
        "supporting_frame_ranges": normalized_ranges,
        "uncertainty_notes": notes.strip(),
    }


def call_qwen(data: Mapping[str, Any], template: str) -> Dict[str, Any]:
    prompt = build_prompt(data, template)
    num_frames = data.get("timeline", {}).get("num_frames")
    last_error: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=build_messages(prompt),
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            message = response.choices[0].message
            content = message.content
            if not content:
                reasoning = getattr(message, "reasoning", None)
                suffix = "; Qwen reasoning was returned" if reasoning else ""
                raise ValueError(f"Qwen returned empty content{suffix}")
            parsed = json.loads(clean_json(content))
            if not isinstance(parsed, Mapping):
                raise ValueError("Qwen output is not a JSON object")
            return validate_caption_result(parsed, num_frames)
        except Exception as exc:
            last_error = exc
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_WAIT_SECONDS * attempt)
    raise RuntimeError(f"Qwen request failed after {MAX_RETRIES} attempts: {last_error}")


def _safe_component(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._") or "unknown"


def output_name(data: Mapping[str, Any]) -> str:
    return "{}__A_{}__B_{}_qwen_description.json".format(
        _safe_component(data.get("scene_id")),
        _safe_component(data.get("agent_A_id")),
        _safe_component(data.get("agent_B_id")),
    )


def build_output(
    data: Mapping[str, Any],
    caption: Mapping[str, Any],
    prompt_file_name: str,
    prompt_sha256: str,
) -> Dict[str, Any]:
    return {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "scene_id": data.get("scene_id"),
        "agent_A_id": data.get("agent_A_id"),
        "agent_B_id": data.get("agent_B_id"),
        "description_short": caption["description_short"],
        "description_detailed": caption["description_detailed"],
        "supporting_frame_ranges": caption["supporting_frame_ranges"],
        "uncertainty_notes": caption["uncertainty_notes"],
        "generation": {
            "model": MODEL_NAME,
            "temperature": TEMPERATURE,
            "prompt_file": prompt_file_name,
            "prompt_sha256": prompt_sha256,
        },
    }


SUMMARY_FIELDS = [
    "scene_id",
    "agent_A_id",
    "agent_B_id",
    "description_short",
    "description_detailed",
    "supporting_frame_ranges",
    "uncertainty_notes",
]


def write_jsonl(handle: Any, value: Mapping[str, Any]) -> None:
    handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def export_summary_csv(outputs: List[Mapping[str, Any]], path: Path) -> int:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for data in outputs:
            writer.writerow({
                "scene_id": data.get("scene_id", ""),
                "agent_A_id": data.get("agent_A_id", ""),
                "agent_B_id": data.get("agent_B_id", ""),
                "description_short": data.get("description_short", ""),
                "description_detailed": data.get("description_detailed", ""),
                "supporting_frame_ranges": json.dumps(
                    data.get("supporting_frame_ranges", []), ensure_ascii=False
                ),
                "uncertainty_notes": data.get("uncertainty_notes", ""),
            })
    return len(outputs)


def read_inputs(args: argparse.Namespace) -> List[Tuple[int, Path, Dict[str, Any]]]:
    if args.input_json is not None:
        data = json.loads(args.input_json.read_text(encoding="utf-8"))
        if not isinstance(data, Mapping):
            raise ValueError("input JSON is not an object")
        return [(1, args.input_json, dict(data))]
    rows = []
    with args.input_jsonl.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            data = json.loads(line)
            if not isinstance(data, Mapping):
                raise ValueError(f"JSONL line {line_number} is not an object")
            rows.append((line_number, args.input_jsonl, dict(data)))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-jsonl", type=Path)
    source.add_argument("--input-json", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=default_prompt_path(),
        help="Interaction prompt file; defaults to prompt/qwen_generate_interaction_descriptions_prompt.txt",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--summary-csv", type=Path, default=None)
    args = parser.parse_args()
    args.workers = max(1, args.workers)
    input_path = args.input_json or args.input_jsonl
    if input_path is None or not input_path.is_file():
        raise SystemExit(f"Input file not found: {input_path}")
    if args.prompt_file is not None and not args.prompt_file.is_file():
        raise SystemExit(f"Prompt file not found: {args.prompt_file}")

    template = load_prompt(args.prompt_file)
    template_hash = prompt_hash(template)
    prompt_name = args.prompt_file.name
    rows = read_inputs(args)
    if args.max_rows is not None and args.max_rows > 0:
        rows = rows[: args.max_rows]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompt_copy = args.output_dir / args.prompt_file.name
    if prompt_copy.resolve() != args.prompt_file.resolve():
        prompt_copy.write_text(template, encoding="utf-8")
    output_jsonl = args.output_dir / "qwen_descriptions.jsonl"
    errors_jsonl = args.output_dir / "qwen_description_errors.jsonl"
    summary_path = args.summary_csv or (args.output_dir / "qwen_descriptions_summary.csv")

    jobs = []
    errors: List[Dict[str, Any]] = []
    for line_number, input_path, data in rows:
        try:
            prompt_input(data)
            output_path = args.output_dir / output_name(data)
            if args.skip_existing and output_path.exists():
                continue
            jobs.append((line_number, input_path, data, output_path))
        except Exception as exc:
            errors.append({"line_number": line_number, "error_type": type(exc).__name__, "error": str(exc)})

    completed: Dict[int, Dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(call_qwen, data, template): (line_number, input_path, data, output_path)
            for line_number, input_path, data, output_path in jobs
        }
        for future in as_completed(futures):
            line_number, input_path, data, output_path = futures[future]
            try:
                caption = future.result()
                output = build_output(data, caption, prompt_name, template_hash)
                completed[line_number] = output
                output_path.write_text(
                    json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8",
                )
                print(f"[SUCCESS] line={line_number} {output_path.name}", flush=True)
            except Exception as exc:
                errors.append({
                    "line_number": line_number,
                    "input_file": str(input_path),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                })
                print(f"[FAILED] line={line_number}: {exc}", flush=True)

    ordered_outputs = [completed[index] for index, _, _, _ in jobs if index in completed]
    output_jsonl.write_text(
        "".join(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n" for item in ordered_outputs),
        encoding="utf-8",
    )
    with errors_jsonl.open("w", encoding="utf-8") as handle:
        for error in errors:
            write_jsonl(handle, error)
    summary_rows = export_summary_csv(ordered_outputs, summary_path)
    print(json.dumps({
        "input_rows": len(rows),
        "submitted": len(jobs),
        "success": len(ordered_outputs),
        "failed": len(errors),
        "output_jsonl": str(output_jsonl),
        "errors_jsonl": str(errors_jsonl),
        "summary_csv": str(summary_path),
        "summary_rows": summary_rows,
    }, ensure_ascii=False))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
