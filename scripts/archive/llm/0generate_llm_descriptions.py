from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any

from openai import OpenAI


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "deepseek-v4-flash"
OUTPUT_SCHEMA_VERSION = "scene_motion_llm_description_v1"

TEMPERATURE = 0.3
MAX_TOKENS = 800
MAX_RETRIES = 3
RETRY_WAIT_SECONDS = 3

SYSTEM_PROMPT = (
    "You are an autonomous-driving traffic scene analyst. "
    "Generate precise and natural traffic interaction descriptions "
    "grounded strictly in the provided structured data."
)


# ============================================================
# DeepSeek client
# ============================================================

API_KEY = os.getenv("DEEPSEEK_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "Environment variable DEEPSEEK_API_KEY is not set.\n"
        "Run:\n"
        "export DEEPSEEK_API_KEY='your_api_key'"
    )

client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.deepseek.com",
)


# ============================================================
# Prompt
# ============================================================

def load_prompt_template(prompt_path: Path) -> str:
    """Load prompt template from an external text file."""
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    if not prompt_path.is_file():
        raise ValueError(f"Prompt path is not a file: {prompt_path}")

    prompt = prompt_path.read_text(encoding="utf-8").strip()

    if not prompt:
        raise ValueError(f"Prompt file is empty: {prompt_path}")

    return prompt


def calculate_file_sha256(file_path: Path) -> str:
    """Calculate SHA256 of the exact prompt file."""
    sha256 = hashlib.sha256()

    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def copy_prompt_to_output(prompt_path: Path, output_dir: Path) -> Path:
    """Copy the exact prompt used in this run into the output directory."""
    destination = output_dir / prompt_path.name

    if prompt_path.resolve() != destination.resolve():
        shutil.copy2(prompt_path, destination)

    return destination


def build_caption_generation_prompt(
    grounded_caption: str,
) -> str:
    """Build the small rewrite prompt from grounded facts only."""
    if not isinstance(grounded_caption, str) or not grounded_caption.strip():
        raise ValueError("grounded_caption must be a non-empty string")
    return (
        "You are rewriting a grounded traffic interaction description.\n\n"
        "Grounded description:\n"
        f"{grounded_caption.strip()}\n\n"
        "Generate a natural description."
    )


def caption_generation_prompt(data: dict[str, Any], prompt_template: str) -> str:
    """Build the LLM prompt from the machine-grounded caption only.

    The extractor output intentionally keeps audit, quality, provenance, and
    schema fields for reproducibility.  They are not model evidence and are
    not sent to the LLM because duplicated low-level fields can compete with
    the grounded caption or expose implementation metadata.
    """
    generation_context = data.get("generation_context", {})
    if not isinstance(generation_context, dict):
        raise ValueError("Missing or invalid generation_context")
    grounded_caption = generation_context.get("grounded_caption")
    if not isinstance(grounded_caption, str) or not grounded_caption.strip():
        raise ValueError("Missing or empty generation_context.grounded_caption")
    grounded_prompt = build_caption_generation_prompt(grounded_caption)
    return f"{prompt_template}\n\n{grounded_prompt}"


def build_prompt(data: dict[str, Any], prompt_template: str) -> str:
    """Backward-compatible alias for :func:`caption_generation_prompt`."""
    return caption_generation_prompt(data, prompt_template)


# ============================================================
# Response processing
# ============================================================

def clean_json_response(content: str) -> str:
    """Remove accidental Markdown code fences."""
    content = content.strip()

    if content.startswith("```json"):
        content = content[len("```json"):].strip()
    elif content.startswith("```"):
        content = content[len("```"):].strip()

    if content.endswith("```"):
        content = content[:-3].strip()

    return content


# ============================================================
# DeepSeek API
# ============================================================

def call_deepseek(
    data: dict[str, Any],
    prompt_template: str,
) -> dict[str, str]:

    prompt = build_prompt(data, prompt_template)
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
                extra_body={"thinking": {"type": "disabled"}},
            )

            content = response.choices[0].message.content

            if not content:
                raise ValueError("DeepSeek returned empty content")

            result = json.loads(clean_json_response(content))

            if not isinstance(result, dict):
                raise ValueError("DeepSeek output is not a JSON object")

            description_short = result.get("description_short")
            description_detailed = result.get("description_detailed")

            if not isinstance(description_short, str) or not description_short.strip():
                raise ValueError("Missing or invalid description_short")

            if not isinstance(description_detailed, str) or not description_detailed.strip():
                raise ValueError("Missing or invalid description_detailed")

            return {
                "description_short": description_short.strip(),
                "description_detailed": description_detailed.strip(),
            }

        except Exception as exc:
            last_error = exc

            if attempt < MAX_RETRIES:
                print(
                    f"[RETRY] attempt={attempt}/{MAX_RETRIES} error={exc}",
                    flush=True,
                )
                time.sleep(RETRY_WAIT_SECONDS)

    raise RuntimeError(
        f"DeepSeek request failed after {MAX_RETRIES} attempts: {last_error}"
    )


# ============================================================
# Output
# ============================================================

def build_output(
    data: dict[str, Any],
    result: dict[str, str],
    input_path: Path,
    prompt_file_name: str,
    prompt_sha256: str,
) -> dict[str, Any]:

    generation_context = data.get("generation_context", {})
    if not isinstance(generation_context, dict):
        generation_context = {}

    return {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "source_input_schema_version": data.get("schema_version"),

        "behavior_record_id": data.get("behavior_record_id"),
        "interaction_id": data.get("interaction_id"),
        "description_short": result["description_short"],
        "description_detailed": result["description_detailed"],

        "source": data.get("source"),
        "behavior": generation_context.get("behavior"),
        "caption_scope": generation_context.get("caption_scope"),
        "caption_window": generation_context.get("caption_window"),

        "generation": {
            "model": MODEL_NAME,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "thinking": "disabled",
            "prompt_file": prompt_file_name,
            "prompt_sha256": prompt_sha256,
            "input_file": input_path.name,
        },
    }


def _filename_component(value: Any) -> str:
    """Keep filename components readable and filesystem-safe."""
    text = "" if value is None else str(value).strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", text)
    return text.strip("-") or "unknown"


def build_description_output_name(
    data: dict[str, Any],
    input_path: Path,
) -> str:
    """Build a unique description filename from the unique input filename.

    The extractor already puts the behavior event and bundle index into the
    input filename. Reusing that stem prevents multiple behavior records from
    overwriting one another while preserving the readable row suffix.
    """
    input_stem = input_path.stem
    if input_stem.endswith("_llm_input"):
        return input_stem[:-len("_llm_input")] + "_llm_description.json"

    # Fallback for older or manually named inputs that do not use the current
    # extractor suffix. The legacy field-based name remains readable.
    source = data.get("source")
    if not isinstance(source, dict):
        source = {}

    context = data.get("generation_context")
    if not isinstance(context, dict):
        context = {}

    behavior = context.get("behavior")
    if not isinstance(behavior, dict):
        behavior = {}

    scene_id = _filename_component(source.get("scene_id"))
    behavior_type = _filename_component(behavior.get("type"))

    audit_context = data.get("audit_context")
    if not isinstance(audit_context, dict):
        audit_context = {}

    # Put the InterHub interaction window in the filename. The more specific
    # behavior/caption window remains available in the JSON and summary CSV.
    window = audit_context.get("interhub_window")
    if not isinstance(window, (list, tuple)) or len(window) != 2:
        window = data.get("interaction_window")
    if not isinstance(window, (list, tuple)) or len(window) != 2:
        window = context.get("caption_window")
    if not isinstance(window, (list, tuple)) or len(window) != 2:
        window = data.get("caption_window")
    if isinstance(window, (list, tuple)) and len(window) == 2:
        window = "{}_{}".format(
            _filename_component(window[0]),
            _filename_component(window[1]),
        )
    else:
        window = "unknown_window"

    agent_ids = [behavior.get("subject_agent_id")]
    reference_agent_id = behavior.get("reference_agent_id")
    if reference_agent_id not in (None, ""):
        agent_ids.append(reference_agent_id)
    agents = "_".join(_filename_component(agent_id) for agent_id in agent_ids)

    row_match = re.search(r"(?:^|_)row_(\d+)(?:_|$)", input_path.stem)
    row = row_match.group(1) if row_match else "unknown"

    return f"{scene_id}_{window}_{behavior_type}_{agents}_row_{row}.json"


SUMMARY_FIELDNAMES = [
    "sceneid",
    "agentid",
    "场景类型",
    "对应窗口",
    "short描述",
    "detail描述",
]


def _summary_value(value: Any) -> str:
    """Convert nested values to compact CSV-safe text."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "-".join(str(item) for item in value)
    return str(value)


def description_to_summary_row(description: dict[str, Any]) -> dict[str, str]:
    """Extract only the six user-facing fields from a description JSON."""
    source = description.get("source")
    if not isinstance(source, dict):
        source = {}

    behavior = description.get("behavior")
    if not isinstance(behavior, dict):
        behavior = {}

    return {
        "sceneid": _summary_value(source.get("scene_id")),
        "agentid": _summary_value(behavior.get("subject_agent_id")),
        "场景类型": _summary_value(behavior.get("type")),
        "对应窗口": _summary_value(description.get("caption_window")),
        "short描述": _summary_value(description.get("description_short")),
        "detail描述": _summary_value(description.get("description_detailed")),
    }


def export_summary_csv(output_dir: Path, summary_csv: Path) -> tuple[int, int]:
    """Export all generated description JSONs in output_dir to a compact CSV."""
    rows: list[dict[str, str]] = []
    invalid_count = 0

    # New runs use the explicit suffix. Keep compatibility with legacy
    # description files generated before that suffix was added.
    description_paths = {
        *output_dir.glob("*_llm_description.json"),
        *output_dir.glob("*.json"),
    }
    for json_path in sorted(description_paths):
        try:
            with json_path.open("r", encoding="utf-8") as f:
                description = json.load(f)
            if not isinstance(description, dict):
                raise ValueError("description JSON is not an object")
            rows.append(description_to_summary_row(description))
        except Exception as exc:
            invalid_count += 1
            print(f"[SUMMARY-SKIP] {json_path.name}: {exc}", flush=True)

    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows), invalid_count


def process_file(
    input_path: Path,
    output_path: Path,
    prompt_template: str,
    prompt_file_name: str,
    prompt_sha256: str,
) -> str:

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    quality = data.get("quality", {})

    if (
        isinstance(quality, dict)
        and quality.get("eligible_for_text_generation") is False
    ):
        return "ineligible"

    generation_context = data.get("generation_context", {})

    if not isinstance(generation_context, dict) or not generation_context:
        raise ValueError("Missing or invalid generation_context")

    result = call_deepseek(
        data,
        prompt_template,
    )

    output = build_output(
        data=data,
        result=result,
        input_path=input_path,
        prompt_file_name=prompt_file_name,
        prompt_sha256=prompt_sha256,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")

    return "success"


# ============================================================
# Main
# ============================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate grounded traffic descriptions from Waymo interaction JSON files."
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing *_llm_input.json files",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory used to save generated descriptions",
    )

    parser.add_argument(
        "--prompt-file",
        type=Path,
        required=True,
        help="Path to the latest prompt text file",
    )

    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files whose output JSON already exists",
    )

    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional maximum number of files to process",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Maximum number of concurrent LLM API calls (default: 4)",
    )

    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help=(
            "Optional compact CSV path. Defaults to "
            "<output-dir>/llm_descriptions_summary.csv"
        ),
    )

    args = parser.parse_args()
    args.workers = max(1, args.workers)

    # --------------------------------------------------------
    # Inputs
    # --------------------------------------------------------

    if not args.input_dir.exists():
        raise SystemExit(f"Input directory not found: {args.input_dir}")

    input_paths = sorted(args.input_dir.glob("*_llm_input.json"))

    if not input_paths:
        raise SystemExit(
            f"No *_llm_input.json files found in {args.input_dir}"
        )

    if args.max_files is not None and args.max_files > 0:
        input_paths = input_paths[:args.max_files]

    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

    prompt_template = load_prompt_template(args.prompt_file)
    prompt_sha256 = calculate_file_sha256(args.prompt_file)

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    args.output_dir.mkdir(parents=True, exist_ok=True)

    output_prompt_path = copy_prompt_to_output(
        args.prompt_file,
        args.output_dir,
    )

    print(f"[PROMPT] source: {args.prompt_file}")
    print(f"[PROMPT] copied to: {output_prompt_path}")
    print(f"[PROMPT] sha256: {prompt_sha256}")

    # --------------------------------------------------------
    # Batch generation
    # --------------------------------------------------------

    total = len(input_paths)
    success_count = 0
    skipped_existing_count = 0
    skipped_ineligible_count = 0
    failed_files: list[dict[str, str]] = []

    # Validate input JSON, derive unique output paths, and handle existing
    # outputs before starting worker threads.  This keeps the concurrent part
    # limited to independent API calls and file writes.
    jobs: list[tuple[int, Path, Path]] = []
    for index, input_path in enumerate(input_paths, start=1):
        try:
            with input_path.open("r", encoding="utf-8") as f:
                input_data = json.load(f)
            if not isinstance(input_data, dict):
                raise ValueError("input JSON is not an object")
            output_name = build_description_output_name(input_data, input_path)
        except Exception as exc:
            failed_files.append(
                {
                    "input_file": str(input_path),
                    "output_file": "",
                    "error_type": type(exc).__name__,
                    "error": f"cannot build output filename: {exc}",
                }
            )
            print(
                f"[{index}/{total}] [ERROR] {input_path.name}: {exc}",
                flush=True,
            )
            continue

        output_path = args.output_dir / output_name

        if args.skip_existing and output_path.exists():
            skipped_existing_count += 1
            print(f"[{index}/{total}] [SKIP] {input_path.name}", flush=True)
            continue

        jobs.append((index, input_path, output_path))

    def run_job(input_path: Path, output_path: Path) -> str:
        return process_file(
            input_path=input_path,
            output_path=output_path,
            prompt_template=prompt_template,
            prompt_file_name=output_prompt_path.name,
            prompt_sha256=prompt_sha256,
        )

    print(f"[CONCURRENCY] workers: {args.workers}", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_job, input_path, output_path): (index, input_path, output_path)
            for index, input_path, output_path in jobs
        }

        for future in as_completed(futures):
            index, input_path, output_path = futures[future]
            try:
                status = future.result()

                if status == "ineligible":
                    skipped_ineligible_count += 1
                    print(
                        f"[{index}/{total}] [INELIGIBLE] {input_path.name}",
                        flush=True,
                    )
                    continue

                success_count += 1
                print(f"[{index}/{total}] [OK] {input_path.name}", flush=True)

            except Exception as exc:
                failed_files.append(
                    {
                        "input_file": str(input_path),
                        "output_file": str(output_path),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )

                print(
                    f"[{index}/{total}] [ERROR] {input_path.name}: {exc}",
                    flush=True,
                )

    # --------------------------------------------------------
    # Failure report
    # --------------------------------------------------------

    failed_csv = args.output_dir / "failed_files.csv"

    with failed_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "input_file",
                "output_file",
                "error_type",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(failed_files)

    # --------------------------------------------------------
    # Compact summary table
    # --------------------------------------------------------

    summary_csv = args.summary_csv or (
        args.output_dir / "llm_descriptions_summary.csv"
    )
    summary_count, summary_invalid_count = export_summary_csv(
        output_dir=args.output_dir,
        summary_csv=summary_csv,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n========== Summary ==========")
    print(f"total: {total}")
    print(f"success: {success_count}")
    print(f"skipped_existing: {skipped_existing_count}")
    print(f"skipped_ineligible: {skipped_ineligible_count}")
    print(f"failed: {len(failed_files)}")
    print(f"prompt_copy: {output_prompt_path}")
    print(f"failed_csv: {failed_csv}")
    print(f"summary_csv: {summary_csv}")
    print(f"summary_rows: {summary_count}")
    print(f"summary_invalid: {summary_invalid_count}")
    print(f"output_dir: {args.output_dir}")

    return 1 if failed_files else 0


if __name__ == "__main__":
    raise SystemExit(main())
