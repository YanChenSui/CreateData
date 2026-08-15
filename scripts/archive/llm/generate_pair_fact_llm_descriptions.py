"""Ask an LLM to verbalize factual pair-timeline evidence.

The input is the compact batch result produced by the remote pair validation.
Only observable timeline evidence is sent to the model. Source behavior labels,
classifier outputs, and InterHub's audit window are deliberately excluded to
avoid leaking semantic answers into the description prompt.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping

DEFAULT_BASE_URL = os.getenv(
    "PAIR_LLM_BASE_URL",
    os.getenv("QWEN_BASE_URL", "http://172.17.0.1:60200/v1"),
)
DEFAULT_API_KEY = os.getenv(
    "PAIR_LLM_API_KEY", os.getenv("QWEN_API_KEY", "EMPTY")
)
DEFAULT_MODEL = os.getenv("PAIR_LLM_MODEL", os.getenv("QWEN_MODEL", "qwen-3.6"))
MAX_RETRIES = 3
RETRY_WAIT_SECONDS = 2.0


SYSTEM_PROMPT = """你是自动驾驶轨迹事实描述器。
你的任务是把输入的车辆对时间线证据改写成简洁、客观的中文描述。

严格规则：
1. 只描述输入中明确给出的事实，不判断 merge、cut-in、overtake、yield 等交互类型。
2. A 和 B 的角色固定，不能交换车辆身份。
3. true 表示该事实被观察到；false 表示该事实在当前分析中没有成立；null 表示没有确认，必须写成“未确认”或“不足以判断”，不能写成“没有发生”。
4. 不要使用或猜测 InterHub 窗口、source behavior、分类器结果；它们不会作为证据提供给你。
5. 不要把 lane segment ID 的变化自动解释成真实变道；只能根据 entered/maintained 等事实描述。
6. 如果没有 pair transition，明确说“没有检测到两车车道关系或前后顺序的明确变化”，但不要说“没有交互”。
7. 输出必须是 JSON 对象，不要输出 Markdown 代码块或额外解释。
"""


def _clean_json(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _evidence(row: Mapping[str, Any]) -> dict[str, Any]:
    """Select only facts and phase relations; intentionally omit labels."""
    return {
        "scene_id": row.get("scene_id"),
        "agent_A": row.get("agent_A"),
        "agent_B": row.get("agent_B"),
        "analysis_window": row.get("analysis_window"),
        "pair_facts": row.get("pair_facts", {}),
        "phase_triplet": row.get("phase_triplet", {}),
    }


def build_prompt(evidence: Mapping[str, Any]) -> str:
    return f"""请根据下面的车辆对事实证据生成中文描述。

返回严格 JSON，字段为：
- description_short：一句话，最多两个分句，只写已确认事实；
- description_detailed：按 Before、During、After 组织，2–5 句；
- evidence_status：只能是 complete、partial 或 insufficient。

事实证据：
{json.dumps(evidence, ensure_ascii=False, indent=2)}
"""


def _call_model(
    client: OpenAI,
    model: str,
    evidence: Mapping[str, Any],
    temperature: float,
    max_tokens: int,
) -> tuple[dict[str, str], str]:
    prompt = build_prompt(evidence)
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            content = response.choices[0].message.content
            if not content:
                raise ValueError("LLM returned empty content")
            raw = content.strip()
            result = json.loads(_clean_json(raw))
            if not isinstance(result, dict):
                raise ValueError("LLM output is not a JSON object")
            short = result.get("description_short")
            detailed = result.get("description_detailed")
            status = result.get("evidence_status")
            if not isinstance(short, str) or not short.strip():
                raise ValueError("Missing description_short")
            if not isinstance(detailed, str) or not detailed.strip():
                raise ValueError("Missing description_detailed")
            if status not in {"complete", "partial", "insufficient"}:
                raise ValueError("evidence_status must be complete/partial/insufficient")
            return {
                "description_short": short.strip(),
                "description_detailed": detailed.strip(),
                "evidence_status": status,
            }, raw
        except Exception as exc:
            last_error = exc
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_WAIT_SECONDS * attempt)
    raise RuntimeError(f"LLM request failed after {MAX_RETRIES} attempts: {last_error}")


def _output_name(index: int, row: Mapping[str, Any]) -> str:
    scene = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(row.get("scene_id", "scene")))
    a = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(row.get("agent_A", "A")))
    b = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(row.get("agent_B", "B")))
    return f"pair_{index:04d}_{scene}_A{a}_B{b}.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    data = json.loads(args.input.read_text(encoding="utf-8"))
    rows = data.get("rows", [])
    if not isinstance(rows, list) or not rows:
        raise SystemExit("Input must contain a non-empty rows list")
    if args.max_rows is not None:
        rows = rows[: args.max_rows]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    prompt_path = args.output_dir / "pair_fact_prompts.jsonl"
    with prompt_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows, 1):
            handle.write(json.dumps({
                "index": index,
                "input_file": row.get("file"),
                "evidence": _evidence(row),
                "prompt": build_prompt(_evidence(row)),
            }, ensure_ascii=False) + "\n")

    if args.dry_run:
        print(f"dry_run: prompts written to {prompt_path}")
        print(f"rows: {len(rows)}")
        return 0

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit(
            "The OpenAI-compatible client is required for live generation. "
            "Install the openai package in the selected environment, or use --dry-run."
        ) from exc
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    def process(index: int, row: Mapping[str, Any]) -> dict[str, Any]:
        evidence = _evidence(row)
        generated, raw = _call_model(
            client, args.model, evidence, args.temperature, args.max_tokens
        )
        return {
            "schema_version": "pair_fact_llm_description_v1",
            "index": index,
            "input_file": row.get("file"),
            "scene_id": row.get("scene_id"),
            "agent_A": row.get("agent_A"),
            "agent_B": row.get("agent_B"),
            "evidence": evidence,
            "llm": generated,
            "raw_response": raw,
            "generation": {
                "model": args.model,
                "base_url": args.base_url,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
            },
        }

    workers = max(1, args.workers)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process, index, row): (index, row)
            for index, row in enumerate(rows, 1)
        }
        for future in as_completed(futures):
            index, row = futures[future]
            try:
                result = future.result()
                results.append(result)
                path = args.output_dir / _output_name(index, row)
                path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                print(f"[{index}/{len(rows)}] success: {path.name}", flush=True)
            except Exception as exc:
                failures.append({
                    "index": index,
                    "input_file": row.get("file"),
                    "scene_id": row.get("scene_id"),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                })
                print(f"[{index}/{len(rows)}] failed: {exc}", flush=True)

    results.sort(key=lambda item: item["index"])
    (args.output_dir / "pair_fact_llm_descriptions.json").write_text(
        json.dumps({
            "schema_version": "pair_fact_llm_description_batch_v1",
            "input_file": str(args.input),
            "model": args.model,
            "base_url": args.base_url,
            "num_input": len(rows),
            "num_success": len(results),
            "num_failed": len(failures),
            "results": results,
            "failures": failures,
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "failed_pairs.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["index", "input_file", "scene_id", "error_type", "error"])
        writer.writeheader()
        writer.writerows(failures)
    print(f"success: {len(results)}")
    print(f"failed: {len(failures)}")
    print(f"output_dir: {args.output_dir}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
