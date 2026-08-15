#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Retry only failed v4 Qwen inputs with an extra no-which constraint."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("qwen_v4_retry_module", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import Qwen runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", required=True, type=Path)
    parser.add_argument("--failed-csv", required=True, type=Path)
    parser.add_argument("--prompt-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    mod = load_module(args.runner)
    template = mod.load_prompt(args.prompt_file)
    prompt_hash = mod.sha256(args.prompt_file)
    strict_template = (
        template
        + "\n\nABSOLUTE RETRY CONSTRAINT: Never use the word 'which' anywhere in either "
        "description. Use two direct sentences or a 'while Vehicle X ...' clause."
    )
    original_call = mod.call_qwen

    def strict_call(data, _template):
        return original_call(data, strict_template)

    mod.call_qwen = strict_call
    failed_rows = []
    with args.failed_csv.open(encoding="utf-8-sig", newline="") as handle:
        failed_rows = list(csv.DictReader(handle))

    success = 0
    errors = []
    for row in failed_rows:
        input_path = Path(row["input_file"])
        output_path = Path(row["output_file"])
        try:
            mod.process_file(
                input_path,
                output_path,
                template,
                args.prompt_file.name,
                prompt_hash,
            )
            success += 1
        except Exception as exc:
            errors.append(
                {
                    "input_file": str(input_path),
                    "output_file": str(output_path),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    retry_log = args.output_dir / "retry_failed_v4.json"
    retry_log.write_text(
        json.dumps(
            {
                "requested": len(failed_rows),
                "success": success,
                "failed": errors,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"requested": len(failed_rows), "success": success, "failed": len(errors)}))
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
