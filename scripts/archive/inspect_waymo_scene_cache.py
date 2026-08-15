from __future__ import annotations

import argparse
import dill
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("path")
    args = p.parse_args()
    with Path(args.path).open("rb") as f:
        value = dill.load(f)
    print(type(value).__name__)
    if isinstance(value, dict):
        print("keys", list(value.keys())[:100])
        print(json.dumps({str(k): repr(v)[:1000] for k, v in value.items()}, ensure_ascii=False, indent=2))
    else:
        print(repr(value)[:10000])


if __name__ == "__main__":
    main()

