from __future__ import annotations

import argparse
import pickletools


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("path")
    args = p.parse_args()
    data = open(args.path, "rb").read()
    ops = list(pickletools.genops(data))
    for index, (op, arg, pos) in enumerate(ops):
        if op.name in {"SHORT_BINUNICODE", "BINUNICODE", "BINSTRING", "STRING"} and "raw_data_idx" in str(arg):
            print("raw_data_idx at", index, pos)
            for item in ops[index:index + 100]:
                print(item[0].name, repr(item[1]))
    print("ops", len(ops))


if __name__ == "__main__":
    main()
