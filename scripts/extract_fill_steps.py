#!/usr/bin/env python3
"""Print each matrix's recorded squaring count from measure_fill_in.py logs.

Counts mark the first density upper bound reaching the target, not measured
density. ``never`` means the bound stays below the target; ``-`` means no result.
"""

import argparse
import re
from pathlib import Path

RESULT = re.compile(
    r"^\s*(\S+/\S+)\s+(\d+|never|-)\s+(?:max_degree=|missing\b|error\b)"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "logs",
        nargs="*",
        type=Path,
        default=[Path(__file__).with_name("fill-log")],
        help="fill logs (default: scripts/fill-log)",
    )
    args = parser.parse_args()
    steps: dict[str, str] = {}
    for path in args.logs:
        try:
            with path.open(encoding="utf-8") as log:
                for line_number, line in enumerate(log, 1):
                    match = RESULT.match(line)
                    if match is None:
                        continue
                    matrix, count = match.groups()
                    if matrix in steps and steps[matrix] != count:
                        parser.error(
                            f"{path}:{line_number}: conflicting counts for {matrix}: "
                            f"{steps[matrix]} and {count}"
                        )
                    steps[matrix] = count
        except OSError as error:
            parser.error(str(error))
    if not steps:
        parser.error("no fill-in summary rows found")
    for matrix, count in sorted(steps.items()):
        print(f"{matrix}\t{count}")


if __name__ == "__main__":
    main()
