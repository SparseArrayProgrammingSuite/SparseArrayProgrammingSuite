#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from saps.metadata import write_metadata_document


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate SAPS metadata")
    parser.add_argument(
        "--statistics",
        action="append",
        default=[],
        type=Path,
        help=(
            "statistics.json file whose fresh statistics tags should be folded "
            "into generated metadata. Can be passed more than once."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        default=Path("metadata.json"),
        type=Path,
        help="Metadata output path (default: metadata.json).",
    )
    args = parser.parse_args()

    document = write_metadata_document(args.output, args.statistics)
    print(f"generated metadata for {len(document['benchmarks'])} benchmarks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
