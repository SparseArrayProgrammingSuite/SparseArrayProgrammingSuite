#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _is_asv_result(document: dict[str, Any]) -> bool:
    return all(
        key in document
        for key in ("commit_hash", "env_name", "params", "result_columns", "results")
    )


def _framework_name(document: dict[str, Any]) -> str | None:
    framework = document.get("env_vars", {}).get("SAPS_FRAMEWORK")
    if framework is None:
        return None
    return Path(framework).stem.removeprefix("saps_")


def combine_results(run_directory: Path) -> dict[str, Any]:
    result_files = []
    for path in sorted((run_directory / "results").rglob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        if _is_asv_result(document):
            result_files.append((path, document))

    return {
        "run_directory": str(run_directory),
        "result_file_count": len(result_files),
        "results": [
            {
                "source": str(path),
                "framework": _framework_name(document),
                "commit_hash": document["commit_hash"],
                "date": document["date"],
                "machine": document["params"].get("machine"),
                "env_name": document["env_name"],
                "params": document["params"],
                "requirements": document.get("requirements", {}),
                "env_vars": document.get("env_vars", {}),
                "result_columns": document["result_columns"],
                "results": document["results"],
            }
            for path, document in result_files
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Combine SAPS competition results")
    parser.add_argument(
        "--run-directory",
        required=True,
        type=Path,
        help="Competition run directory containing ASV results",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=Path,
        help="Combined competition results JSON path",
    )
    args = parser.parse_args()

    document = combine_results(args.run_directory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"combined {document['result_file_count']} result files into {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
