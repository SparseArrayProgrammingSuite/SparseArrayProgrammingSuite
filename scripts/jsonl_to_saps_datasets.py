#!/usr/bin/env python3
"""Print SAPS dataset descriptors from scraper JSONL output.

Each solver gets a list of the dataset constructors for the matrices it
converged on in `scripts/scrape_matrices.py` output:

    pixi run python scripts/jsonl_to_saps_datasets.py \
        .saps/outputs/scrape-matrices/run_<array-job-id>/*.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DATASET_CLASSES = {
    "jacobi": "JacobiDataset",
    "cg": "CGDataset",
    "jacobi_cg": "PCGDataset",
    "block_jacobi_cg": "PCGDataset",
    "lsqr": "LSQRDataset",
    "gmres": "GMRESDataset",
    "amg": "AMGDataset",
}

MAX_SUITESPARSE_RHS_PER_MATRIX = 100


def _result_setting(result: dict[str, Any], new_name: str, old_name: str) -> Any:
    if new_name in result:
        return result[new_name]
    return result[old_name]


def _descriptor(solver: str, record: dict[str, Any], result: dict[str, Any]) -> str:
    source_name = record.get(
        "source_name",
        f"{record['matrix_group']}/{record['matrix_name']}",
    )
    name = json.dumps(source_name)
    rhs_index = record.get("rhs_index")
    args = [
        f"max_iter={int(_result_setting(result, 'max_iter', 'max_iters'))}",
        f"rel_tol={_result_setting(result, 'rel_tol', 'tolerance')!r}",
    ]
    if record.get("rhs_kind") == "suitesparse" and rhs_index is not None:
        args.append(f"rhs_index={int(rhs_index)}")
    args_str = ", " + ", ".join(args) if args else ""
    return f"{DATASET_CLASSES[solver]}({name}{args_str})"


def _format_solver_settings(settings: set[tuple[Any, Any]]) -> str:
    settings_dicts = [
        {"max_iter": max_iter, "rel_tol": rel_tol}
        for max_iter, rel_tol in sorted(settings, key=lambda item: repr(item))
    ]
    if len(settings_dicts) == 1:
        return json.dumps(settings_dicts[0], sort_keys=True)
    return json.dumps(settings_dicts, sort_keys=True)


def _passes_rhs_filter(record: dict[str, Any]) -> bool:
    if record.get("rhs_kind") != "suitesparse":
        return True
    return int(record["rhs_index"]) < MAX_SUITESPARSE_RHS_PER_MATRIX


def _solver_settings(result: dict[str, Any]) -> tuple[Any, Any]:
    return (
        _result_setting(result, "max_iter", "max_iters"),
        _result_setting(result, "rel_tol", "tolerance"),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print SAPS dataset descriptors for converged scraper results."
    )
    parser.add_argument("jsonl", nargs="+", type=Path, help="Scraper JSONL files.")
    args = parser.parse_args()

    descriptors: dict[str, list[tuple[str, str]]] = {
        solver: [] for solver in DATASET_CLASSES
    }
    solver_settings: dict[str, set[tuple[Any, Any]]] = {
        solver: set() for solver in DATASET_CLASSES
    }
    seen: set[tuple[str, str, Any]] = set()

    for path in args.jsonl:
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                if not _passes_rhs_filter(record):
                    continue
                for solver, result in record.get("results", {}).items():
                    solver_settings[solver].add(_solver_settings(result))
                    source_name = record.get(
                        "source_name",
                        f"{record['matrix_group']}/{record['matrix_name']}",
                    )
                    key = (
                        solver,
                        source_name,
                        record.get("rhs_index"),
                    )
                    if key in seen or result.get("converged") is not True:
                        continue
                    seen.add(key)
                    label = source_name
                    if record.get("rhs_kind") == "suitesparse":
                        label = f"{label} rhs{record.get('rhs_index')}"
                    descriptors[solver].append(
                        (label, _descriptor(solver, record, result))
                    )

    for solver, items in descriptors.items():
        print(f"{solver}_settings = {_format_solver_settings(solver_settings[solver])}")
        print()
        print(f"{solver}_datasets = [")
        for _label, descriptor in sorted(items):
            print(f"    {descriptor},")
        print("]")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
