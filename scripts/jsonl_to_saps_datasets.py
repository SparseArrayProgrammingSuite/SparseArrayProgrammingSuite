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
    "jacobi_cg": "CGJacobiDataset",
    "block_jacobi_cg": "CGBlockJacobiDataset",
    "lsqr": "LSQRDataset",
    "gmres": "GMRESDataset",
    "amg": "CGAMGDataset",
    "dae_bdf": "DAEBDFSuiteSparseDataset",
    "ode": "ODESuiteSparseDataset",
}

# These solvers time integrators on model reduction problems instead of
# solving linear systems, and record the steps they pick.
MODEL_REDUCTION_SOLVERS = frozenset({"dae_bdf", "ode"})
# saps.downloaders.suitesparse.SUITESPARSE_PRIMARY, which names a problem's
# primary matrix.
SUITESPARSE_PRIMARY = "primary"

MAX_SUITESPARSE_RHS_PER_MATRIX = 100


def _result_setting(result: dict[str, Any], new_name: str, old_name: str) -> Any:
    if new_name in result:
        return result[new_name]
    return result[old_name]


def _model_reduction_descriptor(
    solver: str, source_name: str, result: dict[str, Any]
) -> str:
    args = [json.dumps(source_name)]
    # List the system's matrices in the order its equations use them.
    system = result["system"]
    for role in ("E", "A", "M", "D", "K", "B"):
        name = system.get(role)
        if name is None:
            continue
        value = (
            "SUITESPARSE_PRIMARY" if name == SUITESPARSE_PRIMARY else json.dumps(name)
        )
        args.append(f"{role}={value}")
    args.append(f"t_max={result['t_max']!r}")
    args.append(f"step={result['step']!r}")
    return f"{DATASET_CLASSES[solver]}({', '.join(args)})"


def _descriptor(solver: str, record: dict[str, Any], result: dict[str, Any]) -> str:
    source_name = record.get(
        "source_name",
        f"{record['matrix_group']}/{record['matrix_name']}",
    )
    if solver in MODEL_REDUCTION_SOLVERS:
        return _model_reduction_descriptor(solver, source_name, result)
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


def _format_solver_settings(solver: str, settings: set[tuple[Any, Any]]) -> str:
    keys = (
        ("step_target_seconds", "hundred_step_limit_seconds")
        if solver in MODEL_REDUCTION_SOLVERS
        else ("max_iter", "rel_tol")
    )
    settings_dicts = [
        dict(zip(keys, item, strict=True))
        for item in sorted(settings, key=lambda item: repr(item))
    ]
    if len(settings_dicts) == 1:
        return json.dumps(settings_dicts[0], sort_keys=True)
    return json.dumps(settings_dicts, sort_keys=True)


def _passes_rhs_filter(record: dict[str, Any]) -> bool:
    if record.get("rhs_kind") != "suitesparse":
        return True
    return int(record["rhs_index"]) < MAX_SUITESPARSE_RHS_PER_MATRIX


def _solver_settings(solver: str, result: dict[str, Any]) -> tuple[Any, Any]:
    if solver in MODEL_REDUCTION_SOLVERS:
        return (result["step_target_seconds"], result["hundred_step_limit_seconds"])
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
                    solver_settings[solver].add(_solver_settings(solver, result))
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
        settings = _format_solver_settings(solver, solver_settings[solver])
        print(f"{solver}_settings = {settings}")
        print()
        print(f"{solver}_datasets = [")
        for _label, descriptor in sorted(items):
            print(f"    {descriptor},")
        print("]")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
