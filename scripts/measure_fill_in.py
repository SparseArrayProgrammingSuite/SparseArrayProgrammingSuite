#!/usr/bin/env python3
"""Bound Boolean fill-in of (I | A) under repeated squaring from row degrees.

One streaming pass over each prepared, row-ordered COO Binsparse file counts
the off-diagonal nonzeros in every row; nothing is squared or traversed. When
no row has more than d of them, at most R(k) = 1 + d + ... + d**k vertices lie
within k edges of any vertex. A row with k0 of them therefore has at most
min(n, 1 + k0 * R(h - 1)) nonzeros in (I | A)**h. Summing over rows bounds the
nonzeros, density, and growth relative to (I | A) after each squaring.

The output leads with one step count per matrix: the fewest squarings whose
density bound reaches --target-density (default 1%). Every earlier squaring is
guaranteed to stay below the target. The bounds hold for every graph with the
same row degrees, so a matrix often stays sparse well past its step count.

Example:
    poetry run python scripts/measure_fill_in.py --output fill-in.json
    poetry run python scripts/measure_fill_in.py --datasets SNAP --output snap.json
    poetry run python scripts/measure_fill_in.py --merge run/*/*.json --output all.json

Dataset names without a slash select every manifest entry in that SuiteSparse
group; the default is all GAP and SNAP matrices. Missing cache files are
reported without downloading multi-gigabyte datasets.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

import h5py

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PREFIX = "suitesparse_matrix."
DEFAULT_GROUPS = ["GAP", "SNAP"]
SEMANTICS = (
    "Upper bounds for Boolean (I OR (A != 0)) raised to 2**squarings, from row"
    " degrees; original edge direction"
)


def _array(dataset, path):
    """Memory-map contiguous arrays; retain h5py slicing for chunked arrays."""
    if not dataset.size:
        return np.empty(dataset.shape, dtype=dataset.dtype)
    offset = dataset.id.get_offset()
    if offset is not None:
        return np.memmap(
            path, mode="r", offset=offset, shape=dataset.shape, dtype=dataset.dtype
        )
    return dataset


@contextmanager
def open_matrix(path):
    """Yield ``(n, rows, columns, values)`` of a prepared COOR adjacency matrix."""
    with h5py.File(path, "r") as file:
        tensor = file["inputs/0"]
        metadata = json.loads(tensor.attrs["binsparse"])["binsparse"]
        if metadata["format"] != "COOR":
            raise ValueError("Expected a prepared row-ordered COO (COOR) matrix")
        shape = metadata["shape"]
        if len(shape) != 2 or shape[0] != shape[1] or shape[0] < 0:
            raise ValueError(f"Expected a square adjacency matrix, got {shape}")
        if metadata.get("fill_value", 0) != 0:
            raise ValueError("Adjacency must have zero implicit fill")
        yield (
            int(shape[0]),
            _array(tensor["indices_0"], path),
            _array(tensor["indices_1"], path),
            _array(tensor["values"], path),
        )


def row_degrees(n, rows, columns, values, chunk=1_000_000):
    """Count each row's off-diagonal nonzeros in one pass over COOR arrays.

    Entries must be sorted by row, then strictly by column, so no coordinate
    repeats. Stored zeros and diagonal entries are not counted; the identity
    supplies the diagonal.
    """
    if any(array.ndim != 1 for array in (rows, columns, values)):
        raise ValueError("Expected one-dimensional COO coordinate/value arrays")
    if not (len(rows) == len(columns) == len(values)):
        raise ValueError("COO coordinate/value lengths differ")
    if rows.dtype.kind not in "iu" or columns.dtype.kind not in "iu":
        raise ValueError("COO coordinates must be integers")
    degree = np.zeros(n, dtype=np.int64)
    previous = (-1, -1)
    for start in range(0, len(rows), chunk):
        row = rows[start : start + chunk]
        column = columns[start : start + chunk]
        value = values[start : start + chunk]
        if row[0] < 0 or row[-1] >= n or np.any(column < 0) or np.any(column >= n):
            raise ValueError("COO coordinates fall outside the matrix shape")
        same = row[1:] == row[:-1]
        if (
            (int(row[0]), int(column[0])) <= previous
            or np.any(row[1:] < row[:-1])
            or np.any(column[1:][same] <= column[:-1][same])
        ):
            raise ValueError(
                "COOR entries must be sorted by row, then column, without duplicates"
            )
        previous = (int(row[-1]), int(column[-1]))
        edges = (value != 0) & (row != column)
        counts = np.bincount(row[edges] - row[0])
        degree[row[0] : row[0] + len(counts)] += counts
    return degree


def reachable_bound(n, degree, hop):
    """Bound the vertices within ``hop`` edges when rows have ``degree`` edges.

    At most 1 + d + ... + d**h vertices are reachable within h hops when every
    row has at most d off-diagonal nonzeros.
    """
    if degree <= 1:
        return min(n, 1 + degree * hop)
    bound = 1
    for _ in range(hop):
        bound = min(n, 1 + degree * bound)
        if bound == n:
            break
    return bound


def bound_fill_in(n, degree, *, target_density=0.01):
    """Bound (I | A)**(2**s) for s = 0, 1, ... until the density bound is reached.

    ``degree`` holds the off-diagonal nonzeros in each row of A. Stops at the
    first squaring whose density bound reaches ``target_density``, or once the
    bound stops growing below it (every row that can grow is already full).
    """
    histogram = np.bincount(degree, minlength=1)
    max_degree = len(histogram) - 1
    # Rows grouped by degree keep the sums exact Python integers.
    groups = [(int(k), int(histogram[k])) for k in np.flatnonzero(histogram)]
    nnz = sum((1 + k) * count for k, count in groups)
    stages = []
    stop_reason = "never"
    step = 0
    while True:
        hop = 2**step
        # A row reaches itself plus, through each of its k neighbors, at most
        # the vertices that neighbor reaches within hop - 1 edges.
        neighbor_bound = reachable_bound(n, max_degree, hop - 1)
        nnz_bound = sum(count * min(n, 1 + k * neighbor_bound) for k, count in groups)
        if stages and nnz_bound == stages[-1]["nnz_bound"]:
            break
        density_bound = nnz_bound / n**2 if n else 0.0
        stages.append(
            {
                "squarings": step,
                "max_path_length": hop,
                "row_nnz_bound": reachable_bound(n, max_degree, hop),
                "nnz_bound": nnz_bound,
                "density_bound": density_bound,
                "growth_factor_bound": nnz_bound / nnz if nnz else None,
                "bool_csr64_bytes_bound": 9 * nnz_bound + 8 * (n + 1),
            }
        )
        if density_bound >= target_density:
            stop_reason = "target"
            break
        step += 1
    return {
        "n": n,
        "nnz": nnz,
        "max_degree": max_degree,
        "target_density": target_density,
        "squarings": step if stop_reason == "target" else None,
        "stop_reason": stop_reason,
        "stages": stages,
    }


def expand_datasets(names, manifest):
    """Expand SuiteSparse group names (no slash) to their manifest entries."""
    datasets = []
    for name in names:
        if "/" in name:
            datasets.append(name)
            continue
        members = sorted(
            key.removeprefix(MANIFEST_PREFIX)
            for key in manifest
            if key.startswith(f"{MANIFEST_PREFIX}{name}/")
        )
        if not members:
            raise ValueError(f"No manifest entries for SuiteSparse group {name}")
        datasets.extend(members)
    return list(dict.fromkeys(datasets))


def steps(matrices):
    """One entry per matrix: the fewest squarings that could reach the target."""
    return [
        {
            "dataset": item["dataset"],
            "squarings": item.get("squarings"),
            "stop_reason": item.get("stop_reason", item["status"]),
            "max_degree": item.get("max_degree"),
            **{
                key: item["stages"][-1][key] if "stages" in item else None
                for key in ("density_bound", "growth_factor_bound")
            },
        }
        for item in matrices
    ]


def print_steps(entries, target_density):
    print(f"Fewest squarings whose density bound reaches {100 * target_density:g}%:")
    width = max((len(entry["dataset"]) for entry in entries), default=0)
    for entry in entries:
        if entry["squarings"] is not None:
            value = str(entry["squarings"])
        else:
            value = "never" if entry["stop_reason"] == "never" else "-"
        if entry["max_degree"] is None:
            detail = entry["stop_reason"]
        else:
            growth = entry["growth_factor_bound"]
            growth_text = "" if growth is None else f"  growth<={growth:.3g}x"
            detail = (
                f"max_degree={entry['max_degree']}{growth_text}"
                f"  density<={entry['density_bound']:.4%}"
            )
        print(f"  {entry['dataset']:<{width}}  {value:>5}  {detail}")


def write(document, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, allow_nan=False, default=str) + "\n")


def merge(paths, output):
    """Combine per-matrix result files (e.g. one per Slurm task) into one list."""
    documents = [json.loads(path.read_text()) for path in paths]
    targets = {document["target_density"] for document in documents}
    if len(targets) != 1:
        raise ValueError(f"Cannot merge different target densities: {targets}")
    matrices = [item for document in documents for item in document["matrices"]]
    document = {
        "semantics": SEMANTICS,
        "target_density": targets.pop(),
        "steps": steps(matrices),
        "merged_from": [str(path) for path in paths],
        "matrices": matrices,
    }
    write(document, output)
    return document


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_GROUPS,
        help="Matrices such as GAP/GAP-road, or whole groups such as SNAP",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(os.getenv("SAPS_CACHE_DIR", REPO_ROOT / ".saps/outputs/cache")),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(os.getenv("SAPS_MANIFEST_PATH", REPO_ROOT / "manifest.json")),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="Print the selected matrix names, one per line, and exit",
    )
    parser.add_argument(
        "--merge",
        nargs="+",
        type=Path,
        metavar="RESULT",
        help="Combine earlier result files into --output instead of measuring",
    )
    parser.add_argument(
        "--target-density",
        type=float,
        default=0.01,
        help="Density the step count is measured against (default: 0.01)",
    )
    args = parser.parse_args(argv)
    if not 0 < args.target_density <= 1:
        parser.error("Require 0 < target-density <= 1")
    if args.output is None and not args.list_datasets:
        parser.error("--output is required")
    if args.merge:
        try:
            document = merge(args.merge, args.output)
        except (OSError, ValueError, KeyError) as error:
            parser.error(str(error))
        print_steps(document["steps"], document["target_density"])
        return int(any(item["status"] != "ok" for item in document["matrices"]))
    manifest = json.loads(args.manifest.read_text())
    try:
        datasets = expand_datasets(args.datasets, manifest)
    except ValueError as error:
        parser.error(str(error))
    if args.list_datasets:
        print("\n".join(datasets))
        return 0
    document = {
        "semantics": SEMANTICS,
        "target_density": args.target_density,
        "steps": [],
        "configuration": {**vars(args), "datasets": datasets},
        "matrices": [],
    }
    for name in datasets:
        key = f"{MANIFEST_PREFIX}{name}"
        record = manifest.get(key)
        result = {"dataset": name}
        if record is None:
            result.update(status="missing", error=f"No manifest entry for {key}")
        else:
            path = (
                args.cache_dir
                / "suitesparse_matrix"
                / name
                / f"{record['digest']}.bsp.h5"
            )
            result.update(path=str(path), digest=record["digest"])
            if not path.is_file():
                result.update(
                    status="missing", error="Prepared dataset is not cached locally"
                )
            else:
                started = time.monotonic()
                try:
                    with open_matrix(path) as (n, rows, columns, values):
                        degree = row_degrees(n, rows, columns, values)
                    result.update(
                        status="ok",
                        **bound_fill_in(n, degree, target_density=args.target_density),
                        seconds=time.monotonic() - started,
                    )
                except (OSError, ValueError, KeyError) as error:
                    result.update(status="error", error=str(error))
        if result["status"] == "ok":
            print(
                f"  {name}: max_degree={result['max_degree']} "
                f"squarings={result['squarings']} ({result['seconds']:.1f}s)",
                flush=True,
            )
        else:
            print(f"  {name}: {result['status']}: {result['error']}", flush=True)
        document["matrices"].append(result)
        document["steps"] = steps(document["matrices"])
        write(document, args.output)
    print_steps(document["steps"], args.target_density)
    print(f"Saved {args.output}")
    return int(any(item["status"] != "ok" for item in document["matrices"]))


if __name__ == "__main__":
    raise SystemExit(main())
