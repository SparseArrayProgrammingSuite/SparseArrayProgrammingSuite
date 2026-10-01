#!/usr/bin/env python3
"""Measure Boolean fill-in of (I | A) after repeated squaring.

Read prepared, row-ordered COO Binsparse files directly from the shared cache.
Breadth-first searches count each selected row of (I | A)**(2**s), without
materializing any squared matrix. Small graphs (or --exact) use every row;
large graphs use uniformly sampled rows with replacement. Confidence bounds
are simultaneous Hoeffding bounds, not assumptions about graph structure.
Traversal limits produce explicit lower/upper bounds, never a false zero.

Example:
    poetry run python scripts/measure_fill_in.py --output fill-in.json
    poetry run python scripts/measure_fill_in.py --datasets GAP/GAP-road --exact

Missing cache files are reported without downloading multi-gigabyte datasets.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

import h5py

REPO_ROOT = Path(__file__).resolve().parents[1]
GAP_DATASETS = [
    f"GAP/GAP-{name}" for name in ("road", "twitter", "web", "kron", "urand")
]


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


def _searchsorted(array, value, side):
    if isinstance(array, np.ndarray):
        info = np.iinfo(array.dtype)
        if value < info.min:
            return 0
        if value > info.max:
            return len(array)
        # A Python int can make NumPy cast the entire int32 coordinate array
        # to int64 for each search. Match the key dtype to keep lookup O(log n).
        return int(np.searchsorted(array, array.dtype.type(value), side=side))
    left, right = 0, len(array)
    while left < right:
        middle = (left + right) // 2
        item = array[middle]
        if item < value or (side == "right" and item == value):
            left = middle + 1
        else:
            right = middle
    return left


class CooGraph:
    def __init__(self, n, rows, columns, values):
        self.n = n
        self.rows = rows
        self.columns = columns
        self.values = values
        self.max_stored_degree = None

    def validate(self):
        if any(array.ndim != 1 for array in (self.rows, self.columns, self.values)):
            raise ValueError("Expected one-dimensional COO coordinate/value arrays")
        if not (len(self.rows) == len(self.columns) == len(self.values)):
            raise ValueError("COO coordinate/value lengths differ")
        if self.rows.dtype.kind not in "iu" or self.columns.dtype.kind not in "iu":
            raise ValueError("COO coordinates must be integers")
        previous = -1
        tail_count = 0
        maximum_degree = 0
        for start in range(0, len(self.rows), 1_000_000):
            rows = self.rows[start : start + 1_000_000]
            columns = self.columns[start : start + 1_000_000]
            if rows[0] < previous or np.any(rows[1:] < rows[:-1]):
                raise ValueError("COOR rows must be sorted for on-disk row lookup")
            if (
                rows[0] < 0
                or rows[-1] >= self.n
                or np.any(columns < 0)
                or np.any(columns >= self.n)
            ):
                raise ValueError("COO coordinates fall outside the matrix shape")
            boundaries = np.flatnonzero(rows[1:] != rows[:-1]) + 1
            counts = np.diff(np.concatenate(([0], boundaries, [len(rows)])))
            if rows[0] == previous:
                counts[0] += tail_count
            maximum_degree = max(maximum_degree, int(counts.max()))
            tail_count = int(counts[-1])
            previous = int(rows[-1])
        self.max_stored_degree = maximum_degree

    def row_bounds(self, row):
        return (
            _searchsorted(self.rows, row, "left"),
            _searchsorted(self.rows, row, "right"),
        )


@contextmanager
def open_graph(path):
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
        graph = CooGraph(
            int(shape[0]),
            _array(tensor["indices_0"], path),
            _array(tensor["indices_1"], path),
            _array(tensor["values"], path),
        )
        graph.validate()
        yield graph


def count_row(graph, source, hops, *, max_visited, max_edges, deadline):
    """Return exact counts at completed depths and bounds at unfinished depths."""
    visited = {int(source)}
    frontier = [int(source)]
    counts = {}
    edges_examined = 0
    reason = None
    for depth in range(1, max(hops) + 1):
        following = []
        for row in frontier:
            if time.monotonic() >= deadline:
                reason = "time_limit"
                break
            left, right = graph.row_bounds(row)
            if edges_examined + right - left > max_edges:
                reason = "edge_limit"
                break
            edges_examined += right - left
            for start in range(left, right, 4096):
                if time.monotonic() >= deadline:
                    reason = "time_limit"
                    break
                end = min(start + 4096, right)
                columns = graph.columns[start:end]
                values = graph.values[start:end]
                new = set(map(int, columns[values != 0])) - visited
                capacity = max_visited - len(visited)
                if len(new) > capacity:
                    visited.update(sorted(new)[:capacity])
                    reason = "visited_limit"
                    break
                visited.update(new)
                following.extend(new)
            if reason:
                break
        if reason:
            break
        if depth in hops:
            counts[depth] = (len(visited), len(visited))
        if not following or len(visited) == graph.n:
            counts.update(
                {hop: (len(visited), len(visited)) for hop in hops if hop > depth}
            )
            break
        frontier = following
    for hop in hops:
        counts.setdefault(hop, (len(visited), graph.n))
    return counts, reason


def measure(
    graph,
    *,
    squarings=2,
    samples=1024,
    seed=0,
    exact=False,
    max_visited=1_000_000,
    max_edges=10_000_000,
    seconds=120,
    confidence=0.95,
    dense_threshold=0.5,
):
    hops = [2**step for step in range(squarings + 1)]
    if graph.max_stored_degree is None:
        graph.validate()
    # At most 1 + d + ... + d**h vertices are reachable within h hops when
    # every row stores at most d entries. Stored zeros/duplicates only loosen
    # this bound. It also tightens the range used by Hoeffding's inequality.
    row_bounds = {}
    for hop in hops:
        degree = graph.max_stored_degree
        bound = min(graph.n, 1 + degree * hop)
        if degree > 1:
            bound = 1
            for _ in range(hop):
                bound = min(graph.n, 1 + degree * bound)
                if bound == graph.n:
                    break
        row_bounds[hop] = bound
    census = exact or graph.n <= samples
    sample_count = graph.n if census else samples
    selected = (
        range(graph.n)
        if census
        else np.random.default_rng(seed).integers(0, graph.n, size=sample_count)
    )
    lower = dict.fromkeys(hops, 0)
    upper = dict.fromkeys(hops, 0)
    completed = dict.fromkeys(hops, 0)
    limits = {}
    started = time.monotonic()
    deadline = started + seconds
    attempted = 0
    for source in selected:
        if time.monotonic() >= deadline:
            break
        counts, reason = count_row(
            graph,
            source,
            hops,
            max_visited=max_visited,
            max_edges=max_edges,
            deadline=deadline,
        )
        attempted += 1
        if reason:
            limits[reason] = limits.get(reason, 0) + 1
        for hop, (lo, hi) in counts.items():
            hi = min(hi, row_bounds[hop])
            lower[hop] += lo
            upper[hop] += hi
            completed[hop] += lo == hi
    missing = sample_count - attempted
    if missing:
        limits["unattempted_time_limit"] = missing
        for hop in hops:
            lower[hop] += missing  # The identity guarantees the source itself.
            upper[hop] += missing * row_bounds[hop]
    margin = (
        0.0
        if census
        else math.sqrt(math.log(2 * len(hops) / (1 - confidence)) / (2 * sample_count))
    )
    stages = []
    for step, hop in enumerate(hops):
        all_counted = completed[hop] == sample_count
        cells = graph.n**2
        denominator = sample_count * graph.n
        row_margin = margin * (row_bounds[hop] - 1) / graph.n if graph.n else 0.0
        density_lo = (
            max(1 / graph.n, lower[hop] / denominator - row_margin)
            if denominator
            else 0.0
        )
        density_hi = (
            min(row_bounds[hop] / graph.n, upper[hop] / denominator + row_margin)
            if denominator
            else 0.0
        )
        nnz = lower[hop] if census and all_counted else None
        estimated = (
            lower[hop] * graph.n / sample_count
            if all_counted and sample_count
            else None
        )
        if nnz is not None:
            estimated = nnz
        nnz_bounds = (
            [lower[hop], upper[hop]]
            if census
            else [math.floor(density_lo * cells), math.ceil(density_hi * cells)]
        )
        stages.append(
            {
                "squarings": step,
                "max_path_length": hop,
                "nnz_exact": nnz,
                "nnz_estimate": estimated,
                "nnz_bounds": nnz_bounds,
                "density_estimate": estimated / cells
                if cells and estimated is not None
                else (0.0 if not cells else None),
                "density_bounds": [density_lo, density_hi],
                "row_nnz_upper_bound": row_bounds[hop],
                "completed_rows": completed[hop],
                "classification": "dense"
                if density_lo >= dense_threshold
                else ("not_dense" if density_hi < dense_threshold else "undetermined"),
                "bool_csr64_bytes_estimate": 9 * estimated + 8 * (graph.n + 1)
                if estimated is not None
                else None,
            }
        )
    initial = stages[0]["nnz_estimate"]
    for stage in stages:
        value = stage["nnz_estimate"]
        stage["fill_factor_estimate"] = (
            value / initial if initial and value is not None else None
        )
    return {
        "status": "bounded" if limits else "ok",
        "n": graph.n,
        "max_stored_out_degree": graph.max_stored_degree,
        "method": "all_rows" if census else "uniform_rows_with_replacement",
        "sample_count": sample_count,
        "attempted_rows": attempted,
        "seed": seed,
        "confidence": confidence,
        "bounds": "deterministic" if census else "simultaneous_hoeffding",
        "limits_hit": limits,
        "traversal_seconds": time.monotonic() - started,
        "dense_bool_bytes": graph.n**2,
        "stages": stages,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--datasets", nargs="+", default=GAP_DATASETS)
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--squarings", type=int, default=2)
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--exact", action="store_true", help="Count all rows (can be very expensive)"
    )
    parser.add_argument(
        "--max-visited",
        type=int,
        default=1_000_000,
        help="Maximum visited vertices per sampled row",
    )
    parser.add_argument(
        "--max-edges",
        type=int,
        default=10_000_000,
        help="Maximum COO entries examined per sampled row",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=120,
        help="Traversal time budget per matrix; excludes input validation",
    )
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument(
        "--dense-threshold",
        type=float,
        default=0.5,
        help="Density fraction called dense (default: 0.5); all densities are reported",
    )
    args = parser.parse_args(argv)
    if (
        args.squarings < 0
        or min(args.samples, args.max_visited, args.max_edges, args.seconds) <= 0
        or not math.isfinite(args.seconds)
    ):
        parser.error(
            "Squarings must be nonnegative; sample counts and budgets must be positive"
        )
    if not 0 < args.confidence < 1 or not 0 < args.dense_threshold <= 1:
        parser.error("Require 0 < confidence < 1 and 0 < dense-threshold <= 1")
    manifest = json.loads(args.manifest.read_text())
    document = {
        "semantics": (
            "Boolean (I OR (A != 0)) raised to 2**squarings; original edge direction"
        ),
        "dense_threshold": args.dense_threshold,
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "matrices": [],
    }
    for name in args.datasets:
        key = f"suitesparse_matrix.{name}"
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
                print(f"Measuring {name}...", flush=True)
                try:
                    with open_graph(path) as graph:
                        result.update(
                            measure(
                                graph,
                                **{
                                    key: getattr(args, key)
                                    for key in (
                                        "squarings",
                                        "samples",
                                        "seed",
                                        "exact",
                                        "max_visited",
                                        "max_edges",
                                        "seconds",
                                        "confidence",
                                        "dense_threshold",
                                    )
                                },
                            )
                        )
                except (OSError, ValueError, KeyError) as error:
                    result.update(status="error", error=str(error))
        document["matrices"].append(result)
        if "stages" in result:
            for stage in result["stages"]:
                density = stage["density_estimate"]
                density_text = "bounded" if density is None else f"{density:.6%}"
                print(
                    f"  {name}: squarings={stage['squarings']} "
                    f"density={density_text} classification={stage['classification']}",
                    flush=True,
                )
        else:
            print(f"  {name}: {result['status']}: {result['error']}", flush=True)
        classifications = [
            item["stages"][-1]["classification"] if "stages" in item else "unmeasured"
            for item in document["matrices"]
        ]
        document["summary"] = {
            label: classifications.count(label)
            for label in (
                "dense",
                "not_dense",
                "undetermined",
                "unmeasured",
            )
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
    print(
        f"After {args.squarings} squarings: {document['summary']}; saved {args.output}"
    )
    return int(any(item["status"] != "ok" for item in document["matrices"]))


if __name__ == "__main__":
    raise SystemExit(main())
