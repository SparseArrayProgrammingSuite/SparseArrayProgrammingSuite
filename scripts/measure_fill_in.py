#!/usr/bin/env python3
"""Count Boolean squarings of (I | A) until its density reaches a target.

Read prepared, row-ordered COO Binsparse files directly from the shared cache.
Level-synchronous breadth-first searches count each selected row of
(I | A)**(2**s) for s = 0, 1, 2, ..., without materializing any squared matrix.
Squaring stops at the first s whose estimated density reaches --target-density
(default 1%), or once the counted rows stop changing (the transitive closure).
The output leads with one step count per matrix. Small graphs (or --exact) use
every row; large graphs use uniformly sampled rows with replacement. Confidence
bounds are simultaneous Hoeffding bounds over every squaring that could be
measured, not assumptions about graph structure. Traversal limits produce
explicit lower/upper bounds, never a false zero.

Example:
    poetry run python scripts/measure_fill_in.py --output fill-in.json
    poetry run python scripts/measure_fill_in.py --datasets GAP/GAP-road --exact
    poetry run python scripts/measure_fill_in.py --merge run/*/*.json --output all.json

Dataset names without a slash select every manifest entry in that SuiteSparse
group; the default is all GAP and SNAP matrices. Missing cache files are
reported without downloading multi-gigabyte datasets.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from contextlib import contextmanager
from functools import partial
from pathlib import Path

import numpy as np

import h5py

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PREFIX = "suitesparse_matrix."
DEFAULT_GROUPS = ["GAP", "SNAP"]
SEMANTICS = "Boolean (I OR (A != 0)) raised to 2**squarings; original edge direction"
# A row reaching this multiple of the target row count already certifies its
# share of the target density, so counting further only adds traversal work.
VISITED_TARGET_MULTIPLE = 4


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


class CooGraph:
    def __init__(self, n, rows, columns, values):
        self.n = n
        self.rows = rows
        self.columns = columns
        self.values = values
        self.indptr = None
        self.max_stored_degree = None

    def validate(self):
        """Check the COOR structure and build CSR row offsets (8 bytes per row)."""
        if any(array.ndim != 1 for array in (self.rows, self.columns, self.values)):
            raise ValueError("Expected one-dimensional COO coordinate/value arrays")
        if not (len(self.rows) == len(self.columns) == len(self.values)):
            raise ValueError("COO coordinate/value lengths differ")
        if self.rows.dtype.kind not in "iu" or self.columns.dtype.kind not in "iu":
            raise ValueError("COO coordinates must be integers")
        indptr = np.zeros(self.n + 1, dtype=np.int64)
        previous = -1
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
            first = int(rows[0])
            counts = np.bincount(rows - rows[0])
            indptr[first + 1 : first + 1 + len(counts)] += counts
            previous = int(rows[-1])
        self.max_stored_degree = int(indptr.max())
        self.indptr = np.cumsum(indptr, out=indptr)

    def neighbors(self, frontier):
        """Return stored nonzero columns of the rows in ``frontier``, with repeats."""
        starts = self.indptr[frontier]
        lengths = self.indptr[frontier + 1] - starts
        if isinstance(self.columns, np.ndarray):
            # Gather every row range at once instead of looping in Python.
            offsets = np.cumsum(lengths) - lengths
            index = np.repeat(starts - offsets, lengths) + np.arange(lengths.sum())
            columns = self.columns[index]
            values = self.values[index]
        else:
            ranges = [
                (start, start + length)
                for start, length in zip(starts.tolist(), lengths.tolist(), strict=True)
                if length
            ]
            if not ranges:
                return np.empty(0, dtype=np.int64)
            columns = np.concatenate([self.columns[s:e] for s, e in ranges])
            values = np.concatenate([self.values[s:e] for s, e in ranges])
        return columns[values != 0]


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


def count_row(graph, source, hop, visited, *, max_visited, max_edges, deadline):
    """Count row ``source`` of (I | A)**hop by breadth-first search.

    Return ``(count, closed, reason)``. The count is exact unless ``reason``
    names the traversal limit that stopped the search; it is then a lower bound.
    ``closed`` means no farther vertex is reachable. ``visited`` is an all-False
    scratch bitmap with one entry per vertex, and is left all-False.
    """
    frontier = np.array([source], dtype=np.int64)
    reached = [frontier]
    visited[frontier] = True
    count = 1
    edges = 0
    reason = None
    closed = False
    try:
        for _ in range(hop):
            if time.monotonic() >= deadline:
                reason = "time_limit"
                break
            lengths = graph.indptr[frontier + 1] - graph.indptr[frontier]
            fits = int(
                np.searchsorted(np.cumsum(lengths), max_edges - edges, side="right")
            )
            if fits < len(frontier):
                frontier = frontier[:fits]
                reason = "edge_limit"
            edges += int(lengths[:fits].sum())
            columns = graph.neighbors(frontier)
            new = np.unique(columns[~visited[columns]])
            if count + len(new) > max_visited:
                new = new[: max_visited - count]
                reason = "visited_limit"
            visited[new] = True
            reached.append(new)
            count += len(new)
            if reason:
                break
            if not len(new) or count == graph.n:
                closed = True
                break
            frontier = new
    finally:
        for vertices in reached:
            visited[vertices] = False
    return count, closed, reason


def reachable_bound(n, degree, hop):
    """Bound the nonzeros in a row of (I | A)**hop.

    At most 1 + d + ... + d**h vertices are reachable within h hops when every
    row stores at most d entries. Stored zeros/duplicates only loosen this
    bound. It also tightens the range used by Hoeffding's inequality.
    """
    if degree <= 1:
        return min(n, 1 + degree * hop)
    bound = 1
    for _ in range(hop):
        bound = min(n, 1 + degree * bound)
        if bound == n:
            break
    return bound


def measure(
    graph,
    *,
    target_density=0.01,
    max_squarings=None,
    samples=1024,
    seed=0,
    exact=False,
    max_visited=None,
    max_edges=10_000_000,
    seconds=120,
    confidence=0.95,
    log=None,
):
    """Square (I | A) until its density reaches ``target_density`` or closes.

    ``result["squarings"]`` is the first squaring count whose density reaches
    the target or, for matrices that close below it, the count at which the
    closure is reached. It is None when limits leave it uncertain;
    ``result["squarings_bounds"]`` then brackets it (None means unbounded).
    """
    if graph.indptr is None:
        graph.validate()
    n = graph.n
    # Paths of n - 1 edges reach every reachable vertex, so no squaring past
    # the first s with 2**s >= n - 1 adds entries.
    last = max(n - 2, 0).bit_length()
    if max_squarings is not None:
        last = min(last, max_squarings)
    if max_visited is None:
        max_visited = max(
            1_000_000, math.ceil(VISITED_TARGET_MULTIPLE * target_density * n)
        )
    census = exact or n <= samples
    sample_count = n if census else samples
    selected = (
        np.arange(n)
        if census
        else np.random.default_rng(seed).integers(0, n, size=sample_count)
    )
    # Row counts only grow with the squaring count. Closed rows keep their
    # counts, and rows stopped by a visited or edge limit would stop at the same
    # place again, so each stage searches only the remaining rows (from scratch).
    lower = np.ones(sample_count, dtype=np.int64)  # The identity: the source.
    closed = np.zeros(sample_count, dtype=bool)
    stuck = np.zeros(sample_count, dtype=bool)
    visited = np.zeros(n, dtype=bool)
    margin = (
        0.0
        if census
        else math.sqrt(math.log(2 * (last + 1) / (1 - confidence)) / (2 * sample_count))
    )
    limits = {}
    stages = []
    totals = []
    stop = "max_squarings"
    started = time.monotonic()
    deadline = started + seconds
    for step in range(last + 1):
        hop = 2**step
        row_bound = reachable_bound(n, graph.max_stored_degree, hop)
        upper = np.where(closed, lower, row_bound)
        for index in np.flatnonzero(~closed & ~stuck):
            if time.monotonic() >= deadline:
                limits["unattempted_time_limit"] = (
                    limits.get("unattempted_time_limit", 0) + 1
                )
                continue
            count, row_closed, reason = count_row(
                graph,
                selected[index],
                hop,
                visited,
                max_visited=max_visited,
                max_edges=max_edges,
                deadline=deadline,
            )
            # A partial search can count fewer vertices than the last stage did.
            lower[index] = max(lower[index], count)
            if reason:
                limits[reason] = limits.get(reason, 0) + 1
                stuck[index] = reason != "time_limit"
            else:
                upper[index] = count
                closed[index] = row_closed or hop >= n - 1
        lo, hi = int(lower.sum()), int(upper.sum())
        totals.append(lo)
        complete = int(np.count_nonzero(lower == upper))
        all_counted = complete == sample_count
        cells = n**2
        denominator = sample_count * n
        estimate_bounds = (
            [lo / denominator, hi / denominator] if denominator else [0.0, 0.0]
        )
        row_margin = margin * (row_bound - 1) / n if n else 0.0
        density_lo = max(1 / n, estimate_bounds[0] - row_margin) if denominator else 0.0
        density_hi = (
            min(row_bound / n, estimate_bounds[1] + row_margin) if denominator else 0.0
        )
        nnz = lo if census and all_counted else None
        estimated = lo * n / sample_count if all_counted and sample_count else None
        if nnz is not None:
            estimated = nnz
        stage = {
            "squarings": step,
            "max_path_length": hop,
            "nnz_exact": nnz,
            "nnz_estimate": estimated,
            "nnz_bounds": [lo, hi]
            if census
            else [math.floor(density_lo * cells), math.ceil(density_hi * cells)],
            "density_estimate": estimated / cells
            if cells and estimated is not None
            else (0.0 if not cells else None),
            # Bounds on the sample estimate itself, from traversal limits only.
            "density_estimate_bounds": estimate_bounds,
            "density_bounds": [density_lo, density_hi],
            "row_nnz_upper_bound": row_bound,
            "completed_rows": complete,
            "target": "reached"
            if estimate_bounds[0] >= target_density
            else ("below" if estimate_bounds[1] < target_density else "undetermined"),
            "bool_csr64_bytes_estimate": 9 * estimated + 8 * (n + 1)
            if estimated is not None
            else None,
            "elapsed_seconds": time.monotonic() - started,
        }
        stages.append(stage)
        if log:
            log(stage)
        if stage["target"] == "reached":
            stop = "target"
            break
        if closed.all():
            stop = "closure"
            break
        if time.monotonic() >= deadline:
            stop = "time_limit"
            break
        if (closed | stuck).all():
            stop = "traversal_limit"
            break
    initial = stages[0]["nnz_estimate"]
    for stage in stages:
        value = stage["nnz_estimate"]
        stage["fill_factor_estimate"] = (
            value / initial if initial and value is not None else None
        )
    possible = next(
        (stage["squarings"] for stage in stages if stage["target"] != "below"), None
    )
    if stop == "target":
        bounds = [possible, len(stages) - 1]
    elif stop == "closure":
        # Every closed row was counted exactly at every stage, and rows only
        # grow, so the first stage with the final total is already closed.
        closure = totals.index(totals[-1])
        bounds = [closure, closure]
    else:
        bounds = [len(stages) if possible is None else possible, None]
    return {
        "status": "bounded" if limits else "ok",
        "n": n,
        "max_stored_out_degree": graph.max_stored_degree,
        "target_density": target_density,
        "squarings": bounds[0] if bounds[0] == bounds[1] else None,
        "squarings_bounds": bounds,
        "stop_reason": stop,
        "method": "all_rows" if census else "uniform_rows_with_replacement",
        "sample_count": sample_count,
        "seed": seed,
        "confidence": confidence,
        "bounds": "deterministic" if census else "simultaneous_hoeffding",
        "max_visited": max_visited,
        "limits_hit": limits,
        "traversal_seconds": time.monotonic() - started,
        "dense_bool_bytes": n**2,
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
    """One entry per matrix: squarings until the target density or closure."""
    return [
        {
            "dataset": item["dataset"],
            "squarings": item.get("squarings"),
            "squarings_bounds": item.get("squarings_bounds"),
            "stop_reason": item.get("stop_reason", item["status"]),
            **{
                key: item["stages"][-1][key] if "stages" in item else None
                for key in ("density_estimate", "density_estimate_bounds")
            },
        }
        for item in matrices
    ]


def print_stage(name, stage):
    density = stage["density_estimate"]
    density_text = "bounded" if density is None else f"{density:.6%}"
    print(
        f"  {name}: squarings={stage['squarings']} density={density_text} "
        f"target={stage['target']} ({stage['elapsed_seconds']:.1f}s)",
        flush=True,
    )


def print_steps(entries, target_density):
    print(f"Squarings until density >= {100 * target_density:g}% (or closure):")
    width = max((len(entry["dataset"]) for entry in entries), default=0)
    for entry in entries:
        bounds = entry["squarings_bounds"]
        if entry["squarings"] is not None:
            value = str(entry["squarings"])
        elif bounds is None:
            value = "-"
        elif bounds[1] is None:
            value = f">={bounds[0]}"
        else:
            value = f"{bounds[0]}-{bounds[1]}"
        density = entry["density_estimate"]
        density_bounds = entry["density_estimate_bounds"]
        if density is not None:
            density_text = f"  density={density:.4%}"
        elif density_bounds is not None:
            density_text = f"  density={density_bounds[0]:.4%}..{density_bounds[1]:.4%}"
        else:
            density_text = ""
        print(
            f"  {entry['dataset']:<{width}}  {value:>5}  "
            f"{entry['stop_reason']}{density_text}"
        )


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
        help="Stop squaring once the density reaches this fraction (default: 0.01)",
    )
    parser.add_argument(
        "--max-squarings",
        type=int,
        help="Stop after this many squarings (default: until the closure)",
    )
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--exact", action="store_true", help="Count all rows (can be very expensive)"
    )
    parser.add_argument(
        "--max-visited",
        type=int,
        help=(
            "Maximum vertices counted per row (default: the larger of 1,000,000"
            f" and {VISITED_TARGET_MULTIPLE}x the target row count)"
        ),
    )
    parser.add_argument(
        "--max-edges",
        type=int,
        default=10_000_000,
        help="Maximum COO entries examined per row search",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=120,
        help="Traversal time budget per matrix across all squarings; excludes"
        " input validation",
    )
    parser.add_argument("--confidence", type=float, default=0.95)
    args = parser.parse_args(argv)
    if (
        (args.max_squarings is not None and args.max_squarings < 0)
        or (args.max_visited is not None and args.max_visited <= 0)
        or min(args.samples, args.max_edges, args.seconds) <= 0
        or not math.isfinite(args.seconds)
    ):
        parser.error(
            "Squarings must be nonnegative; sample counts and budgets must be positive"
        )
    if not 0 < args.confidence < 1 or not 0 < args.target_density <= 1:
        parser.error("Require 0 < confidence < 1 and 0 < target-density <= 1")
    if args.output is None and not args.list_datasets:
        parser.error("--output is required")
    if args.merge:
        try:
            document = merge(args.merge, args.output)
        except (OSError, ValueError, KeyError) as error:
            parser.error(str(error))
        print_steps(document["steps"], document["target_density"])
        return int(any(entry["squarings"] is None for entry in document["steps"]))
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
                print(f"Measuring {name}...", flush=True)
                try:
                    with open_graph(path) as graph:
                        result.update(
                            measure(
                                graph,
                                **{
                                    key: getattr(args, key)
                                    for key in (
                                        "target_density",
                                        "max_squarings",
                                        "samples",
                                        "seed",
                                        "exact",
                                        "max_visited",
                                        "max_edges",
                                        "seconds",
                                        "confidence",
                                    )
                                },
                                log=partial(print_stage, name),
                            )
                        )
                except (OSError, ValueError, KeyError) as error:
                    result.update(status="error", error=str(error))
        if "stages" not in result:
            print(f"  {name}: {result['status']}: {result['error']}", flush=True)
        document["matrices"].append(result)
        document["steps"] = steps(document["matrices"])
        write(document, args.output)
    print_steps(document["steps"], args.target_density)
    print(f"Saved {args.output}")
    return int(any(entry["squarings"] is None for entry in document["steps"]))


if __name__ == "__main__":
    raise SystemExit(main())
