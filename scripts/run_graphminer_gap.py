#!/usr/bin/env python3
"""Compute missing GAP four-clique references with GraphMiner's CPU counter.

Build src/clique's clique_omp_base target first. Prepared SAPS GAP inputs are
loaded through the shared generator (SAPS_CACHE_DIR / SAPS_MANIFEST_PATH apply).
The default selection is GAP graphs without a four-clique reference count.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path

import numpy as np

from binsparse.conversions import to_scipy

from saps.benchmarks.four_clique_counting import _FOUR_CLIQUE_COUNTS
from saps.benchmarks.gap import GAPGraphGenerator, fetch_gap_graph
from saps.util.adjacency import zero_one_adjacency

SEMANTICS = "simple_undirected"


def write_json(path, value):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def export_graph(adjacency, prefix):
    """Write symmetric Boolean CSR in GraphMiner's native-endian binary format.

    GraphMiner orients the graph itself. Giving it the benchmark's strict upper
    triangle would discard edges during degree orientation and undercount.
    Use the benchmark's shared normalization, retaining both edge directions.
    """
    graph = to_scipy(
        zero_one_adjacency(adjacency, bool, simple_undirected=True)
    ).tocsr()
    graph.sort_indices()
    n = graph.shape[0]
    if graph.shape != (n, n) or n > np.iinfo(np.int32).max:
        raise ValueError("GraphMiner requires a square graph with int32 vertex IDs")
    max_degree = int(np.diff(graph.indptr).max(initial=0))
    if not max_degree:
        raise ValueError("GraphMiner requires a graph with at least one edge")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = prefix.with_suffix(".meta.txt")
    metadata_path.unlink(missing_ok=True)
    # Chunk casts avoid allocating another full copy of billion-edge arrays.
    for suffix, values, dtype in (
        (".vertex.bin", graph.indptr, np.int64),
        (".edge.bin", graph.indices, np.int32),
    ):
        with prefix.with_suffix(suffix).open("wb") as output:
            for start in range(0, len(values), 1_000_000):
                values[start : start + 1_000_000].astype(dtype).tofile(output)
    metadata_path.write_text(f"{n}\n{graph.nnz}\n4 8 1 2\n{max_degree}\n0\n0\n0\n")
    return {
        "vertices": n,
        "undirected_edges": int(graph.nnz // 2),
        "csr_nnz": int(graph.nnz),
        "max_degree": max_degree,
        "adjacency": SEMANTICS,
    }


def run_counter(binary, prefix, log_path, threads):
    command = [str(binary), str(prefix), "4"]
    started = time.perf_counter()
    with log_path.open("w") as log:
        subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env={**os.environ, "OMP_NUM_THREADS": str(threads)},
            check=True,
        )
    matches = re.findall(r"^num_4-cliques = (\d+)\s*$", log_path.read_text(), re.M)
    if len(matches) != 1:
        raise ValueError(f"Expected one GraphMiner four-clique count in {log_path}")
    count = int(matches[0])
    if count > np.iinfo(np.int64).max:
        raise ValueError("Four-clique count exceeds the benchmark's int64 output range")
    return count, time.perf_counter() - started, command


def main(argv=None):
    names = [d.name for d in GAPGraphGenerator().datasets]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphminer-bin", type=Path, required=True)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=names,
        default=[name for name in names if name not in _FOUR_CLIQUE_COUNTS],
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path(".saps/outputs/graphminer-gap")
    )
    parser.add_argument(
        "--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip completed results from this binary; assumes unchanged GAP inputs",
    )
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    binary = args.graphminer_bin.resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        parser.error(f"GraphMiner executable not found or not executable: {binary}")
    with binary.open("rb") as source:
        binary_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:
        result_path = output_dir / f"{name}.json"
        expected = _FOUR_CLIQUE_COUNTS.get(name)
        if args.resume and result_path.exists():
            previous = json.loads(result_path.read_text())
            if (
                previous.get("dataset") == name
                and previous.get("adjacency") == SEMANTICS
                and previous.get("binary_sha256") == binary_sha256
                and type(previous.get("four_cliques")) is int
                and previous["four_cliques"] >= 0
                and (expected is None or previous["four_cliques"] == expected)
            ):
                print(
                    f"{name}: reuse {previous['four_cliques']} four-cliques", flush=True
                )
                continue
        # Invalidate a previous result before attempting a fresh run.
        result_path.unlink(missing_ok=True)
        prefix = output_dir / name / "graph"
        log_path = output_dir / f"{name}.log"
        print(f"{name}: preparing GraphMiner input", flush=True)
        started = time.perf_counter()
        raw = fetch_gap_graph(name)
        metadata = export_graph(raw.inputs[0], prefix)
        del raw
        gc.collect()  # Release the Python graph before GraphMiner loads its copy.
        preparation_seconds = time.perf_counter() - started
        print(f"{name}: counting four-cliques; log: {log_path}", flush=True)
        count, elapsed, command = run_counter(binary, prefix, log_path, args.threads)
        if expected is not None and count != expected:
            raise ValueError(f"{name}: got {count} four-cliques; expected {expected}")
        write_json(
            result_path,
            {
                **metadata,
                "dataset": name,
                "four_cliques": count,
                "expected_four_cliques": expected,
                "binary_sha256": binary_sha256,
                "command": command,
                "threads": args.threads,
                "seconds": {"prepare": preparation_seconds, "graphminer": elapsed},
                "log": str(log_path),
            },
        )
        print(f"{name}: {count} four-cliques; result: {result_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
