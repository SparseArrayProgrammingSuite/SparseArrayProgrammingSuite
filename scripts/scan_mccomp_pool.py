#!/usr/bin/env python3
"""List the Model Counting Competition instances that NumPy counts in seconds.

Downloads and unpacks the archives in `MCCOMP_POOL_ARCHIVES`, keeps the DIMACS
CNF instances within the limits below, drops repeats of a formula already kept,
and prints the source paths of one inventory in
`src/saps/benchmarks/mccomp_pool_instance.py`: Track 1's for model counting, or
with --weighted, Track 2's and Track 4's for weighted model counting:

    pixi run python scripts/scan_mccomp_pool.py > model_counting.txt
    pixi run python scripts/scan_mccomp_pool.py --weighted > weighted.txt

A repeat keeps the path of its first copy, taking the archives in the order
`MCCOMP_POOL_ARCHIVES` lists them (competition instances first) and the files of
an archive by path.
"""

from __future__ import annotations

import argparse
import collections
import functools
import hashlib
import lzma
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from saps.benchmarks.weighted_model_counting import parse_format, parse_shown
from saps.downloaders.mccomp import (
    MCCOMP_POOL_ARCHIVES,
    download_mccomp_pool_archive,
    parse_dimacs,
)

# Both benchmarks build one einsum over every variable, which NumPy evaluates as
# a dense Boolean tensor of 2**vars entries, one clause at a time. On an Apple
# M-series laptop, it takes about 0.65 ns per clause per entry once vars reaches
# 24, and peaks at 2.3 GiB with 30 variables.
MAX_VARS = 30
# Their einsum parser nests one call per clause, and from 498 clauses on,
# NumPy's evaluation exceeds Python's default recursion limit. The weighted
# benchmark adds a clause (v or not v) for each shown variable v that no clause
# mentions.
MAX_CLAUSES = 450
# clauses * 2**vars, which keeps each count within about 15 seconds there.
MAX_WORK = 2**34
# The weighted benchmark then multiplies the shown variables' weights over
# their 2**shown assignments, at about 2.3 ns per weight per assignment, so
# shown * 2**shown at most this keeps that within about 10 seconds.
MAX_WEIGHT_WORK = 2**32


def _header(path: Path) -> tuple[int, int] | None:
    with lzma.open(path, "rt", encoding="utf-8", errors="replace") as file:
        for line in file:
            if line.startswith("p "):
                parts = line.split()
                if len(parts) != 4 or parts[1] != "cnf":
                    return None
                return int(parts[2]), int(parts[3])
    return None


def _within_limits(num_vars: int, num_clauses: int) -> bool:
    return (
        num_vars <= MAX_VARS
        and num_clauses <= MAX_CLAUSES
        and num_clauses * 2**num_vars <= MAX_WORK
    )


def _scan(path: Path, weighted: bool) -> tuple[str, str | None]:
    try:
        header = _header(path)
    except (lzma.LZMAError, EOFError, ValueError):
        return "unreadable", None
    if header is None:
        return "not cnf", None
    # The weighted benchmark only adds clauses, so this rules out no instance
    # within the limits.
    if not _within_limits(*header):
        return "over limit", None
    with lzma.open(path, "rt", encoding="utf-8", errors="replace") as file:
        text = file.read()
    if not weighted:
        try:
            num_vars, clauses = parse_dimacs(text)
        except ValueError:
            return "malformed", None
        # The benchmark's einsum only loops over variables in some clause.
        mentioned = {abs(literal) for clause in clauses for literal in clause}
        if len(mentioned) != num_vars:
            return "unused variables", None
        key: tuple = (num_vars, clauses)
    else:
        problem_types = [
            line.split()[2] for line in text.splitlines() if line.startswith("c t ")
        ]
        if problem_types not in (["wmc"], ["pwmc"]):
            return "not weighted", None
        try:
            num_vars, clauses, weights = parse_format(text)
        except ValueError:
            return "malformed", None
        shown = parse_shown(text, num_vars)
        mentioned = {abs(literal) for clause in clauses for literal in clause}
        added = len([var for var in shown if var not in mentioned])
        if (
            not _within_limits(num_vars, len(clauses) + added)
            or len(shown) * 2 ** len(shown) > MAX_WEIGHT_WORK
        ):
            return "over limit", None
        shown_weights = [(weights[var], weights[-var]) for var in shown]
        key = (num_vars, clauses, shown, shown_weights)
    return "kept", hashlib.sha256(repr(key).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--weighted",
        action="store_true",
        help="List Track 2 and Track 4 instances instead of Track 1's.",
    )
    args = parser.parse_args()

    paths: list[Path] = []
    for archive in MCCOMP_POOL_ARCHIVES:
        if ("-track1-" in archive) == args.weighted:
            continue
        directory = download_mccomp_pool_archive(archive)
        paths += sorted(
            path
            for path in directory.rglob("*.xz")
            if path.is_file() and not path.name.startswith("._")
        )
    # Every archive unpacks into a directory of its own name under one root.
    root = directory.parent

    statuses: collections.Counter[str] = collections.Counter()
    kept: dict[str, str] = {}
    scan = functools.partial(_scan, weighted=args.weighted)
    with ProcessPoolExecutor() as executor:
        for path, (status, digest) in zip(
            paths, executor.map(scan, paths, chunksize=64), strict=True
        ):
            if digest is not None and digest in kept:
                status = "repeat"
            statuses[status] += 1
            if status == "kept" and digest is not None:
                kept[digest] = path.relative_to(root).as_posix()

    for status, count in sorted(statuses.items()):
        print(f"{status}: {count}", file=sys.stderr)
    for source_path in sorted(kept.values()):
        print(f'        "{source_path}",')


if __name__ == "__main__":
    main()
