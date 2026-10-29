#!/usr/bin/env python
"""Fill the SuiteSparse source cache from the University of Florida mirror.

sparse.tamu.edu currently answers archive requests with a 403 page. This script
downloads every `suitesparse_matrix` source missing from
`$SAPS_CACHE_DIR/suitesparse/<group>/<name>` from the older UF mirror instead,
using the same layout and lock as `download_suitesparse_matrix`, so later cache
runs find the matrices and never contact TAMU. The UF mirror lacks a few newer
matrices (mostly SNAP); those are reported and left alone.

It also copies the ssgetpy index from `~/.ssgetpy/index.db` to
`$SAPS_CACHE_DIR/suitesparse/.ssgetpy/index.db`, where `_import_ssgetpy` looks
for it, unless the cache already holds a newer index.

    poetry run python scripts/fetch_suitesparse_uf.py --dry-run
    poetry run python scripts/fetch_suitesparse_uf.py
    poetry run python scripts/fetch_suitesparse_uf.py HB/ash85 LPnetlib/lp_afiro
"""

import argparse
import datetime as dt
import shutil
import sqlite3
import sys
import tarfile
import tempfile
from contextlib import closing
from pathlib import Path

import requests  # type: ignore[import-untyped]
from filelock import FileLock

from saps.benchmarks.suitesparse import SuiteSparseMatrixGenerator
from saps.downloaders.cache import source_cache_dir


def index_date(path: Path) -> str:
    """Return the index's last update, or "" when it is missing or never filled."""
    if not path.exists():
        return ""
    with closing(sqlite3.connect(path)) as db:
        try:
            (date,) = db.execute("SELECT MAX(update_date) FROM update_table").fetchone()
        except sqlite3.OperationalError:
            return ""
    return date or ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "source_names",
        nargs="*",
        help="group/name sources to fetch (default: every suitesparse_matrix dataset)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="list what would be copied or downloaded without changing anything",
    )
    args = parser.parse_args()

    root = source_cache_dir("suitesparse").resolve()
    print(f"SuiteSparse source cache: {root}")

    home_index = Path.home() / ".ssgetpy" / "index.db"
    cache_index = root / ".ssgetpy" / "index.db"
    home_date, cache_date = index_date(home_index), index_date(cache_index)
    if not home_date:
        print(f"no usable index at {home_index}; leaving {cache_index} alone")
    elif home_date <= cache_date:
        print(f"index at {cache_index} is current ({cache_date})")
    elif args.dry_run:
        print(f"would copy index ({home_date}) to {cache_index}")
    else:
        cache_index.parent.mkdir(parents=True, exist_ok=True)
        staging = cache_index.with_name(".index.db.saps-staging")
        shutil.copy2(home_index, staging)
        staging.replace(cache_index)
        print(f"copied index ({home_date}) to {cache_index}")
    index_age = dt.datetime.now() - dt.datetime.fromisoformat(
        max(home_date, cache_date) or "1970-01-01"
    )
    if index_age > dt.timedelta(days=90):
        print(
            f"warning: the index is {index_age.days} days old; ssgetpy refetches "
            "indexes older than 90 days from sparse.tamu.edu on import"
        )

    source_names = args.source_names or sorted(
        {dataset.source_name for dataset in SuiteSparseMatrixGenerator().datasets}
    )
    cached, downloaded, absent, failed = [], [], [], []
    for source_name in source_names:
        group, name = source_name.split("/")
        parent = root / group
        matrix_dir = parent / name
        if matrix_dir.exists():
            cached.append(source_name)
            continue
        if args.dry_run:
            print(f"would download {source_name}")
            continue
        parent.mkdir(parents=True, exist_ok=True)
        try:
            with FileLock(parent / f"{name}.lock"):
                if matrix_dir.exists():
                    cached.append(source_name)
                    continue
                with tempfile.TemporaryDirectory(
                    prefix=".saps-", dir=parent
                ) as staging:
                    archive = Path(staging) / f"{name}.tar.gz"
                    with requests.get(
                        f"https://www.cise.ufl.edu/research/sparse/MM/{source_name}.tar.gz",
                        stream=True,
                        timeout=60,
                    ) as response:
                        if response.status_code == 404:
                            print(f"not on UF: {source_name}")
                            absent.append(source_name)
                            continue
                        response.raise_for_status()
                        with archive.open("wb") as file:
                            for chunk in response.iter_content(chunk_size=1 << 20):
                                file.write(chunk)
                    with tarfile.open(archive, "r:gz") as bundle:
                        bundle.extractall(staging, filter="data")
                    extracted = Path(staging) / name
                    if not (extracted / f"{name}.mtx").exists():
                        raise FileNotFoundError(f"archive has no {name}/{name}.mtx")
                    extracted.replace(matrix_dir)
            print(f"downloaded {source_name}")
            downloaded.append(source_name)
        except (OSError, requests.RequestException, tarfile.TarError) as e:
            print(f"failed {source_name}: {e}")
            failed.append(source_name)

    print(
        f"\n{len(downloaded)} downloaded, {len(cached)} already cached, "
        f"{len(absent)} not on UF, {len(failed)} failed"
    )
    for label, names in (("not on UF", absent), ("failed", failed)):
        if names:
            print(f"{label}: {' '.join(names)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
