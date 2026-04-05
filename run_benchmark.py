#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import pkgutil
from pathlib import Path

from asv.benchmarks import Benchmarks
from asv.commands.setup import Setup
from asv.config import Config
from asv.environment import get_environments
from asv.machine import Machine
from asv.repo import get_repo
from asv.results import Results
from asv.runner import run_benchmarks
from asv.console import log


def import_benchmark_modules(benchmark_dir: Path) -> list[str]:
    """Import all modules in benchmarks/ so Benchmark.__init_subclass__ runs."""
    imported: list[str] = []
    for module_info in pkgutil.iter_modules([str(benchmark_dir)]):
        if module_info.ispkg:
            continue
        module_name = module_info.name
        importlib.import_module(f"{benchmark_dir.name}.{module_name}")
        imported.append(module_name)
    return imported


def format_results(results: Results, benchmarks: Benchmarks) -> dict:
    """Return a JSON-serializable snapshot of benchmark results."""
    entries: dict[str, dict] = {}
    for name in sorted(results.get_all_result_keys()):
        benchmark = benchmarks.get(name)
        params = benchmark["params"] if benchmark is not None else []
        entries[name] = {
            "result": results.get_result_value(name, params),
            "stats": results.get_result_stats(name, params),
            "samples": results.get_result_samples(name, params),
            "duration_seconds": results.duration.get(name),
            "started_at": results.started_at.get(name),
            "errcode": results.errcode.get(name),
            "stderr": results.stderr.get(name),
        }

    return {
        "commit_hash": results.commit_hash,
        "date": results.date,
        "env_name": results.env_name,
        "env_vars": results.env_vars,
        "params": results.params,
        "result_count": len(entries),
        "results": entries,
    }


def _load_saps_tags(metadata_path: Path) -> dict[str, set[str]]:
    """Load SAPS benchmark tags keyed by '<module>.<class>' from metadata."""
    if not metadata_path.exists():
        return {}

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    out: dict[str, set[str]] = {}
    for item in payload.get("benchmarks", []):
        if not isinstance(item, dict):
            continue
        benchmark_id = item.get("id")
        if not isinstance(benchmark_id, str):
            continue
        parts = benchmark_id.split(".")
        if len(parts) < 3:
            continue

        # SAPS ids are typically 'benchmarks.<module>.<class>.<name>'
        # and ASV benchmark names are '<module>.<class>.<method>'.
        if parts[0] == "benchmarks":
            key = f"{parts[1]}.{parts[2]}"
        else:
            key = f"{parts[0]}.{parts[1]}"

        tags = item.get("tags", [])
        if isinstance(tags, list):
            out[key] = {str(tag).lower() for tag in tags}
    return out


def filter_benchmarks_by_saps_tags(
    benchmarks: Benchmarks,
    include_tags: list[str],
    exclude_tags: list[str],
    metadata_path: Path,
) -> Benchmarks:
    """Filter ASV benchmarks using SAPS class-level tags from metadata.

    Include semantics: keep benchmark if it has any include tag.
    Exclude semantics: drop benchmark if it has any exclude tag.
    """
    include_set = {tag.strip().lower() for tag in include_tags if tag and tag.strip()}
    exclude_set = {tag.strip().lower() for tag in exclude_tags if tag and tag.strip()}

    if not include_set and not exclude_set:
        return benchmarks

    saps_tags = _load_saps_tags(metadata_path)
    skip: set[str] = set()

    for asv_name in benchmarks.keys():
        parts = asv_name.split(".")
        if len(parts) < 3:
            skip.add(asv_name)
            continue

        class_key = f"{parts[0]}.{parts[1]}"
        tags = saps_tags.get(class_key, set())

        if include_set and not (tags & include_set):
            skip.add(asv_name)
            continue

        if exclude_set and (tags & exclude_set):
            skip.add(asv_name)

    return benchmarks.filter_out(skip)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ASV benchmarks directly via asv.runner.run_benchmarks")
    parser.add_argument(
        "--config",
        default="asv.conf.json",
        help="Path to asv config file (default: asv.conf.json)",
    )
    parser.add_argument(
        "--machine",
        default=None,
        help="Machine name to use (default: host name)",
    )
    parser.add_argument(
        "--bench",
        action="append",
        default=None,
        help="Regex benchmark filter (can be passed more than once)",
    )
    parser.add_argument(
        "--tag",
        action="append",
        default=None,
        help="SAPS tag include filter (can be passed more than once)",
    )
    parser.add_argument(
        "--include-tag",
        action="append",
        default=None,
        help="Include SAPS benchmarks that have any of these tags (can be passed more than once)",
    )
    parser.add_argument(
        "--exclude-tag",
        action="append",
        default=None,
        help="Exclude SAPS benchmarks that have any of these tags (can be passed more than once)",
    )
    parser.add_argument(
        "--show-stderr",
        action="store_true",
        help="Show stderr for benchmark runs",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run each benchmark only once",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    log.enable(args.verbose)

    conf = Config.load(args.config)

    benchmark_dir = Path(conf.benchmark_dir)
    imported_modules = import_benchmark_modules(benchmark_dir)
    print(f"Imported {len(imported_modules)} benchmark modules via saps hook discovery")

    machine_params = Machine.load(
        machine_name=args.machine,
        interactive=True,
        use_defaults=True,
    )
    machine_params.save(conf.results_dir)

    environments = list(get_environments(conf, None))
    if not environments:
        raise RuntimeError("No ASV environments available")

    repo = get_repo(conf)
    commit_hash = repo.get_hash_from_name("HEAD")
    benchmarks = Benchmarks.discover(
        conf=conf,
        repo=repo,
        environments=environments,
        commit_hash=[commit_hash],
        regex=args.bench,
    )

    include_tags = []
    if args.tag:
        include_tags.extend(args.tag)
    if args.include_tag:
        include_tags.extend(args.include_tag)
    exclude_tags = args.exclude_tag or []

    if include_tags or exclude_tags:
        metadata_path = Path("benchmark_metadata.json")
        before = len(benchmarks)
        benchmarks = filter_benchmarks_by_saps_tags(
            benchmarks,
            include_tags=include_tags,
            exclude_tags=exclude_tags,
            metadata_path=metadata_path,
        )
        print(
            "Filtered by SAPS tags "
            f"(include={include_tags or []}, exclude={exclude_tags or []}): "
            f"{before} -> {len(benchmarks)} benchmark entries"
        )

    if len(benchmarks) == 0:
        print("No benchmarks selected after applying filters")
        return 1

    print(f"Discovered {len(benchmarks)} benchmark entries")

    for env in environments:
        Setup.perform_setup([env], parallel=1)


        params = dict(machine_params.__dict__)
        params["python"] = env.python
        params.update(env.requirements)

        results = Results(
            params=params,
            requirements=env.requirements,
            commit_hash=commit_hash,
            date=repo.get_date(commit_hash),
            python=env.python,
            env_name=env.name,
            env_vars=env.env_vars,
        )

        run_benchmarks(
            benchmarks=benchmarks,
            env=env,
            results=results,
            show_stderr=args.show_stderr,
            quick=args.quick,
        )

        print("Results object:", results)
        print(json.dumps(format_results(results, benchmarks), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
