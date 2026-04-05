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
        "params": results.params,
        "result_count": len(entries),
        "results": entries,
    }


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
        "--show-stderr",
        action="store_true",
        help="Show stderr for benchmark runs",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run each benchmark only once",
    )
    args = parser.parse_args()

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

    environments = list(get_environments(conf, ["existing:same"]))
    if not environments:
        raise RuntimeError("No ASV environments available")
    env = environments[0]
    Setup.perform_setup([env], parallel=1)

    repo = get_repo(conf)
    commit_hash = repo.get_hash_from_name("HEAD")

    benchmarks = Benchmarks.discover(
        conf=conf,
        repo=repo,
        environments=[env],
        commit_hash=[commit_hash],
        regex=args.bench,
    )

    print(f"Discovered {len(benchmarks)} benchmark entries")

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
