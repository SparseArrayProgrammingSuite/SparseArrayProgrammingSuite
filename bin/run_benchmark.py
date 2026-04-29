#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
import re
from itertools import product
from pathlib import Path

from asv.benchmarks import Benchmarks
from asv.commands.setup import Setup
from asv.config import Config
from asv.console import log
from asv.environment import get_environments
from asv.machine import Machine
from asv.repo import get_repo
from asv.results import Results
from asv.runner import run_benchmarks

import sparseappbench


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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run ASV benchmarks directly via asv.runner.run_benchmarks"
    )
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
        "--re",
        action="append",
        default=None,
        help=(
            "Regex benchmark filter (can be passed more than once, "
            "applies to datasets, generators, and benchmarks)"
        ),
    )
    parser.add_argument(
        "--no-re",
        action="append",
        default=None,
        help=(
            "Regex benchmark exclude filter (can be passed more than once, "
            "applies to datasets, generators, and benchmarks)"
        ),
    )
    parser.add_argument(
        "--tag",
        action="append",
        default=[],
        help=(
            "SAPS tag include filter (can be passed more than once, "
            "applies to datasets, generators, and benchmarks)"
        ),
    )
    parser.add_argument(
        "--no-tag",
        action="append",
        default=[],
        help=(
            "Exclude SAPS benchmarks that have any of these tags "
            "(can be passed more than once, applies to datasets, "
            "generators, and benchmarks)"
        ),
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
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    log.enable(args.verbose)

    conf = Config.load(args.config)
    
    # Convert relative SAPS_FRAMEWORK paths to absolute paths so child processes can find them
    cwd = os.getcwd()
    for env_config in conf.matrix.get("env_nobuild", {}).get("SAPS_FRAMEWORK", []):
        # This will be handled by updating the environment before discovery
        pass
    
    # Update environment with absolute SAPS_FRAMEWORK paths
    if "env_nobuild" in conf.matrix and "SAPS_FRAMEWORK" in conf.matrix["env_nobuild"]:
        abs_paths = []
        for path in conf.matrix["env_nobuild"]["SAPS_FRAMEWORK"]:
            path_obj = Path(path)
            if path_obj.is_absolute():
                abs_paths.append(path)
            else:
                abs_paths.append(str(Path(cwd) / path_obj))
        conf.matrix["env_nobuild"]["SAPS_FRAMEWORK"] = abs_paths

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
    )

    metadata: dict[str, dict] = {}

    for name in benchmarks:
        module_name, class_name, _method_name = name.rsplit(".", 2)
        benchmark_module = importlib.import_module(
            f"sparseappbench.benchmarks.{module_name}"
        )
        benchmark = getattr(benchmark_module, class_name)()
        assert isinstance(benchmark, sparseappbench.Benchmark)
        metadata[name] = benchmark.metadata

    results_dir = Path(conf.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = results_dir / "benchmarks_meta.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    include_set = {tag.strip().lower() for tag in args.tag if tag and tag.strip()}
    exclude_set = {tag.strip().lower() for tag in args.no_tag if tag and tag.strip()}

    include_res = args.re or []
    exclude_res = args.no_re or []

    def regex_any_match(patterns: list[str], value: str) -> bool:
        return any(re.search(pattern, value) for pattern in patterns)

    def match_target(obj: dict) -> str:
        return obj.get("id", obj["name"])

    def is_include(obj) -> bool:
        if include_res and not regex_any_match(include_res, match_target(obj)):
            return False
        return not (include_set and not include_set.intersection(obj["tags"]))

    def is_exclude(obj) -> bool:
        if exclude_res and regex_any_match(exclude_res, match_target(obj)):
            return True
        return bool(exclude_set and exclude_set.intersection(obj["tags"]))

    skips = []
    benchmarks._benchmark_selection = {}
    for name in benchmarks:
        if name not in metadata:
            log.warning(
                "No SAPS metadata found for benchmark "
                f"'{name}', skipping SAPS tag filtering "
                "for this benchmark"
            )
            skips.append(name)
            continue
        if is_exclude(metadata[name]):
            skips.append(name)
            continue
        benchmarks._benchmark_selection[name] = []
        generators = {gen["name"]: gen for gen in metadata[name]["generators"]}
        param_combos = list(product(*benchmarks[name]["params"]))
        for idx, param in enumerate(param_combos):
            generator, dataset = param[0].split(".")[:2]
            generator = generators.get(generator)
            datasets = {ds["name"]: ds for ds in generator["datasets"]}
            dataset = datasets.get(dataset)
            if (
                is_exclude(generator)
                or is_exclude(dataset)
                or not (
                    is_include(metadata[name])
                    or is_include(generator)
                    or is_include(dataset)
                )
            ):
                continue
            benchmarks._benchmark_selection[name].append(idx)

        if not benchmarks._benchmark_selection[name]:
            skips.append(name)

    benchmarks = benchmarks.filter_out(set(skips))

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
