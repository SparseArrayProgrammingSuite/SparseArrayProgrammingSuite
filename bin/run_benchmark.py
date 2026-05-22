#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import logging
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

import saps

logging.basicConfig(level=logging.INFO)

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


def _upload_all(benchmarks, metadata, backend, is_include, is_exclude) -> int:
    """Walk every (benchmark, generator, dataset) triple and upload."""
    uploaded = failed = skipped = 0
    seen: set[tuple[str, str]] = set()
    for name in benchmarks:
        if name not in metadata:
            continue
        bench_meta = metadata[name]
        if is_exclude(bench_meta):
            continue
        module_name, class_name, _method = name.rsplit(".", 2)
        bench_module = importlib.import_module(f"saps.benchmarks.{module_name}")
        bench = getattr(bench_module, class_name)()
        for gen in bench.generators:
            gen_meta = gen.metadata
            if is_exclude(gen_meta):
                continue
            for ds in gen.datasets:
                ds_meta = ds.metadata
                if is_exclude(ds_meta):
                    continue
                if not (
                    is_include(bench_meta)
                    or is_include(gen_meta)
                    or is_include(ds_meta)
                ):
                    continue
                key = (gen.name, ds.name)
                if key in seen:
                    skipped += 1
                    continue
                seen.add(key)
                label = f"{bench.name} / {gen.name} / {ds.name}"
                try:
                    if backend.upload_dataset(gen, ds):
                        uploaded += 1
                        print(f"[uploaded] {label}")
                    else:
                        failed += 1
                        print(f"[failed]   {label}")
                except Exception as e:
                    failed += 1
                    print(f"[error]    {label}: {e}")
    print(f"upload summary: uploaded={uploaded} failed={failed} skipped={skipped}")
    return 0 if failed == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SAPS benchmarks")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to saps.conf.json (default: auto-detect)",
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
        "--remote-storage-backend",
        default=None,
        help=(
            "Remote storage backend to use for uploading and downloading datasets "
            " (local or s3)"
            "In order to use s3 for upload, you must have AWS credentials configured"
            " (AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY)."
        ),
    )
    parser.add_argument(
        "--remote-storage-bucket",
        default=None,
        help=(
            "Remote storage bucket name to use for uploading and downloading datasets "
            "standard s3 bucket is 'sparse-array-programming-suite'"
            "(local backend will use this as a directory path)"
        ),
    )
    parser.add_argument(
        "--upload-datasets",
        action="store_true",
        help=(
            "Skip benchmark execution. Instead, walk every "
            "(benchmark, generator, dataset) triple, generate the data, "
            "and upload it via the configured storage backend. Honors "
            "--re/--no-re/--tag/--no-tag filters."
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
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help=(
            "Timeout in seconds for each benchmark"
            " (default: config timeout or 5 seconds)"
        ),
    )
    args = parser.parse_args()

    import logging as _logging

    if not _logging.getLogger().handlers:
        _logging.basicConfig(
            level=_logging.INFO,
            format="%(levelname)s %(name)s: %(message)s",
        )

    if args.upload_datasets and saps.xp is None:
        # Upload mode runs in the caller's env (no ASV subproc), so default the
        # framework so benchmark modules can call xp.to_binsparse(...) at generate time.
        import importlib as _importlib

        os.environ.setdefault(
            "SAPS_FRAMEWORK",
            str(Path(__file__).resolve().parent.parent / "frameworks/saps_numpy.py"),
        )
        import saps.framework as _saps_framework

        _importlib.reload(_saps_framework)
        saps.xp = _saps_framework.xp

    log.enable(args.verbose)

    # Get repo root (parent of bin directory where this script is)
    repo_root = Path(__file__).parent.parent

    # Load SAPS configuration
    saps_dir = Path(".saps")
    saps_dir.mkdir(parents=True, exist_ok=True)
    machine_files_dir = saps_dir / "machine_files"
    machine_files_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir = saps_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # Load optional saps.conf.json
    saps_config_data = {}
    if args.config:
        config_file = Path(args.config)
        if config_file.exists():
            with open(config_file) as f:
                saps_config_data = json.load(f)
    else:
        saps_config_file = Path("saps.conf.json")
        if saps_config_file.exists():
            with open(saps_config_file) as f:
                saps_config_data = json.load(f)
    
    matrix = saps_config_data.get("matrix", 
                                    {
                                        "req": {
                                            "numpy": ["2.3"],
                                            "scipy": ["1.16.2"],
                                            "sparse": ["0.17.0"],
                                            "lark": ["1.3.0"],
                                            "ssgetpy": ["1.0rc2"],
                                        },
                                        "env_nobuild": {
                                            "SAPS_FRAMEWORK": [
                                                "frameworks/saps_numpy.py",
                                                "frameworks/saps_scipy.py",
                                                "frameworks/saps_sparse.py",
                                            ],
                                            "SAPS_REPO_ROOT": [str(repo_root)],
                                        },
                                    },)
    if args.remote_storage_backend is not None:
        matrix["env_nobuild"]["REMOTE_STORAGE_BACKEND"] = args.remote_storage_backend
    if args.remote_storage_bucket is not None:
        matrix["env_nobuild"]["REMOTE_STORAGE_BUCKET"] = args.remote_storage_bucket
    if "SAPS_CACHE_DIR" not in matrix["env_nobuild"]:
        matrix["env_nobuild"]["SAPS_CACHE_DIR"] = [str(outputs_dir / "cache")]
    if "SAPS_MANIFEST_PATH" not in matrix["env_nobuild"]:
        matrix["env_nobuild"]["SAPS_MANIFEST_PATH"] = [str(repo_root / "manifest.json")]

    # Construct ASV config dict with all fields visible
    asv_config_dict = {
        "version": 1,
        "project": "saps",
        "project_url": "https://github.com/SparseArrayProgrammingSuite/SparseArrayProgrammingSuite",
        "repo": str(repo_root),
        "branches": "HEAD",
        "environment_type": saps_config_data.get("environment_type", "virtualenv"),
        "install_command": saps_config_data.get(
            "install_command",
            ["in-dir={env_dir} python -mpip install {build_dir} --force-reinstall"],
        ),
        "benchmark_dir": str(repo_root / "src/saps/benchmarks"),
        "env_dir": saps_config_data.get("env_dir", str(saps_dir / "results")),
        "results_dir": saps_config_data.get(
            "results_dir", str(outputs_dir / "results")
        ),
        "html_dir": str(outputs_dir / "html"),
        "matrix": matrix
    }
    log.info(f"Using SAPS config: {saps_config_data}")  
    # Create ASV config from dict
    conf = Config.from_json(asv_config_dict)

    log.info(f"Using SAPS outputs directory: {outputs_dir}")
    log.info(f"Using SAPS machine files directory: {machine_files_dir}")

    # Determine timeout with hierarchy: CLI arg > config > 5 seconds default
    if args.timeout is not None:
        timeout = args.timeout
    elif hasattr(conf, "timeout") and conf.timeout is not None:
        timeout = conf.timeout
    else:
        timeout = 5

    # Convert relative SAPS_FRAMEWORK paths to absolute paths so child processes can
    # find them
    cwd = os.getcwd()
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

    # Save machine file to SAPS machine files directory
    machine_params.save(str(machine_files_dir))

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
        benchmark_module = importlib.import_module(f"saps.benchmarks.{module_name}")
        benchmark = getattr(benchmark_module, class_name)()
        assert isinstance(benchmark, saps.Benchmark)
        metadata[name] = benchmark.metadata

    # Store benchmark metadata in SAPS outputs directory
    results_dir = outputs_dir / "results"
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

    if args.upload_datasets:
        backend = saps.build_storage_backend(
            type=args.remote_storage_backend,
            bucket=args.remote_storage_bucket,
        )
        return _upload_all(
            benchmarks=benchmarks,
            metadata=metadata,
            backend=backend,
            is_include=is_include,
            is_exclude=is_exclude,
        )

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
            if len(param) == 0:
                print(
                    f"Warning: benchmark '{name}' has no data generators, skipping"
                    " benchmark"
                )
                continue
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
    print(f"Using timeout: {timeout} seconds")

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
            extra_params={"timeout": timeout},
        )

        print("Results object:", results)
        print(json.dumps(format_results(results, benchmarks), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
