#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import re
from pathlib import Path

from asv.benchmarks import Benchmarks
from asv.commands.setup import Setup
from asv.config import Config
from asv.console import log
from asv.environment import ExistingEnvironment, get_environments
from asv.machine import Machine
from asv.repo import get_repo
from asv.results import Results
from asv.runner import run_benchmarks

from saps.storage import (
    DEFAULT_REMOTE_STORAGE_BACKEND,
    DEFAULT_REMOTE_STORAGE_BUCKET,
    build_storage_backend,
)

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


def _run_asv_benchmarks(
    benchmarks,
    environments,
    machine_params,
    commit_hash,
    commit_date,
    timeout,
    show_stderr,
    quick,
    install_project=None,
    results_dir=None,
    print_results=False,
    launch_method=None,
):
    failed = 0
    for env in environments:
        Setup.perform_setup([env], parallel=1)
        if install_project is not None:
            conf, repo = install_project
            env.install_project(conf, repo, commit_hash)

        params = dict(machine_params.__dict__)
        params["python"] = env.python
        params.update(env.requirements)

        results = Results(
            params=params,
            requirements=env.requirements,
            commit_hash=commit_hash,
            date=commit_date,
            python=env.python,
            env_name=env.name,
            env_vars=env.env_vars,
        )

        run_benchmarks(
            benchmarks=benchmarks,
            env=env,
            results=results,
            show_stderr=show_stderr,
            quick=quick,
            extra_params={"timeout": timeout},
            launch_method=launch_method,
        )
        failed += sum(
            1 for errcode in results.errcode.values() if errcode not in (None, 0)
        )
        if print_results:
            print("Results object:", results)
            print(
                json.dumps(format_results(results, benchmarks), indent=2, default=str)
            )
        if results_dir is not None:
            results.save(results_dir)
    return failed


def _load_metadata(metadata_path: Path) -> list[dict]:
    if not metadata_path.exists():
        raise RuntimeError(
            f"{metadata_path} does not exist; run generate metadata first with "
            "`poetry run ./bin/generate_metadata.py`."
        )
    document = json.loads(metadata_path.read_text(encoding="utf-8"))
    return document.get("benchmarks", [])


def _filter_metadata(metadata: list[dict], dataset_predicate) -> list[dict]:
    filtered = []
    dataset_number = 0
    for benchmark in metadata:
        generators = []
        for generator in benchmark["generators"]:
            datasets = []
            for dataset in generator["datasets"]:
                if dataset_predicate(benchmark, generator, dataset, dataset_number):
                    datasets.append(dataset)
                dataset_number += 1
            if datasets:
                generators.append({**generator, "datasets": datasets})
        if generators:
            filtered.append({**benchmark, "generators": generators})
    return filtered


def _metadata_to_asv_benchmarks(
    metadata: list[dict], benchmarks: Benchmarks, metrics: list[str]
):
    metadata_by_asv_id = {
        record["asv_ids"][metric]: record
        for record in metadata
        for metric in metrics
        if metric in record.get("asv_ids", {})
    }
    skips = []
    benchmarks._benchmark_selection = {}
    for name in benchmarks:
        bench_meta = metadata_by_asv_id.get(name)
        if bench_meta is None:
            skips.append(name)
            continue
        selected_params = {
            dataset["asv_param"]
            for generator in bench_meta["generators"]
            for dataset in generator["datasets"]
        }
        asv_params = benchmarks[name]["params"][0] if benchmarks[name]["params"] else []
        selection = [
            idx
            for idx, asv_param in enumerate(asv_params)
            if asv_param in selected_params
        ]
        if selection:
            benchmarks._benchmark_selection[name] = selection
        else:
            skips.append(name)
    return benchmarks.filter_out(set(skips))


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
            "(local or s3). "
            "In order to use s3 for upload, you must have AWS credentials configured "
            "(AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY)."
        ),
    )
    parser.add_argument(
        "--remote-storage-bucket",
        default=None,
        help=(
            "Remote storage bucket name to use for uploading and downloading datasets. "
            f"The standard s3 bucket is '{DEFAULT_REMOTE_STORAGE_BUCKET}' "
            "(local backend will use this as a directory path)."
        ),
    )
    parser.add_argument(
        "--cache-datasets",
        action="store_true",
        help=(
            "Skip benchmark execution. Instead, walk every "
            "(benchmark, generator, dataset) triple, generate the data, "
            "and cache it via the configured storage backend. Honors "
            "--re/--no-re/--tag/--no-tag filters."
        ),
    )
    parser.add_argument(
        "--trace-statistics",
        action="store_true",
        help=(
            "Run selected benchmark cases under ASV with the tagger framework "
            "and write generated dataset statistics tags to statistics.json. "
            "Honors --re/--no-re/--tag/--no-tag filters."
        ),
    )
    parser.add_argument(
        "--check-suite",
        action="store_true",
        help=(
            "Run selected benchmark cases once through ASV and print result JSON. "
            "Honors --re/--no-re/--tag/--no-tag filters."
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
        "--metric",
        "--metrics",
        dest="metrics",
        nargs="+",
        choices=("peakmem", "time"),
        default=("time",),
        metavar="METRIC",
        help="Benchmark metric(s) to run: peakmem and/or time. Defaults to time.",
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
    parser.add_argument(
        "--chunk-count",
        type=int,
        default=1,
        help=(
            "Split the selected benchmark datasets into this many chunks. "
            "Run one process per chunk with a distinct --chunk-index."
        ),
    )
    parser.add_argument(
        "--chunk-index",
        type=int,
        default=0,
        help="Zero-based chunk index to run when --chunk-count is greater than 1.",
    )
    args = parser.parse_args()
    if args.chunk_count < 1:
        parser.error("--chunk-count must be at least 1")
    if args.chunk_index < 0 or args.chunk_index >= args.chunk_count:
        parser.error("--chunk-index must be between 0 and --chunk-count - 1")

    import logging as _logging

    if not _logging.getLogger().handlers:
        _logging.basicConfig(
            level=_logging.INFO,
            format="%(levelname)s %(name)s: %(message)s",
        )

    log.enable(args.verbose)
    benchmark_metrics = ["time"] if args.cache_datasets else list(args.metrics)

    # Get repo root (parent of bin directory where this script is)
    repo_root = Path(__file__).parent.parent

    # Load SAPS configuration
    saps_dir = Path(".saps").resolve()
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

    matrix = saps_config_data.get(
        "matrix",
        {
            "req": {
                "array-api-compat": ["1.14"],
                "numpy": ["2.3"],
                "scipy": ["1.16.2"],
                "sparse": ["0.17.0"],
                "lark": ["1.3.0"],
                "ssgetpy": ["1.0rc2"],
                "networkx": ["3.6"],
            },
            "env_nobuild": {
                "SAPS_FRAMEWORK": [
                    "frameworks/saps_numpy.py",
                    "frameworks/saps_scipy.py",
                    "frameworks/saps_sparse.py",
                ],
                "SAPS_REPO_ROOT": [str(repo_root)],
            },
        },
    )
    matrix.setdefault("env_nobuild", {})
    log_path = str(outputs_dir / "results" / "diagnostics.log")
    os.environ["SAPS_LOG_PATH"] = log_path
    matrix["env_nobuild"]["SAPS_LOG_PATH"] = [log_path]
    storage_backend = args.remote_storage_backend or DEFAULT_REMOTE_STORAGE_BACKEND
    storage_bucket = args.remote_storage_bucket or DEFAULT_REMOTE_STORAGE_BUCKET
    if (
        args.remote_storage_backend is not None
        or "REMOTE_STORAGE_BACKEND" not in matrix["env_nobuild"]
    ):
        matrix["env_nobuild"]["REMOTE_STORAGE_BACKEND"] = [storage_backend]
    if (
        args.remote_storage_bucket is not None
        or "REMOTE_STORAGE_BUCKET" not in matrix["env_nobuild"]
    ):
        matrix["env_nobuild"]["REMOTE_STORAGE_BUCKET"] = [storage_bucket]
    cache_dir = str(outputs_dir / "cache")
    os.environ["SAPS_CACHE_DIR"] = cache_dir
    matrix["env_nobuild"]["SAPS_CACHE_DIR"] = [cache_dir]
    persistent_metadata_path = Path(
        os.environ.get("SAPS_METADATA_PATH", str(repo_root / "metadata.json"))
    )
    persistent_statistics_path = Path(
        os.environ.get("SAPS_STATISTICS_PATH", str(repo_root / "statistics.json"))
    )
    manifest_path = str(repo_root / "manifest.json")
    pythonpath = str(repo_root)
    os.environ["SAPS_MANIFEST_PATH"] = manifest_path
    os.environ["PYTHONPATH"] = pythonpath
    os.environ["REMOTE_STORAGE_BACKEND"] = storage_backend
    os.environ["REMOTE_STORAGE_BUCKET"] = storage_bucket
    matrix["env_nobuild"]["SAPS_MANIFEST_PATH"] = [manifest_path]
    if args.cache_datasets:
        os.environ["SAPS_CACHE_DATASETS"] = "1"
        matrix["env_nobuild"]["SAPS_CACHE_DATASETS"] = ["1"]
    else:
        os.environ.pop("SAPS_CACHE_DATASETS", None)
    if args.trace_statistics or args.cache_datasets:
        framework_file = (
            "frameworks/saps_tagger.py"
            if args.trace_statistics
            else "frameworks/saps_numpy.py"
        )
        os.environ["SAPS_FRAMEWORK"] = str(repo_root / framework_file)
        if args.trace_statistics:
            tagger_stats_dir = os.environ.get(
                "SAPS_TAGGER_STATS_DIR", str(outputs_dir / "tagger_stats")
            )
            os.environ["SAPS_TAGGER_STATS_DIR"] = tagger_stats_dir
            os.environ["SAPS_STATISTICS_PATH"] = str(persistent_statistics_path)

    uses_parent_environment = args.trace_statistics or args.cache_datasets

    # Construct ASV config dict with all fields visible
    asv_config_dict = {
        "version": 1,
        "project": "saps",
        "project_url": "https://github.com/SparseArrayProgrammingSuite/SparseArrayProgrammingSuite",
        "repo": str(repo_root),
        "branches": "HEAD",
        "environment_type": (
            "existing:same"
            if uses_parent_environment
            else saps_config_data.get("environment_type", "virtualenv")
        ),
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
        "matrix": matrix,
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
    elif args.cache_datasets:
        timeout = 5000
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

    if uses_parent_environment:
        environments = [ExistingEnvironment(conf, "same", {}, {})]
    else:
        environments = list(get_environments(conf, None))
    if not environments:
        raise RuntimeError("No ASV environments available")

    repo = get_repo(conf)
    commit_hash = repo.get_hash_from_name("HEAD")
    commit_date = repo.get_date(commit_hash)
    discovery_environments = [
        ExistingEnvironment(
            conf,
            "same",
            {},
            {("nobuild", "PYTHONPATH"): pythonpath},
        )
    ]
    benchmarks = Benchmarks.discover(
        conf=conf,
        repo=repo,
        environments=discovery_environments,
        commit_hash=[commit_hash],
    )

    results_dir = outputs_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    include_tags = {tag.strip() for tag in args.tag if tag and tag.strip()}
    exclude_tags = {tag.strip() for tag in args.no_tag if tag and tag.strip()}
    include_patterns = [re.compile(pattern) for pattern in args.re or []]
    exclude_patterns = [re.compile(pattern) for pattern in args.no_re or []]

    def name_matches(patterns, names) -> bool:
        return any(pattern.search(name) for pattern in patterns for name in names)

    def dataset_predicate(
        benchmark: dict, generator: dict, dataset: dict, _dataset_number: int
    ) -> bool:
        dataset_tags = set(dataset["tags"])
        target_names = (benchmark["name"], generator["name"], dataset["name"])
        positive = (
            not include_tags or not dataset_tags.isdisjoint(include_tags),
            not include_patterns or name_matches(include_patterns, target_names),
        )
        negative = (
            exclude_tags and not dataset_tags.isdisjoint(exclude_tags),
            exclude_patterns and name_matches(exclude_patterns, target_names),
        )
        return all(positive) and not any(negative)

    metadata = _filter_metadata(
        _load_metadata(persistent_metadata_path), dataset_predicate
    )
    trace_had_selected_datasets = bool(metadata)
    if args.trace_statistics:
        statistics = (
            json.loads(persistent_statistics_path.read_text(encoding="utf-8"))
            if persistent_statistics_path.exists()
            else {}
        )
        fresh_statistics = {
            (
                benchmark["name"],
                generator["name"],
                dataset["name"],
                dataset["freshness"],
            )
            for benchmark in statistics.get("benchmarks", [])
            for generator in benchmark.get("generators", [])
            for dataset in generator.get("datasets", [])
            if "freshness" in dataset
        }

        def needs_statistics(
            benchmark: dict, generator: dict, dataset: dict, _dataset_number: int
        ) -> bool:
            return (
                benchmark["name"],
                generator["name"],
                dataset["name"],
                dataset["freshness"],
            ) not in fresh_statistics

        metadata = _filter_metadata(metadata, needs_statistics)

    if args.cache_datasets:
        backend = build_storage_backend(
            storage_backend,
            storage_bucket,
            manifest_path=manifest_path,
            cache_dir=cache_dir,
        )
        manifest_file = Path(manifest_path)
        manifest = (
            json.loads(manifest_file.read_text(encoding="utf-8"))
            if manifest_file.exists()
            else {}
        )
        selected_cache_params = set()

        def needs_caching(
            _benchmark: dict, generator: dict, dataset: dict, _dataset_number: int
        ) -> bool:
            if not generator["cacheable"]:
                return False
            manifest_key = dataset["asv_param"]
            record = manifest.get(manifest_key)
            if manifest_key in selected_cache_params:
                return False
            fresh = (
                record is not None
                and all(
                    record.get(key) == dataset[key] for key in ("file", "freshness")
                )
                and "digest" in record
                and backend.manifest_record_exists(manifest_key, record)
            )
            if fresh:
                return False
            selected_cache_params.add(manifest_key)
            return True

        metadata = _filter_metadata(metadata, needs_caching)

    chunk_total = sum(
        len(generator["datasets"])
        for benchmark in metadata
        for generator in benchmark["generators"]
    )
    chunk_kept = chunk_total
    if args.chunk_count > 1:

        def in_chunk(
            _benchmark: dict,
            _generator: dict,
            _dataset: dict,
            dataset_number: int,
        ) -> bool:
            return dataset_number % args.chunk_count == args.chunk_index

        metadata = _filter_metadata(metadata, in_chunk)

    benchmarks = _metadata_to_asv_benchmarks(metadata, benchmarks, benchmark_metrics)
    if args.cache_datasets:
        print(f"Discovered {len(benchmarks)} benchmark entries for caching")
        print(f"Using timeout: {timeout} seconds")
        failed = _run_asv_benchmarks(
            benchmarks=benchmarks,
            environments=environments,
            machine_params=machine_params,
            commit_hash=commit_hash,
            commit_date=commit_date,
            timeout=timeout,
            show_stderr=args.show_stderr,
            quick=True,
        )
        print(f"cache summary: selected={chunk_kept} failed_benchmark_entries={failed}")
        return 0 if failed == 0 else 1

    if args.trace_statistics:
        stats_dir = outputs_dir / "tagger_stats"
        stats_dir.mkdir(parents=True, exist_ok=True)
        for stats_path in stats_dir.glob("*.json"):
            stats_path.unlink()

        print(f"Discovered {len(benchmarks)} benchmark entries for tagger")
        print(f"Using timeout: {timeout} seconds")
        failed = _run_asv_benchmarks(
            benchmarks=benchmarks,
            environments=environments,
            machine_params=machine_params,
            commit_hash=commit_hash,
            commit_date=commit_date,
            timeout=timeout,
            show_stderr=args.show_stderr,
            quick=True,
            results_dir=results_dir,
        )
        tagged = sum(1 for _ in stats_dir.glob("*.json"))
        print(f"tag summary: tagged_records={tagged} failed_benchmark_entries={failed}")
        return 0 if failed == 0 and (tagged > 0 or trace_had_selected_datasets) else 1

    if args.check_suite:
        print(f"Discovered {len(benchmarks)} benchmark entries")
        print(f"Using timeout: {timeout} seconds")
        failed = _run_asv_benchmarks(
            benchmarks=benchmarks,
            environments=environments,
            machine_params=machine_params,
            commit_hash=commit_hash,
            commit_date=commit_date,
            timeout=timeout,
            show_stderr=args.show_stderr,
            quick=True,
            install_project=(conf, repo),
            print_results=True,
            launch_method=None if os.name == "nt" else "forkserver",
        )
        return 0 if failed == 0 else 1

    print(f"Discovered {len(benchmarks)} benchmark entries")
    print(f"Using timeout: {timeout} seconds")

    _run_asv_benchmarks(
        benchmarks=benchmarks,
        environments=environments,
        machine_params=machine_params,
        commit_hash=commit_hash,
        commit_date=commit_date,
        timeout=timeout,
        show_stderr=args.show_stderr,
        quick=args.quick,
        install_project=(conf, repo),
        results_dir=results_dir,
        print_results=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
