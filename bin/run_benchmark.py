#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import re
import time
import traceback
from itertools import product
from pathlib import Path

from asv import util
from asv.benchmarks import Benchmarks
from asv.commands.setup import Setup
from asv.config import Config
from asv.console import log
from asv.environment import ExistingEnvironment, get_environments
from asv.machine import Machine
from asv.repo import get_repo
from asv.results import Results
from asv.runner import run_benchmarks

from saps.storage import DEFAULT_REMOTE_STORAGE_BACKEND, DEFAULT_REMOTE_STORAGE_BUCKET

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


def _apply_chunk_selection(
    benchmarks: Benchmarks, chunk_count: int, chunk_index: int
) -> tuple[int, int]:
    """Restrict selected benchmark parameter cases to one deterministic chunk."""
    if chunk_count == 1:
        total = sum(
            len(benchmarks.benchmark_selection.get(name, [])) for name in benchmarks
        )
        return total, total

    selected_cases: list[tuple[str, int]] = []
    for name in sorted(benchmarks):
        selected_cases.extend(
            (name, idx) for idx in sorted(benchmarks.benchmark_selection.get(name, []))
        )

    kept_by_name: dict[str, list[int]] = {name: [] for name in benchmarks}
    for ordinal, (name, param_index) in enumerate(selected_cases):
        if ordinal % chunk_count == chunk_index:
            kept_by_name[name].append(param_index)

    for name, selected in kept_by_name.items():
        benchmarks._benchmark_selection[name] = selected

    kept = sum(len(selected) for selected in kept_by_name.values())
    return len(selected_cases), kept


def _run_check_suite(benchmarks: Benchmarks, machine_params, commit_hash, commit_date):
    """Run selected benchmark cases once in-process and print result JSON."""
    entries: dict[str, dict] = {}
    for name in sorted(benchmarks):
        module_name, class_name, _method_name = name.rsplit(".", 2)
        benchmark_module = importlib.import_module(f"saps.benchmarks.{module_name}")
        benchmark = getattr(benchmark_module, class_name)()

        param_combos = list(product(*benchmarks[name]["params"]))
        selected = set(
            benchmarks.benchmark_selection.get(name, range(len(param_combos)))
        )
        values = [float("nan")] * len(param_combos)
        stderr = []
        started_at = int(time.time() * 1000)
        start = time.monotonic()

        for idx in sorted(selected):
            param = benchmark.params[idx]
            try:
                benchmark.setup(param)
                benchmark.run(param)
                benchmark.teardown(param)
                values[idx] = 1
            except Exception:  # noqa: BLE001
                stderr.append(f"For parameters: {param}\n{traceback.format_exc()}")
                for attr in (
                    "_output",
                    "_meta",
                    "_ref_outputs",
                    "_ref_meta",
                    "_input",
                ):
                    if hasattr(benchmark, attr):
                        delattr(benchmark, attr)

        entries[name] = {
            "result": values,
            "stats": [None] * len(param_combos),
            "samples": [None] * len(param_combos),
            "duration_seconds": time.monotonic() - start,
            "started_at": started_at,
            "errcode": 1 if stderr else 0,
            "stderr": "\n".join(stderr),
        }

    params = dict(machine_params.__dict__)
    return {
        "commit_hash": commit_hash,
        "date": commit_date,
        "env_name": "check-suite",
        "env_vars": {},
        "params": params,
        "result_count": len(entries),
        "results": entries,
    }


_UPLOAD_DATASET_CODE = r"""
import importlib
import os
import sys

import saps

module_name, class_name, generator_name, dataset_name = sys.argv[1:5]
bench_module = importlib.import_module(f"saps.benchmarks.{module_name}")
bench = getattr(bench_module, class_name)()
generator = next(gen for gen in bench.generators if gen.name == generator_name)
dataset = next(ds for ds in generator.datasets if ds.name == dataset_name)
backend = saps.build_storage_backend(
    type=os.environ.get("REMOTE_STORAGE_BACKEND"),
    bucket=os.environ.get("REMOTE_STORAGE_BUCKET"),
)
digest = backend.check_manifest(generator, dataset)
prefix = None if digest is None else backend.prefix(generator, dataset, digest)
if prefix is not None and backend.file_exists(prefix):
    print("fresh")
    raise SystemExit(0)
raise SystemExit(0 if backend.upload_dataset(generator, dataset) else 1)
"""


def _cache_datasets(benchmarks, metadata, environments, timeout: float | None) -> int:
    """Upload the selected (benchmark, generator, dataset) triples."""
    uploaded = failed = skipped = 0
    seen: set[tuple[str, str]] = set()
    for env in environments:
        Setup.perform_setup([env], parallel=1)
        for name in benchmarks:
            if name not in metadata:
                continue
            selected_pairs = None
            selected_idx = benchmarks.benchmark_selection.get(name)
            if selected_idx is not None:
                selected_pairs = {
                    tuple(param[0].split(".")[:2])
                    for idx, param in enumerate(product(*benchmarks[name]["params"]))
                    if idx in selected_idx and len(param) > 0
                }
            module_name, class_name, _method = name.rsplit(".", 2)
            bench_meta = metadata[name]
            generators = {gen["name"]: gen for gen in bench_meta["generators"]}
            for generator_name, gen_meta in generators.items():
                for ds_meta in gen_meta["datasets"]:
                    dataset_name = ds_meta["name"]
                    key = (generator_name, dataset_name)
                    if selected_pairs is not None and key not in selected_pairs:
                        continue
                    if key in seen:
                        skipped += 1
                        continue
                    seen.add(key)
                    label = f"{bench_meta['name']} / {generator_name} / {dataset_name}"
                    if not gen_meta["cacheable"]:
                        skipped += 1
                        print(f"[uncached] {label}")
                        continue
                    print(f"[caching]  {label}", flush=True)
                    try:
                        output = env.run(
                            [
                                "-c",
                                _UPLOAD_DATASET_CODE,
                                module_name,
                                class_name,
                                generator_name,
                                dataset_name,
                            ],
                            timeout=timeout,
                        )
                        if output.splitlines()[-1:] == ["fresh"]:
                            skipped += 1
                            print(f"[fresh]    {label}")
                        else:
                            uploaded += 1
                            print(f"[uploaded] {label}")
                    except (
                        OSError,
                        RuntimeError,
                        util.ProcessError,
                    ) as e:
                        failed += 1
                        print(f"[error]    {label}: {e}")
    print(f"upload summary: uploaded={uploaded} failed={failed} skipped={skipped}")
    return 0 if failed == 0 else 1


def _benchmark_metadata_name(asv_name: str) -> str:
    method_name = asv_name.rsplit(".", 1)[-1]
    for prefix in ("peakmem_", "time_"):
        if method_name.startswith(prefix):
            return method_name.removeprefix(prefix)
    return method_name


def _load_metadata(metadata_path: Path) -> dict[str, dict]:
    if not metadata_path.exists():
        raise RuntimeError(
            f"{metadata_path} does not exist; run generate metadata first with "
            "`poetry run ./bin/generate_metadata.py`."
        )
    document = json.loads(metadata_path.read_text(encoding="utf-8"))
    return {record["name"]: record for record in document.get("benchmarks", [])}


def _statistics_freshness(statistics_path: Path) -> dict[tuple[str, str, str], str]:
    if not statistics_path.exists():
        return {}
    document = json.loads(statistics_path.read_text(encoding="utf-8"))
    return {
        (benchmark["name"], generator["name"], dataset["name"]): dataset["freshness"]
        for benchmark in document.get("benchmarks", [])
        for generator in benchmark.get("generators", [])
        for dataset in generator.get("datasets", [])
        if "freshness" in dataset
    }


def _trace_statistics(
    benchmarks,
    environments,
    machine_params,
    commit_hash,
    commit_date,
    outputs_dir,
    timeout,
    show_stderr,
    skipped_fresh=0,
) -> int:
    """Run selected benchmarks under ASV with the tagger framework."""
    stats_dir = outputs_dir / "tagger_stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    for stats_path in stats_dir.glob("*.json"):
        stats_path.unlink()

    print(f"Discovered {len(benchmarks)} benchmark entries for tagger")
    print(f"Using timeout: {timeout} seconds")

    failed = 0
    for env in environments:
        Setup.perform_setup([env], parallel=1)

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
            quick=True,
            extra_params={"timeout": timeout},
        )
        failed += sum(
            1 for errcode in results.errcode.values() if errcode not in (None, 0)
        )
        results.save(outputs_dir / "results")

    tagged = sum(1 for _ in stats_dir.glob("*.json"))
    print(
        "tag summary: "
        f"tagged_records={tagged} "
        f"failed_benchmark_entries={failed} "
        f"skipped_fresh={skipped_fresh}"
    )
    return 0 if failed == 0 and (tagged > 0 or skipped_fresh > 0) else 1


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
            "Run selected benchmark cases once in-process and print result JSON. "
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
            "Split the selected benchmark parameter cases into this many chunks. "
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
    benchmark_metrics = args.metrics or ["peakmem", "time"]

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
    persistent_metadata_path = repo_root / "metadata.json"
    persistent_statistics_path = repo_root / "statistics.json"
    manifest_path = str(repo_root / "manifest.json")
    pythonpath = str(repo_root)
    os.environ["SAPS_MANIFEST_PATH"] = manifest_path
    os.environ["PYTHONPATH"] = pythonpath
    os.environ["REMOTE_STORAGE_BACKEND"] = storage_backend
    os.environ["REMOTE_STORAGE_BUCKET"] = storage_bucket
    matrix["env_nobuild"]["SAPS_MANIFEST_PATH"] = [manifest_path]
    if args.trace_statistics or args.cache_datasets:
        framework_file = (
            "frameworks/saps_tagger.py"
            if args.trace_statistics
            else "frameworks/saps_numpy.py"
        )
        os.environ["SAPS_FRAMEWORK"] = str(repo_root / framework_file)
        if args.trace_statistics:
            tagger_stats_dir = str(outputs_dir / "tagger_stats")
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
    metric_prefixes = tuple(f"{metric}_" for metric in benchmark_metrics)
    benchmarks = benchmarks.filter_out(
        {
            name
            for name in benchmarks
            if not name.rsplit(".", 1)[-1].startswith(metric_prefixes)
        }
    )

    persistent_metadata = _load_metadata(persistent_metadata_path)
    metadata: dict[str, dict] = {}
    for name in benchmarks:
        record = persistent_metadata.get(_benchmark_metadata_name(name))
        if record is not None:
            metadata[name] = record

    results_dir = outputs_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    include_set = {tag.strip() for tag in args.tag if tag and tag.strip()}
    exclude_set = {tag.strip() for tag in args.no_tag if tag and tag.strip()}
    include_res = args.re or []
    exclude_res = args.no_re or []
    stats_freshness = (
        _statistics_freshness(persistent_statistics_path)
        if args.trace_statistics
        else {}
    )
    skipped_fresh_statistics = 0

    skips = []
    benchmarks._benchmark_selection = {}
    for name in benchmarks:
        bench_meta = metadata.get(name)
        if bench_meta is None:
            log.warning(
                "No SAPS metadata found for benchmark "
                f"'{name}', skipping SAPS tag filtering "
                "for this benchmark"
            )
            skips.append(name)
            continue
        selection = []
        datasets = {
            (generator["name"], dataset["name"]): dataset
            for generator in bench_meta["generators"]
            for dataset in generator["datasets"]
        }
        param_combos = list(product(*benchmarks[name]["params"]))
        for idx, param in enumerate(param_combos):
            if len(param) == 0:
                print(
                    f"Warning: benchmark '{name}' has no data generators, skipping"
                    " benchmark"
                )
                continue
            generator_name, dataset_name = param[0].split(".")[:2]
            dataset = datasets.get((generator_name, dataset_name))
            if dataset is None:
                continue
            target_names = (bench_meta["name"], generator_name, dataset_name)
            dataset_tags = set(dataset["tags"])
            if include_res and not any(
                re.search(pattern, target)
                for pattern in include_res
                for target in target_names
            ):
                continue
            if include_set and not include_set.intersection(dataset_tags):
                continue
            if exclude_res and any(
                re.search(pattern, target)
                for pattern in exclude_res
                for target in target_names
            ):
                continue
            if exclude_set and exclude_set.intersection(dataset_tags):
                continue
            if args.trace_statistics:
                stats_key = (bench_meta["name"], generator_name, dataset_name)
                if dataset["freshness"] == stats_freshness.get(stats_key):
                    skipped_fresh_statistics += 1
                    continue
            selection.append(idx)

        if selection:
            benchmarks._benchmark_selection[name] = selection
        else:
            skips.append(name)

    benchmarks = benchmarks.filter_out(set(skips))
    chunk_total, chunk_kept = _apply_chunk_selection(
        benchmarks, args.chunk_count, args.chunk_index
    )
    if args.chunk_count > 1:
        benchmarks = benchmarks.filter_out(
            {
                name
                for name in benchmarks
                if not benchmarks.benchmark_selection.get(name)
            }
        )
        print(
            "Selected benchmark chunk "
            f"{args.chunk_index}/{args.chunk_count}: "
            f"{chunk_kept} of {chunk_total} parameter cases"
        )
    if args.cache_datasets:
        return _cache_datasets(
            benchmarks=benchmarks,
            metadata=metadata,
            environments=environments,
            timeout=timeout,
        )

    if args.trace_statistics:
        return _trace_statistics(
            benchmarks=benchmarks,
            environments=environments,
            machine_params=machine_params,
            commit_hash=commit_hash,
            commit_date=commit_date,
            outputs_dir=outputs_dir,
            timeout=timeout,
            show_stderr=args.show_stderr,
            skipped_fresh=skipped_fresh_statistics,
        )

    if args.check_suite:
        print(f"Discovered {len(benchmarks)} benchmark entries")
        result_json = _run_check_suite(
            benchmarks=benchmarks,
            machine_params=machine_params,
            commit_hash=commit_hash,
            commit_date=commit_date,
        )
        print(json.dumps(result_json, indent=2, default=str))
        return 0

    print(f"Discovered {len(benchmarks)} benchmark entries")
    print(f"Using timeout: {timeout} seconds")

    for env in environments:
        Setup.perform_setup([env], parallel=1)
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
            show_stderr=args.show_stderr,
            quick=args.quick,
            extra_params={"timeout": timeout},
        )

        print("Results object:", results)
        print(json.dumps(format_results(results, benchmarks), indent=2, default=str))
        results.save(outputs_dir / "results")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
