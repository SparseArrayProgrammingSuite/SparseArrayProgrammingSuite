#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import importlib
import json
import logging
import os
import re
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
raise SystemExit(0 if backend.upload_dataset(generator, dataset) else 1)
"""


def _cache_datasets(benchmarks, metadata, environments) -> int:
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
                    try:
                        env.run(
                            [
                                "-c",
                                _UPLOAD_DATASET_CODE,
                                module_name,
                                class_name,
                                generator_name,
                                dataset_name,
                            ]
                        )
                        uploaded += 1
                        print(f"[uploaded] {label}")
                    except (
                        OSError,
                        RuntimeError,
                        StopIteration,
                        util.ProcessError,
                    ) as e:
                        failed += 1
                        print(f"[error]    {label}: {e}")
    print(f"upload summary: uploaded={uploaded} failed={failed} skipped={skipped}")
    return 0 if failed == 0 else 1


def _fill_is_nonzero(value) -> bool:
    if value is None:
        return False
    try:
        return bool(value != 0)
    except ValueError:
        return True


def _infer_tags(stats: dict) -> tuple[list[str], list[str]]:
    tensors = stats.get("tensors", [])
    operators = set(stats.get("operators", {}))
    operator_names = {op.rsplit(".", 1)[-1] for op in operators}
    arg_counts = [
        count
        for counts in stats.get("operator_arg_counts", {}).values()
        for count in counts
    ]
    operand_stats = [
        operands
        for invocations in stats.get("operator_operand_stats", {}).values()
        for operands in invocations
    ]

    benchmark_tags: set[str] = set()
    generator_tags: set[str] = set()

    if any(t.get("ndim", 0) >= 5 for t in tensors):
        generator_tags.add("high-dimensional")
    if any(t.get("ndim", 0) >= 3 for t in tensors):
        generator_tags.add("tensor")
    if any(count >= 5 for count in arg_counts):
        benchmark_tags.add("large-query")

    transcendental_ops = {
        "acos",
        "acosh",
        "asin",
        "asinh",
        "atan",
        "atan2",
        "atanh",
        "cos",
        "cosh",
        "exp",
        "expm1",
        "log",
        "log1p",
        "log2",
        "log10",
        "logaddexp",
        "power",
        "sin",
        "sinh",
        "sqrt",
        "tan",
        "tanh",
    }
    shape_ops = {
        "broadcast_to",
        "concatenate",
        "concat",
        "expand_dims",
        "flatten",
        "moveaxis",
        "permute_dims",
        "ravel",
        "reshape",
        "squeeze",
        "stack",
        "transpose",
    }
    fancy_ops = {
        "all",
        "any",
        "bitwise_and",
        "bitwise_invert",
        "bitwise_left_shift",
        "bitwise_or",
        "bitwise_right_shift",
        "bitwise_xor",
        "equal",
        "greater",
        "greater_equal",
        "less",
        "less_equal",
        "logical_and",
        "logical_not",
        "logical_or",
        "logical_xor",
        "max",
        "maximum",
        "min",
        "minimum",
        "not_equal",
        "sort",
        "where",
    }
    index_ops = {"getitem", "setitem", "take", "nonzero", "argwhere"}
    linalg_ops = {
        "cholesky",
        "dot",
        "eig",
        "inv",
        "matmul",
        "norm",
        "pinv",
        "qr",
        "solve",
        "svd",
        "tensordot",
    }
    elementary_ops = {
        "add",
        "divide",
        "floor_divide",
        "multiply",
        "negative",
        "positive",
        "remainder",
        "subtract",
    }

    if operator_names.intersection(transcendental_ops):
        benchmark_tags.add("transcendental-ops")
    if operator_names.intersection(shape_ops):
        benchmark_tags.add("shape-ops")
    if operator_names.intersection(fancy_ops):
        benchmark_tags.add("fancy-ops")
    if operator_names.intersection(index_ops) or {
        "array.getitem",
        "array.setitem",
    }.intersection(operators):
        benchmark_tags.add("index-ops")
    if any(op.startswith("linalg.") for op in operators) or operator_names.intersection(
        linalg_ops
    ):
        benchmark_tags.add("linalg-ops")
    if operator_names.intersection(elementary_ops) and not benchmark_tags.intersection(
        {
            "transcendental-ops",
            "shape-ops",
            "fancy-ops",
            "index-ops",
            "linalg-ops",
        }
    ):
        benchmark_tags.add("elementary-ops")

    if any(_fill_is_nonzero(t.get("fill_value")) for t in tensors):
        benchmark_tags.add("nonzero-fill")

    sparsities = [t.get("sparsity") for t in tensors if t.get("sparsity") is not None]
    if sparsities and all(sparsity == 1 for sparsity in sparsities):
        generator_tags.add("dense")
    if any(t.get("sparsity") is not None and t["sparsity"] <= 0.01 for t in tensors):
        generator_tags.add("hypersparse")
    if any(
        sum(
            1
            for operand in operands
            if operand.get("sparsity") is not None and operand["sparsity"] < 1
        )
        >= 2
        for operands in operand_stats
    ):
        generator_tags.add("dynamic-sparsity")

    return sorted(benchmark_tags), sorted(generator_tags)


def _metadata_generator_record(metadata, benchmark_name, generator_name):
    generators = metadata[benchmark_name]["generators"]
    return next(gen for gen in generators if gen["name"] == generator_name)


def _normalize_tag(tag: str) -> str:
    tag = tag.strip().lower().replace("_", "-")
    return re.sub(r"-+", "-", tag)


def _normalize_tags(tags) -> list[str]:
    return sorted(
        {
            normalized
            for tag in tags
            if isinstance(tag, str) and (normalized := _normalize_tag(tag))
        }
    )


def _merge_generated_tags(record, field: str, tags):
    record[field] = _normalize_tags([*record.get(field, []), *tags])


def _effective_tags(record: dict) -> set[str]:
    return set(record.get("tags", [])).union(
        record.get("topics", []),
        record.get("statistics", []),
    )


def regex_any_match(patterns: list[str], value: str) -> bool:
    return any(re.search(pattern, value) for pattern in patterns)


def _normalize_benchmark_id(benchmark_id: str) -> str:
    if benchmark_id.startswith("benchmarks."):
        return f"saps.{benchmark_id}"
    return benchmark_id


def _apply_tagger_stats(metadata, stats_dir: Path) -> int:
    tagged = 0
    metadata_by_id: dict[str, list[str]] = {}
    for name, bench_meta in metadata.items():
        metadata_by_id.setdefault(
            _normalize_benchmark_id(bench_meta.get("id", name)), []
        ).append(name)
    for stats_path in sorted(stats_dir.glob("*.json")):
        record = json.loads(stats_path.read_text(encoding="utf-8"))
        benchmark_id = _normalize_benchmark_id(record["benchmark_id"])
        generator_name = record["generator_name"]
        if benchmark_id not in metadata_by_id:
            print(f"[missing] {stats_path.name}: {benchmark_id}")
            continue

        benchmark_tags, generator_tags = _infer_tags(record.get("stats", {}))
        for metadata_key in metadata_by_id[benchmark_id]:
            _merge_generated_tags(metadata[metadata_key], "statistics", benchmark_tags)
            generator = _metadata_generator_record(
                metadata, metadata_key, generator_name
            )
            _merge_generated_tags(generator, "statistics", generator_tags)
        tagged += 1
        print(
            f"[tagged]  {benchmark_id} / {generator_name}: "
            f"benchmark={', '.join(benchmark_tags)}; "
            f"generator={', '.join(generator_tags)}"
        )
    return tagged


def _generate_topics(
    metadata: dict[str, dict],
    metadata_path: Path,
    persistent_metadata_path: Path,
) -> int:
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _update_persistent_metadata(
        metadata, persistent_metadata_path, replace_fields={"topics"}
    )
    print(f"generated topics for {len(metadata)} benchmark records")
    return 0


def _record_key(record: dict) -> str:
    benchmark_id = _normalize_benchmark_id(record.get("id", record["name"]))
    match = re.search(r"(?:^|\.)benchmarks\.([^.]+\.[^.]+)(?:\.|$)", benchmark_id)
    if match is not None:
        return match.group(1)
    return benchmark_id


def _merge_generated_field(
    existing: dict,
    incoming: dict,
    field: str,
    replace_fields: set[str],
) -> list[str]:
    if field in replace_fields:
        return _normalize_tags(incoming.get(field, []))
    return _normalize_tags([*existing.get(field, []), *incoming.get(field, [])])


def _merge_dataset_metadata(
    existing: dict, incoming: dict, replace_fields: set[str] | None = None
) -> dict:
    replace_fields = replace_fields or set()
    merged = copy.deepcopy(incoming)
    for key, value in existing.items():
        if key not in {"digest", "tags", "topics", "statistics"} and key not in merged:
            merged[key] = copy.deepcopy(value)
    merged["tags"] = copy.deepcopy(incoming.get("tags", existing.get("tags", [])))
    merged["topics"] = _merge_generated_field(
        existing, incoming, "topics", replace_fields
    )
    merged["statistics"] = _merge_generated_field(
        existing, incoming, "statistics", replace_fields
    )
    merged.pop("digest", None)
    return merged


def _merge_generator_metadata(
    existing: dict, incoming: dict, replace_fields: set[str] | None = None
) -> dict:
    replace_fields = replace_fields or set()
    merged = copy.deepcopy(incoming)
    for key, value in existing.items():
        if key not in {"tags", "topics", "statistics"} and key not in merged:
            merged[key] = copy.deepcopy(value)
    merged["tags"] = copy.deepcopy(incoming.get("tags", existing.get("tags", [])))
    merged["topics"] = _merge_generated_field(
        existing, incoming, "topics", replace_fields
    )
    merged["statistics"] = _merge_generated_field(
        existing, incoming, "statistics", replace_fields
    )
    existing_datasets = {
        dataset["name"]: dataset for dataset in existing.get("datasets", [])
    }
    datasets = []
    for dataset in merged.get("datasets", []):
        old_dataset = existing_datasets.get(dataset["name"])
        if old_dataset is not None:
            dataset = _merge_dataset_metadata(old_dataset, dataset, replace_fields)
        datasets.append(dataset)
    merged["datasets"] = datasets
    return merged


def _merge_benchmark_metadata(
    existing: dict, incoming: dict, replace_fields: set[str] | None = None
) -> dict:
    replace_fields = replace_fields or set()
    merged = copy.deepcopy(incoming)
    for key, value in existing.items():
        if key not in {"tags", "topics", "statistics"} and key not in merged:
            merged[key] = copy.deepcopy(value)
    merged["tags"] = copy.deepcopy(incoming.get("tags", existing.get("tags", [])))
    merged["topics"] = _merge_generated_field(
        existing, incoming, "topics", replace_fields
    )
    merged["statistics"] = _merge_generated_field(
        existing, incoming, "statistics", replace_fields
    )
    existing_generators = {
        generator["name"]: generator for generator in existing.get("generators", [])
    }
    generators = []
    for generator in merged.get("generators", []):
        old_generator = existing_generators.get(generator["name"])
        if old_generator is not None:
            generator = _merge_generator_metadata(old_generator, generator, replace_fields)
        generators.append(generator)
    merged["generators"] = generators
    return merged


def _collapse_metadata(
    metadata: dict[str, dict], replace_fields: set[str] | None = None
) -> list[dict]:
    replace_fields = replace_fields or set()
    collapsed: dict[str, dict] = {}
    for record in metadata.values():
        key = _record_key(record)
        if key in collapsed:
            collapsed[key] = _merge_benchmark_metadata(
                collapsed[key], record, replace_fields
            )
        else:
            collapsed[key] = copy.deepcopy(record)
    return sorted(collapsed.values(), key=_record_key)


def _digest_map(document: dict) -> dict[str, str]:
    digests = dict(document.get("digests", {}))
    for benchmark in document.get("benchmarks", []):
        for generator in benchmark.get("generators", []):
            for dataset in generator.get("datasets", []):
                if "digest" in dataset:
                    digests[f"{generator['name']}.{dataset['name']}"] = dataset[
                        "digest"
                    ]
    return digests


def _strip_nested_digests(record: dict) -> dict:
    stripped = copy.deepcopy(record)
    for generator in stripped.get("generators", []):
        for dataset in generator.get("datasets", []):
            dataset.pop("digest", None)
    return stripped


def _ensure_tag_fields(record: dict) -> dict:
    record.setdefault("tags", [])
    record.setdefault("topics", [])
    record.setdefault("statistics", [])
    record["topics"] = _normalize_tags(record["topics"])
    record["statistics"] = _normalize_tags(record["statistics"])
    for generator in record.get("generators", []):
        _ensure_tag_fields(generator)
        for dataset in generator.get("datasets", []):
            _ensure_tag_fields(dataset)
    return record


def _update_persistent_metadata(
    metadata: dict[str, dict],
    metadata_path: Path,
    *,
    replace_fields: set[str] | None = None,
):
    replace_fields = replace_fields or set()
    current = {"benchmarks": []}
    if metadata_path.exists():
        current = json.loads(metadata_path.read_text(encoding="utf-8"))

    digests = _digest_map(current)
    digests.update(_digest_map({"benchmarks": list(metadata.values())}))

    existing_by_key = {}
    for record in current.get("benchmarks", []):
        key = _record_key(record)
        record = _strip_nested_digests(record)
        if key in existing_by_key:
            record = _merge_benchmark_metadata(existing_by_key[key], record)
        existing_by_key[key] = record

    for record in _collapse_metadata(metadata, replace_fields):
        key = _record_key(record)
        record = _strip_nested_digests(record)
        if key in existing_by_key:
            record = _merge_benchmark_metadata(
                existing_by_key[key], record, replace_fields
            )
        existing_by_key[key] = record

    metadata_path.write_text(
        json.dumps(
            {
                "benchmarks": sorted(
                    (_ensure_tag_fields(record) for record in existing_by_key.values()),
                    key=_record_key,
                ),
                "digests": dict(sorted(digests.items())),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _trace_statistics(
    benchmarks,
    metadata,
    metadata_path,
    persistent_metadata_path,
    environments,
    machine_params,
    commit_hash,
    commit_date,
    outputs_dir,
    timeout,
    show_stderr,
) -> int:
    """Run selected benchmarks under ASV with the tagger framework."""
    _update_persistent_metadata(metadata, persistent_metadata_path)
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

    tagged = _apply_tagger_stats(metadata, stats_dir)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _update_persistent_metadata(
        metadata, persistent_metadata_path, replace_fields={"statistics"}
    )
    print(f"tag summary: tagged_records={tagged} failed_benchmark_entries={failed}")
    return 0 if tagged > 0 else 1


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
            "The standard s3 bucket is 'sparse-array-programming-suite' "
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
            "and replace generated statistics tags in benchmark_metadata.json. "
            "Honors --re/--no-re/--tag/--no-tag filters."
        ),
    )
    parser.add_argument(
        "--generate-topics",
        dest="generate_topics",
        action="store_true",
        help=(
            "Skip benchmark execution. Generate topic tags in benchmark_metadata.json "
            "from pasted ACM CCS XML. Honors --re/--no-re/--tag/--no-tag filters."
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

    log.enable(args.verbose)

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
    storage_backend = args.remote_storage_backend or "s3"
    storage_bucket = args.remote_storage_bucket or "sparse-array-programming-suite"
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
    persistent_metadata_path = repo_root / "benchmark_metadata.json"
    manifest_path = str(persistent_metadata_path)
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
            os.environ["SAPS_ASV_SINGLE_RUN"] = "1"

    uses_parent_environment = (
        args.trace_statistics or args.cache_datasets or args.generate_topics
    )

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
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    include_set = {tag.strip().lower() for tag in args.tag if tag and tag.strip()}
    exclude_set = {tag.strip().lower() for tag in args.no_tag if tag and tag.strip()}
    include_res = args.re or []
    exclude_res = args.no_re or []

    def match_target(obj: dict) -> str:
        return obj.get("id", obj["name"])

    def is_include(obj) -> bool:
        if include_res and not regex_any_match(include_res, match_target(obj)):
            return False
        return not (include_set and not include_set.intersection(_effective_tags(obj)))

    def is_exclude(obj) -> bool:
        if exclude_res and regex_any_match(exclude_res, match_target(obj)):
            return True
        return bool(exclude_set and exclude_set.intersection(_effective_tags(obj)))

    if (
        not (args.cache_datasets or args.trace_statistics or args.generate_topics)
        and persistent_metadata_path.exists()
    ):
        persistent_metadata = json.loads(
            persistent_metadata_path.read_text(encoding="utf-8")
        )
        persistent_by_key = {
            _record_key(record): record
            for record in persistent_metadata.get("benchmarks", [])
        }
        for name, record in list(metadata.items()):
            existing = persistent_by_key.get(_record_key(record))
            if existing is not None:
                metadata[name] = _merge_benchmark_metadata(existing, record)

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

    if args.cache_datasets:
        _update_persistent_metadata(metadata, persistent_metadata_path)
        return _cache_datasets(
            benchmarks=benchmarks,
            metadata=metadata,
            environments=environments,
        )

    if args.generate_topics:
        selected_metadata = {
            name: metadata[name] for name in benchmarks if name in metadata
        }
        return _generate_topics(
            metadata=selected_metadata,
            metadata_path=metadata_path,
            persistent_metadata_path=persistent_metadata_path,
        )

    if args.trace_statistics:
        return _trace_statistics(
            benchmarks=benchmarks,
            metadata=metadata,
            metadata_path=metadata_path,
            persistent_metadata_path=persistent_metadata_path,
            environments=environments,
            machine_params=machine_params,
            commit_hash=commit_hash,
            commit_date=commit_date,
            outputs_dir=outputs_dir,
            timeout=timeout,
            show_stderr=args.show_stderr,
        )

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
