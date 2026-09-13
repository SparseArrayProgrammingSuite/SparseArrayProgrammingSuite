#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from contextlib import suppress
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _reference(table, prefix, record):
    key = (
        prefix
        + "_"
        + hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()[:16]
    )
    table[key] = record
    return key


def _json_value(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    return value


def combine_results(run_directory: Path, metadata: dict) -> tuple[dict, dict]:
    dataset_names = {
        (asv_id, dataset["asv_param"]): (
            benchmark["name"],
            generator["name"],
            dataset["name"],
            metric,
        )
        for benchmark in metadata["benchmarks"]
        for metric, asv_id in benchmark["asv_ids"].items()
        for generator in benchmark["generators"]
        for dataset in generator["datasets"]
    }
    combined = {
        "run": run_directory.name,
        "result_file_count": 0,
        "frameworks": {},
        "sources": {},
        "diagnostics": {},
        "benchmarks": [],
    }
    machines = {}
    outputs = {}
    # Include older task/results/chunk-N layouts as well as direct results/.
    result_roots = [run_directory / "results", *run_directory.glob("task_*/results")]
    for result_root in sorted(result_roots):
        for path in sorted(result_root.rglob("*.json")):
            if any(
                part.startswith(".") for part in path.relative_to(result_root).parts
            ):
                continue
            document = json.loads(path.read_text(encoding="utf-8"))
            if not all(
                key in document
                for key in (
                    "commit_hash",
                    "env_name",
                    "params",
                    "result_columns",
                    "results",
                )
            ):
                continue
            combined["result_file_count"] += 1
            env_vars = document.get("env_vars", {})
            framework_file = env_vars.get("SAPS_FRAMEWORK")
            if framework_file and env_vars.get("SAPS_REPO_ROOT"):
                with suppress(ValueError):
                    framework_file = (
                        Path(framework_file)
                        .relative_to(env_vars["SAPS_REPO_ROOT"])
                        .as_posix()
                    )
            framework = _reference(
                combined["frameworks"],
                "framework",
                {
                    "name": Path(framework_file).stem.removeprefix("saps_")
                    if framework_file
                    else document["env_name"],
                    "file": framework_file,
                    "python": document.get("python", document["params"].get("python")),
                    "requirements": document.get("requirements", {}),
                    "env_vars": {
                        key: value
                        for key, value in env_vars.items()
                        if key
                        not in {
                            "SAPS_FRAMEWORK",
                            "SAPS_REPO_ROOT",
                            "SAPS_LOG_PATH",
                            "SAPS_CACHE_DIR",
                            "SAPS_MANIFEST_PATH",
                            "REMOTE_STORAGE_BACKEND",
                            "REMOTE_STORAGE_BUCKET",
                        }
                    },
                },
            )
            source = _reference(
                combined["sources"],
                "source",
                {
                    "file": path.relative_to(run_directory).as_posix(),
                    "commit_hash": document["commit_hash"],
                    "date": document.get("date"),
                    "env_name": document["env_name"],
                    "env_vars": env_vars,
                    "machine_label": document["params"].get("machine"),
                },
            )
            details = document.get("saps", {})
            fallback_machine = details.get("legacy_machine") or {
                key: value
                for key, value in document["params"].items()
                if key != "python" and key not in document.get("requirements", {})
            }
            fallback_machine["machine"] = fallback_machine.pop(
                "hostname", fallback_machine.get("machine")
            )
            runs = {}
            for run in details.get("runs", []):
                for parameters in run["parameters"]:
                    runs[(run["benchmark"], run.get("version"), tuple(parameters))] = (
                        run
                    )

            for name, raw_row in document["results"].items():
                row = dict(zip(document["result_columns"], raw_row, strict=False))
                parameters = list(itertools.product(*row.get("params", [])))
                values = row.get("result")
                for index, parameter in enumerate(parameters):
                    value = values[index] if values is not None else None
                    run = runs.get((name, row.get("version"), parameter))
                    # ASV marks parameters excluded from a task with NaN. New
                    # diagnostics distinguish those from deliberately skipped runs.
                    is_nan = isinstance(value, float) and math.isnan(value)
                    if run is None and (
                        is_nan
                        or (
                            value is None
                            and details.get("runs")
                            and "legacy_machine" not in details
                        )
                    ):
                        continue
                    key = (name, parameter[0] if parameter else "")
                    if key not in dataset_names:
                        raise ValueError(
                            f"No metadata match for {key!r} in {path}; "
                            "use --metadata with metadata for this run."
                        )
                    benchmark, generator, dataset, metric = dataset_names[key]
                    machine = _reference(
                        machines,
                        "machine",
                        (
                            details["machines"][run["machine"]]
                            if run
                            else fallback_machine
                        ),
                    )
                    diagnostic = None
                    if run is not None:
                        diagnostic = _reference(
                            combined["diagnostics"],
                            "diagnostic",
                            {
                                "source": source,
                                **{
                                    key: val
                                    for key, val in run.items()
                                    if key != "machine"
                                },
                            },
                        )
                    status = (
                        "skipped" if is_nan else "failed" if value is None else "ok"
                    )
                    if isinstance(value, float) and math.isinf(value):
                        status = "nonfinite"
                    result = {
                        "framework": framework,
                        "machine": machine,
                        "source": source,
                        "metric": metric,
                        "result": _json_value(value),
                        "status": status,
                        "diagnostics": diagnostic,
                        "stats": {
                            key.removeprefix("stats_"): _json_value(val[index])
                            for key, val in row.items()
                            if key.startswith("stats_") and val is not None
                        },
                        "samples": _json_value(row["samples"][index])
                        if row.get("samples") is not None
                        else None,
                    }
                    outputs.setdefault((benchmark, generator, dataset), []).append(
                        result
                    )

    # Preserve metadata's hierarchy and ordering, omitting datasets without outputs.
    for benchmark in metadata["benchmarks"]:
        generators = []
        for generator in benchmark["generators"]:
            datasets = []
            for dataset in generator["datasets"]:
                results = outputs.get(
                    (benchmark["name"], generator["name"], dataset["name"])
                )
                if results:
                    datasets.append({"name": dataset["name"], "results": results})
            if datasets:
                generators.append({"name": generator["name"], "datasets": datasets})
        if generators:
            combined["benchmarks"].append(
                {"name": benchmark["name"], "generators": generators}
            )
    return _json_value(combined), _json_value(machines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Tabulate competition results")
    parser.add_argument(
        "--run-directory",
        type=Path,
        help="Run directory (default: highest numbered competition/run_<job-id>)",
    )
    parser.add_argument("--metadata", type=Path, default=REPO_ROOT / "metadata.json")
    parser.add_argument("-o", "--output", type=Path, help="Default: RUN/results.json")
    args = parser.parse_args()
    if args.run_directory is None:
        runs = [
            path
            for path in (REPO_ROOT / "competition").glob("run_*")
            if path.is_dir() and path.name.removeprefix("run_").isdigit()
        ]
        if not runs:
            parser.error("No numbered run directories found in competition/.")
        args.run_directory = max(runs, key=lambda path: int(path.name[4:]))
    run_directory = args.run_directory.resolve()
    if not run_directory.is_dir():
        parser.error(f"Run directory does not exist: {run_directory}")
    output = args.output or run_directory / "results.json"
    machine_output = output.parent / "machines.json"
    if output.resolve() == machine_output.resolve():
        parser.error("--output must differ from machines.json")
    # Array tasks may finish together. Serialize whole-run aggregation so a late
    # writer cannot replace a newer snapshot with an earlier, incomplete one.
    with (run_directory / ".combine.lock").open("a+b") as lock:
        if os.name == "nt":
            import msvcrt

            lock.write(b"0")
            lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        document, machines = combine_results(
            run_directory, json.loads(args.metadata.read_text(encoding="utf-8"))
        )
        document["metadata"] = os.path.relpath(
            args.metadata.resolve(), output.parent.resolve()
        )
        document["run_directory"] = os.path.relpath(
            run_directory, output.parent.resolve()
        )
        document["machines_file"] = machine_output.name
        output.parent.mkdir(parents=True, exist_ok=True)
        for path, content in ((machine_output, machines), (output, document)):
            temporary = path.with_name(path.name + ".tmp")
            temporary.write_text(
                json.dumps(content, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
            temporary.replace(path)
    print(f"combined {document['result_file_count']} result files into {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
