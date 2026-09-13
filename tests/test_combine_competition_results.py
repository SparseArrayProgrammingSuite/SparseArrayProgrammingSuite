from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def combiner():
    module_path = REPO_ROOT / "scripts/combine_competition_results.py"
    spec = importlib.util.spec_from_file_location(
        "combine_competition_results", module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def metadata():
    return {
        "benchmarks": [
            {
                "name": "benchmark",
                "asv_ids": {"time": "bench.time", "peakmem": "bench.peakmem"},
                "generators": [
                    {
                        "name": "generator.with.dots",
                        "datasets": [
                            {"name": "successful.dataset", "asv_param": "p.ok"},
                            {"name": "failed", "asv_param": "p.failed"},
                            {"name": "skipped", "asv_param": "p.skipped"},
                        ],
                    }
                ],
            }
        ]
    }


def write_result(path, values, *, machine="node-a", details=None):
    document = {
        "commit_hash": "abc123",
        "date": 1,
        "python": "3.12",
        "env_name": "test-env",
        "env_vars": {
            "SAPS_FRAMEWORK": "/repo/frameworks/saps_numpy.py",
            "SAPS_REPO_ROOT": "/repo",
        },
        "params": {
            "machine": machine,
            "cpu": "test cpu",
            "python": "3.12",
            "numpy": "2.3",
        },
        "requirements": {"numpy": "2.3"},
        "result_columns": ["result", "params", "version", "stats_repeat", "samples"],
        "results": {
            "bench.time": [
                values,
                [["p.skipped", "p.ok", "p.failed"]],
                "v1",
                [None, 2, None],
            ]
        },
    }
    if details is not None:
        document["saps"] = details
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document))
    return document


@pytest.mark.parametrize("path_type", [PurePosixPath, PureWindowsPath])
def test_combine_maps_parameters_and_cross_references(
    combiner, metadata, tmp_path, monkeypatch, path_type
):
    monkeypatch.setattr(combiner, "Path", path_type)
    details = {
        "machines": {"a": {"machine": "node-a", "cpu": "test cpu"}},
        "runs": [
            {
                "benchmark": "bench.time",
                "version": "v1",
                "parameters": [["p.ok"], ["p.failed"]],
                "machine": "a",
                "errcode": 1,
                "stderr": "For parameters: p.failed\nfailed setup",
            }
        ],
    }
    write_result(
        tmp_path / "task_0/results/chunk-0/alias/abc.json",
        [float("nan"), 0.0, None],
        details=details,
    )
    other = tmp_path / "task_1/results/alias/abc.json"
    document = write_result(other, [float("nan"), float("nan"), 2.0], machine="node-b")
    document["results"]["bench.peakmem"] = [
        [42.0, float("nan"), float("nan")],
        [["p.failed", "p.ok", "p.skipped"]],
        "v1",
    ]
    other.write_text(json.dumps(document))
    (tmp_path / "task_0/results/benchmarks_meta.json").write_text(
        '{"benchmark": "metadata"}'
    )
    write_result(tmp_path / "task_0/results/.save-partial/alias/abc.json", [0, 0, 0])
    write_result(tmp_path / "task_0/env/irrelevant.json", [0, 0, 0])

    combined, machines = combiner.combine_results(tmp_path, metadata)

    assert combined["result_file_count"] == 2
    assert len(combined["frameworks"]) == 1
    framework = next(iter(combined["frameworks"].values()))
    assert framework["name"] == "numpy"
    assert framework["file"] == "frameworks/saps_numpy.py"
    assert framework["requirements"] == {"numpy": "2.3"}
    assert {machine["machine"] for machine in machines.values()} == {"node-a", "node-b"}
    assert all("numpy" not in machine for machine in machines.values())
    (benchmark,) = combined["benchmarks"]
    (generator,) = benchmark["generators"]
    assert generator["name"] == "generator.with.dots"
    assert [item["name"] for item in generator["datasets"]] == [
        "successful.dataset",
        "failed",
    ]
    good = generator["datasets"][0]["results"][0]
    assert good["result"] == 0.0
    assert good["status"] == "ok"
    assert good["stats"] == {"repeat": 2}
    failed, retried, peakmem = generator["datasets"][1]["results"]
    assert failed["status"] == "failed"
    assert failed["result"] is None
    diagnostic = combined["diagnostics"][failed["diagnostics"]]
    assert diagnostic["errcode"] == 1
    assert "failed setup" in diagnostic["stderr"]
    assert good["diagnostics"] == failed["diagnostics"]
    assert retried["result"] == 2.0
    assert retried["diagnostics"] is None  # Legacy ASV files omit error details.
    assert machines[retried["machine"]]["machine"] == "node-b"
    assert peakmem["metric"] == "peakmem"
    assert peakmem["result"] == 42.0
    json.dumps(combined, allow_nan=False)


def test_recorded_skip_is_distinct_from_unselected_nan(combiner, metadata, tmp_path):
    details = {
        "machines": {"a": {"machine": "node-a"}},
        "runs": [
            {
                "benchmark": "bench.time",
                "version": "v1",
                "parameters": [["p.skipped"]],
                "machine": "a",
                "errcode": 0,
                "stderr": "unsupported",
            }
        ],
    }
    write_result(
        tmp_path / "results/node-a/abc.json", [float("nan")] * 3, details=details
    )
    combined, _ = combiner.combine_results(tmp_path, metadata)
    (dataset,) = combined["benchmarks"][0]["generators"][0]["datasets"]
    assert dataset["name"] == "skipped"
    assert dataset["results"][0]["status"] == "skipped"
    assert dataset["results"][0]["result"] is None


def test_default_combines_highest_job_number(combiner, metadata, tmp_path, monkeypatch):
    monkeypatch.setattr(combiner, "REPO_ROOT", tmp_path)
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    for job in (9, 10):
        write_result(
            tmp_path / f"competition/run_{job}/task_0/results/node/abc.json",
            [float("nan"), float(job), float("nan")],
        )
    # Directory timestamps, nonnumeric names, and files do not select a run.
    os.utime(tmp_path / "competition/run_9", (2_000_000_000, 2_000_000_000))
    (tmp_path / "competition/run_local").mkdir()
    (tmp_path / "competition/run_99").write_text("not a directory")
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    monkeypatch.chdir(scripts)
    monkeypatch.setattr(sys, "argv", ["combine_competition_results.py"])

    assert combiner.main() == 0
    run = tmp_path / "competition/run_10"
    document = json.loads((run / "results.json").read_text())
    machines = json.loads((run / "machines.json").read_text())
    assert document["run"] == "run_10"
    assert document["machines_file"] == "machines.json"
    result = document["benchmarks"][0]["generators"][0]["datasets"][0]["results"][0]
    assert result["result"] == 10.0
    assert result["machine"] in machines
    assert not (tmp_path / "competition/run_9/results.json").exists()
    # Rebuilding does not ingest its own combined output.
    assert combiner.main() == 0
    assert json.loads((run / "results.json").read_text())["result_file_count"] == 1


def test_missing_metadata_match_is_reported(combiner, metadata, tmp_path):
    write_result(
        tmp_path / "results/node-a/abc.json", [float("nan"), 1.0, float("nan")]
    )
    metadata["benchmarks"][0]["generators"][0]["datasets"] = []
    with pytest.raises(ValueError, match="No metadata match"):
        combiner.combine_results(tmp_path, metadata)


def test_default_without_runs_reports_error(combiner, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(combiner, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["combine_competition_results.py"])
    with pytest.raises(SystemExit, match="2"):
        combiner.main()
    assert "No numbered run directories" in capsys.readouterr().err


def test_compact_failure_only_includes_attempted_parameters(
    combiner, metadata, tmp_path
):
    details = {
        "machines": {"a": {"machine": "node-a"}},
        "runs": [
            {
                "benchmark": "bench.time",
                "version": "v1",
                "parameters": [["p.failed"]],
                "machine": "a",
                "errcode": -256,
                "stderr": "benchmark timed out",
            }
        ],
    }
    write_result(tmp_path / "results/node-a/abc.json", None, details=details)
    combined, _ = combiner.combine_results(tmp_path, metadata)
    (dataset,) = combined["benchmarks"][0]["generators"][0]["datasets"]
    assert dataset["name"] == "failed"
    assert dataset["results"][0]["status"] == "failed"


def test_resume_attribution_uses_latest_attempt_per_parameter(
    combiner, metadata, tmp_path
):
    details = {
        "machines": {"a": {"machine": "node-a"}, "b": {"machine": "node-b"}},
        "runs": [
            {
                "benchmark": "bench.time",
                "version": "v1",
                "parameters": [["p.ok"], ["p.failed"]],
                "machine": "a",
                "errcode": 1,
                "stderr": "failed setup",
            },
            {
                "benchmark": "bench.time",
                "version": "v1",
                "parameters": [["p.failed"]],
                "machine": "b",
                "errcode": 0,
                "stderr": "",
            },
        ],
    }
    write_result(
        tmp_path / "results/alias/abc.json", [float("nan"), 1.0, 2.0], details=details
    )
    combined, machines = combiner.combine_results(tmp_path, metadata)
    first, second = combined["benchmarks"][0]["generators"][0]["datasets"]
    a, b = first["results"][0], second["results"][0]
    assert machines[a["machine"]]["machine"] == "node-a"
    assert machines[b["machine"]]["machine"] == "node-b"
    assert combined["diagnostics"][b["diagnostics"]]["errcode"] == 0
