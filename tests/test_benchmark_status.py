import json
import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from frameworks.saps_numpy import NumpyFramework
from saps.benchmark_status import STATUS_PREFIX, MeasurementTimeout, parameter_statuses
from saps.benchmarks.bfs import BFSBenchmark


def marker(phase, **details):
    return STATUS_PREFIX + json.dumps({"phase": phase, **details})


@pytest.mark.parametrize(
    ("phases", "expected"),
    [
        ([], "setup_timeout"),
        (["setup"], "setup_timeout"),
        (["setup", "measurement"], "measurement_timeout"),
        (["setup", "measurement", "teardown"], "teardown_timeout"),
        (["setup", "measurement", "teardown", "finished"], "teardown_timeout"),
        (["setup", "measurement", "finished", "setup"], "setup_timeout"),
    ],
)
def test_process_timeout_uses_last_phase(phases, expected):
    output = "\n".join(
        [
            *map(marker, phases),
            STATUS_PREFIX + '{"phase":',
            "asv: benchmark timed out (timeout 10s)",
        ]
    )
    (record,) = parameter_statuses([["data"]], [None], output)
    assert record["status"] == expected
    assert record["timeout_source"] == "process"


def test_mixed_parameter_statuses_keep_measurement_timeout_through_teardown():
    output = "\n".join(
        [
            "For parameters: a",
            marker("setup"),
            "asv: benchmark timed out (timeout 10s)",
            "For parameters: b",
            marker(
                "measurement",
                status="measurement_timeout",
                elapsed_seconds=2,
                timeout_seconds=1,
            ),
            marker("teardown"),
            "asv: benchmark timed out (timeout 10s)",
            "For parameters: c",
            marker("teardown"),
            "AssertionError: wrong answer",
        ]
    )
    records = parameter_statuses([["a", "b", "c", "d"]], [None] * 4, output, [0, 1, 2])
    assert [r["status"] for r in records] == [
        "setup_timeout",
        "measurement_timeout",
        "failed",
    ]
    assert records[1]["timeout_source"] == "measurement"
    assert records[1]["elapsed_seconds"] == 2


@pytest.mark.parametrize("output", [None, "", STATUS_PREFIX + "[]"])
def test_missing_markers_do_not_turn_errors_into_timeouts(output):
    (record,) = parameter_statuses([], [None], output)
    assert record["status"] == "failed"


def test_skips_are_not_failures():
    records = parameter_statuses(
        [["a", "b", "c"]], [float("nan"), float("inf"), 1.0], ""
    )
    assert [r["status"] for r in records] == ["skipped", "failed", "ok"]
    assert "reason" not in records[0]
    assert records[1]["reason"] == "nonfinite"


@pytest.mark.parametrize("metric", ["time", "peakmem"])
def test_measurement_timeout_is_checked_in_teardown(monkeypatch, capsys, metric):
    monkeypatch.setenv("SAPS_MEASUREMENT_TIMEOUT", "1")
    clock = iter([0.0, 2.0])
    monkeypatch.setattr("saps.benchmark.time.perf_counter", lambda: next(clock))
    benchmark = BFSBenchmark()
    param = next(p for p in benchmark.params if p.dataset.name == "basic")
    benchmark.setup(param, use_cache=False, xp=NumpyFramework())
    check = Mock()
    monkeypatch.setattr(benchmark, "check", check)
    for _ in range(2):
        getattr(benchmark, f"{metric}_bfs")(param)
    assert hasattr(benchmark, "_output")
    with pytest.raises(MeasurementTimeout, match="2s of measurement exceeded 1s"):
        benchmark.teardown(param)
    check.assert_not_called()
    assert not hasattr(benchmark, "_output")  # Cleaned up before raising.
    (record,) = parameter_statuses([[repr(param)]], [None], capsys.readouterr().err)
    assert record["status"] == "measurement_timeout"
    assert record["elapsed_seconds"] == 2
    assert record["timeout_seconds"] == 1


def test_measurement_budget_accumulates_between_samples(monkeypatch):
    monkeypatch.setenv("SAPS_MEASUREMENT_TIMEOUT", "1")
    benchmark = BFSBenchmark()
    param = next(p for p in benchmark.params if p.dataset.name == "basic")
    clock = iter([0.0, 0.6, 100.0, 100.5])
    monkeypatch.setattr("saps.benchmark.time.perf_counter", lambda: next(clock))
    benchmark.setup(param, use_cache=False, xp=NumpyFramework())
    benchmark.time_bfs(param)
    benchmark.teardown(param)
    benchmark.setup(param, use_cache=False, xp=NumpyFramework())
    benchmark.time_bfs(param)
    # The 99.4 seconds between measurement phases do not count.
    with pytest.raises(MeasurementTimeout, match="1.1s of measurement exceeded 1s"):
        benchmark.teardown(param)
    benchmark.teardown(param)  # ASV can call teardown again after an exception.


def test_measurement_budget_resets_for_each_dataset(monkeypatch):
    monkeypatch.setenv("SAPS_MEASUREMENT_TIMEOUT", "1")
    benchmark = BFSBenchmark()
    first, second = [p for p in benchmark.params if "test" in p.dataset.suites][:2]
    clock = iter([0.0, 0.6, 100.0, 100.6])
    monkeypatch.setattr("saps.benchmark.time.perf_counter", lambda: next(clock))
    for param in (first, second):
        benchmark.setup(param, use_cache=False, xp=NumpyFramework())
        benchmark.time_bfs(param)
        benchmark.teardown(param)  # 0.6s per dataset stays within the budget.


def test_measurement_limit_allows_exactly_the_budget(monkeypatch):
    monkeypatch.setenv("SAPS_MEASUREMENT_TIMEOUT", "1")
    clock = iter([0.0, 1.0])
    monkeypatch.setattr("saps.benchmark.time.perf_counter", lambda: next(clock))
    benchmark = BFSBenchmark()
    param = next(p for p in benchmark.params if p.dataset.name == "basic")
    benchmark.setup(param, use_cache=False, xp=NumpyFramework())
    benchmark.time_bfs(param)
    benchmark.teardown(param)


@pytest.mark.parametrize(
    ("scenario", "status", "source"),
    [
        ("ok", "ok", None),
        ("slow", "measurement_timeout", "measurement"),
        ("setup", "setup_timeout", "process"),
        ("hang", "measurement_timeout", "process"),
        ("teardown", "teardown_timeout", "process"),
        ("failed", "failed", None),
    ],
)
def test_asv_worker_outcomes(tmp_path, scenario, status, source):
    from asv.runner import run_benchmark
    from asv.util import TIMEOUT_RETCODE

    suite = tmp_path / "benchmarks"
    suite.mkdir()
    (suite / "__init__.py").write_text("")
    run_duration = {"hang": 10, "slow": 0.07}.get(scenario, 0.03)
    (suite / "timing.py").write_text(
        "import time\n"
        "from frameworks.saps_numpy import NumpyFramework\n"
        "from saps.benchmarks.bfs import BFSBenchmark\n"
        "class TimingBFS(BFSBenchmark):\n"
        "    params = [p for p in BFSBenchmark().params if p.dataset.name == 'basic']\n"
        "    def setup(self, param):\n"
        f"        time.sleep({10 if scenario == 'setup' else 0.15})\n"
        "        super().setup(param, use_cache=False, xp=NumpyFramework())\n"
        "    def run(self, param):\n"
        f"        time.sleep({run_duration})\n"
        "        super().run(param)\n"
        "        print('completed benchmark call', flush=True)\n"
        "    def check(self, param):\n"
        f"        time.sleep({10 if scenario == 'teardown' else 0.15})\n"
        f"        assert {scenario != 'failed'}, 'incorrect output'\n"
        "        super().check(param)\n"
    )

    def run(*, name, params_str, profile_path, result_file_name, timeout, **kwargs):
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import sys; from asv_runner.run import _run; _run(sys.argv[1:])",
                    str(suite),
                    name,
                    params_str,
                    profile_path,
                    result_file_name,
                ],
                env={
                    **os.environ,
                    "SAPS_MEASUREMENT_TIMEOUT": "0.1",
                    "SAPS_LOG_PATH": str(tmp_path / "diagnostics.log"),
                },
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return completed.stdout + completed.stderr, completed.returncode
        except subprocess.TimeoutExpired as exc:
            return (exc.stderr or b"").decode(), TIMEOUT_RETCODE

    benchmark = {
        "name": "timing.TimingBFS.time_bfs",
        "params": [["basic"]],
        "timeout": 3,
    }
    result = run_benchmark(
        benchmark,
        SimpleNamespace(run=run),
        profile=False,
        extra_params={"number": 1, "repeat": 1, "rounds": 1, "warmup_time": 0.01},
    )
    (record,) = parameter_statuses(benchmark["params"], result.result, result.stderr)
    assert record["status"] == status, result.stderr
    assert record.get("timeout_source") == source
    if scenario in ("ok", "slow"):
        assert result.stderr.count("completed benchmark call") == 2
    if scenario == "ok":
        assert result.errcode == 0
        assert len(result.samples[0]) == 1  # The warmup is not a timing sample.
    elif scenario == "slow":
        assert result.errcode == 1  # An exception, not a process timeout/signal.
        assert record["elapsed_seconds"] > 0.1
        assert '"phase": "finished"' in result.stderr
