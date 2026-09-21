from unittest.mock import Mock

import pytest

import numpy as np

from binsparse.conversions import from_numpy

from frameworks.saps_numpy import NumpyFramework
from saps.benchmarks.BFS import BreadthFirstSearchBenchmark


@pytest.mark.parametrize("mode", [None, "0", "1"])
@pytest.mark.parametrize("dataset_is_test", [False, True])
@pytest.mark.parametrize("generator_is_test", [False, True])
def test_teardown_checks_require_mode_and_dataset_suite(
    monkeypatch, mode, dataset_is_test, generator_is_test
):
    if mode is None:
        monkeypatch.delenv("SAPS_CHECK_SUITE", raising=False)
    else:
        monkeypatch.setenv("SAPS_CHECK_SUITE", mode)
    benchmark = BreadthFirstSearchBenchmark()
    param = next(p for p in benchmark.params if p.dataset.name == "test_bfs_basic")
    param.dataset._suites = ["test"] if dataset_is_test else ["standard"]
    monkeypatch.setattr(
        type(param.generator),
        "suites",
        property(lambda _: ["test"] if generator_is_test else []),
    )
    benchmark.setup(param, use_cache=False, xp=NumpyFramework())
    benchmark.run(param)
    check = Mock(wraps=benchmark.check)
    monkeypatch.setattr(benchmark, "check", check)
    benchmark.teardown(param)
    if mode == "1" and dataset_is_test:
        check.assert_called_once_with(param)
    else:
        check.assert_not_called()
    for attribute in ("_output", "_input", "_meta", "_xp", "_compiled_benchmark"):
        assert not hasattr(benchmark, attribute)


def test_test_mode_still_rejects_incorrect_output(monkeypatch):
    monkeypatch.setenv("SAPS_CHECK_SUITE", "1")
    benchmark = BreadthFirstSearchBenchmark()
    param = next(p for p in benchmark.params if p.dataset.name == "test_bfs_basic")
    benchmark.setup(param, use_cache=False, xp=NumpyFramework())
    benchmark.run(param)
    benchmark._output = [from_numpy(np.zeros(6, dtype=int))]
    with pytest.raises(AssertionError, match="BFS output mismatch"):
        benchmark.teardown(param)
