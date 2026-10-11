from unittest.mock import Mock

import pytest

import numpy as np

from binsparse.conversions import from_numpy

from frameworks.saps_numpy import NumpyFramework
from saps.benchmarks.bfs import BFSBenchmark


@pytest.mark.parametrize("dataset_is_test", [False, True])
def test_teardown_always_checks_any_dataset(monkeypatch, dataset_is_test):
    benchmark = BFSBenchmark()
    param = next(p for p in benchmark.params if p.dataset.name == "basic")
    param.dataset._suites = ["test"] if dataset_is_test else ["standard"]
    benchmark.setup(param, use_cache=False, xp=NumpyFramework())
    benchmark.run(param)
    check = Mock(wraps=benchmark.check)
    monkeypatch.setattr(benchmark, "check", check)
    benchmark.teardown(param)
    check.assert_called_once_with(param)
    for attribute in ("_output", "_input", "_meta", "_xp", "_compiled_benchmark"):
        assert not hasattr(benchmark, attribute)


def test_teardown_rejects_incorrect_output():
    benchmark = BFSBenchmark()
    param = next(p for p in benchmark.params if p.dataset.name == "basic")
    param.dataset._suites = ["standard"]
    benchmark.setup(param, use_cache=False, xp=NumpyFramework())
    benchmark.run(param)
    benchmark._output = [from_numpy(np.zeros(6, dtype=int))]
    with pytest.raises(AssertionError, match="BFS output mismatch"):
        benchmark.teardown(param)
