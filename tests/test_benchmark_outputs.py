import pytest

import numpy as np

from binsparse.conversions import to_numpy

from frameworks.saps_numpy import NumpyFramework
from saps.benchmark import _as_outputs
from saps.benchmarks.spmv import MatrixVectorBenchmark


def test_single_output_is_wrapped():
    x = np.arange(3)
    (out,) = _as_outputs(x)
    assert out is x


def test_tuple_outputs_pass_through():
    x, y = np.arange(3), np.arange(2)
    assert _as_outputs((x, y)) == (x, y)


def test_empty_tuple_means_no_outputs():
    assert _as_outputs(()) == ()


def test_list_outputs_are_rejected():
    with pytest.raises(TypeError, match="not as a list"):
        _as_outputs([np.arange(3)])


def test_run_passes_inputs_as_explicit_arguments():
    benchmark = MatrixVectorBenchmark()
    param = next(p for p in benchmark.params if p.dataset.name == "small")
    benchmark.setup(param, xp=NumpyFramework(), use_cache=False)
    benchmark.run(param)
    assert len(benchmark._output) == 1
    A, b = (to_numpy(array) for array in benchmark._input)
    np.testing.assert_allclose(to_numpy(benchmark._output[0]), A @ b)
    benchmark.teardown(param)
