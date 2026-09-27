import numpy as np

from binsparse.conversions import to_numpy

from frameworks.saps_numpy import NumpyFramework
from saps.benchmarks.spmv import MatrixVectorBenchmark


def test_run_passes_inputs_as_explicit_arguments():
    benchmark = MatrixVectorBenchmark()
    param = next(p for p in benchmark.params if p.dataset.name == "small")
    benchmark.setup(param, xp=NumpyFramework(), use_cache=False)
    benchmark.run(param)
    assert len(benchmark._output) == 1
    A, b = (to_numpy(array) for array in benchmark._input)
    np.testing.assert_allclose(to_numpy(benchmark._output[0]), A @ b)
    benchmark.teardown(param)
