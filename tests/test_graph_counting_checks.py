import pytest

import numpy as np
from scipy import sparse

from binsparse.conversions import from_numpy, from_scipy, to_numpy

from frameworks.saps_numpy import NumpyFramework
from frameworks.saps_pytorch import PytorchFramework
from frameworks.saps_scipy import SciPyFramework
from frameworks.saps_smart import SmartSparseFramework
from frameworks.saps_sparse import PyDataSparseFramework
from saps.benchmarks.four_clique_counting import FourCliqueCountingBenchmark
from saps.benchmarks.triangle_counting import TriangleCountingBenchmark


@pytest.fixture(params=[TriangleCountingBenchmark, FourCliqueCountingBenchmark])
def benchmark(request):
    return request.param()


@pytest.mark.parametrize("expected", [6_622_234_180_319, 2**53 + 1])
def test_count_checks_require_exact_integer_scalars(benchmark, expected):
    benchmark._ref_outputs = [from_numpy(np.array(expected, dtype=np.int64))]
    for dtype in (np.int64, np.uint64):
        benchmark._output = [from_numpy(np.array(expected, dtype=dtype))]
        benchmark.check(None)
        for delta in (-1, 1):
            benchmark._output = [from_numpy(np.array(expected + delta, dtype=dtype))]
            with pytest.raises(AssertionError, match="Count mismatch"):
                benchmark.check(None)


@pytest.mark.parametrize(
    "output,message",
    [
        (np.array(1.0), "integer dtype"),
        (np.array(True), "integer dtype"),
        (np.array([1], dtype=np.int64), "scalar"),
    ],
)
def test_count_checks_reject_non_integer_or_non_scalar_outputs(
    benchmark, output, message
):
    benchmark._ref_outputs = [from_numpy(np.array(1, dtype=np.int64))]
    benchmark._output = [from_numpy(output)]
    with pytest.raises(AssertionError, match=message):
        benchmark.check(None)


@pytest.mark.parametrize(
    "framework",
    [
        NumpyFramework,
        PytorchFramework,
        SciPyFramework,
        SmartSparseFramework,
        PyDataSparseFramework,
    ],
)
@pytest.mark.parametrize("weighted", [False, True])
def test_counting_contractions_preserve_int64(benchmark, framework, weighted):
    size = 3 if isinstance(benchmark, TriangleCountingBenchmark) else 4
    A = np.triu(np.ones((size, size), dtype=np.int64), k=1)
    # A weighted arithmetic probe exposes float conversion without constructing
    # a graph with quadrillions of cliques. Exactly one product uses this edge.
    expected = 2**53 + 1 if weighted else 1
    A[0, 1] = expected
    xp = framework()
    # PyTorch and SciPy cannot broadcast these sparse operands in 3D/4D.
    # Exercise their supported dense path, and the other backends' sparse path.
    tensor = (
        from_numpy(A)
        if isinstance(xp, (PytorchFramework, SciPyFramework))
        else from_scipy(sparse.coo_array(A))
    )
    output = xp.compute(benchmark.benchmark(xp, {}, xp.from_binsparse(tensor)))
    result = to_numpy(xp.to_binsparse(output))
    assert result.dtype == np.int64
    assert result.item() == expected
