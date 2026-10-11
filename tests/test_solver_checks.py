from types import SimpleNamespace

import pytest

import numpy as np
import scipy.sparse as sps

from binsparse.conversions import from_numpy, from_scipy

from saps.benchmarks.cg import CGBenchmark
from saps.benchmarks.gmres import GMRESBenchmark
from saps.benchmarks.jacobi import JacobiBenchmark
from saps.benchmarks.lsqr import LSQRBenchmark
from saps.benchmarks.pcg import BlockJacobiPCGBenchmark, JacobiPCGBenchmark

_SOLVERS = [
    CGBenchmark,
    JacobiBenchmark,
    GMRESBenchmark,
    BlockJacobiPCGBenchmark,
    JacobiPCGBenchmark,
    LSQRBenchmark,
]
_CHECKS = [pytest.param(cls, None, id=cls.__name__) for cls in _SOLVERS] + [
    pytest.param(GMRESBenchmark, {"residual_tol": 1e-6}, id="gmres_residual"),
    pytest.param(
        BlockJacobiPCGBenchmark, {"check_residual": True}, id="block_pcg_residual"
    ),
    pytest.param(
        JacobiPCGBenchmark, {"check_residual": True}, id="jacobi_pcg_residual"
    ),
    pytest.param(LSQRBenchmark, {"convergence": "residual"}, id="lsqr_residual"),
    pytest.param(LSQRBenchmark, {"convergence": "gradient"}, id="lsqr_gradient"),
]


def _check(benchmark_cls, A, b, x, *, sparse=False, meta=None, ref_meta=None):
    benchmark = benchmark_cls()
    benchmark._input = [
        from_scipy(sps.coo_array(A)) if sparse else from_numpy(A),
        from_numpy(b),
    ]
    if benchmark_cls is not LSQRBenchmark:
        benchmark._input.append(from_numpy(np.zeros_like(x)))
    if benchmark_cls in (BlockJacobiPCGBenchmark, JacobiPCGBenchmark):
        benchmark._input.append(from_numpy(np.eye(A.shape[0], dtype=A.dtype)))
    benchmark._output = [from_numpy(x)]
    benchmark._meta = meta or {}
    benchmark._ref_meta = ref_meta
    benchmark.check(SimpleNamespace(dataset=SimpleNamespace(name="roundoff")))


@pytest.mark.parametrize("benchmark_cls,ref_meta", _CHECKS)
@pytest.mark.parametrize("dtype,power", [(np.float32, 24), (np.float64, 53)])
@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("scale", [2.0**-10, 2.0**10])
def test_solver_checks_allow_cancellation_but_reject_wrong_solutions(
    benchmark_cls, ref_meta, dtype, power, sparse, scale
):
    # The first dot product is exactly one, but (large + 1) - large rounds to zero.
    A = scale * np.array(
        [[2**power, 1, -(2**power)], [0, 1, 0], [0, 0, 1]], dtype=dtype
    )
    b = scale * np.ones(3, dtype=dtype)
    x = np.ones(3, dtype=dtype)
    _check(benchmark_cls, A, b, x, sparse=sparse, ref_meta=ref_meta)
    x[1] = 100
    with pytest.raises(AssertionError):
        _check(benchmark_cls, A, b, x, sparse=sparse, ref_meta=ref_meta)


@pytest.mark.parametrize("benchmark_cls,ref_meta", _CHECKS)
def test_solver_checks_reject_errors_above_convergence_tolerance(
    benchmark_cls, ref_meta
):
    A = np.eye(3)
    b = np.ones(3)
    meta = {"rel_tol": 1e-4}
    _check(benchmark_cls, A, b, b - 1e-7, meta=meta, ref_meta=ref_meta)
    # A factor-of-ten allowance would accept this unconverged solution.
    with pytest.raises(AssertionError):
        _check(benchmark_cls, A, b, b - 2e-4, meta=meta, ref_meta=ref_meta)


@pytest.mark.parametrize(
    "benchmark_cls",
    [CGBenchmark, JacobiBenchmark, BlockJacobiPCGBenchmark, JacobiPCGBenchmark],
)
def test_solver_checks_respect_absolute_tolerance(benchmark_cls):
    A = np.eye(3)
    b = np.zeros(3)
    meta = {"rel_tol": 0.0, "abs_tol": 1.0}
    _check(benchmark_cls, A, b, np.full(3, 0.25), meta=meta)
    with pytest.raises(AssertionError):
        _check(benchmark_cls, A, b, np.full(3, 2.0), meta=meta)


@pytest.mark.parametrize("benchmark_cls", [GMRESBenchmark, LSQRBenchmark])
def test_solver_checks_do_not_add_an_absolute_tolerance(benchmark_cls):
    A = np.eye(3)
    b = np.zeros(3)
    _check(benchmark_cls, A, b, b)
    with pytest.raises(AssertionError):
        _check(benchmark_cls, A, b, np.full(3, 1e-30))


@pytest.mark.parametrize("benchmark_cls,ref_meta", _CHECKS)
@pytest.mark.parametrize("bad_value", [np.nan, np.inf])
def test_solver_checks_reject_nonfinite_solutions(benchmark_cls, ref_meta, bad_value):
    with np.errstate(invalid="ignore"), pytest.raises(AssertionError):
        _check(
            benchmark_cls,
            np.eye(3),
            np.ones(3),
            np.full(3, bad_value),
            ref_meta=ref_meta,
        )


@pytest.mark.parametrize("ref_meta", [None, {"convergence": "gradient"}])
def test_lsqr_checks_stationarity_for_inconsistent_system(ref_meta):
    A = np.ones((2, 1))
    b = np.array([1.0, 3.0])
    _check(LSQRBenchmark, A, b, np.array([2.0]), ref_meta=ref_meta)
    with pytest.raises(AssertionError):
        _check(LSQRBenchmark, A, b, np.array([2.001]), ref_meta=ref_meta)
