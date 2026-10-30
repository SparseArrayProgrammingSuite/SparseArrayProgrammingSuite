import importlib
from types import SimpleNamespace

import pytest

import numpy as np
import scipy.sparse as sps

import sparse as pydata_sparse
from binsparse.conversions import from_numpy, from_scipy, to_numpy, to_scipy

from saps.benchmarks.cg import CGBenchmark
from saps.benchmarks.gmres import GMRESBenchmark
from saps.benchmarks.jacobi import JacobiBenchmark
from saps.benchmarks.lsqr import LSQRBenchmark
from saps.benchmarks.pcg import BlockJacobiPCGBenchmark, JacobiPCGBenchmark
from saps.storage import LocalStorageBackend

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
    benchmark._ref_outputs = [from_scipy(sps.coo_array(np.abs(A)))]
    benchmark._ref_meta = {**(ref_meta or {}), "bnorm": float(np.linalg.norm(b))}
    if benchmark_cls is LSQRBenchmark:
        benchmark._ref_meta["gradient_scale"] = float(np.linalg.norm(A.T @ b))
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


@pytest.mark.parametrize(
    ("benchmark_cls", "generator_index"),
    [
        (benchmark_cls, generator_index)
        for benchmark_cls in _SOLVERS
        for generator_index in (0, 1)
        # This generator fetches its block partition from remote storage.
        if (benchmark_cls, generator_index) != (BlockJacobiPCGBenchmark, 1)
    ],
)
def test_solver_generators_cache_input_dependent_checks(
    monkeypatch, tmp_path, benchmark_cls, generator_index
):
    benchmark = benchmark_cls()
    module = importlib.import_module(benchmark_cls.__module__)
    A = sps.coo_array([[4.0, -1.0], [-1.0, 3.0]])
    monkeypatch.setattr(
        module,
        "fetch_suitesparse_linear_system",
        lambda *args, **kwargs: (from_scipy(A), np.array([1.0, 2.0]), True),
    )
    generator = benchmark.generators[generator_index]
    dataset = generator.datasets[0]
    problem = generator.generate(dataset)
    try:
        A = to_scipy(problem.inputs[0]).toarray()
    except TypeError:
        A = to_numpy(problem.inputs[0])
    b = to_numpy(problem.inputs[1])
    np.testing.assert_array_equal(to_scipy(problem.ref_outputs[0]).toarray(), np.abs(A))
    assert problem.ref_meta["bnorm"] == pytest.approx(np.linalg.norm(b))
    if benchmark_cls is LSQRBenchmark:
        assert problem.ref_meta["gradient_scale"] == pytest.approx(
            np.linalg.norm(A.T @ b)
        )

    backend = LocalStorageBackend(
        tmp_path / "remote", tmp_path / "manifest.json", tmp_path
    )
    path = tmp_path / "solver.bsp.h5"
    backend.serialize_data_to_file(problem, path)
    restored = backend.deserialize_data_from_file(path)
    assert restored.ref_meta == problem.ref_meta
    benchmark._input = restored.inputs
    benchmark._meta = restored.meta
    benchmark._ref_outputs = restored.ref_outputs
    benchmark._ref_meta = restored.ref_meta
    benchmark._output = [from_numpy(np.linalg.lstsq(A, b, rcond=None)[0])]
    b = to_numpy(restored.inputs[1])

    original_abs = np.abs
    original_norm = np.linalg.norm
    original_matmul = pydata_sparse.COO.__matmul__

    def cached_abs(x, *args, **kwargs):
        assert x.ndim != 2, "Recomputed absolute matrix"
        return original_abs(x, *args, **kwargs)

    def cached_norm(x, *args, **kwargs):
        assert not np.shares_memory(x, b), "Recomputed right-hand-side norm"
        return original_norm(x, *args, **kwargs)

    def cached_matmul(matrix, x):
        assert not np.shares_memory(x, b), "Recomputed A.T @ b"
        return original_matmul(matrix, x)

    monkeypatch.setattr(np, "abs", cached_abs)
    monkeypatch.setattr(np.linalg, "norm", cached_norm)
    monkeypatch.setattr(pydata_sparse.COO, "__matmul__", cached_matmul)
    benchmark.check(SimpleNamespace(dataset=dataset))
    benchmark.check(SimpleNamespace(dataset=dataset))
