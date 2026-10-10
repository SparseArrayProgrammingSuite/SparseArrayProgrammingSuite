import importlib
from fractions import Fraction
from types import SimpleNamespace

import pytest

import numpy as np
import scipy.sparse as sps

from binsparse.conversions import from_numpy, from_scipy

from saps.benchmark import DataInstance
from saps.benchmarks.elementwise_multiplication import (
    ElementwiseMultiplicationBenchmark,
)
from saps.benchmarks.four_clique_counting import FourCliqueCountingBenchmark
from saps.benchmarks.matrix_multiplication import MatrixMultiplicationBenchmark
from saps.benchmarks.matrix_vector_multiplication import (
    MatrixVectorMultiplicationBenchmark,
)
from saps.benchmarks.sddmm import SDDMMBenchmark
from saps.benchmarks.triangle_counting import TriangleCountingBenchmark
from saps.benchmarks.weighted_model_counting import WeightedModelCountingBenchmark
from saps.storage import LocalStorageBackend
from saps.util.error_bounds import (
    operation_error_bound,
    summation_error_bound,
)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_bounds_cover_error_against_exact_arithmetic(dtype):
    a = np.array([0.1, 0.3, -0.2, 0.7], dtype=dtype)
    b = np.array([0.2, -0.5, 0.1, 0.4], dtype=dtype)
    exact_a = [Fraction(float(x)) for x in a]
    exact_products = [x * Fraction(float(y)) for x, y in zip(exact_a, b, strict=True)]
    sum_error = abs(Fraction(float(a.sum())) - sum(exact_a))
    dot_error = abs(Fraction(float(a @ b)) - sum(exact_products))
    product_error = abs(Fraction(float(a[0] * b[0])) - exact_products[0])
    assert sum_error <= summation_error_bound(np, len(a), sum(map(abs, exact_a)), dtype)
    assert dot_error <= summation_error_bound(
        np, len(a) + 1, sum(map(abs, exact_products)), dtype
    )
    assert product_error <= operation_error_bound(np, exact_products[0], dtype)


def test_zero_singleton_and_integer_roundoff():
    assert summation_error_bound(np, 0, 0, np.float64) == 0
    assert summation_error_bound(np, 1, 3, np.float64) == 0
    assert operation_error_bound(np, -3.0, np.float64) > 0
    assert summation_error_bound(np, 100, 0, np.float32) == 0
    assert operation_error_bound(np, 50, np.int64) == 0


@pytest.mark.parametrize(
    "dtype_name",
    ["bool", "int64", "uint8", "float32", "float64", "complex64", "complex128"],
)
def test_error_bounds_accept_pytorch_dtypes(dtype_name):
    import array_api_compat.torch as xp

    expected = operation_error_bound(np, 2, getattr(np, dtype_name))
    actual = operation_error_bound(xp, 2, getattr(xp, dtype_name))
    assert actual == pytest.approx(expected, rel=1e-6, abs=0)


def test_invalid_bounds_do_not_silently_accept_any_result():
    with pytest.raises(ValueError, match="nonnegative"):
        summation_error_bound(np, -1, 1, np.float64)
    with pytest.raises(ValueError, match="Too many"):
        summation_error_bound(np, 2**23 + 1, 1, np.float32)
    with pytest.raises(ValueError, match="Too many"):
        summation_error_bound(np, 2**24 + 1, 1, np.float32)


_PRODUCT_BENCHMARKS = [
    MatrixMultiplicationBenchmark,
    MatrixVectorMultiplicationBenchmark,
    SDDMMBenchmark,
]


def _product_case(monkeypatch, benchmark_cls, a, b, *, sparse=False):
    benchmark = benchmark_cls()
    module = importlib.import_module(benchmark_cls.__module__)
    A = a.reshape(1, -1)
    if sparse and benchmark_cls is MatrixVectorMultiplicationBenchmark:
        A = np.repeat(A, 2, axis=0)
    B = b if benchmark_cls is MatrixVectorMultiplicationBenchmark else b.reshape(-1, 1)
    if benchmark_cls is SDDMMBenchmark:
        generator = benchmark.generators[0]
        matrices = iter([sps.coo_array([[2.0]])])
        arrays = iter([A, B])
    elif sparse:
        generator = benchmark.generators[1]
        matrices = iter([sps.coo_array(A), sps.coo_array(B)])
        arrays = iter([B])
    else:
        generator = benchmark.generators[0]
        arrays = iter([A, B])
        matrices = iter([])
    with monkeypatch.context() as patch:
        patch.setattr(
            module,
            "fetch_suitesparse_matrix",
            lambda _: DataInstance([from_scipy(next(matrices))], meta={}),
        )
        patch.setattr(
            np.random,
            "Generator",
            lambda _: SimpleNamespace(random=lambda shape: next(arrays)),
        )
        problem = generator.generate(generator.datasets[0])
    benchmark._ref_meta = problem.ref_meta
    shape = (A.shape[0],) if B.ndim == 1 else (A.shape[0], 1)
    return benchmark, shape


@pytest.mark.parametrize("benchmark_cls", _PRODUCT_BENCHMARKS)
@pytest.mark.parametrize("dtype,power", [(np.float32, 24), (np.float64, 53)])
@pytest.mark.parametrize("sparse", [False, True])
def test_dot_checks_allow_cancellation_and_different_reduction_orders(
    monkeypatch, benchmark_cls, dtype, power, sparse
):
    # (large + 1) - large rounds to zero; (large - large) + 1 is one.
    a = np.array([2**power, 1, -(2**power)], dtype=dtype)
    benchmark, shape = _product_case(
        monkeypatch, benchmark_cls, a, np.ones(3, dtype=dtype), sparse=sparse
    )
    benchmark._ref_outputs = [from_numpy(np.zeros(shape, dtype=dtype))]
    value = 2 if benchmark_cls is SDDMMBenchmark else 1
    benchmark._output = [from_numpy(np.full(shape, value, dtype=dtype))]
    benchmark.check(None)
    benchmark._output = [from_numpy(np.full(shape, 100, dtype=dtype))]
    with pytest.raises(AssertionError):
        benchmark.check(None)


@pytest.mark.parametrize("benchmark_cls", _PRODUCT_BENCHMARKS)
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_dot_checks_reject_incorrect_results(monkeypatch, benchmark_cls, dtype):
    benchmark, shape = _product_case(
        monkeypatch, benchmark_cls, np.ones(2, dtype=dtype), np.ones(2, dtype=dtype)
    )
    value = 4 if benchmark_cls is SDDMMBenchmark else 2
    benchmark._ref_outputs = [from_numpy(np.full(shape, value, dtype=dtype))]
    benchmark._output = [from_numpy(np.full(shape, value, dtype=dtype))]
    benchmark.check(None)
    # Default allclose accepts this float64 error, but roundoff does not.
    error = 1e-6 if np.dtype(dtype).itemsize >= 8 and dtype != np.complex64 else 1e-3
    benchmark._output = [from_numpy(np.full(shape, value + error, dtype=dtype))]
    with pytest.raises(AssertionError):
        benchmark.check(None)


@pytest.mark.parametrize("benchmark_cls", _PRODUCT_BENCHMARKS)
def test_empty_dot_products_have_zero_error_bound(monkeypatch, benchmark_cls):
    benchmark, shape = _product_case(
        monkeypatch, benchmark_cls, np.array([]), np.array([])
    )
    benchmark._ref_outputs = [from_numpy(np.zeros(shape))]
    benchmark._output = [from_numpy(np.zeros(shape))]
    benchmark.check(None)
    benchmark._output = [from_numpy(np.full(shape, 1e-30))]
    with pytest.raises(AssertionError):
        benchmark.check(None)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64])
def test_elementwise_check_uses_dtype_and_rejects_bad_values(monkeypatch, dtype):
    benchmark = ElementwiseMultiplicationBenchmark()
    values = np.array([[-2, 0], [0, 3]], dtype=dtype)
    module = importlib.import_module(benchmark.__module__)
    monkeypatch.setattr(
        module,
        "fetch_suitesparse_matrix",
        lambda _: DataInstance([from_scipy(sps.coo_array(values))], meta={}),
    )
    monkeypatch.setattr(
        module,
        "_matrix_with_overlap",
        lambda *args: sps.coo_array(np.ones_like(values)),
    )
    generator = benchmark.generators[1]
    problem = generator.generate(generator.datasets[0])
    benchmark._ref_meta = problem.ref_meta
    benchmark._ref_outputs = [from_numpy(values)]
    benchmark._output = [from_scipy(sps.coo_array(values))]
    benchmark.check(None)
    if dtype != np.int64:
        rounded = np.nextafter(values, np.full_like(values, np.inf))
        benchmark._output = [from_numpy(rounded)]
        benchmark.check(None)
    for bad_value in (1e-3, np.nan, np.inf):
        bad = values.astype(float)
        bad[0, 1] = bad_value
        benchmark._output = [from_numpy(bad)]
        with pytest.raises(AssertionError):
            benchmark.check(None)


def test_sampled_bound_does_not_materialize_dense_product(monkeypatch):
    benchmark = SDDMMBenchmark()
    mask = sps.coo_array(([2.0, -3.0], ([0, 9999], [9999, 0])), shape=(10000, 10000))
    module = importlib.import_module(benchmark.__module__)
    arrays = iter([np.ones((10000, 2)), np.ones((2, 10000))])
    monkeypatch.setattr(
        module,
        "fetch_suitesparse_matrix",
        lambda _: DataInstance([from_scipy(mask)], meta={}),
    )
    monkeypatch.setattr(
        np.random,
        "Generator",
        lambda _: SimpleNamespace(random=lambda shape: next(arrays)),
    )

    def no_dense(*args, **kwargs):
        pytest.fail("Sparse check attempted to densify")

    monkeypatch.setattr(sps.coo_array, "toarray", no_dense)
    generator = benchmark.generators[0]
    problem = generator.generate(generator.datasets[0])
    benchmark._ref_meta = problem.ref_meta
    benchmark._ref_outputs = problem.ref_outputs
    benchmark._output = [from_scipy(mask * 2)]
    monkeypatch.setattr(
        module, "sampled_product", lambda *args: pytest.fail("Recomputed bound")
    )
    benchmark.check(None)
    benchmark.check(None)


@pytest.mark.parametrize("generator_index", [0, 1])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_sddmm_bound_accounts_for_scaling_and_cancellation(
    monkeypatch, generator_index, dtype
):
    benchmark = SDDMMBenchmark()
    module = importlib.import_module(benchmark.__module__)
    generator = benchmark.generators[generator_index]

    def generate_bound(mask_value, terms):
        mask = sps.coo_array([[mask_value]])
        arrays = iter(
            [np.array([terms], dtype=dtype), np.ones((2, 1), dtype=dtype)]
        )
        rng = SimpleNamespace(random=lambda shape: next(arrays))
        with monkeypatch.context() as patch:
            patch.setattr(np.random, "Generator", lambda _: rng)
            patch.setattr(np.random, "default_rng", lambda _: rng)
            patch.setattr(sps, "random_array", lambda *args, **kwargs: mask)
            patch.setattr(
                module,
                "fetch_suitesparse_matrix",
                lambda _: DataInstance([from_scipy(mask)], meta={}),
            )
            return generator.generate(generator.datasets[0]).ref_meta["error_bound"]

    cancelled = generate_bound(1.0, [1, -1])
    positive = generate_bound(1.0, [1, 1])
    # The absolute term sums match, but cancellation reduces multiplication error.
    assert 0 < cancelled < positive
    for mask_value in (0.0, -(2.0**-20), -(2.0**20)):
        assert generate_bound(mask_value, [1, -1]) == pytest.approx(
            np.abs(mask_value) * cancelled, rel=1e-14, abs=0
        )
        assert generate_bound(mask_value, [1, 1]) == pytest.approx(
            np.abs(mask_value) * positive, rel=1e-14, abs=0
        )


@pytest.mark.parametrize(
    "benchmark_cls", [TriangleCountingBenchmark, FourCliqueCountingBenchmark]
)
def test_count_checks_reject_one_missing_count(benchmark_cls):
    benchmark = benchmark_cls()
    benchmark._ref_outputs = [from_numpy(np.array(1_000_000))]
    benchmark._output = [from_numpy(np.array(1_000_000.0))]
    benchmark.check(None)
    benchmark._output = [from_numpy(np.array(999_999.0))]
    with pytest.raises(AssertionError):
        benchmark.check(None)


def test_weighted_count_check_rejects_fixed_tolerance_error():
    benchmark = WeightedModelCountingBenchmark()
    param = benchmark.params[0]
    problem = param.generator.generate(param.dataset)
    benchmark._ref_meta = problem.ref_meta
    benchmark._ref_outputs = [from_numpy(np.array(0.8))]
    benchmark._output = [from_numpy(np.array(np.nextafter(0.8, 1.0)))]
    benchmark.check(param)
    benchmark._output = [from_numpy(np.array(0.8 + 1e-9))]
    with pytest.raises(AssertionError):
        benchmark.check(param)


def test_generated_bound_round_trips_through_storage(tmp_path):
    benchmark = MatrixMultiplicationBenchmark()
    generator = benchmark.generators[0]
    problem = generator.generate(generator.datasets[0])
    backend = LocalStorageBackend(
        tmp_path / "remote", tmp_path / "manifest.json", tmp_path / "cache"
    )
    path = tmp_path / "dataset.bsp.h5"
    backend.serialize_data_to_file(problem, path)
    restored = backend.deserialize_data_from_file(path)
    assert restored.ref_meta == problem.ref_meta
    assert restored.ref_meta["error_bound"] > 0
    benchmark._ref_meta = restored.ref_meta
    benchmark._ref_outputs = restored.ref_outputs
    benchmark._output = restored.ref_outputs
    benchmark.check(None)
