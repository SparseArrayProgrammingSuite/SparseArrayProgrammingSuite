import math

import pytest

import numpy as np

from binsparse import CustomTensor, ElementLevel, SparseLevel
from binsparse.conversions import from_numpy
from numba import get_num_threads, set_num_threads

from saps.benchmark import DataInstance, Param
from saps.benchmarks import cp_als, frostt, hosvd
from saps.util import tensor_checks


def _coo(shape, coords, values):
    return CustomTensor(
        shape, len(values), level=SparseLevel(len(shape), ElementLevel(values), coords)
    )


def _sparse(array):
    coords = array.nonzero()
    return _coo(array.shape, coords, array[coords])


def _benchmark(tensor, factors, small, kind, tolerance, metadata=None):
    n = len(factors)
    if kind == "cp":
        benchmark = getattr(cp_als, f"CPALS{n}DBenchmark")()
        benchmark._input = [tensor]
        benchmark._meta = {"rank": len(small)}
        benchmark._output = [from_numpy(f) for f in factors] + [from_numpy(small)]
    else:
        benchmark = getattr(hosvd, f"HOSVD{n}DBenchmark")()
        benchmark._input = [tensor, from_numpy(np.array(small.shape))]
        benchmark._output = [from_numpy(small)] + [from_numpy(f) for f in factors]
    benchmark._ref_meta = {"rel_error_tol": tolerance, **(metadata or {})}
    return benchmark


def _inner(tensor, factors, small, kind):
    modes = tuple(zip(tensor.level.indices, factors, strict=True))
    values = tensor.level.level.values
    n = len(factors)
    if kind == "cp":
        modes += (modes[0],) * (5 - n)
        return cp_als._sparse_cp_inner(values, modes, n, small)
    modes = ((modes[0][0], np.ones((1, 1))),) * (5 - n) + modes
    core = small.reshape((1,) * (5 - n) + small.shape)
    return hosvd._sparse_tucker_inner(values, modes, n, core)


def _assert_threshold_matches_dense(tensor, array, factors, small, kind, model):
    squared_error = math.fsum(((array.astype(float) - model) ** 2).ravel())
    input_norm = math.fsum((array.astype(float) ** 2).ravel())
    tolerance = math.sqrt(squared_error / input_norm)
    benchmark = _benchmark(tensor, factors, small, kind, tolerance * (1 + 1e-6))
    benchmark.check(None)
    benchmark._ref_meta["rel_error_tol"] = tolerance * (1 - 1e-6)
    with pytest.raises(AssertionError, match="residual"):
        benchmark.check(None)


@pytest.mark.parametrize("n", [3, 4, 5])
@pytest.mark.parametrize("kind", ["cp", "tucker"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64, np.bool_])
@pytest.mark.parametrize("sparse_input", [False, True])
def test_check_threshold_matches_explicit_dense_oracle(n, kind, dtype, sparse_input):
    rng = np.random.default_rng(7)
    shape = tuple(range(2, n + 2))
    array = rng.integers(-2, 3, size=shape).astype(dtype)
    factors = [rng.standard_normal((size, 2)).astype(np.float32) for size in shape]
    dims = "ijklm"[:n]
    if kind == "cp":
        small = np.array([1.0, -2.0])
        expression = ",".join(["r"] + [d + "r" for d in dims])
    else:
        small = rng.standard_normal((2,) * n)
        ranks = "abcde"[:n]
        expression = ",".join(
            [ranks] + [d + r for d, r in zip(dims, ranks, strict=True)]
        )
    model = np.einsum(
        expression + "->" + dims, small, *[f.astype(float) for f in factors]
    )
    tensor = _sparse(array) if sparse_input else from_numpy(array)
    _assert_threshold_matches_dense(tensor, array, factors, small, kind, model)


@pytest.mark.parametrize("n", [3, 4, 5])
@pytest.mark.parametrize("kind", ["cp", "tucker"])
def test_parallel_kernel_handles_mixed_dtypes_layouts_and_unequal_ranks(n, kind):
    rng = np.random.default_rng(31)
    shape = tuple(range(3, n + 3))
    ranks = (3,) * n if kind == "cp" else (2, 1, 3, 2, 1)[:n]
    factors = []
    for mode, (size, rank) in enumerate(zip(shape, ranks, strict=True)):
        factor = rng.standard_normal((size, 2 * rank))[:, ::2]
        if mode % 2:
            factor = np.asfortranarray(factor, dtype=np.float32)
        factors.append(factor)
    array = rng.integers(-2, 3, size=shape)
    coords = tuple(
        index.astype(np.int32) if mode % 2 else index
        for mode, index in enumerate(array.nonzero())
    )
    tensor = _coo(shape, coords, array[coords])
    small = rng.standard_normal((3,) if kind == "cp" else ranks)[::-1]
    dims, labels = "ijklm"[:n], "abcde"[:n]
    expression = (
        ",".join(["r"] + [d + "r" for d in dims])
        if kind == "cp"
        else ",".join([labels] + [d + r for d, r in zip(dims, labels, strict=True)])
    )
    model = np.einsum(
        expression + "->" + dims, small, *[f.astype(float) for f in factors]
    )
    assert _inner(tensor, factors, small, kind) == pytest.approx(
        math.fsum((array * model).ravel()), rel=1e-12, abs=1e-10
    )
    _assert_threshold_matches_dense(tensor, array, factors, small, kind, model)


@pytest.mark.parametrize("kind", ["cp", "tucker"])
def test_parallel_reduction_matches_dense_oracle_across_thread_counts(kind):
    rng = np.random.default_rng(921)
    shape = (8,) * 5
    ranks = (3,) * 5 if kind == "cp" else (3, 2, 1, 2, 3)
    factors = [rng.standard_normal((s, r)) for s, r in zip(shape, ranks, strict=True)]
    array = rng.integers(-2, 3, size=shape).astype(float)
    tensor = _sparse(array)
    if kind == "cp":
        small = rng.standard_normal(3)
        model = np.einsum("r,ir,jr,kr,lr,mr->ijklm", small, *factors)
    else:
        small = rng.standard_normal(ranks)
        model = np.einsum("abcde,ia,jb,kc,ld,me->ijklm", small, *factors)
    expected_inner = math.fsum((array * model).ravel())
    saved_threads = get_num_threads()
    try:
        for threads in sorted({1, min(2, saved_threads), min(4, saved_threads)}):
            set_num_threads(threads)
            inner = _inner(tensor, factors, small, kind)
            assert inner == _inner(tensor, factors, small, kind)
            assert inner == pytest.approx(expected_inner, rel=1e-12, abs=1e-10)
            _assert_threshold_matches_dense(tensor, array, factors, small, kind, model)
    finally:
        set_num_threads(saved_threads)


@pytest.mark.parametrize("kind", ["cp", "tucker"])
def test_parallel_empty_input(kind):
    tensor = _sparse(np.zeros((2, 2, 2)))
    factors = [np.ones((2, 1)) for _ in range(3)]
    small = np.zeros(1) if kind == "cp" else np.zeros((1, 1, 1))
    assert _inner(tensor, factors, small, kind) == 0
    _benchmark(tensor, factors, small, kind, 0).check(None)


@pytest.mark.parametrize("kind", ["cp", "tucker"])
def test_huge_logical_shape_never_materializes_tensor(kind, monkeypatch):
    shape = (10000,) * 5
    assert math.prod(shape) > np.iinfo(np.int64).max
    coords = tuple(np.array([0]) for _ in shape)
    tensor = _coo(shape, coords, np.array([3], dtype=np.int64))
    factors = [np.zeros((size, 1)) for size in shape]
    for factor in factors:
        factor[0, 0] = 1
    small = np.array([2.0]) if kind == "cp" else np.full((1,) * 5, 2.0)
    original_to_numpy = tensor_checks.to_numpy

    def no_dense_input(value):
        assert value is not tensor, "Input tensor was converted to dense"
        return original_to_numpy(value)

    monkeypatch.setattr(tensor_checks, "to_numpy", no_dense_input)
    _benchmark(tensor, factors, small, kind, 1 / 3).check(None)
    # A model which fits stored entries but leaks into implicit zeros must fail.
    factors[0][1, 0] = 100
    with pytest.raises(AssertionError, match="residual"):
        _benchmark(tensor, factors, small, kind, 1).check(None)


def test_input_norms_sum_duplicates_without_integer_overflow():
    shape = (2, 2, 2)
    coords = tuple(np.array([1, 0, 1, 0, 1, 0, 1, 0, 1]) for _ in shape)
    values = np.full(9, 2**32, dtype=np.int64)
    metadata = tensor_checks.input_norms(_coo(shape, coords, values))
    assert metadata["input_squared_norm"] == float((4 * 2**32) ** 2 + (5 * 2**32) ** 2)


@pytest.mark.parametrize("kind", ["cp", "tucker"])
def test_raw_frostt_caches_norms_and_checks_skip_sorting(kind, monkeypatch):
    shape = (2, 2, 2)
    coords = tuple(np.array([1, 0, 1, 0]) for _ in shape)
    values = np.array([2.0, 1e8, -1.0, -1e8])
    monkeypatch.setattr(
        frostt,
        "load_frostt_tensor",
        lambda *args, **kwargs: (coords, values, {"shape": shape}),
    )
    generator = frostt.FROSTTTensorGenerator()
    raw = generator.generate(generator.datasets[0])
    assert raw.ref_meta["input_squared_norm"] == 1
    assert raw.ref_meta["input_absolute_squared_norm"] >= (2e8) ** 2

    def no_sort(*args, **kwargs):
        pytest.fail("Cached residual check sorted coordinates again")

    monkeypatch.setattr(np, "lexsort", no_sort)
    factors = [np.array([[0.0], [1.0]]) for _ in shape]
    small = np.ones(1) if kind == "cp" else np.ones((1, 1, 1))
    _benchmark(raw.inputs[0], factors, small, kind, 0, raw.ref_meta).check(None)


@pytest.mark.parametrize("kind", ["cp", "tucker"])
def test_check_promotes_float32_outputs_before_products(kind):
    # Two such float32 values overflow when multiplied in float32, but the
    # reconstruction and squared residual remain representable in float64.
    value = np.float32(1e20)
    factors = [np.full((1, 1), value, dtype=np.float32) for _ in range(3)]
    shape = (1,) if kind == "cp" else (1, 1, 1)
    small = np.full(shape, value, dtype=np.float32)
    tensor = _sparse(np.full((1, 1, 1), float(value) ** 4))
    _benchmark(tensor, factors, small, kind, 0).check(None)


@pytest.mark.parametrize("kind", ["cp", "tucker"])
def test_check_refuses_overflowed_arithmetic(kind):
    tensor = _sparse(np.ones((2, 2, 2)))
    factors = [np.full((2, 1), 1e200) for _ in range(3)]
    small = np.ones(1) if kind == "cp" else np.ones((1, 1, 1))
    with (
        np.errstate(over="ignore", invalid="ignore"),
        pytest.raises(ValueError, match=r"-inf \+ inf in fsum"),
    ):
        _benchmark(tensor, factors, small, kind, 1).check(None)


@pytest.mark.parametrize("kind", ["cp", "tucker"])
def test_roundoff_handles_exact_fit_and_rejects_perturbation(kind):
    shape = (4, 3, 2)
    factors = [np.linspace(0.1, 0.9, size)[:, None] for size in shape]
    tensor = _sparse(np.einsum("ir,jr,kr->ijk", *factors))
    small = np.ones(1) if kind == "cp" else np.ones((1, 1, 1))
    _benchmark(tensor, factors, small, kind, 0).check(None)
    factors[0] = factors[0] * 1.01
    with pytest.raises(AssertionError, match="residual"):
        _benchmark(tensor, factors, small, kind, 1e-5).check(None)


@pytest.mark.parametrize("kind", ["cp", "tucker"])
def test_check_accepts_cancellation_between_large_components(kind):
    factors = [np.ones((2, 2)) for _ in range(3)]
    factors[0][:, 1] += 1e-8
    if kind == "cp":
        small = np.array([1e8, -1e8])
        array = np.einsum("r,ir,jr,kr->ijk", small, *factors)
    else:
        small = np.zeros((2, 2, 2))
        small[0, 0, 0], small[1, 1, 1] = 1e8, -1e8
        array = np.einsum("abc,ia,jb,kc->ijk", small, *factors)
    _benchmark(_sparse(array), factors, small, kind, 0).check(None)


@pytest.mark.parametrize("kind", ["cp", "tucker"])
@pytest.mark.parametrize("sparse_input", [False, True])
def test_zero_input(kind, sparse_input):
    array = np.zeros((2, 2, 2))
    tensor = _sparse(array) if sparse_input else from_numpy(array)
    factors = [np.ones((2, 1)) for _ in range(3)]
    small = np.zeros(1) if kind == "cp" else np.zeros((1, 1, 1))
    _benchmark(tensor, factors, small, kind, 0).check(None)
    with pytest.raises(AssertionError, match="residual"):
        _benchmark(tensor, factors, np.ones_like(small), kind, 1).check(None)


@pytest.mark.parametrize("n", [3, 4, 5])
@pytest.mark.parametrize("module,prefix", [(cp_als, "CPALS"), (hosvd, "HOSVD")])
def test_frostt_generators_enable_sparse_quality_checks(n, module, prefix, monkeypatch):
    shape = (6,) * n
    # A rank-one component plus a disjoint outlier is deliberately not rank one.
    coords = tuple(np.array([0, 1]) for _ in shape)
    tensor = _coo(shape, coords, np.array([10.0, 1.0]))
    monkeypatch.setattr(
        module,
        "fetch_frostt_tensor",
        lambda _: DataInstance([tensor], {"shape": shape}),
    )
    generator = getattr(module, f"{prefix}FROSTTGenerator")(n)
    # Use an existing dataset's metadata, replacing only the rank to avoid downloads.
    dataset = generator.datasets[0]
    if prefix == "CPALS":
        dataset.rank = 1
    else:
        dataset.ranks = (1,) * n
    problem = generator.generate(dataset)
    benchmark = getattr(module, f"{prefix}{n}DBenchmark")()
    benchmark._input, benchmark._meta = problem.inputs, problem.meta
    benchmark._ref_meta = problem.ref_meta
    factors = [np.zeros((size, 1)) for size in shape]
    for factor in factors:
        factor[0, 0] = 1
    if prefix == "CPALS":
        benchmark._output = [from_numpy(f) for f in factors] + [
            from_numpy(np.array([10.0]))
        ]
    else:
        benchmark._output = [from_numpy(np.full((1,) * n, 10.0))] + [
            from_numpy(f) for f in factors
        ]
    benchmark.check(Param(generator, dataset))
    for bad_value in (0, np.nan, 1000):
        benchmark._output[0] = from_numpy(
            np.full(benchmark._output[0].shape, bad_value)
        )
        with pytest.raises(AssertionError):
            benchmark.check(None)
