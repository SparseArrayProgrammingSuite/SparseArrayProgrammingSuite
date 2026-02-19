import pytest

import numpy as np

from sparseappbench.benchmarks.HOSVD import (
    benchmark_hosvd,
    dg_hosvd_sparse_small,
)
from sparseappbench.binsparse_format import BinsparseFormat
from sparseappbench.frameworks.numpy_framework import NumpyFramework


@pytest.fixture
def xp_numpy():
    return NumpyFramework()


def reconstruct_tensor(core, factors):
    """
    Helper method to reconstruct tensor from Tucker decomposition.
    """
    num_modes = len(factors)

    core_idx = "".join([chr(65 + m) for m in range(num_modes)])  # ABC
    result_idx = "".join([chr(97 + m) for m in range(num_modes)])  # abc

    terms = [core_idx]
    operands = [core]

    for m in range(num_modes):
        terms.append(f"{result_idx[m]}{core_idx[m]}")
        operands.append(factors[m])

    subscripts = f"{','.join(terms)}->{result_idx}"
    return np.einsum(subscripts, *operands)


def test_manual_example_1_diagonal(xp_numpy):
    """
    Test with manually created diagonal tensor.
    """
    dims = (10, 10, 10)
    ranks = (2, 2, 2)
    rng = np.random.default_rng(1)

    core_true = rng.random(ranks)

    X_dense = np.zeros(dims)
    X_dense[: ranks[0], : ranks[1], : ranks[2]] = core_true

    X_bin = BinsparseFormat.from_numpy(X_dense)
    ranks_bin = BinsparseFormat.from_numpy(np.array(ranks))

    core_bin, factors_bin = benchmark_hosvd(xp_numpy, X_bin, ranks_bin, max_iter=10)
    core_res = xp_numpy.from_benchmark(core_bin)
    factors_res = [xp_numpy.from_benchmark(f) for f in factors_bin]

    X_rec = reconstruct_tensor(core_res, factors_res)

    assert np.allclose(X_dense, X_rec, atol=1e-5)


def test_manual_example_2_rank_one(xp_numpy):
    """
    Test with manually created rank-one tensor.
    """
    dims = (10, 10, 10)
    ranks = (1, 1, 1)
    rng = np.random.default_rng(2)

    a = rng.random(dims[0])
    b = rng.random(dims[1])
    c = rng.random(dims[2])

    X_dense = np.einsum("i,j,k->ijk", a, b, c)
    X_bin = BinsparseFormat.from_numpy(X_dense)
    ranks_bin = BinsparseFormat.from_numpy(np.array(ranks))

    core_bin, factors_bin = benchmark_hosvd(xp_numpy, X_bin, ranks_bin, max_iter=10)
    core_res = xp_numpy.from_benchmark(core_bin)
    factors_res = [xp_numpy.from_benchmark(f) for f in factors_bin]
    X_rec = reconstruct_tensor(core_res, factors_res)

    assert np.allclose(X_dense, X_rec, atol=1e-5)


def test_manual_example_3_structured(xp_numpy):
    """
    Test with manually created structured tensor.
    """
    dims = (5, 5, 5)
    ranks = (2, 2, 2)

    rng = np.random.default_rng(3)

    def get_orth(n, r):
        U, _, _ = np.linalg.svd(rng.standard_normal((n, n)))
        return U[:, :r]

    A = get_orth(dims[0], ranks[0])
    B = get_orth(dims[1], ranks[1])
    C = get_orth(dims[2], ranks[2])
    G = rng.standard_normal(ranks)

    X_dense = np.einsum("pqr,ip,jq,kr->ijk", G, A, B, C)
    X_bin = BinsparseFormat.from_numpy(X_dense)
    ranks_bin = BinsparseFormat.from_numpy(np.array(ranks))

    core_bin, factors_bin = benchmark_hosvd(xp_numpy, X_bin, ranks_bin, max_iter=20)
    core_res = xp_numpy.from_benchmark(core_bin)
    factors_res = [xp_numpy.from_benchmark(f) for f in factors_bin]
    X_rec = reconstruct_tensor(core_res, factors_res)

    assert np.allclose(X_dense, X_rec, atol=1e-5)

    for f in factors_res:
        identity = f.T @ f
        assert np.allclose(identity, np.eye(f.shape[1]), atol=1e-5)


def test_hosvd_sparse_input(xp_numpy):
    """
    Test with sparse input.
    """
    X_bin, ranks_bin = dg_hosvd_sparse_small()
    ranks = tuple(xp_numpy.from_benchmark(ranks_bin).astype(int))

    core_bin, factors_bin = benchmark_hosvd(xp_numpy, X_bin, ranks_bin, max_iter=5)
    core_res = xp_numpy.from_benchmark(core_bin)
    factors_res = [xp_numpy.from_benchmark(f) for f in factors_bin]

    assert core_res.shape == tuple(ranks)
    for i, f in enumerate(factors_res):
        assert f.shape == (X_bin.data["shape"][i], ranks[i])
