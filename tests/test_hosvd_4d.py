import pytest

import numpy as np

import saps.benchmarks.HOSVD_4d as hosvd_4d
from saps_framework import BinsparseFormat
from frameworks.saps_numpy import NumpyFramework


@pytest.fixture
def xp_numpy():
    return NumpyFramework()


def run_hosvd_benchmark(xp_numpy, X, ranks, max_iter=50, tolerance=1e-8):
    benchmark = hosvd_4d.HOSVD4DBenchmark()
    prev_xp = getattr(hosvd_4d, "xp", None)
    hosvd_4d.xp = xp_numpy
    try:
        output = benchmark.benchmark(
            [X, ranks],
            {
                "max_iter": max_iter,
                "tolerance": tolerance,
            },
        )
    finally:
        hosvd_4d.xp = prev_xp
    return output[0], output[1:]


def reconstruct_tensor(core, factors):
    """
    Helper method to reconstruct tensor from Tucker decomposition.
    """
    num_modes = len(factors)

    core_idx = "".join([chr(65 + m) for m in range(num_modes)])
    result_idx = "".join([chr(97 + m) for m in range(num_modes)])

    terms = [core_idx]
    operands = [core]

    for m in range(num_modes):
        terms.append(f"{result_idx[m]}{core_idx[m]}")
        operands.append(factors[m])

    subscripts = f"{','.join(terms)}->{result_idx}"
    return np.einsum(subscripts, *operands)


def test_hosvd_dense_generator_reconstruction(xp_numpy):
    """
    Test with the dense low-rank generator.
    """
    data, meta = hosvd_4d.HOSVD4DDenseGenerator().generate(
        hosvd_4d.HOSVD4DDenseGenerator().datasets[0]
    )
    X_bin, ranks_bin = data
    X_dense = xp_numpy.from_binsparse(X_bin)
    ranks = xp_numpy.from_binsparse(ranks_bin).astype(int)

    core_res, factors_res = run_hosvd_benchmark(
        xp_numpy, X_dense, ranks, max_iter=meta["max_iter"]
    )
    X_rec = reconstruct_tensor(core_res, factors_res)
    error = np.linalg.norm(X_dense - X_rec) / np.linalg.norm(X_dense)

    assert error < 1e-5


def test_manual_example_1_diagonal(xp_numpy):
    """
    Test with manually created diagonal tensor.
    """
    dims = (10, 10, 10, 10)
    ranks = (2, 2, 2, 2)
    rng = np.random.default_rng(1)

    core_true = rng.random(ranks)

    X_dense = np.zeros(dims)
    X_dense[: ranks[0], : ranks[1], : ranks[2], : ranks[3]] = core_true

    X_bin = BinsparseFormat.from_numpy(X_dense)
    ranks_bin = BinsparseFormat.from_numpy(np.array(ranks))

    core_res, factors_res = run_hosvd_benchmark(
        xp_numpy,
        xp_numpy.from_binsparse(X_bin),
        xp_numpy.from_binsparse(ranks_bin),
        max_iter=10,
    )

    X_rec = reconstruct_tensor(core_res, factors_res)

    assert np.allclose(X_dense, X_rec, atol=1e-5)


def test_manual_example_2_rank_one(xp_numpy):
    """
    Test with manually created rank-one tensor.
    """
    dims = (10, 10, 10, 10)
    ranks = (1, 1, 1, 1)
    rng = np.random.default_rng(2)

    a = rng.random(dims[0])
    b = rng.random(dims[1])
    c = rng.random(dims[2])
    d = rng.random(dims[3])

    X_dense = np.einsum("i,j,k,l->ijkl", a, b, c, d)
    X_bin = BinsparseFormat.from_numpy(X_dense)
    ranks_bin = BinsparseFormat.from_numpy(np.array(ranks))

    core_res, factors_res = run_hosvd_benchmark(
        xp_numpy,
        xp_numpy.from_binsparse(X_bin),
        xp_numpy.from_binsparse(ranks_bin),
        max_iter=10,
    )
    X_rec = reconstruct_tensor(core_res, factors_res)

    assert np.allclose(X_dense, X_rec, atol=1e-5)


def test_manual_example_3_structured(xp_numpy):
    """
    Test with manually created structured tensor.
    """
    dims = (5, 5, 5, 5)
    ranks = (2, 2, 2, 2)

    rng = np.random.default_rng(3)

    def get_orth(n, r):
        U, _, _ = np.linalg.svd(rng.standard_normal((n, n)))
        return U[:, :r]

    A = get_orth(dims[0], ranks[0])
    B = get_orth(dims[1], ranks[1])
    C = get_orth(dims[2], ranks[2])
    D = get_orth(dims[3], ranks[3])
    G = rng.standard_normal(ranks)

    X_dense = np.einsum("pqrs,ip,jq,kr,ls->ijkl", G, A, B, C, D)
    X_bin = BinsparseFormat.from_numpy(X_dense)
    ranks_bin = BinsparseFormat.from_numpy(np.array(ranks))

    core_res, factors_res = run_hosvd_benchmark(
        xp_numpy,
        xp_numpy.from_binsparse(X_bin),
        xp_numpy.from_binsparse(ranks_bin),
        max_iter=20,
    )
    X_rec = reconstruct_tensor(core_res, factors_res)

    assert np.allclose(X_dense, X_rec, atol=1e-5)

    for f in factors_res:
        identity = f.T @ f
        assert np.allclose(identity, np.eye(f.shape[1]), atol=1e-5)


def test_hosvd_sparse_input(xp_numpy):
    """
    Test with sparse input.
    """
    data, meta = hosvd_4d.HOSVD4DSparseGenerator().generate(
        hosvd_4d.HOSVD4DSparseGenerator().datasets[0]
    )
    X_bin, ranks_bin = data
    X_dense = xp_numpy.from_binsparse(X_bin)
    ranks = xp_numpy.from_binsparse(ranks_bin).astype(int)

    core_res, factors_res = run_hosvd_benchmark(
        xp_numpy, X_dense, ranks, max_iter=5, tolerance=meta["tolerance"]
    )

    assert core_res.shape == tuple(ranks)
    for i, f in enumerate(factors_res):
        assert f.shape == (X_dense.shape[i], ranks[i])
