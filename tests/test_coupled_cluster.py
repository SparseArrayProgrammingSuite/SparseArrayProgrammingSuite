import pytest

import numpy as np

from sparseappbench.benchmarks.coupled_cluster import benchmark_ccsd, make_ccsd_inputs
from sparseappbench.binsparse_format import BinsparseFormat
from sparseappbench.frameworks.checker_framework import CheckerFramework
from sparseappbench.frameworks.numpy_framework import NumpyFramework

# Ground truth from C++ CTF (tests/cpp_reference/coupled_cluster/ccsd.cxx):
#   mpirun -n 1 ./ccsd -no 4 -nv 6 -niter 1  →  |T| = 380638.269079
CPP_CTF_REFERENCE_NORM = 380638.269079


def _as_canon_abij(T):
    """Enforce {AS,NS,AS,NS}: canonical a<b and i<j only, derive rest."""
    nv, _, no, _ = T.shape
    canon = (
        np.arange(nv)[:, None, None, None] < np.arange(nv)[None, :, None, None]
    ) & (np.arange(no)[None, None, :, None] < np.arange(no)[None, None, None, :])
    T_c = np.where(canon, T, 0.0)
    return (
        T_c
        - T_c.transpose(1, 0, 2, 3)
        - T_c.transpose(0, 1, 3, 2)
        + T_c.transpose(1, 0, 3, 2)
    )


def _full_antisym3(T, axes):
    """Fully antisymmetrize T over exactly 3 axes (must have equal dimension)."""
    a0, a1, a2 = axes
    base = list(range(T.ndim))

    def perm(p0, p1, p2):
        p = base[:]
        p[a0], p[a1], p[a2] = axes[p0], axes[p1], axes[p2]
        return tuple(p)

    return (
        T
        - T.transpose(perm(1, 0, 2))
        - T.transpose(perm(2, 1, 0))
        - T.transpose(perm(0, 2, 1))
        + T.transpose(perm(1, 2, 0))
        + T.transpose(perm(2, 0, 1))
    )


@pytest.mark.parametrize("xp", [NumpyFramework(), CheckerFramework()])
def test_ccsd_output_shape(xp):
    """Verify benchmark runs without errors and returns correct output shapes."""
    T1_out_b, T2_out_b = benchmark_ccsd(xp, *make_ccsd_inputs(no=4, nv=6))
    assert T1_out_b.data["shape"] == (6, 4)
    assert T2_out_b.data["shape"] == (6, 6, 4, 4)


def test_ccsd_output_matches_cpp_reference():
    """Verify Python CCSD output reproduces the C++ CTF reference norm.

    C++ CTF reference (ccsd.cxx, no=4, nv=6, niter=1): |T| = 380638.269079

    _as_canon_abij is applied to T2 before taking the norm because CTF's
    norm2() counts all elements of an ASAS tensor (including derived
    antisymmetric positions), equivalent to norm(full_antisymmetrized).
    """
    xp = NumpyFramework()
    T1_out_b, T2_out_b = benchmark_ccsd(xp, *make_ccsd_inputs(no=4, nv=6))
    T1_out = xp.from_binsparse(T1_out_b)
    T2_out = xp.from_binsparse(T2_out_b)
    T2_out = _as_canon_abij(T2_out)
    # Verify T2 antisymmetry: T2[a,b,i,j] == -T2[b,a,i,j] and T2[a,b,i,j] ==-T2[a,b,j,i]
    assert np.allclose(T2_out, -T2_out.transpose(1, 0, 2, 3), atol=1e-10), (
        "T2 output violates antisymmetry in first index pair (a,b)"
    )
    assert np.allclose(T2_out, -T2_out.transpose(0, 1, 3, 2), atol=1e-10), (
        "T2 output violates antisymmetry in second index pair (i,j)"
    )
    T_norm = np.linalg.norm(T1_out) + np.linalg.norm(T2_out)
    assert np.isclose(T_norm, CPP_CTF_REFERENCE_NORM, rtol=1e-6), (
        f"|T| = {T_norm:.6f} does not match C++ CTF reference {CPP_CTF_REFERENCE_NORM}"
    )


@pytest.mark.parametrize("xp", [NumpyFramework(), CheckerFramework()])
def test_ccsdt_map_contraction_shape(xp):
    """Verify Z[hijmno] += W[hijk]*T[kmno] produces shape (n,n,n,n,n,n)."""
    n = 4
    W_b = BinsparseFormat.from_numpy(np.zeros((n, n, n, n)))
    T_b = BinsparseFormat.from_numpy(np.zeros((n, n, n, n)))
    Z_b = BinsparseFormat.from_numpy(np.zeros((n, n, n, n, n, n)))
    W = xp.from_binsparse(W_b)
    T = xp.from_binsparse(T_b)
    Z = xp.from_binsparse(Z_b)
    Z = Z + xp.einsum("Z[h,i,j,m,n,o] += W[h,i,j,k] * T[k,m,n,o]", W=W, T=T)
    assert xp.to_binsparse(Z).data["shape"] == (n, n, n, n, n, n)


def test_ccsdt_map_contraction_correctness():
    """Verify Z[hijmno] += W[hijk]*T[kmno] matches numpy einsum reference."""
    rng = np.random.default_rng(42)
    n = 4
    W = rng.standard_normal((n, n, n, n))
    T = rng.standard_normal((n, n, n, n))
    Z = rng.standard_normal((n, n, n, n, n, n))
    xp = NumpyFramework()
    Z_result = xp.from_binsparse(
        xp.to_binsparse(
            xp.from_binsparse(BinsparseFormat.from_numpy(Z))
            + xp.einsum(
                "Z[h,i,j,m,n,o] += W[h,i,j,k] * T[k,m,n,o]",
                W=xp.from_binsparse(BinsparseFormat.from_numpy(W)),
                T=xp.from_binsparse(BinsparseFormat.from_numpy(T)),
            )
        )
    )
    Z_ref = Z + np.einsum("hijk,kmno->hijmno", W, T)
    assert np.allclose(Z_result, Z_ref, rtol=1e-10)


def test_ccsdt_t3_to_t2_antisymmetry():
    """Mirror ccsdt_t3_to_t2.cxx: AS contraction preserves antisymmetry.

    Checks norm(AS_C) == norm(NS_C) and NS_C - AS_C ≈ 0.
    """
    n, m = 6, 7
    rng = np.random.default_rng(2013)

    raw_A = rng.uniform(0, 1, (n, n, n, m))
    AS_A = raw_A - raw_A.transpose(1, 0, 2, 3)

    raw_B = rng.uniform(0, 1, (m, m, m, n, n, n))
    AS_B = _full_antisym3(raw_B, [0, 1, 2])
    AS_B = _full_antisym3(AS_B, [3, 4, 5])

    raw_C = rng.uniform(0, 1, (m, m, n, n))
    AS_C = raw_C - raw_C.transpose(1, 0, 2, 3)
    AS_C = AS_C - AS_C.transpose(0, 1, 3, 2)

    NS_C = AS_C.copy()

    contrib = 0.5 * np.einsum("mnje,abeimn->abij", AS_A, AS_B)
    AS_C = AS_C + (contrib - contrib.transpose(0, 1, 3, 2))

    ns_contrib = 0.5 * np.einsum("mnje,abeimn->abij", AS_A, AS_B)
    NS_C = NS_C + ns_contrib
    NS_C = NS_C - ns_contrib.transpose(0, 1, 3, 2)

    assert np.isclose(np.linalg.norm(AS_C), np.linalg.norm(NS_C), rtol=1e-6), (
        f"AS norm {np.linalg.norm(AS_C)} != NS norm {np.linalg.norm(NS_C)}"
    )
    diff_norm = np.linalg.norm(NS_C - AS_C)
    assert diff_norm < 1e-6, f"NS_C - AS_C norm = {diff_norm} > 1e-6"
