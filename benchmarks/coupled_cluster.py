"""
Name: Coupled Cluster Singles and Doubles (CCSD)
Author: Tarun Devi
Motivation:
"Coupled cluster theory is one of the most accurate and widely used methods in
quantum chemistry for computing ground-state energies of molecular systems."
G. D. Purvis and R. J. Bartlett, "A full coupled-cluster singles and doubles
model: The inclusion of disconnected triples," J. Chem. Phys., vol. 76, no. 4,
pp. 1910-1918, 1982, doi: 10.1063/1.443164.
Role of sparsity:
The two-electron integral tensors (Vabef, Vabij, etc.) are antisymmetric, which
means roughly 3/4 of entries are redundant. Exploiting this antisymmetry reduces
both storage and compute by up to 8x for 4-index tensors.
Here antisymmetry means swapping an antisymmetric index pair flips the sign, for
example T[a,b,i,j] = -T[b,a,i,j] and T[a,b,i,j] = -T[a,b,j,i].
Implementation (Where did the reference algorithm come from? With citation.):
Ported from the CTF (Cyclops Tensor Framework) CCSD reference implementation:
E. Solomonik, D. Matthews, J. R. Hammond, J. F. Stanton, and J. Demmel,
"A massively parallel tensor contraction framework for coupled-cluster
computations," J. Parallel Distrib. Comput., vol. 74, no. 12, pp. 3176-3190,
2014, doi: 10.1016/j.jpdc.2014.06.002.
Data Generation (How is the data generated? Why is it realistic?):
Inputs are generated using the same deterministic pseudorandom fill as the C++
CTF reference (ccsd.cxx): canonical antisymmetric elements are set to
((flat_index * multiplier + tensor_id) % 13077) / 13077 - 0.5, then reflected
via antisymmetry. This exactly reproduces the C++ reference output |T| = 380638.
Statement on the use of Generative AI:
Generative AI was used to assist in debugging the benchmark.
This statement was written by hand.
"""

import numpy as np

from ..src.saps.binsparse_format import BinsparseFormat


def benchmark_ccsd(
    xp,
    Vme_bench,  # (no, nv)
    Vae_bench,  # (nv, nv)
    Vmi_bench,  # (no, no)
    Vai_bench,  # (nv, no)
    Vmnef_bench,  # (no, no, nv, nv)
    Vabef_bench,  # (nv, nv, nv, nv)
    Vabij_bench,  # (nv, nv, no, no)
    Vabei_bench,  # (nv, nv, nv, no)
    Vmnij_bench,  # (no, no, no, no)
    Vmnei_bench,  # (no, no, nv, no)
    Vamei_bench,  # (nv, no, nv, no)
    Vamij_bench,  # (nv, no, no, no)
    Vanef_bench,  # (nv, no, nv, nv)
    Vmnfi_bench,  # (no, no, nv, no)
    Vamef_bench,  # (nv, no, nv, nv)
    Vaeim_bench,  # (nv, nv, no, no)
    T1_bench,  # (nv, no)
    T2_bench,  # (nv, nv, no, no)
    D1_bench,
    D2_bench,
):

    Vme = xp.from_benchmark(Vme_bench)
    Vae = xp.from_benchmark(Vae_bench)
    Vmi = xp.from_benchmark(Vmi_bench)
    Vai = xp.from_benchmark(Vai_bench)
    Vmnef = xp.from_benchmark(Vmnef_bench)
    Vabef = xp.from_benchmark(Vabef_bench)
    Vabij = xp.from_benchmark(Vabij_bench)
    Vabei = xp.from_benchmark(Vabei_bench)
    Vmnij = xp.from_benchmark(Vmnij_bench)
    Vmnei = xp.from_benchmark(Vmnei_bench)
    Vamei = xp.from_benchmark(Vamei_bench)
    Vamij = xp.from_benchmark(Vamij_bench)
    Vanef = xp.from_benchmark(Vanef_bench)
    Vmnfi = xp.from_benchmark(Vmnfi_bench)
    Vamef = xp.from_benchmark(Vamef_bench)
    Vaeim = xp.from_benchmark(Vaeim_bench)
    T1 = xp.from_benchmark(T1_bench)
    T2 = xp.from_benchmark(T2_bench)
    D1 = xp.from_benchmark(D1_bench)
    D2 = xp.from_benchmark(D2_bench)

    outer = xp.einsum("outer[a,b,i,j] += 0.5 * T1[a,i] * T1[b,j]", T1=T1)
    T21 = T2 + _asas_full(xp, outer)

    # CTF initializes each intermediate via copy constructor (adds 1x integral)
    # plus explicit "+=" lines.  Fme: 1 copy + 1 "+=" = 2x Vme.
    Fme = 2 * Vme + xp.einsum(
        "Fme[m,e] += Vmnef[m,n,e,f] * T1[f,n]", Vmnef=Vmnef, T1=T1
    )

    rest_Fae = (
        -xp.einsum("Fae[a,e] += Fme[m,e] * T1[a,m]", Fme=Fme, T1=T1)
        - xp.einsum(
            "Fae[a,e] += 0.5 * Vmnef[m,n,e,f] * T2[a,f,m,n]", Vmnef=Vmnef, T2=T2
        )
        + xp.einsum("Fae[a,e] += Vanef[a,n,e,f] * T1[f,n]", Vanef=Vanef, T1=T1)
    )
    Fae = 2 * Vae + _as2d_full(xp, rest_Fae)

    rest_Fmi = (
        xp.einsum("Fmi[m,i] += Fme[m,e] * T1[e,i]", Fme=Fme, T1=T1)
        + xp.einsum(
            "Fmi[m,i] += 0.5 * Vmnef[m,n,e,f] * T2[e,f,i,n]", Vmnef=Vmnef, T2=T2
        )
        + xp.einsum("Fmi[m,i] += Vmnfi[m,n,f,i] * T1[f,n]", Vmnfi=Vmnfi, T1=T1)
    )
    Fmi = 2 * Vmi + _as2d_full(xp, rest_Fmi)

    R_Wmnei = xp.einsum(
        "Wmnei[m,n,e,i] += Vmnef[m,n,e,f] * T1[f,i]", Vmnef=Vmnef, T1=T1
    )
    Wmnei = 3 * Vmnei + R_Wmnei

    R_Wmnij = xp.einsum(
        "Wmnij[m,n,i,j] += Vmnei[m,n,e,i] * T1[e,j]", Vmnei=Vmnei, T1=T1
    )
    S_Wmnij = xp.einsum(
        "Wmnij[m,n,i,j] += Vmnef[m,n,e,f] * T21[e,f,i,j]", Vmnef=Vmnef, T21=T21
    )
    Wmnij = 2 * Vmnij - _antisym_dims23(xp, R_Wmnij) + S_Wmnij

    Wamei = (
        2 * Vamei
        - xp.einsum("Wamei[a,m,e,i] += Wmnei[m,n,e,i] * T1[a,n]", Wmnei=Wmnei, T1=T1)
        + xp.einsum("Wamei[a,m,e,i] += Vamef[a,m,e,f] * T1[f,i]", Vamef=Vamef, T1=T1)
        + xp.einsum(
            "Wamei[a,m,e,i] += 0.5 * Vmnef[m,n,e,f] * T2[a,f,i,n]",
            Vmnef=Vmnef,
            T2=T2,
        )
    )

    R1_Wamij = xp.einsum(
        "Wamij[a,m,i,j] += Vamei[a,m,e,i] * T1[e,j]", Vamei=Vamei, T1=T1
    )
    R2_Wamij = xp.einsum(
        "Wamij[a,m,i,j] += Vamef[a,m,e,f] * T2[e,f,i,j]", Vamef=Vamef, T2=T2
    )
    Wamij = 2 * Vamij + _antisym_dims23(xp, R1_Wamij) + R2_Wamij

    T1_new = (
        2 * Vai
        - xp.einsum("T1_new[a,i] += Fmi[m,i] * T1[a,m]", Fmi=Fmi, T1=T1)
        + xp.einsum("T1_new[a,i] += Vae[a,e] * T1[e,i]", Vae=Vae, T1=T1)
        + xp.einsum("T1_new[a,i] += Vamei[a,m,e,i] * T1[e,m]", Vamei=Vamei, T1=T1)
        + xp.einsum("T1_new[a,i] += Vaeim[a,e,i,m] * Fme[m,e]", Vaeim=Vaeim, Fme=Fme)
        + xp.einsum(
            "T1_new[a,i] += 0.5 * Vamef[a,m,e,f] * T21[e,f,i,m]",
            Vamef=Vamef,
            T21=T21,
        )
        - xp.einsum(
            "T1_new[a,i] += 0.5 * Wmnei[m,n,e,i] * T21[e,a,m,n]",
            Wmnei=Wmnei,
            T21=T21,
        )
    )
    R1_Z = xp.einsum("T2_new[a,b,i,j] += Vabei[a,b,e,i] * T1[e,j]", Vabei=Vabei, T1=T1)
    R2_Z = xp.einsum(
        "T2_new[a,b,i,j] += Wamei[a,m,e,i] * T2[e,b,m,j]", Wamei=Wamei, T2=T2
    )
    R3_Z = xp.einsum("T2_new[a,b,i,j] += Wamij[a,m,i,j] * T1[b,m]", Wamij=Wamij, T1=T1)
    R4_Z = xp.einsum("T2_new[a,b,i,j] += Fae[a,e] * T2[e,b,i,j]", Fae=Fae, T2=T2)
    R5_Z = xp.einsum("T2_new[a,b,i,j] += Fmi[m,i] * T2[a,b,m,j]", Fmi=Fmi, T2=T2)
    R6_Z = xp.einsum(
        "T2_new[a,b,i,j] += 0.5 * Vabef[a,b,e,f] * T21[e,f,i,j]",
        Vabef=Vabef,
        T21=T21,
    )
    R7_Z = xp.einsum(
        "T2_new[a,b,i,j] += 0.5 * Wmnij[m,n,i,j] * T21[a,b,m,n]",
        Wmnij=Wmnij,
        T21=T21,
    )
    T2_new = (
        2 * Vabij
        + _antisym_dims23(xp, R1_Z)
        + _asas_full(xp, R2_Z)
        - _antisym_dims01(xp, R3_Z)
        + _antisym_dims01(xp, R4_Z)
        - _antisym_dims23(xp, R5_Z)
        + R6_Z
        + R7_Z
    )

    T1_final = T1_new / D1
    T2_final = 2 * T2_new / D2

    T1_out, T2_out = xp.compute((T1_final, T2_final))

    return xp.to_benchmark(T1_out), xp.to_benchmark(T2_out)


def _as2d_full(xp, F):
    """Full 2D antisymmetrizer: F[a,e] - F[e,a]."""
    F_T = xp.einsum("FT[i,j] = F[j,i]", F=F)
    return F - F_T


def _asas_full(xp, T):
    """Fully antisymmetrize a 4D tensor in both index pairs (0,1) and (2,3).

    Here antisymmetry means swapping either pair flips the sign:
    T[a,b,i,j] = -T[b,a,i,j] and T[a,b,i,j] = -T[a,b,j,i].
    This helper applies the full ASAS combination
    T - T_ba - T_ji + T_baji without assuming canonical masking.
    """
    T_ba = xp.einsum("Tba[a,b,i,j] = T[b,a,i,j]", T=T)
    T_ji = xp.einsum("Tji[a,b,i,j] = T[a,b,j,i]", T=T)
    T_baji = xp.einsum("Tbaji[a,b,i,j] = T[b,a,j,i]", T=T)
    return T - T_ba - T_ji + T_baji


def _antisym_dims01(xp, T):
    """Antisymmetrize T in dims 0,1: T - T[b,a,i,j]."""
    T_ba = xp.einsum("Tba[a,b,i,j] = T[b,a,i,j]", T=T)
    return T - T_ba


def _antisym_dims23(xp, T):
    """Antisymmetrize T in dims 2,3: T - T[a,b,j,i]."""
    T_ji = xp.einsum("Tji[a,b,i,j] = T[a,b,j,i]", T=T)
    return T - T_ji


def _ctf_col_major_idx(shape):
    return np.arange(int(np.prod(shape)), dtype=np.int64).reshape(shape, order="F")


def _ctf_rand(shape, tensor_id, multiplier=16):
    """NS fill: all elements independent, matching CTF fill_rand."""
    idx = _ctf_col_major_idx(shape)
    values = ((idx * multiplier + tensor_id) % 13077) / 13077.0 - 0.5
    return BinsparseFormat.from_numpy(values)


def _make_as2d(shape, tensor_id, multiplier=16):
    """Build a 2D antisymmetric tensor from canonical entries with d0 < d1.

    Values are generated only on the upper-triangular canonical half using the
    CTF-style deterministic fill, then mirrored with a sign flip so the final
    tensor satisfies A[i, j] = -A[j, i].
    """
    d0, d1 = shape
    idx = _ctf_col_major_idx(shape)
    vals = ((idx * multiplier + tensor_id) % 13077) / 13077.0 - 0.5
    canon = np.arange(d0)[:, None] < np.arange(d1)[None, :]
    result = np.where(canon, vals, 0.0)
    result = result - result.T
    return BinsparseFormat.from_numpy(result)


def _make_asns_asns(shape, tensor_id, multiplier=16):
    """Build a 4D tensor antisymmetric in both index pairs (0,1) and (2,3).

    Only canonical entries with d0 < d1 and d2 < d3 are filled directly. The
    remaining positions are derived by reflecting across each antisymmetric pair
    with the appropriate sign changes, matching the CCSD {AS,NS,AS,NS} layout.
    """
    d0, d1, d2, d3 = shape
    idx = _ctf_col_major_idx(shape)
    vals = ((idx * multiplier + tensor_id) % 13077) / 13077.0 - 0.5
    canon = (
        np.arange(d0)[:, None, None, None] < np.arange(d1)[None, :, None, None]
    ) & (np.arange(d2)[None, None, :, None] < np.arange(d3)[None, None, None, :])
    result = np.where(canon, vals, 0.0)
    result = result - result.transpose(1, 0, 2, 3)
    result = result - result.transpose(0, 1, 3, 2)
    return BinsparseFormat.from_numpy(result)


def _make_asns_nsns(shape, tensor_id, multiplier=16):
    """Build a 4D tensor antisymmetric only in the first index pair (0,1).

    Canonical values are placed where d0 < d1, then reflected across the first
    pair with a sign flip. The last two dimensions are left nonsymmetric.
    """
    d0, d1 = shape[0], shape[1]
    idx = _ctf_col_major_idx(shape)
    vals = ((idx * multiplier + tensor_id) % 13077) / 13077.0 - 0.5
    canon = np.arange(d0)[:, None, None, None] < np.arange(d1)[None, :, None, None]
    result = np.where(canon, vals, 0.0)
    result = result - result.transpose(1, 0, 2, 3)
    return BinsparseFormat.from_numpy(result)


def _make_nsns_asns(shape, tensor_id, multiplier=16):
    """Build a 4D tensor antisymmetric only in the last index pair (2,3).

    Canonical values are placed where d2 < d3, then reflected across the last
    pair with a sign flip. The first two dimensions are left nonsymmetric.
    """
    d2, d3 = shape[2], shape[3]
    idx = _ctf_col_major_idx(shape)
    vals = ((idx * multiplier + tensor_id) % 13077) / 13077.0 - 0.5
    canon = np.arange(d2)[None, None, :, None] < np.arange(d3)[None, None, None, :]
    result = np.where(canon, vals, 0.0)
    result = result - result.transpose(0, 1, 3, 2)
    return BinsparseFormat.from_numpy(result)


def make_ccsd_inputs(no, nv):
    """Generate deterministic antisymmetric CCSD inputs matching ccsd.cxx fill_rand."""
    Vae_b = _make_as2d((nv, nv), 2)
    Vai_b = _ctf_rand((nv, no), 3)
    Vme_b = _ctf_rand((no, nv), 4)
    Vmi_b = _make_as2d((no, no), 5)
    Vabef_b = _make_asns_asns((nv, nv, nv, nv), 6)
    Vabei_b = _make_asns_nsns((nv, nv, nv, no), 7)
    Vanef_b = _make_nsns_asns((nv, no, nv, nv), 8)
    Vamei_b = _ctf_rand((nv, no, nv, no), 9)
    Vabij_b = _make_asns_asns((nv, nv, no, no), 10)
    Vmnef_b = _make_asns_asns((no, no, nv, nv), 11)
    Vamij_b = _make_nsns_asns((nv, no, no, no), 12)
    Vmnei_b = _make_asns_nsns((no, no, nv, no), 13)
    Vmnij_b = _make_asns_asns((no, no, no, no), 14)

    # Tensor aliases (same underlying CTF storage in C++)
    Vamef_b = Vanef_b
    Vaeim_b = Vabij_b
    Vmnfi_b = Vmnei_b

    T1_b = _ctf_rand((nv, no), 0, multiplier=13)
    T2_b = _make_asns_asns((nv, nv, no, no), 1, multiplier=13)

    aa = ((np.arange(nv) * 16 + 0) % 13077) / 13077.0 - 0.5
    ii = ((np.arange(no) * 16 + 1) % 13077) / 13077.0 - 0.5
    D1 = ii[None, :] - aa[:, None]
    D2 = (
        ii.reshape(1, 1, no, 1)
        + ii.reshape(1, 1, 1, no)
        - aa.reshape(nv, 1, 1, 1)
        - aa.reshape(1, nv, 1, 1)
    )

    return (
        Vme_b,
        Vae_b,
        Vmi_b,
        Vai_b,
        Vmnef_b,
        Vabef_b,
        Vabij_b,
        Vabei_b,
        Vmnij_b,
        Vmnei_b,
        Vamei_b,
        Vamij_b,
        Vanef_b,
        Vmnfi_b,
        Vamef_b,
        Vaeim_b,
        T1_b,
        T2_b,
        BinsparseFormat.from_numpy(D1),
        BinsparseFormat.from_numpy(D2),
    )


def dg_ccsd_small():
    """no=4, nv=6 — matches the C++ CTF reference (ccsd.cxx)."""
    return make_ccsd_inputs(no=4, nv=6)


def dg_ccsd_medium():
    """no=8, nv=12."""
    return make_ccsd_inputs(no=8, nv=12)


def dg_ccsd_large():
    """no=16, nv=24."""
    return make_ccsd_inputs(no=16, nv=24)
