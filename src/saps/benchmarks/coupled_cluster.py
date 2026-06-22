# BEGIN COPIED TEST FILE: tests/test_coupled_cluster.py
# import pytest
#
# import numpy as np
#
# import saps.benchmarks.coupled_cluster as cc
# from frameworks.saps_numpy import NumpyFramework
# from saps.benchmarks.coupled_cluster import CCSD, make_ccsd_inputs
# from saps_framework.binsparse_format import BinsparseFormat
#
# # Ground truth from C++ CTF (tests/cpp_reference/coupled_cluster/ccsd.cxx):
# #   mpirun -n 1 ./ccsd -no 4 -nv 6 -niter 1  →  |T| = 380638.269079
# CPP_CTF_REFERENCE_NORM = 380638.269079
#
#
# def benchmark_ccsd(xp, data):
#     cc.xp = xp
#     data = [xp.from_binsparse(d) for d in data]
#     return CCSD().benchmark(data, {})
#
#
# def _as_canon_abij(T):
#     """Enforce {AS,NS,AS,NS}: canonical a<b and i<j only, derive rest."""
#     nv, _, no, _ = T.shape
#     canon = (
#         np.arange(nv)[:, None, None, None] < np.arange(nv)[None, :, None, None]
#     ) & (np.arange(no)[None, None, :, None] < np.arange(no)[None, None, None, :])
#     T_c = np.where(canon, T, 0.0)
#     return (
#         T_c
#         - T_c.transpose(1, 0, 2, 3)
#         - T_c.transpose(0, 1, 3, 2)
#         + T_c.transpose(1, 0, 3, 2)
#     )
#
#
# def _full_antisym3(T, axes):
#     """Fully antisymmetrize T over exactly 3 axes (must have equal dimension)."""
#     a0, a1, a2 = axes
#     base = list(range(T.ndim))
#
#     def perm(p0, p1, p2):
#         p = base[:]
#         p[a0], p[a1], p[a2] = axes[p0], axes[p1], axes[p2]
#         return tuple(p)
#
#     return (
#         T
#         - T.transpose(perm(1, 0, 2))
#         - T.transpose(perm(2, 1, 0))
#         - T.transpose(perm(0, 2, 1))
#         + T.transpose(perm(1, 2, 0))
#         + T.transpose(perm(2, 0, 1))
#     )
#
#
# @pytest.mark.parametrize("xp", [NumpyFramework()])
# def test_ccsd_output_shape(xp):
#     """Verify benchmark runs without errors and returns correct output shapes."""
#     T1_out, T2_out = benchmark_ccsd(xp, make_ccsd_inputs(no=4, nv=6))
#     assert T1_out.shape == (6, 4)
#     assert T2_out.shape == (6, 6, 4, 4)
#
#
# def test_ccsd_output_matches_cpp_reference():
#     """Verify Python CCSD output reproduces the C++ CTF reference norm.
#
#     C++ CTF reference (ccsd.cxx, no=4, nv=6, niter=1): |T| = 380638.269079
#
#     _as_canon_abij is applied to T2 before taking the norm because CTF's
#     norm2() counts all elements of an ASAS tensor (including derived
#     antisymmetric positions), equivalent to norm(full_antisymmetrized).
#     """
#     xp = NumpyFramework()
#     T1_out, T2_out = benchmark_ccsd(xp, make_ccsd_inputs(no=4, nv=6))
#     T2_out = _as_canon_abij(T2_out)
#     # Verify T2 antisymmetry: T2[a,b,i,j] == -T2[b,a,i,j] and T2[a,b,i,j] ==-T2[a,b,j,i]
#     assert np.allclose(T2_out, -T2_out.transpose(1, 0, 2, 3), atol=1e-10), (
#         "T2 output violates antisymmetry in first index pair (a,b)"
#     )
#     assert np.allclose(T2_out, -T2_out.transpose(0, 1, 3, 2), atol=1e-10), (
#         "T2 output violates antisymmetry in second index pair (i,j)"
#     )
#     T_norm = np.linalg.norm(T1_out) + np.linalg.norm(T2_out)
#     assert np.isclose(T_norm, CPP_CTF_REFERENCE_NORM, rtol=1e-6), (
#         f"|T| = {T_norm:.6f} does not match C++ CTF reference {CPP_CTF_REFERENCE_NORM}"
#     )
#
#
# @pytest.mark.parametrize("xp", [NumpyFramework()])
# def test_ccsdt_map_contraction_shape(xp):
#     """Verify Z[hijmno] += W[hijk]*T[kmno] produces shape (n,n,n,n,n,n)."""
#     n = 4
#     W_b = BinsparseFormat.from_numpy(np.zeros((n, n, n, n)))
#     T_b = BinsparseFormat.from_numpy(np.zeros((n, n, n, n)))
#     Z_b = BinsparseFormat.from_numpy(np.zeros((n, n, n, n, n, n)))
#     W = xp.from_binsparse(W_b)
#     T = xp.from_binsparse(T_b)
#     Z = xp.from_binsparse(Z_b)
#     Z = Z + xp.einsum("Z[h,i,j,m,n,o] += W[h,i,j,k] * T[k,m,n,o]", W=W, T=T)
#     assert xp.to_binsparse(Z).data["shape"] == (n, n, n, n, n, n)
#
#
# def test_ccsdt_map_contraction_correctness():
#     """Verify Z[hijmno] += W[hijk]*T[kmno] matches numpy einsum reference."""
#     rng = np.random.default_rng(42)
#     n = 4
#     W = rng.standard_normal((n, n, n, n))
#     T = rng.standard_normal((n, n, n, n))
#     Z = rng.standard_normal((n, n, n, n, n, n))
#     xp = NumpyFramework()
#     Z_result = xp.from_binsparse(
#         xp.to_binsparse(
#             xp.from_binsparse(BinsparseFormat.from_numpy(Z))
#             + xp.einsum(
#                 "Z[h,i,j,m,n,o] += W[h,i,j,k] * T[k,m,n,o]",
#                 W=xp.from_binsparse(BinsparseFormat.from_numpy(W)),
#                 T=xp.from_binsparse(BinsparseFormat.from_numpy(T)),
#             )
#         )
#     )
#     Z_ref = Z + np.einsum("hijk,kmno->hijmno", W, T)
#     assert np.allclose(Z_result, Z_ref, rtol=1e-10)
#
#
# def test_ccsdt_t3_to_t2_antisymmetry():
#     """Mirror ccsdt_t3_to_t2.cxx: AS contraction preserves antisymmetry.
#
#     Checks norm(AS_C) == norm(NS_C) and NS_C - AS_C ≈ 0.
#     """
#     n, m = 6, 7
#     rng = np.random.default_rng(2013)
#
#     raw_A = rng.uniform(0, 1, (n, n, n, m))
#     AS_A = raw_A - raw_A.transpose(1, 0, 2, 3)
#
#     raw_B = rng.uniform(0, 1, (m, m, m, n, n, n))
#     AS_B = _full_antisym3(raw_B, [0, 1, 2])
#     AS_B = _full_antisym3(AS_B, [3, 4, 5])
#
#     raw_C = rng.uniform(0, 1, (m, m, n, n))
#     AS_C = raw_C - raw_C.transpose(1, 0, 2, 3)
#     AS_C = AS_C - AS_C.transpose(0, 1, 3, 2)
#
#     NS_C = AS_C.copy()
#
#     contrib = 0.5 * np.einsum("mnje,abeimn->abij", AS_A, AS_B)
#     AS_C = AS_C + (contrib - contrib.transpose(0, 1, 3, 2))
#
#     ns_contrib = 0.5 * np.einsum("mnje,abeimn->abij", AS_A, AS_B)
#     NS_C = NS_C + ns_contrib
#     NS_C = NS_C - ns_contrib.transpose(0, 1, 3, 2)
#
#     assert np.isclose(np.linalg.norm(AS_C), np.linalg.norm(NS_C), rtol=1e-6), (
#         f"AS norm {np.linalg.norm(AS_C)} != NS norm {np.linalg.norm(NS_C)}"
#     )
#     diff_norm = np.linalg.norm(NS_C - AS_C)
#     assert diff_norm < 1e-6, f"NS_C - AS_C norm = {diff_norm} > 1e-6"
# END COPIED TEST FILE: tests/test_coupled_cluster.py

import numpy as np

import saps
from saps.benchmark import (
    Author,
    Benchmark,
    Contributor,
    DataInstance,
    Dataset,
    Generator,
    Ref,
)
from saps_framework.binsparse_format import BinsparseFormat

xp = saps.xp


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


class CCSDDataset(Dataset):
    def __init__(self, name, pretty_name, description, suites, no, nv):
        self._name = name
        self._pretty_name = pretty_name
        self._description = description
        self._suites = suites
        self.no = no
        self.nv = nv

    @property
    def name(self) -> str:
        return self._name

    @property
    def pretty_name(self) -> str:
        return self._pretty_name

    @property
    def description(self) -> str:
        return self._description

    @property
    def suites(self) -> list[str]:
        return self._suites

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"


class CCSDGenerator(Generator[CCSDDataset]):
    @property
    def name(self) -> str:
        return "ccsd_inputs"

    @property
    def pretty_name(self) -> str:
        return "CCSD Input Generator"

    @property
    def description(self) -> str:
        return (
            "Inputs are generated using the same deterministic pseudorandom fill as"
            " the C++ CTF reference (ccsd.cxx): canonical antisymmetric elements are"
            " set to ((flat_index * multiplier + tensor_id) % 13077) / 13077 - 0.5,"
            " then reflected via antisymmetry. This exactly reproduces the C++"
            " reference output |T| = 380638."
        )

    @property
    def suites(self) -> list[str]:
        return []

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def authors(self) -> list[Contributor]:
        return [Contributor("Tarun Devi", "tdevi3@gatech.edu")]

    @property
    def references(self) -> list[Ref]:
        return [
            Ref(
                title=(
                    "A full coupled-cluster singles and doubles model:"
                    " The inclusion of disconnected triples"
                ),
                authors=[Author("G. D. Purvis"), Author("R. J. Bartlett")],
                year=1982,
                url="https://doi.org/10.1063/1.443164",
                journal="J. Chem. Phys.",
                volume=76,
                number=4,
                pages="1910-1918",
            ),
            Ref(
                title=(
                    "A massively parallel tensor contraction framework"
                    " for coupled-cluster computations"
                ),
                authors=[
                    Author("E. Solomonik"),
                    Author("D. Matthews"),
                    Author("J. R. Hammond"),
                    Author("J. F. Stanton"),
                    Author("J. Demmel"),
                ],
                year=2014,
                url="https://doi.org/10.1016/j.jpdc.2014.06.002",
                journal="J. Parallel Distrib. Comput.",
                volume=74,
                number=12,
                pages="3176-3190",
            ),
        ]

    @property
    def ai_disclosure(self) -> str:
        return (
            "Generative AI was used to assist in debugging the benchmark. This"
            " statement was written by hand."
        )

    @property
    def motivation(self) -> str:
        return ""

    @property
    def datasets(self) -> list[CCSDDataset]:
        return [
            CCSDDataset(
                name="ccsd_small",
                pretty_name="CCSD Small",
                description="no=4, nv=6 — matches the C++ CTF reference.",
                suites=[],
                no=4,
                nv=6,
            ),
            CCSDDataset(
                name="ccsd_medium",
                pretty_name="CCSD Medium",
                description="no=8, nv=12.",
                suites=[],
                no=8,
                nv=12,
            ),
            CCSDDataset(
                name="ccsd_large",
                pretty_name="CCSD Large",
                description="no=16, nv=24.",
                suites=[],
                no=16,
                nv=24,
            ),
        ]

    def generate(self, dataset: CCSDDataset):
        return DataInstance(inputs=make_ccsd_inputs(dataset.no, dataset.nv), meta={})


class CCSD(Benchmark):
    @property
    def name(self) -> str:
        return "ccsd"

    @property
    def pretty_name(self) -> str:
        return "Coupled Cluster Singles and Doubles (CCSD)"

    @property
    def description(self) -> str:
        return (
            "Computes T1 and T2 amplitudes for CCSD. Ported from the CTF (Cyclops"
            " Tensor Framework) CCSD reference implementation"
        )

    @property
    def suites(self) -> list[str]:
        return []

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def authors(self) -> list[Contributor]:
        return [Contributor("Tarun Devi", "tdevi3@gatech.edu")]

    @property
    def references(self) -> list[Ref]:
        return [
            Ref(
                title=(
                    "A full coupled-cluster singles and doubles model:"
                    " The inclusion of disconnected triples"
                ),
                authors=[Author("G. D. Purvis"), Author("R. J. Bartlett")],
                year=1982,
                url="https://doi.org/10.1063/1.443164",
                journal="J. Chem. Phys.",
                volume=76,
                number=4,
                pages="1910-1918",
            ),
            Ref(
                title=(
                    "A massively parallel tensor contraction framework"
                    " for coupled-cluster computations"
                ),
                authors=[
                    Author("E. Solomonik"),
                    Author("D. Matthews"),
                    Author("J. R. Hammond"),
                    Author("J. F. Stanton"),
                    Author("J. Demmel"),
                ],
                year=2014,
                url="https://doi.org/10.1016/j.jpdc.2014.06.002",
                journal="J. Parallel Distrib. Comput.",
                volume=74,
                number=12,
                pages="3176-3190",
            ),
        ]

    @property
    def ai_disclosure(self) -> str:
        return (
            "Generative AI was used to assist in debugging the benchmark. This"
            " statement was written by hand."
        )

    @property
    def motivation(self) -> str:
        return (
            "Coupled cluster theory is one of the most accurate and widely "
            "used methods in quantum chemistry for computing ground-state "
            "energies of molecular systems. The two-electron integral tensors "
            "(Vabef, Vabij, etc.) are antisymmetric, which means roughly 3/4 "
            "of entries are redundant. Exploiting this antisymmetry reduces "
            "both storage and compute by up to 8x for 4-index tensors. "
            "Here antisymmetry means swapping an antisymmetric index pair flips "
            "the sign, for example T[a,b,i,j] = -T[b,a,i,j] and "
            "T[a,b,i,j] = -T[a,b,j,i]."
        )

    @property
    def generators(self):
        return [CCSDGenerator()]

    def benchmark(self, data, meta):
        (
            Vme,  # (no, nv)
            Vae,  # (nv, nv)
            Vmi,  # (no, no)
            Vai,  # (nv, no)
            Vmnef,  # (no, no, nv, nv)
            Vabef,  # (nv, nv, nv, nv)
            Vabij,  # (nv, nv, no, no)
            Vabei,  # (nv, nv, nv, no)
            Vmnij,  # (no, no, no, no)
            Vmnei,  # (no, no, nv, no)
            Vamei,  # (nv, no, nv, no)
            Vamij,  # (nv, no, no, no)
            Vanef,  # (nv, no, nv, nv)
            Vmnfi,  # (no, no, nv, no)
            Vamef,  # (nv, no, nv, nv)
            Vaeim,  # (nv, nv, no, no)
            T1,  # (nv, no)
            T2,  # (nv, nv, no, no)
            D1,
            D2,
        ) = data

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
            - xp.einsum(
                "Wamei[a,m,e,i] += Wmnei[m,n,e,i] * T1[a,n]", Wmnei=Wmnei, T1=T1
            )
            + xp.einsum(
                "Wamei[a,m,e,i] += Vamef[a,m,e,f] * T1[f,i]", Vamef=Vamef, T1=T1
            )
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
            + xp.einsum(
                "T1_new[a,i] += Vaeim[a,e,i,m] * Fme[m,e]", Vaeim=Vaeim, Fme=Fme
            )
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
        R1_Z = xp.einsum(
            "T2_new[a,b,i,j] += Vabei[a,b,e,i] * T1[e,j]", Vabei=Vabei, T1=T1
        )
        R2_Z = xp.einsum(
            "T2_new[a,b,i,j] += Wamei[a,m,e,i] * T2[e,b,m,j]", Wamei=Wamei, T2=T2
        )
        R3_Z = xp.einsum(
            "T2_new[a,b,i,j] += Wamij[a,m,i,j] * T1[b,m]", Wamij=Wamij, T1=T1
        )
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

        return (T1_final, T2_final)
