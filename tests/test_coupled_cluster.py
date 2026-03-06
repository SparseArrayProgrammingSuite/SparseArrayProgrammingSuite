import numpy as np

from sparseappbench.benchmarks.coupled_cluster import benchmark_ccsd
from sparseappbench.binsparse_format import BinsparseFormat
from sparseappbench.frameworks.numpy_framework import NumpyFramework


def test_ccsd_correctness():
    xp = NumpyFramework()
    no = 3
    nv = 4

    # Helper to create random binsparse using C++ logic
    def ctf_rand(shape, tensor_id, multiplier=16):
        size = np.prod(shape)
        indices = np.arange(size)
        # Formula from C++: ((indices[j]*multiplier+i)%13077)/13077. -.5
        values = ((indices * multiplier + tensor_id) % 13077) / 13077.0 - 0.5
        arr = values.reshape(shape)
        return BinsparseFormat.from_numpy(arr), arr

    # Generate Integrals (multiplier 16)
    # IDs: aa=0, ii=1, ab=2, ai=3, ia=4, ij=5
    # abcd=6, abci=7, aibc=8, aibj=9, abij=10, ijab=11, aijk=12, ijak=13, ijkl=14

    _, aa = ctf_rand((nv,), 0)
    _, ii = ctf_rand((no,), 1)

    Vae_b, Vae = ctf_rand((nv, nv), 2)  # ab
    Vai_b, Vai = ctf_rand((nv, no), 3)  # ai
    Vme_b, Vme = ctf_rand((no, nv), 4)  # ia
    Vmi_b, Vmi = ctf_rand((no, no), 5)  # ij

    Vabef_b, Vabef = ctf_rand((nv, nv, nv, nv), 6)  # abcd
    Vabei_b, Vabei = ctf_rand((nv, nv, nv, no), 7)  # abci
    Vanef_b, Vanef = ctf_rand((nv, no, nv, nv), 8)  # aibc
    Vamef_b, Vamef = Vanef_b, Vanef  # aibc reused

    Vamei_b, Vamei = ctf_rand((nv, no, nv, no), 9)  # aibj

    Vabij_b, Vabij = ctf_rand((nv, nv, no, no), 10)  # abij
    Vaeim_b, Vaeim = Vabij_b, Vabij  # abij reused

    Vmnef_b, Vmnef = ctf_rand((no, no, nv, nv), 11)  # ijab

    Vamij_b, Vamij = ctf_rand((nv, no, no, no), 12)  # aijk

    Vmnei_b, Vmnei = ctf_rand((no, no, nv, no), 13)  # ijak
    Vmnfi_b, Vmnfi = Vmnei_b, Vmnei  # ijak reused

    Vmnij_b, Vmnij = ctf_rand((no, no, no, no), 14)  # ijkl

    # Generate Amplitudes (multiplier 13)
    # IDs: ai=0, abij=1
    T1_b, T1 = ctf_rand((nv, no), 0, multiplier=13)
    T2_b, T2 = ctf_rand((nv, nv, no, no), 1, multiplier=13)

    # Construct Denominators
    # D1[a, i] = ii[i] - aa[a]
    D1 = ii[None, :] - aa[:, None]
    D1_b = BinsparseFormat.from_numpy(D1)

    # D2[a, b, i, j] = ii[i] + ii[j] - aa[a] - aa[b]
    D2 = (
        ii.reshape(1, 1, no, 1)
        + ii.reshape(1, 1, 1, no)
        - aa.reshape(nv, 1, 1, 1)
        - aa.reshape(1, nv, 1, 1)
    )
    D2_b = BinsparseFormat.from_numpy(D2)

    # Run benchmark
    T1_out_b, T2_out_b = benchmark_ccsd(
        xp,
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
        D1_b,
        D2_b,
    )

    T1_out = xp.from_benchmark(T1_out_b)
    T2_out = xp.from_benchmark(T2_out_b)

    # Reference implementation (copy of logic using standard numpy einsum)
    T21 = T2.copy()
    T21 += np.einsum("ai,bj->abij", T1, T1) * 0.5

    Fme = np.einsum("me->me", Vme)
    Fme += np.einsum("mnef,fn->me", Vmnef, T1)

    Fae = np.einsum("ae->ae", Vae)
    Fae -= np.einsum("me,am->ae", Fme, T1)
    Fae -= 0.5 * np.einsum("mnef,afmn->ae", Vmnef, T2)
    Fae += np.einsum("anef,fn->ae", Vanef, T1)

    Fmi = np.einsum("mi->mi", Vmi)
    Fmi += np.einsum("me,ei->mi", Fme, T1)
    Fmi += 0.5 * np.einsum("mnef,efin->mi", Vmnef, T2)
    Fmi += np.einsum("mnfi,fn->mi", Vmnfi, T1)

    Wmnei = np.einsum("mnei->mnei", Vmnei)
    Wmnei += np.einsum("mnei->mnei", Vmnei)
    Wmnei += np.einsum("mnef,fi->mnei", Vmnef, T1)

    Wmnij = np.einsum("mnij->mnij", Vmnij)
    Wmnij -= np.einsum("mnei,ej->mnij", Wmnei, T1)
    Wmnij += np.einsum("mnef,efij->mnij", Vmnef, T21)

    Wamei = np.einsum("amei->amei", Vamei)
    Wamei -= np.einsum("mnei,an->amei", Wmnei, T1)
    Wamei += np.einsum("amef,fi->amei", Vamef, T1)
    Wamei += 0.5 * np.einsum("mnef,afin->amei", Vmnef, T2)

    Wamij = np.einsum("amij->amij", Vamij)
    Wamij += np.einsum("amei,ej->amij", Wamei, T1)
    Wamij += np.einsum("amef,efij->amij", Vamef, T2)

    T1_new = np.einsum("ai->ai", Vai)
    T1_new -= np.einsum("mi,am->ai", Fmi, T1)
    T1_new += np.einsum("ae,ei->ai", Vae, T1)
    T1_new += np.einsum("amei,em->ai", Vamei, T1)
    T1_new += np.einsum("aeim,me->ai", Vaeim, Fme)
    T1_new += 0.5 * np.einsum("amef,efim->ai", Vamef, T21)
    T1_new -= 0.5 * np.einsum("mnei,eamn->ai", Wmnei, T21)

    T2_new = np.einsum("abij->abij", Vabij)
    T2_new += np.einsum("abei,ej->abij", Vabei, T1)
    T2_new += np.einsum("amei,ebmj->abij", Wamei, T2)
    T2_new -= np.einsum("amij,bm->abij", Wamij, T1)
    T2_new += np.einsum("ae,ebij->abij", Fae, T2)
    T2_new -= np.einsum("mi,abmj->abij", Fmi, T2)
    T2_new += 0.5 * np.einsum("abef,efij->abij", Vabef, T21)
    T2_new += 0.5 * np.einsum("mnij,abmn->abij", Wmnij, T21)

    T1_ref = T1_new / D1
    T2_ref = T2_new / D2

    assert np.allclose(T1_out, T1_ref)
    assert np.allclose(T2_out, T2_ref)
