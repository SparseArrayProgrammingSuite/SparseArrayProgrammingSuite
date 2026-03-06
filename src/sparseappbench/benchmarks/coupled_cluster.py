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

    T21 = xp.copy(T2)
    T21 += xp.einsum("T21[a,b,i,j] += 0.5 * T1[a,i] * T1[b,j]", T1=T1)

    # Fme intermediate
    Fme = xp.einsum("Fme[m,e] = Vme[m,e]", Vme=Vme)
    Fme += xp.einsum("Fme[m,e] += Vmnef[m,n,e,f] * T1[f,n]", Vmnef=Vmnef, T1=T1)

    # Fae intermediate
    Fae = xp.einsum("Fae[a,e] = Vae[a,e]", Vae=Vae)
    Fae -= xp.einsum("Fae[a,e] += Fme[m,e] * T1[a,m]", Fme=Fme, T1=T1)
    Fae -= xp.einsum(
        "Fae[a,e] += 0.5 * Vmnef[m,n,e,f] * T2[a,f,m,n]", Vmnef=Vmnef, T2=T2
    )
    Fae += xp.einsum("Fae[a,e] += Vanef[a,n,e,f] * T1[f,n]", Vanef=Vanef, T1=T1)

    # Fmi intermediate
    Fmi = xp.einsum("Fmi[m,i] = Vmi[m,i]", Vmi=Vmi)
    Fmi += xp.einsum("Fmi[m,i] += Fme[m,e] * T1[e,i]", Fme=Fme, T1=T1)
    Fmi += xp.einsum(
        "Fmi[m,i] += 0.5 * Vmnef[m,n,e,f] * T2[e,f,i,n]", Vmnef=Vmnef, T2=T2
    )
    Fmi += xp.einsum("Fmi[m,i] += Vmnfi[m,n,f,i] * T1[f,n]", Vmnfi=Vmnfi, T1=T1)

    # Wmnei intermediate
    Wmnei = xp.einsum("Wmnei[m,n,e,i] = Vmnei[m,n,e,i]", Vmnei=Vmnei)
    Wmnei += xp.einsum(
        "Wmnei[m,n,e,i] += Vmnei[m,n,e,i]", Vmnei=Vmnei
    )  # Replicating the double addition from C++
    Wmnei += xp.einsum("Wmnei[m,n,e,i] += Vmnef[m,n,e,f] * T1[f,i]", Vmnef=Vmnef, T1=T1)

    # Wmnij intermediate
    Wmnij = xp.einsum("Wmnij[m,n,i,j] = Vmnij[m,n,i,j]", Vmnij=Vmnij)
    Wmnij -= xp.einsum("Wmnij[m,n,i,j] += Wmnei[m,n,e,i] * T1[e,j]", Wmnei=Wmnei, T1=T1)
    Wmnij += xp.einsum(
        "Wmnij[m,n,i,j] += Vmnef[m,n,e,f] * T21[e,f,i,j]", Vmnef=Vmnef, T21=T21
    )

    # Wamei intermediate
    Wamei = xp.einsum("Wamei[a,m,e,i] = Vamei[a,m,e,i]", Vamei=Vamei)
    Wamei -= xp.einsum("Wamei[a,m,e,i] += Wmnei[m,n,e,i] * T1[a,n]", Wmnei=Wmnei, T1=T1)
    Wamei += xp.einsum("Wamei[a,m,e,i] += Vamef[a,m,e,f] * T1[f,i]", Vamef=Vamef, T1=T1)
    Wamei += xp.einsum(
        "Wamei[a,m,e,i] += 0.5 * Vmnef[m,n,e,f] * T2[a,f,i,n]", Vmnef=Vmnef, T2=T2
    )

    # Wamij intermediate
    Wamij = xp.einsum("Wamij[a,m,i,j] = Vamij[a,m,i,j]", Vamij=Vamij)
    Wamij += xp.einsum("Wamij[a,m,i,j] += Wamei[a,m,e,i] * T1[e,j]", Wamei=Wamei, T1=T1)
    Wamij += xp.einsum(
        "Wamij[a,m,i,j] += Vamef[a,m,e,f] * T2[e,f,i,j]", Vamef=Vamef, T2=T2
    )

    # Zai (T1 update)
    T1_new = xp.einsum("T1_new[a,i] = Vai[a,i]", Vai=Vai)
    T1_new -= xp.einsum("T1_new[a,i] += Fmi[m,i] * T1[a,m]", Fmi=Fmi, T1=T1)
    T1_new += xp.einsum("T1_new[a,i] += Vae[a,e] * T1[e,i]", Vae=Vae, T1=T1)
    T1_new += xp.einsum("T1_new[a,i] += Vamei[a,m,e,i] * T1[e,m]", Vamei=Vamei, T1=T1)
    T1_new += xp.einsum(
        "T1_new[a,i] += Vaeim[a,e,i,m] * Fme[m,e]", Vaeim=Vaeim, Fme=Fme
    )
    T1_new += xp.einsum(
        "T1_new[a,i] += 0.5 * Vamef[a,m,e,f] * T21[e,f,i,m]", Vamef=Vamef, T21=T21
    )
    T1_new -= xp.einsum(
        "T1_new[a,i] += 0.5 * Wmnei[m,n,e,i] * T21[e,a,m,n]", Wmnei=Wmnei, T21=T21
    )

    # Zabij (T2 update)
    T2_new = xp.einsum("T2_new[a,b,i,j] = Vabij[a,b,i,j]", Vabij=Vabij)
    T2_new += xp.einsum(
        "T2_new[a,b,i,j] += Vabei[a,b,e,i] * T1[e,j]", Vabei=Vabei, T1=T1
    )
    T2_new += xp.einsum(
        "T2_new[a,b,i,j] += Wamei[a,m,e,i] * T2[e,b,m,j]", Wamei=Wamei, T2=T2
    )
    T2_new -= xp.einsum(
        "T2_new[a,b,i,j] += Wamij[a,m,i,j] * T1[b,m]", Wamij=Wamij, T1=T1
    )
    T2_new += xp.einsum("T2_new[a,b,i,j] += Fae[a,e] * T2[e,b,i,j]", Fae=Fae, T2=T2)
    T2_new -= xp.einsum("T2_new[a,b,i,j] += Fmi[m,i] * T2[a,b,m,j]", Fmi=Fmi, T2=T2)
    T2_new += xp.einsum(
        "T2_new[a,b,i,j] += 0.5 * Vabef[a,b,e,f] * T21[e,f,i,j]", Vabef=Vabef, T21=T21
    )
    T2_new += xp.einsum(
        "T2_new[a,b,i,j] += 0.5 * Wmnij[m,n,i,j] * T21[a,b,m,n]", Wmnij=Wmnij, T21=T21
    )

    # Apply denominators (D1, D2)
    T1_final = T1_new / D1
    T2_final = T2_new / D2

    T1_out, T2_out = xp.compute((T1_final, T2_final))

    return xp.to_benchmark(T1_out), xp.to_benchmark(T2_out)
