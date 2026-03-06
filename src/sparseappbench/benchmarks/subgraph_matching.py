def benchmark_subgraph_matching(xp, expr: str, sp_mats: dict):
    for key, val in sp_mats.items():
        sp_mats[key] = xp.from_benchmark(val)
    count = xp.einsum(expr, **sp_mats)
    return count