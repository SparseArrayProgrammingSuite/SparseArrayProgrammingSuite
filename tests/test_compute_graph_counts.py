import json
from itertools import combinations
from types import SimpleNamespace

import numpy as np
from scipy import sparse

from binsparse.conversions import from_scipy

from saps.benchmark import DataInstance
from scripts import compute_graph_counts as counts


def test_counting_script_replaces_old_results_and_resumes(monkeypatch, tmp_path):
    A = np.triu(np.ones((4, 4), dtype=np.int64), k=1)
    calls = []

    def generate(_):
        calls.append("example")
        return DataInstance(inputs=[from_scipy(sparse.coo_array(A))], meta={})

    datasets = [(SimpleNamespace(generate=generate), SimpleNamespace(name="example"))]
    monkeypatch.setattr(counts, "_datasets", lambda: iter(datasets))
    monkeypatch.setattr(counts, "declared_edges", lambda _: 6)
    monkeypatch.setattr(counts, "fast_triangle_sum", counts.triangle_sum)
    monkeypatch.setattr(counts, "fast_four_clique_sum", counts.four_clique_sum)
    output = tmp_path / "counts.json"
    output.write_text(
        json.dumps({"example": {"triangle_sum": 24, "four_clique_sum": 24}})
    )
    monkeypatch.setattr(
        "sys.argv", ["compute_graph_counts.py", "--output", str(output)]
    )
    counts.main()
    result = json.loads(output.read_text())["example"]
    assert result["triangles"] == 4
    assert result["four_cliques"] == 1
    assert result["nnz"] == 6
    assert result["adjacency"] == "simple_undirected_upper"
    assert calls == ["example"]
    counts.main()
    assert calls == ["example"]


def test_upper_triangular_counts_match_vertex_subsets():
    rng = np.random.default_rng(0)
    for n in range(3, 10):
        A = rng.integers(0, 2, (n, n))
        A = np.maximum(A, A.T)
        np.fill_diagonal(A, 0)
        for size, counter in [(3, counts.triangle_sum), (4, counts.four_clique_sum)]:
            expected = sum(
                all(A[i, j] for i, j in combinations(vertices, 2))
                for vertices in combinations(range(n), size)
            )
            for order in (np.arange(n), rng.permutation(n)):
                upper = sparse.csr_array(np.triu(A[order][:, order], k=1))
                assert counter(upper) == expected
