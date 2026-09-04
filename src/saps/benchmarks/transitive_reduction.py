import numpy as np

from binsparse.conversions import from_numpy, to_numpy

from saps.benchmark import (
    Author,
    Benchmark,
    Contributor,
    DataInstance,
    Dataset,
    Generator,
    Ref,
)


class TransitiveReductionDataset(Dataset):
    def __init__(self, name, edges, expected_edges, suites=None):
        self._name = name
        self.edges = edges
        self.expected_edges = expected_edges
        self._suites = suites or []

    @property
    def name(self) -> str:
        return self._name

    @property
    def pretty_name(self) -> str:
        return f"Transitive Reduction {self._name}"

    @property
    def description(self) -> str:
        return "Small overlap graph with expected transitive reduction output."

    @property
    def suites(self) -> list[str]:
        return self._suites

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"


class TransitiveReductionTestGenerator(Generator[TransitiveReductionDataset]):
    @property
    def cacheable(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return "transitive_reduction_test_inputs"

    @property
    def pretty_name(self) -> str:
        return "Transitive Reduction Test Inputs"

    @property
    def description(self) -> str:
        return "Small test graphs with expected reduced edges."

    @property
    def suites(self) -> list[str]:
        return ["test", "trace"]

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def authors(self) -> list[Contributor]:
        return []

    @property
    def references(self) -> list[Ref]:
        return []

    @property
    def ai_disclosure(self) -> str:
        return "No generative AI was used to construct the benchmark function."

    @property
    def motivation(self) -> str:
        return ""

    @property
    def datasets(self) -> list[TransitiveReductionDataset]:
        return [
            TransitiveReductionDataset(
                "remove_long_direct_edge",
                edges=[(0, 1, 10.0), (1, 2, 10.0), (0, 2, 30.0)],
                expected_edges=[(0, 1, 10.0), (1, 2, 10.0)],
                suites=["test", "trace"],
            ),
            TransitiveReductionDataset(
                "keep_short_direct_edge",
                edges=[(0, 1, 10.0), (1, 2, 10.0), (0, 2, 15.0)],
                expected_edges=[(0, 1, 10.0), (1, 2, 10.0), (0, 2, 15.0)],
                suites=["test", "trace"],
            ),
            TransitiveReductionDataset(
                "keep_when_indirect_is_long",
                edges=[(0, 1, 40.0), (1, 2, 40.0), (0, 2, 30.0)],
                expected_edges=[(0, 1, 40.0), (1, 2, 40.0), (0, 2, 30.0)],
                suites=["test", "trace"],
            ),
            TransitiveReductionDataset(
                "remove_equal_direct_edge",
                edges=[(0, 1, 10.0), (1, 2, 10.0), (0, 2, 20.0)],
                expected_edges=[(0, 1, 10.0), (1, 2, 10.0)],
                suites=["test", "trace"],
            ),
        ]

    def generate(self, dataset: TransitiveReductionDataset):
        R = np.full((3, 3), np.inf)
        for i, j, value in dataset.edges:
            R[i, j] = value
        expected = np.full((3, 3), np.inf)
        for i, j, value in dataset.expected_edges:
            expected[i, j] = value
        return DataInstance(
            inputs=[from_numpy(R)],
            meta={"x": 1, "max_iters": 5},
            ref_outputs=[from_numpy(expected)],
        )


class TransitiveReductionBenchmark(Benchmark):
    @property
    def name(self) -> str:
        return "transitive_reduction"

    @property
    def pretty_name(self) -> str:
        return "diBELLA Transitive Reduction Algorithm"

    @property
    def description(self) -> str:
        return (
            "Iterative transitive reduction on a sparse overlap graph, following "
            "the diBELLA reduction step."
        )

    @property
    def suites(self) -> list[str]:
        return []

    @property
    def concepts(self) -> str:
        return """
<ccs2012>
<concept>
<concept_id>10002950.10003705</concept_id>
<concept_desc>Mathematics of computing~Mathematical software</concept_desc>
<concept_significance>500</concept_significance>
</concept>
<concept>
<concept_id>10002950.10003705.10011686</concept_id>
<concept_desc>Mathematics of computing~Mathematical software performance</concept_desc>
<concept_significance>500</concept_significance>
</concept>
<concept>
<concept_id>10002950.10003624.10003633.10010917</concept_id>
<concept_desc>Mathematics of computing~Graph algorithms</concept_desc>
<concept_significance>500</concept_significance>
</concept>
<concept>
<concept_id>10002950.10003624.10003633.10003640</concept_id>
<concept_desc>Mathematics of computing~Paths and connectivity problems</concept_desc>
<concept_significance>500</concept_significance>
</concept>
</ccs2012>
"""

    @property
    def authors(self) -> list[Contributor]:
        return [Contributor("Jaehun Baek", "jbaek90@gatech.edu")]

    @property
    def references(self) -> list[Ref]:
        return [
            Ref(
                title=(
                    "Parallel String Graph Construction and Transitive Reduction "
                    "for De Novo Genome Assembly"
                ),
                authors=[
                    Author("Giulia Guidi"),
                    Author("Oguz Selvitopi"),
                    Author("Marquita Ellis"),
                    Author("Leonid Oliker"),
                    Author("Katherine Yelick"),
                    Author("Aydin Buluc"),
                ],
                conference=(
                    "IEEE International Parallel and Distributed Processing "
                    "Symposium (IPDPS)"
                ),
                year=2021,
                pages="517-526",
                doi="10.1109/IPDPS49936.2021.00060",
            )
        ]

    @property
    def ai_disclosure(self) -> str:
        return (
            "No generative AI was used to construct the benchmark function. This "
            "statement was written by hand."
        )

    @property
    def motivation(self) -> str:
        return (
            "This benchmark implements the iterative transitive reduction step from "
            "the diBELLA 2D paper. The overlap graph R is a sparse matrix where "
            "R[i, j] represents the suffix length of an overlap between read i and "
            "read j. The reduction is implemented as sparse (min, +) semiring "
            "SpGEMM to find shortest 2-hop paths."
        )

    @property
    def generators(self):
        return [TransitiveReductionTestGenerator()]

    def benchmark(self, xp, data, meta):
        R = data[0]
        x = meta.get("x", 1)
        max_iters = meta.get("max_iters", 10)

        R_nnz_prev_tensor = xp.sum(np.inf != R)
        R_nnz_prev = R_nnz_prev_tensor[()]

        for _i in range(max_iters):
            N = xp.einsum("N[i, j] min= R[i, k] + R[k, j]", R=R)

            R_for_max = xp.where(np.inf == R, -1.0, R)
            v = xp.max(R_for_max, axis=1)
            v = v + x

            v_expanded = xp.expand_dims(v, axis=1)
            M = v_expanded

            is_transitive = M >= N
            common_sparsity = xp.logical_and(np.inf != R, np.inf != N)
            edges_to_remove = xp.logical_and(common_sparsity, is_transitive)

            R = xp.where(edges_to_remove, np.inf, R)
            R_nnz_new_tensor = xp.sum(np.inf != R)
            R_nnz_new = R_nnz_new_tensor[()]

            if R_nnz_new == R_nnz_prev:
                break

            R_nnz_prev = R_nnz_new

        return [R]

    def check(self, param):
        super().check(param)
        expected = to_numpy(self._ref_outputs[0])
        actual = to_numpy(self._output[0])
        assert np.array_equal(actual, expected)
