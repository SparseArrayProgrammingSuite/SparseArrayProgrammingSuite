import os
from typing import Any, cast

import numpy as np

from saps.benchmark import (
    Author,
    Benchmark,
    Contributor,
    DataInstance,
    Dataset,
    Generator,
    Ref,
)
from saps_framework import BinsparseFormat


class GCNDataset(Dataset):
    def __init__(
        self,
        name: str,
        description: str = "",
        source_name: str | None = None,
        feature_dim: int = 16,
        hidden_dim: int = 8,
        out_dim: int = 1,
        suites: list[str] | None = None,
        adjacency: np.ndarray | None = None,
        features: np.ndarray | None = None,
        weights1: np.ndarray | None = None,
        bias1: np.ndarray | None = None,
        weights2: np.ndarray | None = None,
        bias2: np.ndarray | None = None,
        expected: np.ndarray | None = None,
    ):
        self._suites = suites or []
        self.dataset_name = name
        self.dataset_description = description
        self.source_name = source_name if source_name is not None else name
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.adjacency = adjacency
        self.features = features
        self.weights1 = weights1
        self.bias1 = bias1
        self.weights2 = weights2
        self.bias2 = bias2
        self.expected = expected

    @property
    def name(self) -> str:
        return self.dataset_name

    @property
    def pretty_name(self) -> str:
        return f"GCN {self.dataset_name}"

    @property
    def description(self) -> str:
        return self.dataset_description or f"SuiteSparse matrix {self.source_name}."

    @property
    def suites(self) -> list[str]:
        return self._suites

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def metadata(self) -> dict[str, Any]:
        data = super().metadata
        data["source_name"] = self.source_name
        data["feature_dim"] = self.feature_dim
        data["hidden_dim"] = self.hidden_dim
        data["out_dim"] = self.out_dim
        return data


def gcn_reference_np(adjacency, features, weights1, bias1, weights2, bias2):
    h1 = adjacency @ features
    h1 = h1 @ weights1 + bias1
    h1 = np.maximum(h1, 0)
    h2 = adjacency @ h1
    return h2 @ weights2 + bias2


class GCNTestGenerator(Generator[GCNDataset]):
    @property
    def name(self) -> str:
        return "gcn_test_inputs"

    @property
    def pretty_name(self) -> str:
        return "GCN Test Input Generator"

    @property
    def description(self) -> str:
        return "Small inlined GCN forward-pass examples."

    @property
    def suites(self) -> list[str]:
        return ["test"]

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def authors(self) -> list[Contributor]:
        return [Contributor("Tarun Devi", "tdevi3@gatech.edu")]

    @property
    def references(self) -> list[Ref]:
        return GCNBenchmark().references

    @property
    def ai_disclosure(self) -> str:
        return GCNBenchmark().ai_disclosure

    @property
    def motivation(self) -> str:
        return "Uses small graph examples to verify the GCN forward pass."

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def datasets(self) -> list[GCNDataset]:
        return [
            GCNDataset(
                "test_gcn_3node",
                suites=["test"],
                adjacency=np.array(
                    [[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32
                ),
                features=np.array(
                    [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32
                ),
                weights1=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
                bias1=np.array([0.0, 0.0], dtype=np.float32),
                weights2=np.array([[1.0], [1.0]], dtype=np.float32),
                bias2=np.array([0.0], dtype=np.float32),
            ),
            GCNDataset(
                "test_gcn_simple_2node",
                suites=["test"],
                adjacency=np.array([[0, 1], [1, 0]], dtype=np.float32),
                features=np.array([[1.0], [2.0]], dtype=np.float32),
                weights1=np.array([[2.0]], dtype=np.float32),
                bias1=np.array([0.0], dtype=np.float32),
                weights2=np.array([[3.0]], dtype=np.float32),
                bias2=np.array([0.0], dtype=np.float32),
                expected=np.array([[6.0], [12.0]], dtype=np.float32),
            ),
            GCNDataset(
                "test_gcn_simple_3node_line",
                suites=["test"],
                adjacency=np.array(
                    [[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32
                ),
                features=np.array([[1.0], [0.0], [1.0]], dtype=np.float32),
                weights1=np.array([[1.0]], dtype=np.float32),
                bias1=np.array([0.0], dtype=np.float32),
                weights2=np.array([[1.0]], dtype=np.float32),
                bias2=np.array([0.0], dtype=np.float32),
                expected=np.array([[2.0], [0.0], [2.0]], dtype=np.float32),
            ),
            GCNDataset(
                "test_gcn_with_relu_activation",
                suites=["test"],
                adjacency=np.array([[0, 1], [1, 0]], dtype=np.float32),
                features=np.array([[1.0], [-1.0]], dtype=np.float32),
                weights1=np.array([[1.0]], dtype=np.float32),
                bias1=np.array([0.0], dtype=np.float32),
                weights2=np.array([[2.0]], dtype=np.float32),
                bias2=np.array([0.0], dtype=np.float32),
                expected=np.array([[2.0], [0.0]], dtype=np.float32),
            ),
        ]

    def generate(self, dataset: GCNDataset):
        required = (
            dataset.adjacency,
            dataset.features,
            dataset.weights1,
            dataset.bias1,
            dataset.weights2,
            dataset.bias2,
        )
        if any(item is None for item in required):
            raise ValueError("GCN test datasets must define all input arrays.")

        arrays = tuple(cast(np.ndarray, item) for item in required)
        expected = dataset.expected
        expected = gcn_reference_np(*arrays) if expected is None else expected

        inputs = [BinsparseFormat.from_numpy(item) for item in arrays]
        return DataInstance(
            inputs=inputs,
            meta={},
            ref_outputs=[BinsparseFormat.from_numpy(expected)],
            ref_meta={"rtol": 1e-10},
        )

class GCNGenerator(Generator[GCNDataset]):
    @property
    def name(self) -> str:
        return "gcn_weights"

    @property
    def pretty_name(self) -> str:
        return "Graph Convolutional Network Weights"

    @property
    def description(self) -> str:
        return "Generates random weights for a 2-layer Graph Convolutional Network."

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
                title="Scorch: A Library for Sparse Deep Learning",
                authors=[
                    Author("Bobby Yan"),
                    Author("Alexander J. Root"),
                    Author("Trevor Gale"),
                    Author("David Broman"),
                    Author("Fredrik Kjolstad"),
                ],
                journal="Arxiv",
                volume="arXiv:2405.16883",
                year=2024,
                url="https://anonymous.4open.science/r/scorch/README.md",
            ),
            Ref(
                title="Open Graph Benchmark: Datasets for Machine Learning on Graphs",
                authors=[
                    Author("Weihua Hu"),
                    Author("Matthias Fey"),
                    Author("Marinka Zitnik"),
                    Author("Yuxiao Dong"),
                    Author("Hongyu Ren"),
                    Author("Bowen Liu"),
                    Author("Michele Catasta"),
                    Author("Jure Leskovec"),
                ],
                journal="Arxiv",
                volume="arXiv:2005.00687",
                year=2020,
                url="https://arxiv.org/abs/2005.00687",
            ),
        ]

    @property
    def ai_disclosure(self) -> str:
        return (
            "No generative AI was used to construct the benchmark function itself."
            " Generative AI might have been used to construct tests. This statement was"
            " written by hand."
        )

    @property
    def motivation(self) -> str:
        return (
            "Graphs are widely used for abstracting systems of interacting objects,"
            " such as social networks (Easley et al., 2010), knowledge graphs (Nickel"
            " et al., 2015), molecular graphs (Wu et al., 2018), and biological"
            " networks (Barabasi & Oltvai, 2004), as well as for modeling 3D objects"
            " (Simonovsky & Komodakis, 2017), manifolds (Bronstein et al., 2017), and"
            " source code (Allamanis et al., 2017). Machine learning (ML), especially"
            " deep learning, on graphs is an emerging field (Hamilton et al., 2017b;"
            " Bronstein et al., 2017). W. Hu et al. To represent a graph, an adjaceny"
            " matrix is used, which is inherently sparse."
        )

    @property
    def datasets(self) -> list[GCNDataset]:
        return [
            GCNDataset(
                "dg_gcn_social_1",
                "Small social network graph.",
                "karate",
                feature_dim=16,
                hidden_dim=8,
                out_dim=1,
            ),
            GCNDataset(
                "dg_gcn_social_2",
                "Medium social network graph.",
                "dolphins",
                feature_dim=16,
                hidden_dim=8,
                out_dim=1,
            ),
            GCNDataset(
                "dg_gcn_social_3",
                "Larger social network graph.",
                "ca-GrQc",
                feature_dim=8,
                hidden_dim=16,
                out_dim=1,
            ),
            GCNDataset(
                "dg_gcn_road_1",
                "Small road network graph.",
                "chesapeake",
                feature_dim=8,
                hidden_dim=4,
                out_dim=1,
            ),
            GCNDataset(
                "dg_gcn_road_2",
                "Medium road network graph.",
                "road_central",
                feature_dim=4,
                hidden_dim=8,
                out_dim=1,
            ),
            GCNDataset(
                "dg_gcn_molecular_1",
                "Small molecular graph. - Email network.",
                "email",
                feature_dim=4,
                hidden_dim=8,
                out_dim=1,
            ),
            GCNDataset(
                "dg_gcn_molecular_2",
                "Medium molecular graph - PDDB protein structure.",
                "Chebyshev3",
                feature_dim=6,
                hidden_dim=12,
                out_dim=1,
            ),
            GCNDataset(
                "dg_gcn_citation_1",
                "Large citation network graph (AIDS-like size).",
                "ca-HepPh",
                feature_dim=16,
                hidden_dim=32,
                out_dim=1,
            ),
            GCNDataset(
                "dg_gcn_large_2",
                "Very large road network.",
                "road_usa",
                feature_dim=16,
                hidden_dim=32,
                out_dim=1,
            ),
            GCNDataset(
                "dg_gcn_bcsstk01",
                "Original small structural engineering matrix"
                " (for backward compatibility).",
                "bcsstk01",
                feature_dim=16,
                hidden_dim=8,
                out_dim=1,
            ),
        ]

    def generate(self, dataset: GCNDataset):
        from scipy.io import mmread

        import ssgetpy

        feature_dim = dataset.feature_dim
        hidden_dim = dataset.hidden_dim
        out_dim = dataset.out_dim

        source = dataset.source_name
        matrices = ssgetpy.search(name=source)
        if not matrices:
            raise ValueError(f"No matrix found with name '{source}'")
        matrix = matrices[0]
        (path, archive) = matrix.download(extract=True)
        matrix_path = os.path.join(path, matrix.name + ".mtx")
        if matrix_path and os.path.exists(matrix_path):
            A = mmread(matrix_path)
        else:
            raise FileNotFoundError(f"Matrix file not found at {matrix_path}")
        rng = np.random.default_rng(0)
        A = A.tocoo()

        # Create feature/weight arrays using the RNG (deterministic)
        n = A.shape[0]
        features = rng.standard_normal((n, feature_dim), dtype=np.float32)
        weights1 = rng.standard_normal((feature_dim, hidden_dim), dtype=np.float32)
        bias1 = np.zeros((hidden_dim,), dtype=np.float32)
        weights2 = rng.standard_normal((hidden_dim, out_dim), dtype=np.float32)
        bias2 = np.zeros((out_dim,), dtype=np.float32)

        A_bin = BinsparseFormat.from_coo(
            (A.row, A.col), A.data.astype(np.float32, copy=False), A.shape
        )
        features_b = BinsparseFormat.from_numpy(features)
        weights1_b = BinsparseFormat.from_numpy(weights1)
        bias1_b = BinsparseFormat.from_numpy(bias1)
        weights2_b = BinsparseFormat.from_numpy(weights2)
        bias2_b = BinsparseFormat.from_numpy(bias2)
        return DataInstance(
            inputs=[A_bin, features_b, weights1_b, bias1_b, weights2_b, bias2_b],
            meta={},
        )


class GCNBenchmark(Benchmark):
    @property
    def name(self) -> str:
        return "gcn"

    @property
    def pretty_name(self) -> str:
        return "Graph Convolutional Network Inference"

    @property
    def description(self) -> str:
        return (
            "Computes a 2-layer Graph Convolutional Network forward pass: "
            "h1 = ReLU(adjacency @ features @ weights1 + bias1) "
            "output = adjacency @ h1 @ weights2 + bias2 "
            "Implementation based on Scorch (Yan et al., 2024)."
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
                title="Scorch: A Library for Sparse Deep Learning",
                authors=[
                    Author("Bobby Yan"),
                    Author("Alexander J. Root"),
                    Author("Trevor Gale"),
                    Author("David Broman"),
                    Author("Fredrik Kjolstad"),
                ],
                journal="Arxiv",
                volume="arXiv:2405.16883",
                year=2024,
                url="https://anonymous.4open.science/r/scorch/README.md",
            ),
            Ref(
                title="Open Graph Benchmark: Datasets for Machine Learning on Graphs",
                authors=[
                    Author("Weihua Hu"),
                    Author("Matthias Fey"),
                    Author("Marinka Zitnik"),
                    Author("Yuxiao Dong"),
                    Author("Hongyu Ren"),
                    Author("Bowen Liu"),
                    Author("Michele Catasta"),
                    Author("Jure Leskovec"),
                ],
                journal="Arxiv",
                volume="arXiv:2005.00687",
                year=2020,
                url="https://arxiv.org/abs/2005.00687",
            ),
        ]

    @property
    def ai_disclosure(self) -> str:
        return (
            "No generative AI was used to construct the benchmark function itself."
            " Generative AI might have been used to construct tests. This statement was"
            " written by hand."
        )

    @property
    def motivation(self) -> str:
        return (
            "Graphs are widely used for abstracting systems of interacting objects,"
            " such as social networks (Easley et al., 2010), knowledge graphs (Nickel"
            " et al., 2015), molecular graphs (Wu et al., 2018), and biological"
            " networks (Barabasi & Oltvai, 2004), as well as for modeling 3D objects"
            " (Simonovsky & Komodakis, 2017), manifolds (Bronstein et al., 2017), and"
            " source code (Allamanis et al., 2017). Machine learning (ML), especially"
            " deep learning, on graphs is an emerging field (Hamilton et al., 2017b;"
            " Bronstein et al., 2017). W. Hu et al. To represent a graph, an adjaceny"
            " matrix is used, which is inherently sparse."
        )

    @property
    def generators(self):
        return [GCNTestGenerator(), GCNGenerator()]

    def check(self, param):
        for item in self._output:
            assert isinstance(item, BinsparseFormat), (
                "Output must be in binsparse format"
            )

        if self._ref_outputs is None:
            return

        result = self._output[0].data["values"].reshape(self._output[0].data["shape"])
        expected = self._ref_outputs[0].data["values"].reshape(
            self._ref_outputs[0].data["shape"]
        )
        np.testing.assert_allclose(
            result,
            expected,
            rtol=self._ref_meta["rtol"],
            err_msg=f"GCN output mismatch for {param.dataset.name}",
        )

    """
    Args:
    ----
    xp : array_api
        Array API module (e.g. numpy, cupy, torch)
    adjacency_bench : BinsparseFormat
        Sparse adjacency matrix of the graph
    features_bench : BinsparseFormat
        Node feature matrix
    weights1_bench : BinsparseFormat
        Weights for first GCN layer
    bias1_bench : BinsparseFormat
        Bias for first GCN layer
    weights2_bench : BinsparseFormat
        Weights for second GCN layer
    bias2_bench : BinsparseFormat
        Bias for second GCN layer

    Returns:
    -------
    BinsparseFormat
        Output node embeddings after 2-layer GCN
    """

    def benchmark(self, xp, data: list, meta: dict):
        (
            adjacency,
            features,
            weights1,
            bias1,
            weights2,
            bias2,
        ) = data

        # Layer 1: adjacency @ features -> linear transform -> ReLU
        h1 = adjacency @ features
        h1 = h1 @ weights1 + bias1
        h1 = xp.maximum(h1, 0)  # ReLU activation

        # Layer 2: adjacency @ h1 -> linear transform
        h2 = adjacency @ h1
        output = h2 @ weights2 + bias2

        solution = output
        return [solution]
