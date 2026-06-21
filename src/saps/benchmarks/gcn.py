import os
from typing import Any

import numpy as np

import saps
from saps.benchmark import (
    Author,
    Benchmark,
    Contributor,
    Dataset,
    Generator,
    Ref,
)
from saps_framework import BinsparseFormat

xp = saps.xp


class GCNDataset(Dataset):
    def __init__(
        self,
        name: str,
        description: str = "",
        source_name: str | None = None,
        feature_dim: int = 16,
        hidden_dim: int = 8,
        out_dim: int = 1,
    ):
        self._tags: list[str] = []
        self.dataset_name = name
        self.dataset_description = description
        self.source_name = source_name if source_name is not None else name
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

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
    def tags(self) -> list[str]:
        return self._tags

    @property
    def metadata(self) -> dict[str, Any]:
        data = super().metadata
        data["source_name"] = self.source_name
        data["feature_dim"] = self.feature_dim
        data["hidden_dim"] = self.hidden_dim
        data["out_dim"] = self.out_dim
        return data


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
    def tags(self) -> list[str]:
        return []

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
                title="Open Graph Benchmark: Datasets for Machine Learning on Graph",
                authors=[
                    Author("Wenbing Hu"),
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
                year=2021,
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
                feature_dim=32,
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
                feature_dim=16,
                hidden_dim=8,
                out_dim=1,
            ),
            GCNDataset(
                "dg_gcn_molecular_1",
                "Small molecular graph. - Email network.",
                "email",
                feature_dim=16,
                hidden_dim=8,
                out_dim=1,
            ),
            GCNDataset(
                "dg_gcn_molecular_2",
                "Medium molecular graph - PDDB protein structure.",
                "Chebyshev3",
                feature_dim=24,
                hidden_dim=12,
                out_dim=1,
            ),
            GCNDataset(
                "dg_gcn_citation_1",
                "Large citation network graph (AIDS-like size).",
                "ca-HepPh",
                feature_dim=64,
                hidden_dim=32,
                out_dim=1,
            ),
            GCNDataset(
                "dg_gcn_large_2",
                "Very large road network.",
                "road_usa",
                feature_dim=64,
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
        features = rng.standard_normal((n, feature_dim))
        weights1 = rng.standard_normal((feature_dim, hidden_dim))
        bias1 = np.zeros((hidden_dim,))
        weights2 = rng.standard_normal((hidden_dim, out_dim))
        bias2 = np.zeros((out_dim,))

        A_bin = BinsparseFormat.from_coo((A.row, A.col), A.data, A.shape)
        features_b = BinsparseFormat.from_numpy(features)
        weights1_b = BinsparseFormat.from_numpy(weights1)
        bias1_b = BinsparseFormat.from_numpy(bias1)
        weights2_b = BinsparseFormat.from_numpy(weights2)
        bias2_b = BinsparseFormat.from_numpy(bias2)
        return (A_bin, features_b, weights1_b, bias1_b, weights2_b, bias2_b), {}


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
    def tags(self) -> list[str]:
        return []

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
                title="Open Graph Benchmark: Datasets for Machine Learning on Graph",
                authors=[
                    Author("Wenbing Hu"),
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
                year=2021,
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
        return [GCNGenerator()]

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

    def benchmark(self, data: list, meta: dict):
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
