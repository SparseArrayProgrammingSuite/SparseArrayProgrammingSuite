from typing import Any, cast

import numpy as np

from binsparse import BinsparseTensor, COORMatrix
from binsparse.conversions import from_numpy, to_numpy, to_scipy

from saps.benchmark import (
    Author,
    Benchmark,
    Contributor,
    DataInstance,
    Dataset,
    Generator,
    Ref,
)
from saps.benchmarks.ogb import OGBNodePropGenerator, fetch_ogb_nodeprop_dataset
from saps.benchmarks.suitesparse import SuiteSparseDataset, fetch_suitesparse_matrix


def _from_binsparse(array):
    try:
        return to_numpy(array)
    except TypeError:
        return to_scipy(array).toarray()


def _gcn_loss(adjacency, features, weights1, bias1, weights2, bias2, targets):
    z1 = adjacency @ features
    h1_pre = z1 @ weights1 + bias1
    h1 = np.maximum(h1_pre, 0)
    z2 = adjacency @ h1
    predictions = z2 @ weights2 + bias2
    diff = predictions - targets
    return np.sum(diff * diff) / predictions.shape[0]


def _targets_from_ogb_labels(labels: np.ndarray, num_outputs: int) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.ndim == 1:
        labels = labels.reshape(-1, 1)
    if labels.ndim != 2:
        raise ValueError("OGB labels must be a vector or matrix.")

    if labels.shape[1] == num_outputs:
        return np.nan_to_num(labels.astype(np.float32, copy=False))

    if labels.shape[1] != 1:
        raise ValueError(
            f"Cannot convert OGB labels with shape {labels.shape} to "
            f"{num_outputs} outputs."
        )

    flat_labels = labels[:, 0]
    if num_outputs == 1:
        return np.nan_to_num(flat_labels.astype(np.float32)).reshape(-1, 1)

    targets = np.zeros((labels.shape[0], num_outputs), dtype=np.float32)
    valid = np.isfinite(flat_labels)
    label_ids = flat_labels[valid].astype(np.int64)
    if np.any(label_ids < 0) or np.any(label_ids >= num_outputs):
        raise ValueError("OGB labels contain class IDs outside num_outputs.")
    targets[np.nonzero(valid)[0], label_ids] = 1.0
    return targets


class GCNTrainingDataset(SuiteSparseDataset):
    def __init__(
        self,
        name: str,
        description: str | None = None,
        source_name: str | None = None,
        *,
        feature_dim: int = 1,
        hidden_dim: int = 4,
        out_dim: int = 1,
        num_iterations: int = 10,
        learning_rate: float = 0.01,
        suites: list[str] | None = None,
        adjacency: np.ndarray | None = None,
        features: np.ndarray | None = None,
        weights1: np.ndarray | None = None,
        bias1: np.ndarray | None = None,
        weights2: np.ndarray | None = None,
        bias2: np.ndarray | None = None,
        targets: np.ndarray | None = None,
        ref_meta: dict[str, Any] | None = None,
    ):
        super().__init__(
            name,
            source_name=source_name or name,
            pretty_name=f"GCN {name}",
            description=description,
            suites=suites,
        )
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.adjacency = adjacency
        self.features = features
        self.weights1 = weights1
        self.bias1 = bias1
        self.weights2 = weights2
        self.bias2 = bias2
        self.targets = targets
        self.ref_meta = ref_meta or {}

    @property
    def metadata(self) -> dict[str, Any]:
        data = super().metadata
        data["source_name"] = self.source_name
        data["feature_dim"] = self.feature_dim
        data["hidden_dim"] = self.hidden_dim
        data["out_dim"] = self.out_dim
        data["num_iterations"] = self.num_iterations
        data["learning_rate"] = self.learning_rate
        return data


class OGBGCNTrainingDataset(Dataset):
    """A full-graph GCN training dataset sourced from OGB."""

    def __init__(
        self,
        name: str,
        *,
        source_name: str,
        hidden_dim: int = 256,
        num_iterations: int = 10,
        learning_rate: float = 0.01,
        description: str,
        suites: list[str] | None = None,
    ):
        self._name = name
        self.source_name = source_name
        self.hidden_dim = hidden_dim
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self._description = description
        self._suites = suites or []

    @property
    def name(self) -> str:
        return self._name

    @property
    def pretty_name(self) -> str:
        return f"GCN Backward {self.source_name}"

    @property
    def description(self) -> str:
        return self._description

    @property
    def suites(self) -> list[str]:
        return self._suites

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def metadata(self) -> dict[str, Any]:
        data = super().metadata
        data.update(
            {
                "source_name": self.source_name,
                "hidden_dim": self.hidden_dim,
                "num_iterations": self.num_iterations,
                "learning_rate": self.learning_rate,
            }
        )
        return data


class GCNTrainingTestGenerator(Generator[GCNTrainingDataset]):
    @property
    def name(self) -> str:
        return "gcn_backward_test_inputs"

    @property
    def pretty_name(self) -> str:
        return "GCN Backward Test Input Generator"

    @property
    def description(self) -> str:
        return "Small inlined GCN training examples."

    @property
    def suites(self) -> list[str]:
        return ["test", "trace"]

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def authors(self) -> list[Contributor]:
        return [Contributor("Tarun Devi", "tdevi3@gatech.edu")]

    @property
    def references(self) -> list[Ref]:
        return GCNBackwardBenchmark().references

    @property
    def ai_disclosure(self) -> str:
        return GCNBackwardBenchmark().ai_disclosure

    @property
    def motivation(self) -> str:
        return "Uses small graph examples to verify the GCN training loop."

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def datasets(self) -> list[GCNTrainingDataset]:
        rng = np.random.default_rng(42)
        degree_train_adj = np.array(
            [
                [0, 1, 1, 1, 1, 0, 0],
                [1, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=np.float32,
        )
        degree_targets = degree_train_adj.sum(axis=1, keepdims=True)
        degree_targets = degree_targets / degree_targets.max()
        degree_test_adj = np.array(
            [
                [0, 1, 1, 1, 0, 0],
                [1, 0, 1, 0, 0, 0],
                [1, 1, 0, 0, 1, 0],
                [1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ],
            dtype=np.float32,
        )
        degree_test_targets = degree_test_adj.sum(axis=1, keepdims=True)
        degree_test_targets = degree_test_targets / degree_test_targets.max()
        degree_weights1 = rng.standard_normal((1, 4), dtype=np.float32) * 0.5
        degree_weights2 = rng.standard_normal((4, 1), dtype=np.float32) * 0.5
        return [
            GCNTrainingDataset(
                "test_gcn_backward_2node",
                suites=["test", "trace"],
                adjacency=np.array([[0, 1], [1, 0]], dtype=np.float32),
                features=np.array([[1.0], [2.0]], dtype=np.float32),
                weights1=np.array([[1.0]], dtype=np.float32),
                bias1=np.array([0.0], dtype=np.float32),
                weights2=np.array([[1.0]], dtype=np.float32),
                bias2=np.array([0.0], dtype=np.float32),
                targets=np.array([[2.0], [1.0]], dtype=np.float32),
                num_iterations=10,
                learning_rate=0.01,
                ref_meta={
                    "check_loss_reduction": True,
                    "output_shapes": [(1,), (1, 1), (1,), (1, 1), (1,)],
                },
            ),
            GCNTrainingDataset(
                "test_gcn_backward_multidim",
                suites=["test", "trace"],
                adjacency=np.array(
                    [
                        [0, 1, 1, 0],
                        [1, 0, 1, 1],
                        [1, 1, 0, 1],
                        [0, 1, 1, 0],
                    ],
                    dtype=np.float32,
                ),
                features=np.array(
                    [
                        [1.0, 0.5],
                        [0.0, 1.0],
                        [1.0, 1.0],
                        [0.5, 0.5],
                    ],
                    dtype=np.float32,
                ),
                weights1=np.array([[0.5, 0.3, 0.1], [0.2, 0.4, 0.6]], dtype=np.float32),
                bias1=np.zeros(3, dtype=np.float32),
                weights2=np.array(
                    [[0.5, 0.5], [0.3, 0.7], [0.2, 0.8]], dtype=np.float32
                ),
                bias2=np.zeros(2, dtype=np.float32),
                targets=np.array(
                    [
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [1.0, 1.0],
                        [0.5, 0.5],
                    ],
                    dtype=np.float32,
                ),
                num_iterations=100,
                learning_rate=0.01,
                ref_meta={
                    "check_loss_reduction": True,
                    "output_shapes": [(1,), (2, 3), (3,), (3, 2), (2,)],
                },
            ),
            GCNTrainingDataset(
                "test_gcn_backward_degree_loss",
                suites=["test", "trace"],
                adjacency=degree_train_adj,
                features=np.ones((7, 1), dtype=np.float32),
                weights1=degree_weights1.copy(),
                bias1=np.zeros(4, dtype=np.float32),
                weights2=degree_weights2.copy(),
                bias2=np.zeros(1, dtype=np.float32),
                targets=degree_targets,
                num_iterations=500,
                learning_rate=0.01,
                ref_meta={
                    "check_loss_reduction": True,
                    "output_shapes": [(1,), (1, 4), (4,), (4, 1), (1,)],
                },
            ),
            GCNTrainingDataset(
                "test_gcn_backward_degree_test_graph_loss",
                suites=["test", "trace"],
                adjacency=degree_test_adj,
                features=np.ones((6, 1), dtype=np.float32),
                weights1=degree_weights1.copy(),
                bias1=np.zeros(4, dtype=np.float32),
                weights2=degree_weights2.copy(),
                bias2=np.zeros(1, dtype=np.float32),
                targets=degree_test_targets,
                num_iterations=500,
                learning_rate=0.01,
                ref_meta={
                    "check_loss_reduction": True,
                    "output_shapes": [(1,), (1, 4), (4,), (4, 1), (1,)],
                },
            ),
        ]

    def generate(self, dataset: GCNTrainingDataset):
        required = (
            dataset.adjacency,
            dataset.features,
            dataset.weights1,
            dataset.bias1,
            dataset.weights2,
            dataset.bias2,
            dataset.targets,
        )
        if any(item is None for item in required):
            raise ValueError("GCN backward test datasets must define all arrays.")

        ref_meta = dict(dataset.ref_meta)
        adjacency = cast(np.ndarray, dataset.adjacency)
        features = cast(np.ndarray, dataset.features)
        weights1 = cast(np.ndarray, dataset.weights1)
        bias1 = cast(np.ndarray, dataset.bias1)
        weights2 = cast(np.ndarray, dataset.weights2)
        bias2 = cast(np.ndarray, dataset.bias2)
        targets = cast(np.ndarray, dataset.targets)

        return DataInstance(
            inputs=[
                from_numpy(adjacency),
                from_numpy(features),
                from_numpy(weights1),
                from_numpy(bias1),
                from_numpy(weights2),
                from_numpy(bias2),
                from_numpy(targets),
            ],
            meta={
                "num_iterations": dataset.num_iterations,
                "learning_rate": dataset.learning_rate,
            },
            ref_meta=ref_meta,
        )


class GCNTrainingGenerator(Generator[GCNTrainingDataset]):
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
    def datasets(self) -> list[GCNTrainingDataset]:
        return [
            GCNTrainingDataset(
                "dg_gcn_social_1",
                "Small social network graph.",
                "Newman/karate",
                feature_dim=16,
                hidden_dim=8,
                out_dim=1,
            ),
            GCNTrainingDataset(
                "dg_gcn_social_2",
                "Medium social network graph.",
                "Newman/dolphins",
                feature_dim=16,
                hidden_dim=8,
                out_dim=1,
            ),
            GCNTrainingDataset(
                "dg_gcn_social_3",
                "Larger social network graph.",
                "SNAP/ca-GrQc",
                feature_dim=8,
                hidden_dim=16,
                out_dim=1,
            ),
            GCNTrainingDataset(
                "dg_gcn_road_1",
                "Small road network graph.",
                "DIMACS10/chesapeake",
                feature_dim=8,
                hidden_dim=4,
                out_dim=1,
            ),
            GCNTrainingDataset(
                "dg_gcn_molecular_1",
                "Small molecular graph. - Email network.",
                "Arenas/email",
                feature_dim=4,
                hidden_dim=8,
                out_dim=1,
            ),
            GCNTrainingDataset(
                "dg_gcn_molecular_2",
                "Medium molecular graph - PDDB protein structure.",
                "Muite/Chebyshev3",
                feature_dim=6,
                hidden_dim=12,
                out_dim=1,
            ),
            GCNTrainingDataset(
                "dg_gcn_citation_1",
                "Large citation network graph (AIDS-like size).",
                "SNAP/ca-HepPh",
                feature_dim=16,
                hidden_dim=32,
                out_dim=1,
            ),
        ]

    @property
    def cacheable(self) -> bool:
        return False

    def generate(self, dataset: GCNTrainingDataset):
        feature_dim = dataset.feature_dim
        hidden_dim = dataset.hidden_dim
        out_dim = dataset.out_dim

        raw = fetch_suitesparse_matrix(dataset.source_name)
        coo = to_scipy(raw.inputs[0]).tocoo()
        rng = np.random.default_rng(0)

        # Create feature/weight arrays using the RNG (deterministic)
        n = raw.meta["shape"][0]
        features = rng.standard_normal((n, feature_dim), dtype=np.float32)
        weights1 = rng.standard_normal((feature_dim, hidden_dim), dtype=np.float32)
        bias1 = np.zeros((hidden_dim,), dtype=np.float32)
        weights2 = rng.standard_normal((hidden_dim, out_dim), dtype=np.float32)
        bias2 = np.zeros((out_dim,), dtype=np.float32)
        targets = rng.standard_normal((n, out_dim), dtype=np.float32)

        row, col = coo.row, coo.col
        shape = coo.shape
        values_f32 = coo.data.astype(np.float32, copy=False)
        A_bin = COORMatrix(
            shape,
            len(values_f32),
            indices_0=row,
            indices_1=col,
            values=values_f32,
        )
        features_b = from_numpy(features)
        weights1_b = from_numpy(weights1)
        bias1_b = from_numpy(bias1)
        weights2_b = from_numpy(weights2)
        bias2_b = from_numpy(bias2)
        targets_b = from_numpy(targets)
        return DataInstance(
            inputs=[
                A_bin,
                features_b,
                weights1_b,
                bias1_b,
                weights2_b,
                bias2_b,
                targets_b,
            ],
            meta={
                "num_iterations": dataset.num_iterations,
                "learning_rate": dataset.learning_rate,
            },
        )


class OGBGCNTrainingGenerator(Generator[OGBGCNTrainingDataset]):
    @property
    def name(self) -> str:
        return "gcn_backward_ogb_inputs"

    @property
    def pretty_name(self) -> str:
        return "Open Graph Benchmark GCN Backward Inputs"

    @property
    def description(self) -> str:
        return "Loads full OGB node-property graphs for 2-layer GCN training."

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
        return GCNBackwardBenchmark().references

    @property
    def ai_disclosure(self) -> str:
        return (
            "No generative AI was used to construct the benchmark function itself. "
            "Generative AI was used to help implement and audit OGB input plumbing, "
            "tests, documentation, and debugging."
        )

    @property
    def motivation(self) -> str:
        return (
            "Uses complete real-world node features, graph structure, and labels "
            "from the Open Graph Benchmark for the GCN training loop. SAPS caching "
            "is disabled for these benchmark-specific training tensors because the "
            "shared OGB shell generator caches the prepared source graph."
        )

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def datasets(self) -> list[OGBGCNTrainingDataset]:
        return [
            OGBGCNTrainingDataset(
                dataset.name,
                source_name=dataset.source_name,
                description=dataset.description,
                suites=["standard"],
            )
            for dataset in OGBNodePropGenerator().datasets
        ]

    def generate(self, dataset: OGBGCNTrainingDataset) -> DataInstance:
        graph = fetch_ogb_nodeprop_dataset(dataset.source_name)
        feature_dim = graph.num_features
        out_dim = graph.num_outputs
        hidden_dim = dataset.hidden_dim
        targets = _targets_from_ogb_labels(graph.labels, out_dim)

        rng = np.random.default_rng(0)
        weights1 = rng.standard_normal((feature_dim, hidden_dim), dtype=np.float32)
        weights2 = rng.standard_normal((hidden_dim, out_dim), dtype=np.float32)
        meta = {
            **graph.metadata,
            "num_nodes": graph.num_nodes,
            "num_raw_edges": graph.num_raw_edges,
            "num_features": feature_dim,
            "num_outputs": out_dim,
            "num_iterations": dataset.num_iterations,
            "learning_rate": dataset.learning_rate,
        }
        return DataInstance(
            inputs=[
                graph.adjacency,
                from_numpy(graph.features),
                from_numpy(weights1),
                from_numpy(np.zeros(hidden_dim, dtype=np.float32)),
                from_numpy(weights2),
                from_numpy(np.zeros(out_dim, dtype=np.float32)),
                from_numpy(targets),
            ],
            meta=meta,
        )


class GCNBackwardBenchmark(Benchmark):
    @property
    def name(self) -> str:
        return "gcn_backward"

    @property
    def pretty_name(self) -> str:
        return "Graph Convolutional Network Inference"

    @property
    def description(self) -> str:
        return """
Benchmarks 2-layer GCN training loop (forward, backward, weight updates).

Each iteration:
    Forward pass:
        Z1 = A @ X
        H1_pre = Z1 @ W1 + b1
        H1 = ReLU(H1_pre)
        Z2 = A @ H1
        Y = Z2 @ W2 + b2

    Backward pass (MSE loss):
        dY = (2/N) * (Y - T)
        dW2 = Z2.T @ dY
        db2 = sum(dY, axis=0)
        dZ2 = dY @ W2.T
        dH1 = A.T @ dZ2
        dH1_pre = dH1 * (H1_pre > 0)
        dW1 = Z1.T @ dH1_pre
        db1 = sum(dH1_pre, axis=0)

    Weight updates:
        W1 = W1 - lr * dW1
        b1 = b1 - lr * db1
        W2 = W2 - lr * dW2
        b2 = b2 - lr * db2
        """

    @property
    def suites(self) -> list[str]:
        return []

    @property
    def concepts(self) -> str:
        return """
        <ccs2012>
        <concept>
        <concept_id>10010147.10010257.10010293.10010294</concept_id>
        <concept_desc>Computing methodologies~Neural networks</concept_desc>
        <concept_significance>500</concept_significance>
        </concept>
        <concept>
        <concept_id>10002950.10003624.10003633.10010917</concept_id>
        <concept_desc>Mathematics of computing~Graph algorithms</concept_desc>
        <concept_significance>500</concept_significance>
        </concept>
        </ccs2012>
        """

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
        return [
            GCNTrainingTestGenerator(),
            GCNTrainingGenerator(),
            OGBGCNTrainingGenerator(),
        ]

    def check(self, param):
        for item in self._output:
            assert isinstance(item, BinsparseTensor), (
                "Output must be in binsparse format"
            )

        if not self._ref_meta:
            return

        outputs = [_from_binsparse(item) for item in self._output]

        output_shapes = self._ref_meta.get("output_shapes")
        if output_shapes is not None:
            for output, shape in zip(outputs, output_shapes, strict=True):
                assert output.shape == tuple(shape)

        if self._ref_meta.get("check_loss_reduction"):
            (
                adjacency,
                features,
                initial_w1,
                initial_b1,
                initial_w2,
                initial_b2,
                targets,
            ) = [_from_binsparse(item) for item in self._input]
            reported_loss, final_w1, final_b1, final_w2, final_b2 = outputs

            initial_loss = _gcn_loss(
                adjacency,
                features,
                initial_w1,
                initial_b1,
                initial_w2,
                initial_b2,
                targets,
            )
            final_weight_loss = _gcn_loss(
                adjacency,
                features,
                final_w1,
                final_b1,
                final_w2,
                final_b2,
                targets,
            )
            assert reported_loss.item() < initial_loss, (
                f"Reported loss should decrease: {reported_loss.item()} < "
                f"{initial_loss}"
            )
            assert final_weight_loss < initial_loss, (
                f"Final weights should reduce loss: {final_weight_loss} < "
                f"{initial_loss}"
            )

    """
    Args:
    ----
    xp : array_api
        Array API module (e.g. numpy, cupy, torch)
    adjacency_bench : BinsparseTensor
        Sparse adjacency matrix A (N x N)
    features_bench : BinsparseTensor
        Node feature matrix X (N x F)
    weights1_bench : BinsparseTensor
        Initial weights for first GCN layer W1 (F x H)
    bias1_bench : BinsparseTensor
        Initial bias for first GCN layer b1 (H,)
    weights2_bench : BinsparseTensor
        Initial weights for second GCN layer W2 (H x O)
    bias2_bench : BinsparseTensor
        Initial bias for second GCN layer b2 (O,)
    targets_bench : BinsparseTensor
        Target values T (N x O) for MSE loss
    num_iterations : int
        Number of training iterations (default 10)
    learning_rate : float
        Learning rate for gradient descent (default 0.01)

    Returns:
    -------
    tuple
        (final_loss, final_W1, final_b1, final_W2, final_b2)
    """

    def benchmark(self, xp, data: list, meta: dict):
        adjacency, features, weights1, bias1, weights2, bias2, targets = data
        adjacency_T = adjacency.T
        num_iterations = meta["num_iterations"]
        learning_rate = meta["learning_rate"]

        for _ in range(num_iterations):
            # Forward pass
            Z1 = adjacency @ features  # (N × F)
            H1_pre = Z1 @ weights1 + bias1  # (N × H)
            H1 = xp.maximum(H1_pre, 0)  # (N × H) ReLU
            Z2 = adjacency @ H1  # (N × H)
            Y = Z2 @ weights2 + bias2  # (N × O)

            # MSE loss
            N = Y.shape[0]
            diff = Y - targets
            loss = xp.sum(diff * diff) / N

            # Backward pass
            dY = (2 / N) * diff

            # Layer 2 gradients
            dW2 = Z2.T @ dY  # (H × O)
            db2 = xp.sum(dY, axis=0)  # (O,)
            dZ2 = dY @ weights2.T  # (N × H)

            # Backprop through adjacency
            dH1 = adjacency_T @ dZ2  # (N × H)

            # Backprop through ReLU
            dH1_pre = dH1 * (H1_pre > 0)  # (N × H)

            # Layer 1 gradients
            dW1 = Z1.T @ dH1_pre  # (F × H)
            db1 = xp.sum(dH1_pre, axis=0)  # (H,)

            weights1 = weights1 - learning_rate * dW1
            bias1 = bias1 - learning_rate * db1
            weights2 = weights2 - learning_rate * dW2
            bias2 = bias2 - learning_rate * db2

        # Compute final outputs
        loss_out = xp.asarray([loss])
        weights1_out = weights1
        bias1_out = bias1
        weights2_out = weights2
        bias2_out = bias2

        return (
            loss_out,
            weights1_out,
            bias1_out,
            weights2_out,
            bias2_out,
        )
