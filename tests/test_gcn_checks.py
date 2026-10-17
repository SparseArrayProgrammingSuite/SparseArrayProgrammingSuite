import json
from fractions import Fraction
from types import SimpleNamespace

import pytest

import numpy as np
import scipy.sparse as sps

from binsparse.conversions import from_numpy, from_scipy, to_numpy, to_scipy

from frameworks.saps_numpy import NumpyFramework
from frameworks.saps_smart import SmartSparseFramework
from frameworks.saps_sparse import PyDataSparseFramework
from saps.benchmark import DataInstance, Param
from saps.benchmarks.gcn import (
    GCNBenchmark,
    GCNDataset,
    GCNOGBGenerator,
    GCNSuiteSparseGenerator,
    GCNTestGenerator,
    gcn_reference_with_error,
)
from saps.downloaders.ogb import normalized_undirected_adjacency
from saps.storage import LocalStorageBackend


def _benchmark_with_output(output, *, expected):
    benchmark = GCNBenchmark()
    output = to_numpy(output) if not isinstance(output, np.ndarray) else output
    expected = to_numpy(expected) if not isinstance(expected, np.ndarray) else expected
    benchmark._output = [from_numpy(output)]
    benchmark._ref_outputs = [from_numpy(expected), from_numpy(np.zeros_like(expected))]
    return benchmark


def test_gcn_check_accepts_reference_output():
    dataset = GCNTestGenerator().datasets[1]
    problem = GCNTestGenerator().generate(dataset)
    expected = problem.ref_outputs[0]
    benchmark = _benchmark_with_output(expected, expected=expected)

    benchmark.check(SimpleNamespace(dataset=dataset))


def test_gcn_check_rejects_wrong_output():
    dataset = GCNTestGenerator().datasets[1]
    problem = GCNTestGenerator().generate(dataset)
    expected = problem.ref_outputs[0]
    wrong = to_numpy(expected) + 1.0
    benchmark = _benchmark_with_output(wrong, expected=expected)

    with pytest.raises(AssertionError, match="GCN output mismatch"):
        benchmark.check(SimpleNamespace(dataset=dataset))


@pytest.mark.parametrize("bad_value", [np.nan, np.inf])
def test_gcn_check_rejects_nonfinite_output(bad_value):
    dataset = GCNTestGenerator().datasets[1]
    problem = GCNTestGenerator().generate(dataset)
    expected = to_numpy(problem.ref_outputs[0])
    benchmark = _benchmark_with_output(
        np.full_like(expected, bad_value), expected=expected
    )

    with pytest.raises(AssertionError, match="nonfinite"):
        benchmark.check(SimpleNamespace(dataset=dataset))


def test_gcn_check_requires_reference_output():
    dataset = GCNTestGenerator().datasets[1]
    benchmark = GCNBenchmark()
    benchmark._output = [from_numpy(np.zeros((2, 1), dtype=np.float32))]
    benchmark._ref_outputs = None
    benchmark._ref_meta = None

    with pytest.raises(AssertionError, match="No reference output"):
        benchmark.check(SimpleNamespace(dataset=dataset))


def _multiclass_dataset(dtype=np.float32):
    # A X = [[1,-1],[-1,3],[-1,2]]; ReLU(A X W1 + b1) =
    # [[0,0],[4,5],[2,4]]. Applying A, W2 and b2 gives the literal scores below.
    return GCNDataset(
        "multiclass",
        adjacency=np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=dtype),
        features=np.array([[1, -2], [0, 1], [-1, 2]], dtype=dtype),
        weights1=np.array([[1, -1], [2, 1]], dtype=dtype),
        bias1=np.array([-1, 1], dtype=dtype),
        weights2=np.array([[1, -1, 2], [-1, 2, 1]], dtype=dtype),
        bias2=np.array([0.5, -1, -2], dtype=dtype),
        expected=np.array([[-0.5, 5, 11], [-2.5, 11, 19], [-1.5, 5, 6]]),
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("framework", [NumpyFramework, SmartSparseFramework])
def test_gcn_multiclass_forward_and_check(dtype, framework):
    dataset = _multiclass_dataset(dtype)
    param = Param(GCNTestGenerator(), dataset)
    benchmark = GCNBenchmark()
    benchmark.setup(param, use_cache=False, xp=framework())
    benchmark.run(param)
    np.testing.assert_array_equal(to_numpy(benchmark._output[0]), dataset.expected)
    benchmark.check(param)

    # Corrupt a non-winning score: the predicted class stays the same.
    wrong = to_numpy(benchmark._output[0]).copy()
    wrong[0, 0] += 1
    benchmark._output = [from_numpy(wrong)]
    with pytest.raises(AssertionError, match="GCN output mismatch"):
        benchmark.check(param)


def test_gcn_generator_checks_hand_calculated_expected_output():
    dataset = _multiclass_dataset()
    dataset.expected[0, 0] += 1
    with pytest.raises(AssertionError):
        GCNTestGenerator().generate(dataset)


@pytest.mark.parametrize("dtype,power", [(np.float32, 24), (np.float64, 53)])
@pytest.mark.parametrize("scale", [2.0**-10, 1.0, 2.0**10])
def test_gcn_check_allows_cancellation_but_rejects_wrong_scores(dtype, power, scale):
    dataset = GCNDataset(
        "cancellation",
        adjacency=np.eye(1, dtype=dtype),
        features=scale * np.array([[2**power, 1, -(2**power)]], dtype=dtype),
        weights1=np.ones((3, 1), dtype=dtype),
        bias1=np.array([-0.5 * scale], dtype=dtype),
        weights2=np.array([[2, -3]], dtype=dtype),
        bias2=np.array([0, scale], dtype=dtype),
    )
    param = Param(GCNTestGenerator(), dataset)
    benchmark = GCNBenchmark()
    benchmark.setup(param, use_cache=False, xp=NumpyFramework())
    benchmark.run(param)
    benchmark.check(param)

    # Both association orders of large + small - large are valid floating
    # point evaluations; their discrepancy must propagate through ReLU/W2.
    benchmark._output = [from_numpy(np.array([[scale, -0.5 * scale]], dtype=dtype))]
    benchmark.check(param)
    benchmark._output = [from_numpy(np.array([[0, scale]], dtype=dtype))]
    benchmark.check(param)
    benchmark._output = [from_numpy(np.array([[1e6, -1e6]], dtype=dtype) * scale)]
    with pytest.raises(AssertionError, match="GCN output mismatch"):
        benchmark.check(param)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("scale", [2.0**-10, 1.0, 2.0**10])
def test_gcn_check_accepts_roundoff_but_rejects_small_errors(dtype, scale):
    dataset = GCNDataset(
        "small_error",
        adjacency=np.ones((1, 1), dtype=dtype),
        features=np.array([[scale]], dtype=dtype),
        weights1=np.ones((1, 1), dtype=dtype),
        bias1=np.zeros(1, dtype=dtype),
        weights2=np.ones((1, 1), dtype=dtype),
        bias2=np.zeros(1, dtype=dtype),
        expected=np.array([[scale]], dtype=dtype),
    )
    param = Param(GCNTestGenerator(), dataset)
    benchmark = GCNBenchmark()
    benchmark.setup(param, use_cache=False, xp=NumpyFramework())
    benchmark.run(param)
    np.testing.assert_array_equal(to_numpy(benchmark._output[0]), [[scale]])
    benchmark.check(param)

    rounded = np.nextafter(dtype(scale), dtype(np.inf))
    benchmark._output = [from_numpy(np.array([[rounded]], dtype=dtype))]
    benchmark.check(param)
    # Four singleton products and two bias additions: roundoff is about
    # 6*eps*scale. A 32*eps error is excessive, even though default allclose
    # or a tenfold inflated bound would accept it. Do not derive this error
    # from the generated bound, or an inflated bound could evade the test.
    wrong = scale * (1 + 32 * np.finfo(dtype).eps)
    benchmark._output = [from_numpy(np.array([[wrong]], dtype=dtype))]
    with pytest.raises(AssertionError, match="GCN output mismatch"):
        benchmark.check(param)


@pytest.mark.parametrize("dtype,power", [(np.float32, 24), (np.float64, 53)])
@pytest.mark.parametrize("scale", [2.0**-10, 1.0, 2.0**10])
@pytest.mark.parametrize("framework", [NumpyFramework, PyDataSparseFramework])
def test_gcn_graph_aggregation_allows_cancellation(dtype, power, scale, framework):
    # Node 1 aggregates large + small - large; node 0 receives that result
    # in the second layer. Nodes 2--4 have no incoming contributions.
    adjacency = sps.coo_array(
        (np.ones(4, dtype=dtype), ([0, 1, 1, 1], [1, 2, 3, 4])), shape=(5, 5)
    )
    dataset = GCNDataset(
        "graph_cancellation",
        adjacency=adjacency.toarray(),
        features=scale
        * np.array([[0], [0], [2**power], [1], [-(2**power)]], dtype=dtype),
        weights1=np.ones((1, 1), dtype=dtype),
        bias1=np.array([-0.5 * scale], dtype=dtype),
        weights2=np.array([[2, -3]], dtype=dtype),
        bias2=np.array([0, scale], dtype=dtype),
    )
    param = Param(GCNTestGenerator(), dataset)
    benchmark = GCNBenchmark()
    benchmark.setup(param, use_cache=False, xp=framework())
    benchmark._input[0] = from_scipy(adjacency)
    benchmark.run(param)
    benchmark.check(param)

    # Aggregating before cancellation can lose the small term; cancelling
    # first preserves it. Both outcomes must pass after ReLU and W2.
    scores = np.tile(np.array([0, scale], dtype=dtype), (5, 1))
    benchmark._output = [from_numpy(scores.copy())]
    benchmark.check(param)
    scores[0] = [scale, -0.5 * scale]
    benchmark._output = [from_numpy(scores.copy())]
    benchmark.check(param)
    scores[0] = [1e6 * scale, -1e6 * scale]
    benchmark._output = [from_numpy(scores)]
    with pytest.raises(AssertionError, match="GCN output mismatch"):
        benchmark.check(param)


def test_gcn_check_rejects_wrong_shape():
    benchmark = _benchmark_with_output(np.array(1.0), expected=np.ones((2, 3)))
    with pytest.raises(AssertionError, match="shape"):
        benchmark.check(SimpleNamespace(dataset=SimpleNamespace(name="shape")))


@pytest.mark.parametrize("bound", [-1.0, np.nan, np.inf])
def test_gcn_check_rejects_invalid_error_bound(bound):
    benchmark = _benchmark_with_output(np.zeros((2, 1)), expected=np.zeros((2, 1)))
    benchmark._ref_outputs[1] = from_numpy(np.full((2, 1), bound))
    with pytest.raises(AssertionError, match="error bound"):
        benchmark.check(SimpleNamespace(dataset=SimpleNamespace(name="bound")))


def test_gcn_suitesparse_generator_provides_sparse_reference(monkeypatch):
    adjacency = sps.coo_array([[1, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=np.float32)
    raw = DataInstance(inputs=[from_scipy(adjacency)], meta={"shape": [3, 3]})
    monkeypatch.setattr("saps.benchmarks.gcn.fetch_suitesparse_matrix", lambda _: raw)
    generator = GCNSuiteSparseGenerator()
    dataset = GCNDataset("tiny", feature_dim=2, hidden_dim=3, out_dim=2)
    instance = generator.generate(dataset)
    assert instance.ref_outputs is not None
    json.dumps(instance.ref_meta)
    assert sps.issparse(to_scipy(instance.inputs[0]))

    param = Param(generator, dataset)
    benchmark = GCNBenchmark()
    benchmark.setup(param, use_cache=False, xp=SmartSparseFramework())
    benchmark.run(param)
    benchmark.teardown(param)


def test_gcn_sparse_reference_does_not_densify_adjacency(monkeypatch):
    adjacency = sps.eye(100_000, format="coo", dtype=np.float32)
    raw = DataInstance(
        inputs=[from_scipy(adjacency)], meta={"shape": list(adjacency.shape)}
    )
    monkeypatch.setattr("saps.benchmarks.gcn.fetch_suitesparse_matrix", lambda _: raw)

    def forbid_dense(*args, **kwargs):
        raise AssertionError("Reference must not densify adjacency")

    for cls in (sps.coo_array, sps.csr_array, sps.coo_matrix, sps.csr_matrix):
        monkeypatch.setattr(cls, "toarray", forbid_dense)
    generator = GCNSuiteSparseGenerator()
    instance = generator.generate(GCNDataset("large", feature_dim=1, hidden_dim=1))
    assert instance.ref_outputs is not None
    assert instance.ref_outputs[0].shape == (100_000, 1)


def test_gcn_ogb_hub_graph_rejects_broken_forward_passes(monkeypatch):
    n = 30_001
    adjacency = normalized_undirected_adjacency(
        np.stack([np.zeros(n - 1, dtype=np.int64), np.arange(1, n)]), num_nodes=n
    )
    features = np.random.default_rng(0).standard_normal((n, 128), dtype=np.float32)
    graph = SimpleNamespace(
        adjacency=adjacency,
        features=features,
        num_features=128,
        num_outputs=40,
        metadata={},
    )
    monkeypatch.setattr(
        "saps.benchmarks.gcn.fetch_ogb_nodeprop_dataset", lambda _: graph
    )
    generator = GCNOGBGenerator()
    dataset = generator.datasets[0]
    param = Param(generator, dataset)
    benchmark = GCNBenchmark()
    benchmark.setup(param, use_cache=False, xp=PyDataSparseFramework())
    A = to_scipy(adjacency).tocsr()
    W1, b1, W2, b2 = [to_numpy(x) for x in benchmark._input[2:]]
    hidden = (A @ features) @ W1 + b1
    correct = (A @ np.maximum(hidden, 0)) @ W2 + b2
    benchmark._output = [from_numpy(correct)]
    benchmark.check(param)
    for wrong in (
        np.zeros_like(correct),
        (A @ hidden) @ W2 + b2,
        np.maximum(hidden, 0) @ W2 + b2,
    ):
        benchmark._output = [from_numpy(wrong)]
        with pytest.raises(AssertionError, match="GCN output mismatch"):
            benchmark.check(param)


@pytest.mark.parametrize("promote", ["bias1", "weights2", "bias2"])
def test_gcn_mixed_precision_checks_actual_operation_precision(promote):
    dataset = GCNDataset(
        "mixed_precision",
        adjacency=np.ones((1, 1), dtype=np.float32),
        features=np.array([[1.1]], dtype=np.float32),
        weights1=np.array([[1.1]], dtype=np.float32),
        bias1=np.zeros(1, dtype=np.float32),
        weights2=np.ones((1, 1), dtype=np.float32),
        bias2=np.zeros(1, dtype=np.float32),
    )
    setattr(dataset, promote, getattr(dataset, promote).astype(np.float64))
    param = Param(GCNTestGenerator(), dataset)
    benchmark = GCNBenchmark()
    benchmark.setup(param, use_cache=False, xp=NumpyFramework())
    benchmark.run(param)
    benchmark.check(param)
    wrong = to_numpy(benchmark._output[0]) + 1e-4
    benchmark._output = [from_numpy(wrong)]
    with pytest.raises(AssertionError, match="GCN output mismatch"):
        benchmark.check(param)


@pytest.mark.parametrize("dtype,power", [(np.float32, 24), (np.float64, 53)])
def test_gcn_bound_covers_long_neighbor_accumulation(dtype, power):
    n = 257
    adjacency = np.zeros((n, n), dtype=dtype)
    adjacency[0] = 1
    features = np.ones((n, 1), dtype=dtype)
    features[0, 0] = 2**power
    expected, bound = gcn_reference_with_error(
        sps.csr_array(adjacency),
        features,
        np.ones((1, 1), dtype=dtype),
        np.zeros(1, dtype=dtype),
        np.ones((1, 1), dtype=dtype),
        np.zeros(1, dtype=dtype),
    )
    # Sequentially accumulating the 256 ones after 2**power loses all of
    # them; summing the ones first preserves them. Both orders are valid.
    for first_score in (2**power, 2**power + 256):
        scores = np.zeros((n, 1), dtype=dtype)
        scores[0, 0] = first_score
        benchmark = _benchmark_with_output(scores, expected=expected)
        benchmark._ref_outputs[1] = from_numpy(bound)
        benchmark.check(SimpleNamespace(dataset=SimpleNamespace(name="long_sum")))


@pytest.mark.parametrize("dtype,power", [(np.float32, 24), (np.float64, 53)])
@pytest.mark.parametrize("channels,gain", [(1, 1024), (128, 1)])
def test_gcn_bound_propagates_weight_amplification_and_channel_sums(
    dtype, power, channels, gain
):
    expected, bound = gcn_reference_with_error(
        np.ones((1, 1), dtype=dtype),
        np.array([[2**power, 1, -(2**power)]], dtype=dtype),
        np.ones((3, channels), dtype=dtype),
        np.full(channels, -0.5, dtype=dtype),
        np.full((channels, 1), gain, dtype=dtype),
        np.zeros(1, dtype=dtype),
    )
    # First-layer cancellation gives either zero or one before bias/ReLU.
    # The second layer amplifies that difference and sums across channels.
    for score in (0, 0.5 * channels * gain):
        benchmark = _benchmark_with_output(
            np.array([[score]], dtype=dtype), expected=expected
        )
        benchmark._ref_outputs[1] = from_numpy(bound)
        benchmark.check(SimpleNamespace(dataset=SimpleNamespace(name="amplification")))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_gcn_reference_and_forward_against_exact_arithmetic(dtype):
    rng = np.random.default_rng(13)
    for _ in range(10):
        arrays = [
            rng.uniform(-2, 2, size=shape).astype(dtype)
            for shape in [(4, 4), (4, 3), (3, 5), (5,), (5, 2), (2,)]
        ]
        arrays[0][rng.random((4, 4)) < 0.5] = 0
        A, X, W1, b1, W2, b2 = [
            np.vectorize(lambda x: Fraction(float(x)), otypes=[object])(x)
            for x in arrays
        ]
        exact = (A @ np.maximum((A @ X) @ W1 + b1, 0)) @ W2 + b2
        expected, bound = gcn_reference_with_error(
            sps.csr_array(arrays[0]), *arrays[1:]
        )
        output = GCNBenchmark().benchmark(NumpyFramework(), {}, *arrays)
        for index in np.ndindex(exact.shape):
            for result in (expected, output):
                assert abs(Fraction(float(result[index])) - exact[index]) <= Fraction(
                    float(bound[index])
                )


def test_gcn_elementwise_bounds_round_trip_and_do_not_mask_other_scores(tmp_path):
    dataset = GCNDataset(
        "local_bounds",
        adjacency=np.eye(2, dtype=np.float32),
        features=np.array([[1e8, 0], [0, 1]], dtype=np.float32),
        weights1=np.eye(2, dtype=np.float32),
        bias1=np.zeros(2, dtype=np.float32),
        weights2=np.eye(2, dtype=np.float32),
        bias2=np.zeros(2, dtype=np.float32),
    )
    problem = GCNTestGenerator().generate(dataset)
    backend = LocalStorageBackend(
        tmp_path / "remote", tmp_path / "manifest.json", tmp_path
    )
    path = tmp_path / "gcn.bsp.h5"
    backend.serialize_data_to_file(problem, path)
    restored = backend.deserialize_data_from_file(path)
    for result, expected in zip(restored.ref_outputs, problem.ref_outputs, strict=True):
        np.testing.assert_array_equal(to_numpy(result), to_numpy(expected))
    benchmark = GCNBenchmark()
    benchmark._ref_outputs = restored.ref_outputs
    expected = to_numpy(restored.ref_outputs[0])
    benchmark._output = [from_numpy(expected.copy())]
    param = SimpleNamespace(dataset=dataset)
    benchmark.check(param)
    for index in ((0, 1), (1, 1)):
        wrong = expected.copy()
        wrong[index] += 1
        benchmark._output = [from_numpy(wrong)]
        with pytest.raises(AssertionError, match="GCN output mismatch"):
            benchmark.check(param)


@pytest.mark.parametrize("underflow_stage", ["adjacency", "weights"])
def test_gcn_bound_propagates_underflow_before_amplification(underflow_stage):
    dataset = GCNDataset(
        "underflow",
        adjacency=np.array(
            [[2.0**-70 if underflow_stage == "adjacency" else 1]], dtype=np.float32
        ),
        features=np.array([[2.0**-80]], dtype=np.float32),
        weights1=np.array(
            [[2.0**-70 if underflow_stage == "weights" else 1]], dtype=np.float32
        ),
        bias1=np.zeros(1, dtype=np.float32),
        weights2=np.array([[2.0**120]], dtype=np.float32),
        bias2=np.zeros(1, dtype=np.float32),
    )
    param = Param(GCNTestGenerator(), dataset)
    benchmark = GCNBenchmark()
    benchmark.setup(param, use_cache=False, xp=NumpyFramework())
    benchmark.run(param)
    np.testing.assert_array_equal(to_numpy(benchmark._output[0]), [[0]])
    benchmark.check(param)
    benchmark._output = [from_numpy(np.ones((1, 1), dtype=np.float32))]
    with pytest.raises(AssertionError, match="GCN output mismatch"):
        benchmark.check(param)
