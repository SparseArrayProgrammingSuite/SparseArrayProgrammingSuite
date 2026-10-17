from types import SimpleNamespace

import pytest

import numpy as np
import scipy.sparse as sps

from binsparse.conversions import from_numpy, from_scipy, to_numpy

from frameworks.saps_numpy import NumpyFramework
from frameworks.saps_smart import SmartSparseFramework
from saps.benchmark import DataInstance, Param
from saps.benchmarks.gcn import (
    GCNBenchmark,
    GCNDataset,
    GCNSuiteSparseGenerator,
    GCNTestGenerator,
)
from saps.storage import LocalStorageBackend


def _dataset(dtype=np.float32, **arrays):
    # A one-node identity network, unless a test supplies different inputs.
    inputs = {
        "adjacency": [[1]],
        "features": [[1]],
        "weights1": [[1]],
        "bias1": [0],
        "weights2": [[1]],
        "bias2": [0],
    }
    inputs.update(arrays)
    return GCNDataset(
        "test", **{key: np.array(value, dtype=dtype) for key, value in inputs.items()}
    )


def _check(problem, output):
    benchmark = GCNBenchmark()
    benchmark._ref_outputs = problem.ref_outputs
    benchmark._output = [from_numpy(np.asarray(output))]
    benchmark.check(SimpleNamespace(dataset=SimpleNamespace(name="test")))


def _forward(problem):
    return GCNBenchmark().benchmark(
        NumpyFramework(), {}, *[to_numpy(value) for value in problem.inputs]
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("framework", [NumpyFramework, SmartSparseFramework])
def test_multiclass_scores(dtype, framework):
    dataset = _dataset(
        dtype,
        adjacency=[[1, 1, 0], [0, 1, 1], [0, 0, 1]],
        features=[[1, -2], [0, 1], [-1, 2]],
        weights1=[[1, -1], [2, 1]],
        bias1=[-1, 1],
        weights2=[[1, -1, 2], [-1, 2, 1]],
        bias2=[0.5, -1, -2],
    )
    # Hand calculation: hidden values after ReLU are [[0,0],[4,5],[2,4]].
    dataset.expected = np.array([[-0.5, 5, 11], [-2.5, 11, 19], [-1.5, 5, 6]])
    param = Param(GCNTestGenerator(), dataset)
    benchmark = GCNBenchmark()
    benchmark.setup(param, use_cache=False, xp=framework())
    benchmark.run(param)
    np.testing.assert_array_equal(to_numpy(benchmark._output[0]), dataset.expected)
    benchmark.teardown(param)

    problem = GCNTestGenerator().generate(dataset)
    wrong_score = dataset.expected.copy()
    wrong_score[0, 0] += 1  # Same winning class, but an incorrect score.
    for wrong in (wrong_score, np.zeros_like(dataset.expected)):
        with pytest.raises(AssertionError, match="GCN output mismatch"):
            _check(problem, wrong)
    dataset.expected[0, 0] += 1
    with pytest.raises(AssertionError):
        GCNTestGenerator().generate(dataset)


@pytest.mark.parametrize("output", [np.array(1.0), [[np.nan]], [[np.inf]]])
def test_rejects_invalid_output(output):
    problem = GCNTestGenerator().generate(_dataset())
    with pytest.raises(AssertionError):
        _check(problem, output)


@pytest.mark.parametrize("bound", [None, -1.0, np.nan, np.inf])
def test_requires_valid_reference(bound):
    problem = GCNTestGenerator().generate(_dataset())
    if bound is None:
        problem.ref_outputs = None
    else:
        problem.ref_outputs[1] = from_numpy(np.array([[bound]]))
    with pytest.raises(AssertionError):
        _check(problem, [[1.0]])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("scale", [2.0**-10, 2.0**10])
def test_accepts_roundoff_but_rejects_small_errors(dtype, scale):
    problem = GCNTestGenerator().generate(_dataset(dtype, features=[[scale]]))
    _check(problem, [[np.nextafter(dtype(scale), dtype(np.inf))]])
    # Independent threshold: even a tenfold inflated tolerance must fail.
    with pytest.raises(AssertionError, match="GCN output mismatch"):
        _check(problem, [[scale * (1 + 32 * np.finfo(dtype).eps)]])


@pytest.mark.parametrize("dtype,power", [(np.float32, 24), (np.float64, 53)])
@pytest.mark.parametrize("channels,gain", [(1, 1024), (128, 1)])
def test_cancellation_error_propagates_through_weights(dtype, power, channels, gain):
    dataset = _dataset(
        dtype,
        features=[[2**power, 1, -(2**power)]],
        weights1=np.ones((3, channels)),
        bias1=np.full(channels, -0.5),
        weights2=np.full((channels, 1), gain),
    )
    problem = GCNTestGenerator().generate(dataset)
    _check(problem, _forward(problem))
    # Different summation orders produce zero or one before bias and ReLU.
    for score in (0, 0.5 * channels * gain):
        _check(problem, [[score]])
    with pytest.raises(AssertionError, match="GCN output mismatch"):
        _check(problem, [[1e6 * channels * gain]])


@pytest.mark.parametrize("dtype,power", [(np.float32, 24), (np.float64, 53)])
def test_neighbor_accumulation_roundoff(dtype, power):
    adjacency = np.zeros((257, 257), dtype=dtype)
    adjacency[0] = 1
    features = np.ones((257, 1), dtype=dtype)
    features[0, 0] = 2**power
    problem = GCNTestGenerator().generate(
        _dataset(dtype, adjacency=adjacency, features=features)
    )
    # Summing the 256 ones first preserves them; adding them one by one loses them.
    for score in (2**power, 2**power + 256):
        output = np.zeros((257, 1), dtype=dtype)
        output[0, 0] = score
        _check(problem, output)


@pytest.mark.parametrize("promote", ["bias1", "weights2", "bias2"])
def test_later_float64_promotion_preserves_float32_roundoff(promote):
    dataset = _dataset(features=[[1.1]], weights1=[[1.1]])
    setattr(dataset, promote, getattr(dataset, promote).astype(np.float64))
    problem = GCNTestGenerator().generate(dataset)
    output = _forward(problem)
    _check(problem, output)
    with pytest.raises(AssertionError, match="GCN output mismatch"):
        _check(problem, output + 1e-4)


@pytest.mark.parametrize("stage", ["adjacency", "weights1"])
def test_underflow_error_propagates_through_weights(stage):
    dataset = _dataset(features=[[2.0**-80]], weights2=[[2.0**120]])
    setattr(dataset, stage, np.array([[2.0**-70]], dtype=np.float32))
    problem = GCNTestGenerator().generate(dataset)
    output = _forward(problem)
    np.testing.assert_array_equal(output, [[0]])
    _check(problem, output)
    with pytest.raises(AssertionError, match="GCN output mismatch"):
        _check(problem, [[1.0]])


def test_saved_bounds_do_not_mask_errors_in_other_scores(tmp_path):
    problem = GCNTestGenerator().generate(
        _dataset(
            adjacency=np.eye(2),
            features=[[1e8, 0], [0, 1]],
            weights1=np.eye(2),
            bias1=[0, 0],
            weights2=np.eye(2),
            bias2=[0, 0],
        )
    )
    backend = LocalStorageBackend(
        tmp_path / "remote", tmp_path / "manifest.json", tmp_path
    )
    path = tmp_path / "gcn.bsp.h5"
    backend.serialize_data_to_file(problem, path)
    restored = backend.deserialize_data_from_file(path)
    for actual, expected in zip(restored.ref_outputs, problem.ref_outputs, strict=True):
        np.testing.assert_array_equal(to_numpy(actual), to_numpy(expected))
    expected = np.array([[1e8, 0], [0, 1]], dtype=np.float32)
    _check(restored, expected)
    for index in ((0, 1), (1, 1)):
        wrong = expected.copy()
        wrong[index] += 1
        with pytest.raises(AssertionError, match="GCN output mismatch"):
            _check(restored, wrong)


def test_suitesparse_reference_and_teardown_without_dense_adjacency(monkeypatch):
    adjacency = sps.coo_array([[1, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=np.float32)
    raw = DataInstance(inputs=[from_scipy(adjacency)], meta={"shape": [3, 3]})
    monkeypatch.setattr("saps.benchmarks.gcn.fetch_suitesparse_matrix", lambda _: raw)

    def forbid_dense(*args, **kwargs):
        raise AssertionError("Reference must not densify adjacency")

    for cls in (sps.coo_array, sps.csr_array, sps.coo_matrix, sps.csr_matrix):
        monkeypatch.setattr(cls, "toarray", forbid_dense)
    param = Param(
        GCNSuiteSparseGenerator(),
        GCNDataset("tiny", feature_dim=2, hidden_dim=3, out_dim=2),
    )
    benchmark = GCNBenchmark()
    benchmark.setup(param, use_cache=False, xp=SmartSparseFramework())
    benchmark.run(param)
    benchmark.teardown(param)
