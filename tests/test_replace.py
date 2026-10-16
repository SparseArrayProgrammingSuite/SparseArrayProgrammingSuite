import importlib
from pathlib import Path

import pytest

import numpy as np
import scipy.sparse as sps

import sparse as sp
import torch
from binsparse.conversions import from_numpy, from_sparse

from frameworks.saps_numpy import NumpyFramework
from frameworks.saps_pytorch import PytorchFramework
from frameworks.saps_scipy import SciPyFramework
from frameworks.saps_smart import SmartSparseFramework, SmartSparseKernels
from frameworks.saps_sparse import PyDataSparseFramework
from saps.benchmarks.jacobi import JacobiBenchmark
from saps.benchmarks.pcg import JacobiPCGBenchmark


@pytest.fixture(
    params=[
        NumpyFramework,
        SciPyFramework,
        PyDataSparseFramework,
        SmartSparseFramework,
        PytorchFramework,
        "tagger",
    ]
)
def xp(request, monkeypatch):
    if request.param == "tagger":
        monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "frameworks"))
        return importlib.import_module("frameworks.saps_tagger").TaggerFramework()
    return request.param()


def dense(array):
    if hasattr(array, "array"):
        array = array.array
    if isinstance(array, torch.Tensor):
        return array.to_dense().detach().cpu().numpy()
    if hasattr(array, "todense"):
        return np.asarray(array.todense())
    return np.asarray(array)


@pytest.mark.parametrize(
    "old,new,expected",
    [
        (0, 7, [7, 2, np.nan, 2]),
        (2, 0, [0, 0, np.nan, 0]),
        (np.nan, 0, [0, 2, 0, 2]),
        (9, 1, [0, 2, np.nan, 2]),
    ],
)
def test_replace_values_without_mutating_input(xp, old, new, expected):
    values = np.array([0.0, 2.0, np.nan, 2.0])
    array = xp.from_binsparse(from_numpy(values.copy()))
    result = xp.replace(array, old, new)
    np.testing.assert_array_equal(dense(result), expected)
    np.testing.assert_array_equal(dense(array), values)


def test_replace_promotes_integer_values(xp):
    array = xp.from_binsparse(from_numpy(np.array([0, 2], dtype=np.int64)))
    np.testing.assert_array_equal(dense(xp.replace(array, 2, 1.5)), [0, 1.5])


@pytest.mark.parametrize("shape", [(), (0, 3)])
def test_replace_scalar_and_empty_arrays(xp, shape):
    array = xp.from_binsparse(from_numpy(np.ones(shape)))
    result = dense(xp.replace(array, 1, 2))
    np.testing.assert_array_equal(result, np.full(shape, 2))


@pytest.mark.parametrize("framework", [PyDataSparseFramework, SmartSparseKernels])
@pytest.mark.parametrize("format", ["coo", "gcxs", "dok"])
def test_replace_sparse_fill_and_stored_values(framework, format):
    array = sp.COO(
        [[1, 2, 3]], [0.0, 2.0, 7.0], shape=(1_000_000_000,), fill_value=7
    ).asformat(format)
    result = framework().replace(array, 7, 3)
    assert isinstance(result, sp.SparseArray)
    assert result.fill_value == 3
    assert result.nnz == 2
    np.testing.assert_array_equal(
        result.asformat("coo")[[0, 1, 2, 3]].todense(), [3, 0, 2, 3]
    )
    assert array.fill_value == 7


@pytest.mark.parametrize("framework", [PyDataSparseFramework, SmartSparseFramework])
@pytest.mark.parametrize("format", ["coo", "gcxs", "dok"])
def test_replace_nan_fill_and_explicit_nan(framework, format):
    array = sp.COO([[1, 2]], [np.nan, 5.0], shape=(4,), fill_value=np.nan).asformat(
        format
    )
    result = framework().replace(array, np.nan, 0)
    assert result.fill_value == 0
    np.testing.assert_array_equal(result.todense(), [0, 0, 5, 0])
    assert np.isnan(array.fill_value)


@pytest.mark.parametrize("framework", ["scipy", "torch"])
@pytest.mark.parametrize("format", ["coo", "csr", "csc"])
def test_replace_zero_fill_backends(framework, format):
    values = np.array([[0.0, 2.0, np.nan], [2.0, 0.0, 4.0]])
    if framework == "scipy":
        xp = SciPyFramework()
        array = getattr(sps, f"{format}_array")(values)
    else:
        xp = PytorchFramework()
        array = torch.as_tensor(values.copy()).to_sparse(
            layout=getattr(torch, f"sparse_{format}")
        )
    changed = xp.replace(array, 2, 0)
    assert (
        sps.issparse(changed)
        if framework == "scipy"
        else changed.layout == array.layout
    )
    np.testing.assert_array_equal(dense(changed), [[0, 0, np.nan], [0, 0, 4]])
    np.testing.assert_array_equal(
        dense(xp.replace(array, 0, 3)), [[3, 2, np.nan], [2, 3, 4]]
    )
    np.testing.assert_array_equal(
        dense(xp.replace(array, np.nan, 0)), [[0, 2, 0], [2, 0, 4]]
    )
    np.testing.assert_array_equal(dense(array), values)


def test_replace_combines_duplicate_sparse_coordinates_first():
    scipy_array = sps.coo_array(([2.0, 3.0], ([0, 0], [1, 1])), shape=(2, 3))
    torch_array = torch.sparse_coo_tensor([[0, 0], [1, 1]], [2.0, 3.0], (2, 3))
    for xp, array in (
        (SciPyFramework(), scipy_array),
        (PytorchFramework(), torch_array),
    ):
        np.testing.assert_array_equal(dense(xp.replace(array, 5, 0)), np.zeros((2, 3)))
        assert dense(array)[0, 1] == 5


def test_tagger_records_replace_and_nonzero_fill(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "frameworks"))
    xp = importlib.import_module("frameworks.saps_tagger").TaggerFramework()
    array = xp.from_binsparse(from_sparse(sp.asarray([0.0, 2.0])))
    result = xp.replace(array, 0, 1)
    np.testing.assert_array_equal(dense(result), [1, 2])
    assert xp.stats["operators"]["replace"] == 1
    assert xp.stats["operator_arg_counts"]["replace"] == [3]
    assert result.elementwise_ops_since_reduction == 1
    assert {"feature-nonzero-fill", "feature-fancy-ops"} <= set(xp.tags)


@pytest.mark.parametrize(
    "xp", [NumpyFramework, PyDataSparseFramework, SmartSparseFramework], indirect=True
)
def test_jacobi_keeps_implicit_diagonal_fill_behavior(xp):
    matrix = xp.from_binsparse(from_sparse(sp.asarray([[0.0, 0.0], [0.0, 3.0]])))
    result = JacobiBenchmark().benchmark(
        xp, {}, matrix, xp.asarray([0.0, 6.0]), xp.asarray([0.0, 0.0])
    )
    np.testing.assert_array_equal(dense(result), [0, 2])


@pytest.mark.parametrize(
    "xp", [NumpyFramework, PyDataSparseFramework, SmartSparseFramework], indirect=True
)
def test_jacobi_preconditioning_replaces_nan(xp):
    matrix = xp.from_binsparse(
        from_sparse(sp.asarray([[0.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]))
    )
    preconditioner = xp.from_binsparse(from_sparse(sp.asarray([0.0, 2.0, 3.0])))
    with np.errstate(invalid="ignore", divide="ignore"):
        result = JacobiPCGBenchmark().benchmark(
            xp,
            {},
            matrix,
            xp.asarray([0.0, 4.0, 0.0]),
            xp.asarray([0.0, 0.0, 0.0]),
            preconditioner,
        )
    np.testing.assert_array_equal(dense(result), [0, 2, 0])
