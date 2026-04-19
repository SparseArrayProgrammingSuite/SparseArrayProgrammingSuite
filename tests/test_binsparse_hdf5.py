import json
from pathlib import Path

import pytest

import numpy as np
import scipy.sparse as scipy_sparse

import h5py

from sparseappbench.binsparse_format import BinsparseFormat
from sparseappbench.frameworks.sparse_framework import PyDataSparseFramework
from sparseappbench.io.binsparse_hdf5 import (
    _dtype_to_str,
    _open_group,
    _read_attr,
    _str_to_dtype,
    _write_attr,
    load_benchmark,
    read_binsparse,
    save_benchmark,
    write_binsparse,
)


def _make_coo(ndim, nnz=10, shape=None):
    rng = np.random.default_rng(42)
    if shape is None:
        shape = tuple(20 for _ in range(ndim))
    coords = set()
    while len(coords) < nnz:
        coords.add(tuple(int(rng.integers(0, shape[i])) for i in range(ndim)))
    ordered = np.array(sorted(coords), dtype=np.int32)
    indices = tuple(ordered[:, i] for i in range(ndim))
    values = rng.random(nnz).astype(np.float64)
    return BinsparseFormat.from_coo(indices, values, shape)


def _coo_as_dict(bf):
    ndim = len(bf.data["shape"])
    result = {}
    for pos, value in zip(
        zip(*(bf.data[f"indices_{i}"] for i in range(ndim)), strict=True),
        bf.data["values"],
        strict=True,
    ):
        result[tuple(int(i) for i in pos)] = value
    return result


def test_dtype_roundtrip():
    for dtype in [np.int32, np.int64, np.float32, np.float64, np.bool_]:
        name = _dtype_to_str(np.dtype(dtype))
        recovered = _str_to_dtype(name)
        assert recovered == np.dtype(dtype)


def test_dtype_unknown_raises():
    with pytest.raises(ValueError, match="unrecognized dtype"):
        _dtype_to_str(np.dtype("complex64"))


def test_write_read_attr_h5py(tmp_path):
    path = tmp_path / "attrs.h5"
    with h5py.File(path, "w") as hf:
        grp = hf.require_group("test")
        _write_attr(grp, "binsparse", {"format": "COO", "shape": [10, 20]})
        result = _read_attr(grp, "binsparse")
    assert result == {"format": "COO", "shape": [10, 20]}


def test_open_group_local(tmp_path):
    path = str(tmp_path / "test.h5")
    with _open_group(path, "w") as group:
        group.require_group("a")
    with _open_group(path, "r") as group:
        assert "a" in group


@pytest.mark.parametrize("ndim", [1, 2, 3, 4, 5])
def test_coo_roundtrip(tmp_path, ndim):
    bf = _make_coo(ndim)
    path = str(tmp_path / "test.h5")
    with h5py.File(path, "w") as file:
        write_binsparse(file, "tensor", bf)
    with h5py.File(path, "r") as file:
        result = read_binsparse(file, "tensor")
    assert result.data["format"] == "COO"
    assert result.data["shape"] == bf.data["shape"]
    assert _coo_as_dict(result) == _coo_as_dict(bf)


def test_coo_write_sorts_lexicographically(tmp_path):
    bf = BinsparseFormat.from_coo(
        (np.array([1, 0, 1], dtype=np.int32), np.array([0, 1, 1], dtype=np.int32)),
        np.array([10.0, 20.0, 30.0]),
        (2, 2),
    )
    path = str(tmp_path / "sorted.h5")
    with h5py.File(path, "w") as file:
        write_binsparse(file, "X", bf)
        np.testing.assert_array_equal(file["X"]["indices_0"][:], np.array([0, 1, 1]))
        np.testing.assert_array_equal(file["X"]["indices_1"][:], np.array([1, 0, 1]))
        np.testing.assert_array_equal(
            file["X"]["values"][:], np.array([20.0, 10.0, 30.0])
        )


def test_coo_rejects_duplicate_coordinates(tmp_path):
    bf = BinsparseFormat.from_coo(
        (np.array([0, 0], dtype=np.int32), np.array([1, 1], dtype=np.int32)),
        np.array([1.0, 2.0]),
        (2, 2),
    )
    with (
        h5py.File(tmp_path / "dupe.h5", "w") as file,
        pytest.raises(ValueError, match="duplicates"),
    ):
        write_binsparse(file, "X", bf)


def test_coo_rejects_out_of_bounds_indices(tmp_path):
    bf = BinsparseFormat.from_coo(
        (np.array([0, 2], dtype=np.int32), np.array([1, 1], dtype=np.int32)),
        np.array([1.0, 2.0]),
        (2, 2),
    )
    with (
        h5py.File(tmp_path / "bounds.h5", "w") as file,
        pytest.raises(ValueError, match="out-of-bounds"),
    ):
        write_binsparse(file, "X", bf)


def test_coo_metadata_spec_compliant(tmp_path):
    bf = _make_coo(3)
    path = str(tmp_path / "test.h5")
    with h5py.File(path, "w") as file:
        write_binsparse(file, "tensor", bf)
    with h5py.File(path, "r") as file:
        meta = json.loads(file["tensor"].attrs["binsparse"])
    assert meta["version"] == "0.1"
    assert meta["format"] == "COO"
    assert "shape" in meta
    assert "number_of_stored_values" in meta
    assert "data_types" in meta


@pytest.mark.parametrize("shape,expected_fmt", [((100,), "DVEC"), ((10, 20), "DMAT")])
def test_dense_roundtrip(tmp_path, shape, expected_fmt):
    rng = np.random.default_rng(0)
    arr = rng.random(shape).astype(np.float64)
    bf = BinsparseFormat.from_numpy(arr)
    path = str(tmp_path / "test.h5")
    with h5py.File(path, "w") as file:
        write_binsparse(file, "arr", bf)
    with h5py.File(path, "r") as file:
        result = read_binsparse(file, "arr")
        meta = json.loads(file["arr"].attrs["binsparse"])
    assert meta["format"] == expected_fmt
    assert meta["number_of_stored_values"] == arr.size
    assert result.data["format"] == "dense"
    assert result.data["shape"] == bf.data["shape"]
    np.testing.assert_array_equal(result.data["values"], bf.data["values"])


def test_dense_rank_greater_than_two_rejected(tmp_path):
    bf = BinsparseFormat.from_numpy(np.zeros((2, 3, 4), dtype=np.float64))
    with (
        h5py.File(tmp_path / "dense3d.h5", "w") as file,
        pytest.raises(ValueError, match="rank-1 DVEC and rank-2 DMAT"),
    ):
        write_binsparse(file, "X", bf)


def test_read_coor_alias(tmp_path):
    path = tmp_path / "coor.h5"
    with h5py.File(path, "w") as file:
        grp = file.require_group("X")
        _write_attr(
            grp,
            "binsparse",
            {
                "version": "0.1",
                "format": "COOR",
                "shape": [2, 3],
                "number_of_stored_values": 2,
                "data_types": {
                    "indices_0": "int32",
                    "indices_1": "int32",
                    "values": "float64",
                },
            },
        )
        grp.create_dataset("indices_0", data=np.array([0, 1], dtype=np.int32))
        grp.create_dataset("indices_1", data=np.array([2, 0], dtype=np.int32))
        grp.create_dataset("values", data=np.array([4.0, 5.0]))
    with h5py.File(path, "r") as file:
        result = read_binsparse(file, "X")
    assert result.data["format"] == "COO"
    assert result.data["shape"] == (2, 3)
    assert _coo_as_dict(result) == {(0, 2): 4.0, (1, 0): 5.0}


def test_read_dmatr_alias(tmp_path):
    path = tmp_path / "dmatr.h5"
    with h5py.File(path, "w") as file:
        grp = file.require_group("X")
        _write_attr(
            grp,
            "binsparse",
            {
                "version": "0.1",
                "format": "DMATR",
                "shape": [2, 3],
                "number_of_stored_values": 6,
                "data_types": {"values": "float64"},
            },
        )
        grp.create_dataset("values", data=np.arange(6.0))
    with h5py.File(path, "r") as file:
        result = read_binsparse(file, "X")
    assert result.data["format"] == "dense"
    assert result.data["shape"] == (2, 3)
    np.testing.assert_array_equal(result.data["values"], np.arange(6.0))


def test_write_binsparse_unsupported_format(tmp_path):
    bf = BinsparseFormat(
        {"format": "CSR", "shape": (10, 10), "values": np.array([1.0])}
    )
    with (
        h5py.File(tmp_path / "unsupported.h5", "w") as file,
        pytest.raises(ValueError, match="Unsupported BinsparseFormat format"),
    ):
        write_binsparse(file, "X", bf)


def test_read_binsparse_unsupported_format(tmp_path):
    path = tmp_path / "unsupported_read.h5"
    with h5py.File(path, "w") as file:
        grp = file.require_group("X")
        _write_attr(
            grp,
            "binsparse",
            {
                "version": "0.1",
                "format": "CSR",
                "shape": [10, 10],
                "number_of_stored_values": 0,
                "data_types": {},
            },
        )
    with (
        h5py.File(path, "r") as file,
        pytest.raises(ValueError, match="Unsupported binsparse format"),
    ):
        read_binsparse(file, "X")


def test_binsparse_format_save_load(tmp_path):
    bf = _make_coo(3, nnz=15, shape=(8, 10, 6))
    path = str(tmp_path / "bf.h5")
    bf.save(path, "X")
    result = BinsparseFormat.load(path, "X")
    assert _coo_as_dict(result) == _coo_as_dict(bf)


def test_zarr_write_read_binsparse(tmp_path):
    zarr = pytest.importorskip("zarr")
    bf = _make_coo(3, nnz=15, shape=(10, 12, 6))
    store_path = str(tmp_path / "data.zarr")
    store_w = zarr.open_group(store_path, mode="w")
    write_binsparse(store_w, "X", bf)

    store_r = zarr.open_group(store_path, mode="r")
    result = read_binsparse(store_r, "X")

    assert result.data["format"] == "COO"
    assert result.data["shape"] == bf.data["shape"]
    assert _coo_as_dict(result) == _coo_as_dict(bf)


@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
@pytest.mark.parametrize("value_dtype", [np.float32, np.float64])
def test_coo_dtype_preservation(tmp_path, index_dtype, value_dtype):
    rng = np.random.default_rng(9)
    shape = (20, 15)
    coords = np.array([(0, 0), (1, 2), (5, 6), (10, 3)], dtype=index_dtype)
    values = rng.random(len(coords)).astype(value_dtype)
    bf = BinsparseFormat.from_coo((coords[:, 0], coords[:, 1]), values, shape)
    path = str(tmp_path / "dtype.h5")
    with h5py.File(path, "w") as file:
        write_binsparse(file, "T", bf)
    with h5py.File(path, "r") as file:
        result = read_binsparse(file, "T")
    assert result.data["indices_0"].dtype == np.dtype(index_dtype)
    assert result.data["values"].dtype == np.dtype(value_dtype)


def test_save_load_benchmark_mixed(tmp_path):
    rng = np.random.default_rng(0)
    matrix = scipy_sparse.random(
        50, 50, density=0.1, format="coo", dtype=np.float64, random_state=rng
    )
    matrix.sum_duplicates()
    A_bf = BinsparseFormat.from_coo((matrix.row, matrix.col), matrix.data, matrix.shape)
    b_bf = BinsparseFormat.from_numpy(rng.random(50).astype(np.float64))
    x_bf = BinsparseFormat.from_numpy(rng.random(50).astype(np.float64))

    path = str(tmp_path / "cg.h5")
    save_benchmark(path, (A_bf, b_bf, x_bf), ["A", "b", "x"])
    A_r, b_r, x_r = load_benchmark(path, ["A", "b", "x"])

    assert _coo_as_dict(A_r) == _coo_as_dict(A_bf)
    np.testing.assert_array_equal(b_r.data["values"], b_bf.data["values"])
    np.testing.assert_array_equal(x_r.data["values"], x_bf.data["values"])


def test_save_load_benchmark_tensor(tmp_path):
    X_bf = _make_coo(3, nnz=20, shape=(10, 12, 8))
    rng = np.random.default_rng(1)
    A_bf = BinsparseFormat.from_numpy(rng.random((10, 4)).astype(np.float64))
    B_bf = BinsparseFormat.from_numpy(rng.random((12, 4)).astype(np.float64))
    C_bf = BinsparseFormat.from_numpy(rng.random((8, 4)).astype(np.float64))

    path = str(tmp_path / "cp_als.h5")
    save_benchmark(path, (X_bf, A_bf, B_bf, C_bf), ["X", "A", "B", "C"])
    X_r, A_r, B_r, C_r = load_benchmark(path, ["X", "A", "B", "C"])

    assert _coo_as_dict(X_r) == _coo_as_dict(X_bf)
    np.testing.assert_array_equal(A_r.data["values"], A_bf.data["values"])
    np.testing.assert_array_equal(B_r.data["values"], B_bf.data["values"])
    np.testing.assert_array_equal(C_r.data["values"], C_bf.data["values"])


def test_read_binsparse_loads_into_pydatasparse(tmp_path):
    bf = _make_coo(2, nnz=4, shape=(5, 6))
    path = tmp_path / "pydata.h5"
    with h5py.File(path, "w") as file:
        write_binsparse(file, "X", bf)
    with h5py.File(path, "r") as file:
        loaded = read_binsparse(file, "X")

    sparse_array = PyDataSparseFramework().from_benchmark(loaded)
    assert sparse_array.shape == bf.data["shape"]
    expected = PyDataSparseFramework().from_benchmark(bf).todense()
    np.testing.assert_array_equal(sparse_array.todense(), expected)


def test_optional_isaac_binsparse_interop(tmp_path):
    binsparse = pytest.importorskip("binsparse")
    matrix = scipy_sparse.coo_matrix(
        (np.array([1.0, 2.0]), (np.array([0, 1]), np.array([2, 0]))),
        shape=(2, 3),
    )
    path = tmp_path / "isaac.h5"
    with h5py.File(path, "w") as file:
        binsparse.write(file, "X", matrix)
    with h5py.File(path, "r") as file:
        try:
            loaded = read_binsparse(file, "X")
        except ValueError as exc:
            pytest.skip(f"Isaac binsparse wrote an unsupported format: {exc}")
    assert loaded.data["shape"] == matrix.shape


def test_optional_finch_interop():
    # Reads a pre-generated fixture that matches Finch.jl binsparse output format.
    # 2x3 matrix: A[0,2]=1.0, A[1,0]=2.0 (lexicographically sorted COO)
    fixture = Path(__file__).parent / "fixtures" / "finch_reference.h5"
    with h5py.File(fixture, "r") as f:
        bf = read_binsparse(f, "A")
    assert bf.data["shape"] == (2, 3)
    np.testing.assert_array_equal(bf.data["indices_0"], [0, 1])
    np.testing.assert_array_equal(bf.data["indices_1"], [2, 0])
    np.testing.assert_array_equal(bf.data["values"], [1.0, 2.0])
