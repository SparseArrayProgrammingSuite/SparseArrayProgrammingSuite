import json
from contextlib import contextmanager
from functools import reduce
from operator import mul

import numpy as np

_DTYPE_TO_STR = {
    np.dtype("int8"): "int8",
    np.dtype("int16"): "int16",
    np.dtype("int32"): "int32",
    np.dtype("int64"): "int64",
    np.dtype("uint8"): "uint8",
    np.dtype("uint16"): "uint16",
    np.dtype("uint32"): "uint32",
    np.dtype("uint64"): "uint64",
    np.dtype("float32"): "float32",
    np.dtype("float64"): "float64",
    np.dtype("bool"): "bint8",
}
_STR_TO_DTYPE = {v: k for k, v in _DTYPE_TO_STR.items()}


def _dtype_to_str(dtype: np.dtype) -> str:
    result = _DTYPE_TO_STR.get(dtype)
    if result is None:
        raise ValueError(f"unrecognized dtype: {dtype}")
    return result


def _str_to_dtype(name: str) -> np.dtype:
    result = _STR_TO_DTYPE.get(name)
    if result is None:
        raise ValueError(f"unrecognized dtype string: {name}")
    return result


def _write_attr(group, key: str, value: dict) -> None:
    """Write a dict as a JSON-encoded HDF5 attribute, or direct Zarr attr."""
    try:
        import h5py

        if isinstance(group, (h5py.Group, h5py.Dataset)):
            group.attrs[key] = json.dumps(value)
            return
    except ImportError:
        pass
    # Zarr stores dicts directly
    group.attrs[key] = value


def _read_attr(group, key: str) -> dict:
    """Read a binsparse attribute from h5py or Zarr group."""
    try:
        import h5py

        if isinstance(group, (h5py.Group, h5py.Dataset)):
            return json.loads(group.attrs[key])
    except ImportError:
        pass
    # Zarr stores dicts directly; json.loads handles if stored as string
    raw = group.attrs[key]
    return json.loads(raw) if isinstance(raw, str) else dict(raw)


@contextmanager
def _open_group(path: str, mode: str):
    """Open an h5py File or Zarr group depending on path prefix."""
    if path.startswith("s3://"):
        try:
            import s3fs
            import zarr
        except ImportError as e:
            raise ImportError(
                "S3 support requires zarr and s3fs: pip install zarr s3fs"
            ) from e
        fs = s3fs.S3FileSystem()
        store = s3fs.S3Map(path, s3=fs)
        yield zarr.open_group(store, mode=mode)
    else:
        try:
            import h5py
        except ImportError as e:
            raise ImportError(
                "Local HDF5 support requires h5py: pip install h5py"
            ) from e
        with h5py.File(path, mode) as f:
            yield f


def write_binsparse(store, key: str, bf) -> None:
    """Write one BinsparseFormat to an h5py or Zarr group under key."""
    fmt = bf.data["format"]
    if fmt == "COO":
        _write_coo(store, key, _validated_coo_data(bf.data))
    elif fmt == "dense":
        _write_dense(store, key, _validated_dense_data(bf.data))
    else:
        raise ValueError(f"Unsupported BinsparseFormat format: {fmt!r}")


def read_binsparse(store, key: str):
    """Read one BinsparseFormat from an h5py or Zarr group at key."""
    from sparseappbench.binsparse_format import BinsparseFormat

    meta = _read_attr(store[key], "binsparse")
    _validate_meta(meta)
    fmt = meta["format"]
    if fmt in ("COO", "COOR"):
        return _read_coo(store, key, meta, BinsparseFormat)
    if fmt in ("DVEC", "DMAT", "DMATR"):
        return _read_dense(store, key, meta, BinsparseFormat)
    raise ValueError(f"Unsupported binsparse format in file: {fmt!r}")


def _validate_meta(meta: dict) -> None:
    required = ("version", "format", "shape", "number_of_stored_values", "data_types")
    missing = [key for key in required if key not in meta]
    if missing:
        raise ValueError(f"binsparse metadata missing required keys: {missing}")
    if meta["version"] != "0.1":
        raise ValueError(f"Unsupported binsparse version: {meta['version']!r}")
    if not isinstance(meta["shape"], list | tuple) or not meta["shape"]:
        raise ValueError("binsparse metadata shape must be a non-empty list")
    if any(not isinstance(dim, int) or dim < 0 for dim in meta["shape"]):
        raise ValueError(f"Invalid binsparse shape: {meta['shape']!r}")


def _validated_coo_data(data: dict) -> dict:
    shape = tuple(data["shape"])
    if not shape:
        raise ValueError("COO shape must be non-empty")
    values = np.asarray(data["values"])
    nnz = len(values)

    indices = []
    for i, dim in enumerate(shape):
        key = f"indices_{i}"
        if key not in data:
            raise ValueError(f"COO data missing {key!r}")
        arr = np.asarray(data[key])
        if len(arr) != nnz:
            raise ValueError(f"{key} length must match values length")
        if not np.issubdtype(arr.dtype, np.integer):
            raise ValueError(f"{key} dtype must be an integer dtype")
        if len(arr) and (np.any(arr < 0) or np.any(arr >= dim)):
            raise ValueError(f"{key} contains out-of-bounds indices for dimension {i}")
        indices.append(arr)

    if nnz:
        order = np.lexsort(tuple(reversed(indices)))
        sorted_indices = [arr[order] for arr in indices]
        sorted_values = values[order]
        coords = np.stack(sorted_indices, axis=1)
        if len(coords) > 1 and np.any(np.all(coords[1:] == coords[:-1], axis=1)):
            raise ValueError("COO coordinates must not contain duplicates")
    else:
        sorted_indices = indices
        sorted_values = values

    return {
        "shape": shape,
        "values": sorted_values,
        **{f"indices_{i}": arr for i, arr in enumerate(sorted_indices)},
    }


def _validated_dense_data(data: dict) -> dict:
    shape = tuple(data["shape"])
    if len(shape) not in (1, 2):
        raise ValueError(
            "Dense binsparse writing only supports rank-1 DVEC and rank-2 DMAT"
        )
    values = np.asarray(data["values"])
    n_elements = reduce(mul, shape, 1)
    if len(values) != n_elements:
        raise ValueError("Dense values length must match product of shape")
    return {"shape": shape, "values": values}


def _write_coo(store, key: str, data: dict) -> None:
    ndim = len(data["shape"])
    nnz = len(data["values"])
    meta = {
        "version": "0.1",
        "format": "COO",
        "shape": list(data["shape"]),
        "number_of_stored_values": nnz,
        "data_types": {
            **{
                f"indices_{i}": _dtype_to_str(data[f"indices_{i}"].dtype)
                for i in range(ndim)
            },
            "values": _dtype_to_str(data["values"].dtype),
        },
    }
    grp = store.require_group(key)
    _write_attr(grp, "binsparse", meta)
    for i in range(ndim):
        arr = data[f"indices_{i}"]
        grp.create_dataset(f"indices_{i}", data=arr, shape=arr.shape)
    arr = data["values"]
    grp.create_dataset("values", data=arr, shape=arr.shape)


def _read_coo(store, key: str, meta: dict, BinsparseFormat):
    ndim = len(meta["shape"])
    expected = {f"indices_{i}" for i in range(ndim)} | {"values"}
    missing = sorted(name for name in expected if name not in store[key])
    if missing:
        raise ValueError(f"COO group missing required datasets: {missing}")
    data = {"format": "COO", "shape": tuple(meta["shape"])}
    for i in range(ndim):
        data[f"indices_{i}"] = np.array(store[key][f"indices_{i}"])
    data["values"] = np.array(store[key]["values"])
    validated = _validated_coo_data(data)
    return BinsparseFormat({"format": "COO", **validated})


def _write_dense(store, key: str, data: dict) -> None:
    shape = data["shape"]
    fmt = "DVEC" if len(shape) == 1 else "DMAT"
    n_elements = reduce(mul, shape, 1)
    meta = {
        "version": "0.1",
        "format": fmt,
        "shape": list(shape),
        "number_of_stored_values": n_elements,
        "data_types": {"values": _dtype_to_str(data["values"].dtype)},
    }
    grp = store.require_group(key)
    _write_attr(grp, "binsparse", meta)
    arr = data["values"]
    grp.create_dataset("values", data=arr, shape=arr.shape)


def _read_dense(store, key: str, meta: dict, BinsparseFormat):
    if meta["format"] == "DVEC" and len(meta["shape"]) != 1:
        raise ValueError("DVEC binsparse data must have rank 1")
    if meta["format"] in ("DMAT", "DMATR") and len(meta["shape"]) != 2:
        raise ValueError("DMAT binsparse data must have rank 2")
    if "values" not in store[key]:
        raise ValueError("Dense group missing required dataset: 'values'")
    values = np.array(store[key]["values"])
    expected = reduce(mul, tuple(meta["shape"]), 1)
    if len(values) != expected:
        raise ValueError("Dense values length must match product of shape")
    return BinsparseFormat(
        {
            "format": "dense",
            "shape": tuple(meta["shape"]),
            "values": values,
        }
    )


def save_benchmark(
    path: str,
    data: tuple,
    keys: list,
) -> None:
    """Save a tuple of BinsparseFormat objects to one file."""
    if len(data) != len(keys):
        raise ValueError(f"len(data)={len(data)} must equal len(keys)={len(keys)}")
    with _open_group(path, "w") as store:
        for key, bf in zip(keys, data, strict=True):
            write_binsparse(store, key, bf)


def load_benchmark(
    path: str,
    keys: list,
) -> tuple:
    """Load a tuple of BinsparseFormat objects from a file."""
    with _open_group(path, "r") as store:
        return tuple(read_binsparse(store, key) for key in keys)
