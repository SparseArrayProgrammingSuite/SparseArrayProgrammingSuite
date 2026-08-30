"""Small SAPS integration helpers for the reference ``binsparse`` package."""

import json
from typing import Any

import numpy as np

from binsparse import (
    BinsparseTensor,
    COORMatrix,
    CustomTensor,
    ElementLevel,
    InMemoryBinsparseContainer,
    SparseLevel,
)
from binsparse.conversions import from_numpy as _from_numpy
from binsparse.conversions import to_numpy, to_scipy


def from_numpy(array: np.ndarray) -> BinsparseTensor:
    return _from_numpy(np.asarray(array))


def from_coo(
    indices: tuple[np.ndarray, ...], values: np.ndarray, shape: tuple[int, ...]
) -> BinsparseTensor:
    values = np.asarray(values)
    if len(shape) == 2:
        return COORMatrix(
            tuple(shape),
            int(values.size),
            indices_0=np.asarray(indices[0]),
            indices_1=np.asarray(indices[1]),
            values=values,
        )
    return CustomTensor(
        tuple(shape),
        int(values.size),
        level=SparseLevel(
            len(shape),
            ElementLevel(values),
            tuple(np.asarray(index) for index in indices),
        ),
    )


def _coo_arrays(
    tensor: BinsparseTensor,
) -> tuple[tuple[np.ndarray, ...], np.ndarray]:
    if (
        isinstance(tensor, CustomTensor)
        and tensor.transpose is None
        and isinstance(tensor.level, SparseLevel)
        and tensor.level.rank == len(tensor.shape)
        and tensor.level.pointers_to_next is None
        and isinstance(tensor.level.level, ElementLevel)
    ):
        return tensor.level.indices, tensor.level.level.values
    if all(hasattr(tensor, name) for name in ("indices_0", "indices_1", "values")):
        return (tensor.indices_0, tensor.indices_1), tensor.values
    raise TypeError(f"{type(tensor).__name__} is not a flat COO tensor")


def to_coo(tensor: BinsparseTensor) -> BinsparseTensor:
    try:
        indices, values = _coo_arrays(tensor)
        return from_coo(indices, values, tensor.shape)
    except TypeError:
        try:
            array = to_numpy(tensor)
            indices = np.nonzero(array)
            return from_coo(indices, array[indices], array.shape)
        except TypeError:
            coo = to_scipy(tensor).tocoo()
            return from_coo((coo.row, coo.col), coo.data, coo.shape)


def tensor_data(tensor: BinsparseTensor) -> dict[str, Any]:
    try:
        array = to_numpy(tensor, copy=False)
    except TypeError:
        indices, values = _coo_arrays(tensor)
        return {
            "format": "COO",
            "shape": tensor.shape,
            "values": values,
            **{f"indices_{i}": index for i, index in enumerate(indices)},
        }
    return {"format": "dense", "shape": tensor.shape, "values": array.reshape(-1)}


def serialize(tensor: BinsparseTensor) -> str:
    container = InMemoryBinsparseContainer()
    tensor.serialize(container)
    buffers = [
        (str(buffer.dtype), buffer.tobytes().hex()) for buffer in container.buffers
    ]
    return json.dumps({"header": container.header, "buffers": buffers}, sort_keys=True)


def deserialize(value: str) -> BinsparseTensor:
    encoded = json.loads(value)
    if "header" not in encoded:
        data = {
            key: (
                np.frombuffer(bytes.fromhex(item[1]), dtype=item[0])
                if isinstance(item, list)
                and len(item) == 2
                and all(isinstance(part, str) for part in item)
                else tuple(item)
                if isinstance(item, list)
                else item
            )
            for key, item in encoded.items()
        }
        if data["format"] == "dense":
            return from_numpy(data["values"].reshape(data["shape"]))
        indices = tuple(data[f"indices_{i}"] for i in range(len(data["shape"])))
        return from_coo(indices, data["values"], data["shape"])
    buffers = [
        np.frombuffer(bytes.fromhex(item), dtype=dtype)
        for dtype, item in encoded["buffers"]
    ]
    return BinsparseTensor.parse(
        InMemoryBinsparseContainer(encoded["header"], buffers)
    )


def to_scipy_coo(tensor: BinsparseTensor) -> Any:
    import scipy.sparse as sp

    indices, values = _coo_arrays(to_coo(tensor))
    return sp.coo_matrix((values, (indices[0], indices[1])), shape=tensor.shape)


def diagonal(tensor: BinsparseTensor) -> np.ndarray:
    indices, values = _coo_arrays(to_coo(tensor))
    result = np.zeros(min(tensor.shape), dtype=values.dtype)
    on_diagonal = indices[0] == indices[1]
    np.add.at(result, indices[0][on_diagonal], values[on_diagonal])
    return result


def equal(left: BinsparseTensor, right: BinsparseTensor) -> bool:
    left_data, right_data = tensor_data(left), tensor_data(right)
    return left_data.keys() == right_data.keys() and all(
        np.array_equal(left_data[key], right_data[key])
        if isinstance(left_data[key], np.ndarray)
        else left_data[key] == right_data[key]
        for key in left_data
    )


# Remove once benchmark bodies use tensor attributes/converters throughout. These
# properties do not wrap or replace reference objects; inputs remain BinsparseTensor.
BinsparseTensor.data = property(tensor_data)
BinsparseTensor.from_numpy = staticmethod(from_numpy)
BinsparseTensor.from_coo = staticmethod(from_coo)
BinsparseTensor.to_coo = staticmethod(to_coo)
BinsparseTensor.to_scipy_coo = to_scipy_coo
BinsparseTensor.diagonal = diagonal


def _tensor_classes(root: type[BinsparseTensor]):
    for child in root.__subclasses__():
        yield child
        yield from _tensor_classes(child)


for _tensor_class in _tensor_classes(BinsparseTensor):
    _tensor_class.__eq__ = equal
