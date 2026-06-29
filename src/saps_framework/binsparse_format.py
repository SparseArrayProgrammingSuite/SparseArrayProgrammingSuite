import json

import numpy as np

from pyparsing import Any


def canonicalize_coo(
    indices: tuple[np.ndarray, ...], values: np.ndarray
) -> tuple[tuple[np.ndarray, ...], np.ndarray]:
    """Return COO (indices, values) sorted lexicographically with duplicate
    coordinates summed.
    """
    indices = [np.asarray(idx) for idx in indices]
    values = np.asarray(values)
    if values.size == 0:
        return tuple(indices), values

    # lexsort's last key is primary, so reverse to sort by indices[0] first,
    # then indices[1], and so on.
    order = np.lexsort(tuple(reversed(indices)))
    indices = [idx[order] for idx in indices]
    values = values[order]

    # After sorting, duplicate coordinates are adjacent. Mark the first entry of
    # each distinct coordinate and accumulate values per group.
    new_group = np.zeros(values.shape[0], dtype=bool)
    new_group[0] = True
    for idx in indices:
        new_group[1:] |= idx[1:] != idx[:-1]
    group_ids = np.cumsum(new_group) - 1
    summed = np.zeros(int(group_ids[-1]) + 1, dtype=values.dtype)
    np.add.at(summed, group_ids, values)
    first = np.nonzero(new_group)[0]
    return tuple(idx[first] for idx in indices), summed


class BinsparseFormat:
    def __init__(self, data):
        self.data = data

    @staticmethod
    def from_numpy(array: np.ndarray) -> "BinsparseFormat":
        data: dict[str, Any] = {}
        data["format"] = "dense"
        data["shape"] = array.shape
        data["values"] = array.flatten()
        return BinsparseFormat(data)

    @staticmethod
    def from_coo(
        I_tuple: tuple[np.ndarray, ...], V: np.ndarray, shape: tuple[int, ...]
    ) -> "BinsparseFormat":
        I_tuple, V = canonicalize_coo(I_tuple, V)  # sanity check
        data: dict[str, Any] = {}
        data["format"] = "COO"
        for i in range(len(I_tuple)):
            data["indices_" + str(i)] = I_tuple[i]
        data["values"] = V
        data["shape"] = shape
        return BinsparseFormat(data)

    @staticmethod
    def to_coo(binsparse: "BinsparseFormat") -> "BinsparseFormat":
        if binsparse.data["format"] == "COO":
            return binsparse
        if binsparse.data["format"] == "dense":
            shape = binsparse.data["shape"]
            values = binsparse.data["values"].reshape(shape)
            indices = np.nonzero(values)
            V = values[indices]
            return BinsparseFormat.from_coo(indices, V, shape)
        raise ValueError("Unsupported format: " + binsparse.data["format"])

    def serialize(self) -> str:
        bytes_data = {}
        for k, v in self.data.items():
            if isinstance(v, np.ndarray):
                bytes_data[k] = (str(v.dtype), v.tobytes().hex())
            else:
                bytes_data[k] = v
        return json.dumps(bytes_data, sort_keys=True)

    @staticmethod
    def deserialize(string_data: str) -> "BinsparseFormat":
        data = json.loads(string_data)
        for k, v in data.items():
            if isinstance(v, list):
                data[k] = tuple(v)
                v = data[k]
            if (
                isinstance(v, tuple)
                and len(v) == 2
                and isinstance(v[0], str)
                and isinstance(v[1], str)
            ):
                data[k] = np.frombuffer(bytes.fromhex(v[1]), dtype=v[0])
        return BinsparseFormat(data)

    def __eq__(self, value):
        if not isinstance(value, BinsparseFormat):
            return NotImplemented
        for key in self.data:
            if key not in value.data:
                return False
            if isinstance(self.data[key], np.ndarray) and isinstance(
                value.data[key], np.ndarray
            ):
                if not np.array_equal(self.data[key], value.data[key]):
                    return False
            else:
                if self.data[key] != value.data[key]:
                    return False

        return all(key in self.data for key in value.data)
