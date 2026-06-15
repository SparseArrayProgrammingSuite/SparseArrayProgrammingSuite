import json

import numpy as np

from pyparsing import Any


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
