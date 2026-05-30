import numpy as np

import array_api_compat
import torch

from saps_framework import BinsparseFormat, Framework, einsum


class PytorchFramework(Framework):
    # TODO: Check if pyproject restricted dependencies is necessary
    def __init__(self, sparse_layout: str = "COO"):
        self.sparse_layout = sparse_layout

    def from_binsparse(self, array):
        if array.data["format"] == "dense":
            return torch.from_numpy(
                np.asarray(array.data["values"]).reshape(array.data["shape"])
            )
        if array.data["format"] == "COO":
            indices = []
            idx_dim = 0
            while "indices_" + str(idx_dim) in array.data:
                indices.append(array.data["indices_" + str(idx_dim)])
                idx_dim += 1

            coo = torch.sparse_coo_tensor(
                torch.from_numpy(np.stack(indices).astype(np.int64, copy=False)),
                torch.from_numpy(np.asarray(array.data["values"])),
                size=array.data["shape"],
            ).coalesce()

            if self.sparse_layout == "CSR":
                if len(array.data["shape"]) != 2:
                    raise ValueError("PyTorch CSR only works for 2D matrices")
                return coo.to_sparse_csr()

            return coo

        raise ValueError("Unsupported format: " + array.data["format"])

    def to_binsparse(self, array):
        return BinsparseFormat.from_pytorch(array)

    def lazy(self, array):
        return array

    def compute(self, array):
        return array

    def einsum(self, prgm, **kwargs):
        xp = array_api_compat.array_namespace(*kwargs.values(), use_compat=True)
        return einsum(xp, prgm, **kwargs)

    def with_fill_value(self, array, value):
        return array

    def vecdot(self, x, y):
        return torch.sum(x * y)

    def __getattr__(self, name):
        return getattr(torch, name)


xp = PytorchFramework()
