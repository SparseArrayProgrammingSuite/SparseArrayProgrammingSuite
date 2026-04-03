import numpy as np

import torch

from ..binsparse_format import BinsparseFormat
from .abstract_framework import AbstractFramework


class PytorchFramework(AbstractFramework):
    def __init__(self, sparse_layout: str = "COO"):
        self.sparse_layout = sparse_layout

    def from_benchmark(self, array):
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

    def to_benchmark(self, array):
        layout = array.layout
        if layout in {torch.sparse_coo, torch.sparse_csr, torch.strided}:
            return BinsparseFormat.from_pytorch(array)
        raise ValueError("Unsupported array type: " + str(layout))

    def lazy(self, array):
        return array

    def compute(self, array):
        return array

    def einsum(self, prgm, **kwargs):
        pass

    def with_fill_value(self, array, value):
        return array

    def __getattr__(self, name):
        return getattr(torch, name)
