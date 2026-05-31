import numpy as np

import array_api_compat
import torch

from saps_framework import BinsparseFormat, Framework, einsum


def _is_sparse_tensor(array):
    return array.layout in {
        torch.sparse_coo,
        torch.sparse_csr,
        torch.sparse_csc,
    }


class PytorchLinalg:
    @staticmethod
    def _dense(array):
        return array.to_dense() if _is_sparse_tensor(array) else array

    @staticmethod
    def solve(A, b, **kwargs):
        return torch.linalg.solve(PytorchLinalg._dense(A), b, **kwargs)

    @staticmethod
    def norm(x, **kwargs):
        return torch.linalg.norm(PytorchLinalg._dense(x), **kwargs)

    @staticmethod
    def lstsq(A, b, **kwargs):
        result = torch.linalg.lstsq(PytorchLinalg._dense(A), b, **kwargs)
        return result.solution, result.residuals, result.rank, result.singular_values


class PytorchFramework(Framework):
    def __init__(self, sparse_layout: str = "COO"):
        self.sparse_layout = sparse_layout

    @property
    def linalg(self):
        return PytorchLinalg

    def from_binsparse(self, array):
        if array.data["format"] == "dense":
            values = np.asarray(array.data["values"]).copy()
            return torch.from_numpy(values.reshape(array.data["shape"]))
        if array.data["format"] == "COO":
            indices = []
            idx_dim = 0
            while "indices_" + str(idx_dim) in array.data:
                indices.append(array.data["indices_" + str(idx_dim)])
                idx_dim += 1

            values = np.asarray(array.data["values"]).copy()
            coo = torch.sparse_coo_tensor(
                torch.from_numpy(np.stack(indices).astype(np.int64, copy=True)),
                torch.from_numpy(values),
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
        if not hasattr(xp, "power"):
            xp.power = torch.pow
        return einsum(xp, prgm, **kwargs)

    def with_fill_value(self, array, value):
        return array

    def maximum(self, x, y):
        if not torch.is_tensor(y):
            y = torch.as_tensor(y, dtype=x.dtype, device=x.device)
        return torch.maximum(x, y)

    def minimum(self, x, y):
        if not torch.is_tensor(y):
            y = torch.as_tensor(y, dtype=x.dtype, device=x.device)
        return torch.minimum(x, y)

    def vecdot(self, x, y):
        return torch.sum(torch.conj(x) * y, dim=-1)

    def __getattr__(self, name):
        return getattr(torch, name)


xp = PytorchFramework()
