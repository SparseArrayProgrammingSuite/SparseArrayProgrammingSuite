import numpy as np
import scipy.sparse.linalg as spla

import sparse as sp

from saps_framework import BinsparseFormat, Framework, einsum


class PyDataSparseLinalg:
    @staticmethod
    def solve(A, b):
        if hasattr(A, "to_scipy_sparse"):
            A = A.to_scipy_sparse()

        if hasattr(b, "todense"):
            b = np.asarray(b.todense()).ravel()
        else:
            b = np.asarray(b).ravel()

        x = spla.spsolve(A, b)
        return sp.asarray(x)


class PyDataSparseFramework(Framework):
    def __init__(self):
        pass

    def from_binsparse(self, array):
        if array.data["format"] == "dense":
            return sp.asarray(array.data["values"].reshape(array.data["shape"]))
        if array.data["format"] == "COO":
            indices = []
            idx_dim = 0
            while "indices_" + str(idx_dim) in array.data:
                indices.append(array.data["indices_" + str(idx_dim)])
                idx_dim += 1
            V = array.data["values"]
            shape = array.data["shape"]
            return sp.COO(tuple(indices), V, shape=shape, fill_value=0)
        raise ValueError("Unsupported format: " + array.data["format"])

    def to_binsparse(self, array):
        if isinstance(array, sp.COO):
            return BinsparseFormat.from_coo(array.coords, array.data, array.shape)
        if isinstance(array, sp.SparseArray):
            return self.to_benchmark(array.tocoo())
        if isinstance(array, np.ndarray):
            return BinsparseFormat.from_numpy(array)
        raise ValueError("Unsupported array type: " + str(type(array)))

    def lazy(self, array):
        return array

    def compute(self, array):
        return array

    def einsum(self, prgm, **kwargs):
        return einsum(sp, prgm, **kwargs)

    def with_fill_value(self, array, value):
        if isinstance(array, sp.SparseArray):
            res = array.copy(deep=False)
            res.fill_value = array.dtype.type(value)
            return res
        return array

    @property
    def linalg(self):
        return PyDataSparseLinalg

    def __getattr__(self, name):
        return getattr(sp, name)


xp = PyDataSparseFramework()
