import numpy as np
import scipy as sp
import scipy.linalg as la
import scipy.sparse as sps
import scipy.sparse.linalg as spla

import array_api_compat
import array_api_compat.numpy as compat_np

from saps_framework import BinsparseFormat, Framework, einsum


class ScipyLinalg:
    @staticmethod
    def solve(A, b, **kwargs):
        b_dense = np.asarray(b).ravel()

        if sps.issparse(A):
            return spla.spsolve(A, b_dense, **kwargs)
        return la.solve(A, b_dense, **kwargs)

    @staticmethod
    def norm(x, **kwargs):
        if sps.issparse(x):
            return spla.norm(x, **kwargs)
        return np.linalg.norm(x, **kwargs)


class SciPyFramework(Framework):
    def __init__(self):
        self._modules = [sps, compat_np, sp, np]

    @staticmethod
    def _array_namespace(*arrays):
        return array_api_compat.array_namespace(*arrays, use_compat=True)

    @property
    def linalg(self):
        return ScipyLinalg

    def from_binsparse(self, array):
        if array.data["format"] == "dense":
            return array.data["values"].reshape(array.data["shape"])
        if array.data["format"] == "COO":
            indices = []
            idx_dim = 0
            while "indices_" + str(idx_dim) in array.data:
                indices.append(array.data["indices_" + str(idx_dim)])
                idx_dim += 1
            return sp.sparse.coo_array(
                (array.data["values"], tuple(indices)), shape=array.data["shape"]
            ).tocsr()
        raise ValueError(f"Unsupported format: {array.data['format']}")

    def to_binsparse(self, array):
        if sp.sparse.issparse(array):
            coo = array.tocoo()
            return BinsparseFormat.from_coo((coo.row, coo.col), coo.data, coo.shape)
        if isinstance(array, np.ndarray):
            return BinsparseFormat.from_numpy(array)
        if isinstance(array, np.matrix):
            return BinsparseFormat.from_numpy(np.array(array))
        raise TypeError(f"Type {type(array)} is not a recognized SciPy/NumPy format.")

    def lazy(self, array):
        return array

    def compute(self, array):
        return array

    def einsum(self, prgm, **kwargs):
        xp = self._array_namespace(*kwargs.values())
        return einsum(xp, prgm, **kwargs)

    def diagonal(self, a, *args, **kwargs):
        if sps.issparse(a):
            return a.diagonal(*args, **kwargs)
        xp = self._array_namespace(a)
        return xp.diagonal(a, *args, **kwargs)

    def matmul(self, x1, x2, /, **kwargs):
        if sps.issparse(x1) or sps.issparse(x2):
            return x1 @ x2
        xp = self._array_namespace(x1, x2)
        return xp.matmul(x1, x2, **kwargs)

    def with_fill_value(self, array, value):
        return array

    def __getattr__(self, name):
        for module in self._modules:
            if hasattr(module, name):
                return getattr(module, name)

        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")


xp = SciPyFramework()
