import numpy as np
import scipy as sp
import scipy.linalg as la
import scipy.sparse as sps
import scipy.sparse.linalg as spla

from ..binsparse_format import BinsparseFormat
from .abstract_framework import AbstractFramework
from .einsum import einsum


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


class SciPyFramework(AbstractFramework):
    def __init__(self):
        self._modules = [sp, sps, np]

    @property
    def linalg(self):
        return ScipyLinalg

    def from_benchmark(self, array):
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

    def to_benchmark(self, array):
        return BinsparseFormat.from_scipy(array)

    def lazy(self, array):
        return array

    def compute(self, array):
        return array

    def diagonal(self, array, **kwargs):
        if hasattr(array, "diagonal"):
            return array.diagonal(**kwargs)
        return np.diagonal(array, **kwargs)

    def einsum(self, prgm, **kwargs):
        dense_kwargs = {}
        for key, value in kwargs.items():
            if hasattr(value, "toarray"):
                dense_kwargs[key] = value.toarray()
            else:
                dense_kwargs[key] = value
        return einsum(np, prgm, **dense_kwargs)

    def with_fill_value(self, array, value):
        return array

    def __getattr__(self, name):
        for module in self._modules:
            if hasattr(module, name):
                return getattr(module, name)

        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")
