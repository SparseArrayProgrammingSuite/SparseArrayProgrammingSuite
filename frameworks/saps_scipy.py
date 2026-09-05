import numpy as np
import scipy as sp
import scipy.linalg as la
import scipy.sparse as sps
import scipy.sparse.linalg as spla

import array_api_compat
import array_api_compat.numpy as compat_np
from binsparse import (
    CustomTensor,
    DenseLevel,
    DMATCMatrix,
    DMATRMatrix,
    DVECVector,
    ElementLevel,
)
from binsparse.conversions import from_numpy, from_scipy, to_numpy, to_scipy

from saps_framework import Framework, einsum


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

    @staticmethod
    def lstsq(a, b, rcond=None, **kwargs):
        if rcond is not None:
            kwargs["cond"] = rcond
        return la.lstsq(a, b, **kwargs)


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
        match array:
            case DVECVector() | DMATRMatrix() | DMATCMatrix():
                return to_numpy(array)
            case CustomTensor(shape=(), transpose=None, level=ElementLevel()):
                return to_numpy(array)
            case CustomTensor(
                shape=shape,
                transpose=None,
                level=DenseLevel(rank=rank, level=ElementLevel()),
            ) if rank == len(shape):
                return to_numpy(array)
            case _:
                return to_scipy(array).tocsr()

    def to_binsparse(self, array):
        if sp.sparse.issparse(array):
            return from_scipy(array)
        if isinstance(array, np.ndarray):
            return from_numpy(array)
        if isinstance(array, np.matrix):
            return from_numpy(np.array(array))
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
