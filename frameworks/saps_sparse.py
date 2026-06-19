import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla

import sparse as sp

from saps_framework import BinsparseFormat, Framework, einsum


class PyDataSparseLinalg:
    @staticmethod
    def _scipy_sparse(array):
        if hasattr(array, "to_scipy_sparse"):
            if array.ndim == 2:
                return array.to_scipy_sparse()
            if array.ndim == 1:
                return array.reshape((array.shape[0], 1)).to_scipy_sparse()
            raise ValueError(
                "SciPy sparse linalg only supports one- or two-dimensional arrays."
            )
        array = np.asarray(array)
        if array.ndim == 1:
            return sps.coo_matrix(array.reshape((-1, 1)))
        if array.ndim == 2:
            return sps.coo_matrix(array)
        raise ValueError(
            "SciPy sparse linalg only supports one- or two-dimensional arrays."
        )

    @staticmethod
    def _wrap_result(result):
        if sps.issparse(result):
            return sp.asarray(result)
        if isinstance(result, np.ndarray):
            return sp.asarray(result)
        if isinstance(result, tuple):
            return tuple(PyDataSparseLinalg._wrap_result(item) for item in result)
        return result

    @staticmethod
    def solve(A, b):
        A = PyDataSparseLinalg._scipy_sparse(A).tocsc()
        b = PyDataSparseLinalg._scipy_sparse(b)
        x = spla.spsolve(A, b)
        return PyDataSparseLinalg._wrap_result(x)

    @staticmethod
    def svd(A, full_matrices=False, k=None, **kwargs):
        if full_matrices:
            raise ValueError("Sparse SVD does not support full_matrices=True.")

        A = PyDataSparseLinalg._scipy_sparse(A)
        min_dim = min(A.shape)
        if min_dim <= 1:
            raise ValueError("Sparse SVD requires min(A.shape) > 1.")
        if k is None:
            k = min_dim - 1

        U, S, Vt = spla.svds(A, k=k, **kwargs)
        order = np.argsort(S)[::-1]
        return (
            sp.asarray(U[:, order]),
            sp.asarray(S[order]),
            sp.asarray(Vt[order, :]),
        )

    def __getattr__(self, name):
        attr = getattr(spla, name)

        def wrapped(*args, **kwargs):
            args = tuple(
                self._scipy_sparse(arg) if hasattr(arg, "ndim") else arg
                for arg in args
            )
            kwargs = {
                key: self._scipy_sparse(value) if hasattr(value, "ndim") else value
                for key, value in kwargs.items()
            }
            return self._wrap_result(attr(*args, **kwargs))

        return wrapped


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
        return PyDataSparseLinalg()

    def __getattr__(self, name):
        return getattr(sp, name)


xp = PyDataSparseFramework()
