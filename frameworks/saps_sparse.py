import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla

import array_api_compat
import array_api_compat.numpy as compat_np
import sparse as sp

from saps_framework import BinsparseFormat, Framework, einsum

sparse_namespace = sp.asarray([0]).__array_namespace__()


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
                self._scipy_sparse(arg) if hasattr(arg, "ndim") else arg for arg in args
            )
            kwargs = {
                key: self._scipy_sparse(value) if hasattr(value, "ndim") else value
                for key, value in kwargs.items()
            }
            return self._wrap_result(attr(*args, **kwargs))

        return wrapped


class PyDataSparseFramework(Framework):
    _sparse_first = {"asarray", "eye", "ones"}

    def __init__(self):
        self._modules = [sp, sparse_namespace, np]

    @staticmethod
    def _has_sparse_arg(*args, **kwargs):
        return any(isinstance(arg, sp.SparseArray) for arg in args) or any(
            isinstance(value, sp.SparseArray) for value in kwargs.values()
        )

    @staticmethod
    def _array_namespace(*arrays):
        return array_api_compat.array_namespace(*arrays, use_compat=True)

    def from_binsparse(self, array):
        if array.data["format"] == "dense":
            return np.asarray(array.data["values"]).reshape(array.data["shape"])
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
        if all(not isinstance(value, sp.SparseArray) for value in kwargs.values()):
            xp = self._array_namespace(*kwargs.values())
            return einsum(xp, prgm, **kwargs)
        return einsum(sp, prgm, **kwargs)

    def diagonal(self, a, *args, **kwargs):
        if isinstance(a, sp.SparseArray):
            return sp.diagonal(a, *args, **kwargs)
        xp = self._array_namespace(a)
        return xp.diagonal(a, *args, **kwargs)

    def matmul(self, x1, x2, /, **kwargs):
        if isinstance(x1, sp.SparseArray) or isinstance(x2, sp.SparseArray):
            return x1 @ x2
        xp = self._array_namespace(x1, x2)
        return xp.matmul(x1, x2, **kwargs)

    def zeros(self, shape, *args, **kwargs):
        return compat_np.zeros(shape, *args, **kwargs)

    def arange(self, *args, **kwargs):
        return compat_np.arange(*args, **kwargs)

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
        sparse_attr = getattr(sp, name, None)
        compat_attr = getattr(sparse_namespace, name, None)
        if name in self._sparse_first and sparse_attr is not None:
            return sparse_attr
        if sparse_attr is not None and compat_attr is not None:

            def wrapped(*args, **kwargs):
                attr = (
                    sparse_attr
                    if self._has_sparse_arg(*args, **kwargs)
                    else compat_attr
                )
                return attr(*args, **kwargs)

            return wrapped

        for attr in (sparse_attr, compat_attr, getattr(np, name, None)):
            if attr is not None:
                return attr

        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")


xp = PyDataSparseFramework()
