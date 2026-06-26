import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla

import array_api_compat
import array_api_compat.numpy as compat_np
import sparse as sp

from saps_framework import BinsparseFormat, Framework, einsum


class PyDataSparseLinalg:
    @staticmethod
    def _dense(array):
        if hasattr(array, "todense"):
            return np.asarray(array.todense())
        if hasattr(array, "toarray"):
            return np.asarray(array.toarray())
        return np.asarray(array)

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
    def pinv(A, **kwargs):
        return np.linalg.pinv(PyDataSparseLinalg._dense(A), **kwargs)

    @staticmethod
    def lstsq(A, b, **kwargs):
        return np.linalg.lstsq(
            PyDataSparseLinalg._dense(A), PyDataSparseLinalg._dense(b), **kwargs
        )

    @staticmethod
    def svd(A, full_matrices=False, k=None, **kwargs):
        if not isinstance(A, sp.SparseArray):
            U, S, Vt = np.linalg.svd(
                np.asarray(A), full_matrices=full_matrices, **kwargs
            )
            if k is not None:
                U = U[:, :k]
                S = S[:k]
                Vt = Vt[:k, :]
            return U, S, Vt

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
    _sparse_first: set[str] = set()
    _dtype_attrs = {
        "bool",
        "float32",
        "float64",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
    }

    def __init__(self):
        self._modules = [sp, compat_np, np]

    @staticmethod
    def _has_sparse_arg(*args, **kwargs):
        return any(isinstance(arg, sp.SparseArray) for arg in args) or any(
            isinstance(value, sp.SparseArray) for value in kwargs.values()
        )

    @staticmethod
    def _array_namespace(*arrays):
        return array_api_compat.array_namespace(*arrays, use_compat=True)

    @staticmethod
    def _fill_value_is_zero(array):
        fill_value = getattr(array, "fill_value", 0)
        return np.all(np.asarray(fill_value) == 0)

    @staticmethod
    def _dense(array):
        if hasattr(array, "todense"):
            return np.asarray(array.todense())
        if hasattr(array, "toarray"):
            return np.asarray(array.toarray())
        return np.asarray(array)

    @staticmethod
    def _sparse_compatible_arg(arg):
        if isinstance(arg, np.ndarray):
            return sp.asarray(arg)
        return arg

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
            if array.ndim == 0 or not self._fill_value_is_zero(array):
                return BinsparseFormat.from_numpy(self._dense(array))
            return BinsparseFormat.from_coo(array.coords, array.data, array.shape)
        if isinstance(array, sp.SparseArray):
            if array.ndim == 0 or not self._fill_value_is_zero(array):
                return BinsparseFormat.from_numpy(self._dense(array))
            return self.to_binsparse(array.tocoo())
        if isinstance(array, np.ndarray):
            return BinsparseFormat.from_numpy(array)
        if np.isscalar(array):
            return BinsparseFormat.from_numpy(np.asarray(array))
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

    def asarray(self, obj, *args, **kwargs):
        if isinstance(obj, sp.SparseArray):
            return sp.asarray(obj, *args, **kwargs)
        return compat_np.asarray(obj, *args, **kwargs)

    def array(self, obj, *args, **kwargs):
        if isinstance(obj, sp.SparseArray):
            return sp.asarray(obj, *args, **kwargs)
        return np.array(obj, *args, **kwargs)

    def eye(self, *args, **kwargs):
        return compat_np.eye(*args, **kwargs)

    def ones(self, *args, **kwargs):
        return compat_np.ones(*args, **kwargs)

    def expand_dims(self, a, axis):
        if isinstance(a, sp.SparseArray):
            return sp.expand_dims(a, axis=axis)
        xp = self._array_namespace(a)
        return xp.expand_dims(a, axis=axis)

    def take(self, x, indices, /, *args, **kwargs):
        if isinstance(indices, sp.SparseArray):
            indices = self._dense(indices)
        if isinstance(x, sp.SparseArray):
            return sp.take(x, indices, *args, **kwargs)
        xp = self._array_namespace(x)
        return xp.take(x, indices, *args, **kwargs)

    def item(self, array):
        if isinstance(array, sp.SparseArray):
            return self._dense(array).item()
        return array.item()

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
        compat_attr = getattr(compat_np, name, None)
        if name in self._dtype_attrs and compat_attr is not None:
            return compat_attr
        if name in self._sparse_first and sparse_attr is not None:
            return sparse_attr
        if callable(sparse_attr) and callable(compat_attr):

            def wrapped(*args, **kwargs):
                if self._has_sparse_arg(*args, **kwargs):
                    args = tuple(self._sparse_compatible_arg(arg) for arg in args)
                    kwargs = {
                        key: self._sparse_compatible_arg(value)
                        for key, value in kwargs.items()
                    }
                    attr = sparse_attr
                else:
                    attr = compat_attr
                return attr(*args, **kwargs)

            return wrapped

        if sparse_attr is not None and compat_attr is not None:
            return compat_attr

        for attr in (sparse_attr, compat_attr, getattr(np, name, None)):
            if attr is not None:
                return attr

        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")


xp = PyDataSparseFramework()
