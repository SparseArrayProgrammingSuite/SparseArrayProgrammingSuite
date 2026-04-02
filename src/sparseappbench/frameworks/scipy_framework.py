import numpy as np
import scipy as sp
import scipy.linalg as la
import scipy.sparse as sps
import scipy.sparse.linalg as spla

from ..binsparse_format import BinsparseFormat
from .abstract_framework import AbstractFramework
from .einsum import Access, Call, parse_einsum


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
        expr = parse_einsum(prgm)
        return self._einsum_sparse_2d(expr, prgm, **kwargs)

    def _einsum_sparse_2d(self, expr, prgm, **kwargs):
        if (
            expr.op in {"+", "add"}
            and isinstance(expr.arg, Call)
            and expr.arg.func in {"*", "mul", "multiply"}
            and len(expr.arg.args) == 2
            and isinstance(expr.arg.args[0], Access)
            and isinstance(expr.arg.args[1], Access)
        ):
            a = expr.arg.args[0]
            b = expr.arg.args[1]
            A = kwargs[a.tns]
            B = kwargs[b.tns]

            # Detects matmul
            if len(expr.idxs) == 2:
                if (
                    len(a.idxs) == 2
                    and len(b.idxs) == 2
                    and a.idxs[1] == b.idxs[0]
                    and expr.idxs == [a.idxs[0], b.idxs[1]]
                ):
                    return A @ B

            # Detects matvec
            elif len(expr.idxs) == 1:
                out = expr.idxs[0]
                if (
                    len(a.idxs) == 2
                    and len(b.idxs) == 1
                    and a.idxs[0] == out
                    and a.idxs[1] == b.idxs[0]
                ):
                    return A @ B
                if (
                    len(a.idxs) == 1
                    and len(b.idxs) == 2
                    and b.idxs[1] == out
                    and a.idxs[0] == b.idxs[0]
                ):
                    return A @ B

            # Detects vecdot
            elif (
                len(expr.idxs) == 0
                and len(a.idxs) == 1
                and len(b.idxs) == 1
                and a.idxs == b.idxs
            ):
                return A @ B

        raise NotImplementedError(
            f"SciPy sparse einsum does not support '{prgm}' without densifying."
        )

    def with_fill_value(self, array, value):
        return array

    def __getattr__(self, name):
        for module in self._modules:
            if hasattr(module, name):
                return getattr(module, name)

        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")
