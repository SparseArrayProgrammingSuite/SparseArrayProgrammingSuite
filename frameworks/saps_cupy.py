import math
import operator

import numpy as np

import array_api_compat.cupy as compat_cp
import cupy
import cupyx.scipy
import cupyx.scipy.sparse as cusp
import cupyx.scipy.sparse.linalg as cuspla
from binsparse import (
    CustomTensor,
    DenseLevel,
    DMATCMatrix,
    DMATRMatrix,
    DVECVector,
    ElementLevel,
)
from binsparse.conversions import from_numpy, from_scipy, to_numpy, to_scipy

from saps_framework import Framework, einsum, normalize_unfold_args


class CuPyLinalg:
    @staticmethod
    def solve(A, b, **kwargs):
        b_dense = cupy.asarray(b).ravel()

        if cusp.issparse(A):
            return cuspla.spsolve(A, b_dense, **kwargs)
        return cupy.linalg.solve(A, b_dense, **kwargs)

    @staticmethod
    def norm(x, **kwargs):
        if cusp.issparse(x):
            return cuspla.norm(x, **kwargs)
        return cupy.linalg.norm(x, **kwargs)

    @staticmethod
    def lstsq(a, b, rcond=None, **kwargs):
        return cupy.linalg.lstsq(a, b, rcond=rcond, **kwargs)

    def __getattr__(self, name):
        # Use dense CuPy linalg for operations without a sparse variant.
        for module in (cuspla, cupy.linalg):
            if hasattr(module, name):
                return getattr(module, name)
        raise AttributeError(f"'CuPyLinalg' has no attribute '{name}'")


# Instance lookup is required for __getattr__.
_linalg = CuPyLinalg()


# cupyx sparse supports these value dtypes; other dtypes are cast only losslessly.
_SPARSE_DTYPES = frozenset(
    np.dtype(name) for name in ("bool", "float32", "float64", "complex64", "complex128")
)


def _as_sparse_dtype(matrix):
    if matrix.dtype in _SPARSE_DTYPES:
        return matrix

    target_dtype = np.complex128 if matrix.dtype.kind == "c" else np.float64
    with np.errstate(over="ignore", invalid="ignore"):
        converted = matrix.astype(target_dtype)
        roundtrip = converted.data.astype(matrix.dtype)
    equal_nan = matrix.dtype.kind in "fc"
    if not np.array_equal(matrix.data, roundtrip, equal_nan=equal_nan):
        raise ValueError(
            f"Cannot convert sparse {matrix.dtype} values to {target_dtype} "
            "without loss"
        )
    return converted


class CuPyFramework(Framework):
    def __init__(self):
        self._modules = [cusp, compat_cp, cupyx.scipy, cupy]

    @staticmethod
    def _array_namespace(*arrays):
        # Framework inputs are CuPy arrays, so dense operations use this namespace.
        return compat_cp

    @property
    def linalg(self):
        return _linalg

    def from_binsparse(self, array):
        match array:
            case DVECVector() | DMATRMatrix() | DMATCMatrix():
                return cupy.asarray(to_numpy(array))
            case CustomTensor(shape=(), transpose=None, level=ElementLevel()):
                return cupy.asarray(to_numpy(array))
            case CustomTensor(
                shape=shape,
                transpose=None,
                level=DenseLevel(rank=rank, level=ElementLevel()),
            ) if rank == len(shape):
                return cupy.asarray(to_numpy(array))
            case _:
                return cusp.csr_array(_as_sparse_dtype(to_scipy(array).tocsr()))

    def to_binsparse(self, array):
        # binsparse has no CuPy conversion, so results return through SciPy/NumPy.
        if cusp.issparse(array):
            # binsparse supports CSR/CSC/COO; cupyx operations may return DIA.
            if array.format not in ("csr", "csc", "coo"):
                array = array.tocsr()
            return from_scipy(array.get())
        if isinstance(array, cupy.ndarray):
            return from_numpy(cupy.asnumpy(array))
        if isinstance(array, np.ndarray):
            return from_numpy(array)
        if np.isscalar(array):
            return from_numpy(np.asarray(array))
        raise TypeError(f"Type {type(array)} is not a recognized CuPy format.")

    def lazy(self, array):
        return array

    def compute(self, array):
        # Complete asynchronous kernels before returning to the caller.
        cupy.cuda.get_current_stream().synchronize()
        return array

    def einsum(self, prgm, **kwargs):
        return einsum(self, prgm, **kwargs)

    def permute_dims(self, a, axes):
        if cusp.issparse(a):
            # Sparse transpose takes no axes; rank 2 only needs a swap.
            if tuple(axes) == tuple(range(a.ndim)):
                return a
            return a.transpose()
        return compat_cp.permute_dims(a, axes)

    def expand_dims(self, a, axis):
        if cusp.issparse(a):
            if not -a.ndim - 1 <= axis <= a.ndim:
                raise IndexError(f"axis {axis} is out of bounds for expand_dims")
            axis %= a.ndim + 1
            shape = a.shape[:axis] + (1,) + a.shape[axis:]
            return self.reshape(a, shape)
        return compat_cp.expand_dims(a, axis=axis)

    def multiply(self, x1, x2):
        if cusp.issparse(x1):
            return x1.multiply(x2)
        if cusp.issparse(x2):
            return x2.multiply(x1)
        return compat_cp.multiply(x1, x2)

    def unfold(
        self,
        x,
        kernel_shape,
        *,
        axes=None,
        strides=None,
        dilations=None,
        padding=None,
        fill_value=0,
    ):
        # Unfolded results are rank > 2, so densify cupyx sparse inputs on-device.
        array = x.toarray() if cusp.issparse(x) else x
        array = cupy.asarray(array)
        kernel_t = tuple(int(size) for size in kernel_shape)
        axes_t, strides_t, dilations_t, padding_t = normalize_unfold_args(
            array.ndim,
            kernel_t,
            axes,
            strides,
            dilations,
            padding,
        )
        effective_kernel = tuple(
            (kernel - 1) * dilation + 1
            for kernel, dilation in zip(kernel_t, dilations_t, strict=True)
        )

        pad_width = [(0, 0)] * array.ndim
        for axis, pad_pair in zip(axes_t, padding_t, strict=True):
            pad_width[axis] = pad_pair
        if any(pair != (0, 0) for pair in pad_width):
            array = cupy.pad(
                array,
                pad_width,
                mode="constant",
                constant_values=fill_value,
            )

        windows = cupy.lib.stride_tricks.sliding_window_view(
            array,
            effective_kernel,
            axis=axes_t,
        )
        slices: list[slice] = [slice(None)] * windows.ndim
        for axis, step in zip(axes_t, strides_t, strict=True):
            slices[axis] = slice(None, None, step)
        for window_axis, dilation in enumerate(dilations_t, start=array.ndim):
            slices[window_axis] = slice(None, None, dilation)
        return windows[tuple(slices)]

    def diagonal(self, a, *args, **kwargs):
        if cusp.issparse(a):
            return a.diagonal(*args, **kwargs)
        xp = self._array_namespace(a)
        return xp.diagonal(a, *args, **kwargs)

    def matmul(self, x1, x2, /, **kwargs):
        if cusp.issparse(x1) or cusp.issparse(x2):
            return x1 @ x2
        xp = self._array_namespace(x1, x2)
        return xp.matmul(x1, x2, **kwargs)

    # CuPy ufuncs reject sparse operands; use sparse operators for those cases.
    # multiply above uses cupyx's elementwise sparse method.

    def add(self, x1, x2, /, **kwargs):
        if cusp.issparse(x1) or cusp.issparse(x2):
            return x1 + x2
        xp = self._array_namespace(x1, x2)
        return xp.add(x1, x2, **kwargs)

    def subtract(self, x1, x2, /, **kwargs):
        if cusp.issparse(x1) or cusp.issparse(x2):
            return x1 - x2
        xp = self._array_namespace(x1, x2)
        return xp.subtract(x1, x2, **kwargs)

    def divide(self, x1, x2, /, **kwargs):
        if cusp.issparse(x1) or cusp.issparse(x2):
            return x1 / x2
        xp = self._array_namespace(x1, x2)
        return xp.divide(x1, x2, **kwargs)

    def maximum(self, x1, x2, /, **kwargs):
        if cusp.issparse(x1):
            return x1.maximum(x2)
        if cusp.issparse(x2):
            return x2.maximum(x1)
        xp = self._array_namespace(x1, x2)
        return xp.maximum(x1, x2, **kwargs)

    def minimum(self, x1, x2, /, **kwargs):
        if cusp.issparse(x1):
            return x1.minimum(x2)
        if cusp.issparse(x2):
            return x2.minimum(x1)
        xp = self._array_namespace(x1, x2)
        return xp.minimum(x1, x2, **kwargs)

    def power(self, x1, x2, /, **kwargs):
        # cupyx sparse exponentiation supports only scalar exponents.
        if cusp.issparse(x1) and np.isscalar(x2):
            return x1**x2
        if cusp.issparse(x1) and getattr(x2, "ndim", None) == 0:
            return x1 ** cupy.asarray(x2).item()
        xp = self._array_namespace(x1, x2)
        return xp.power(x1, x2, **kwargs)

    def reshape(self, x, /, shape, **kwargs):
        if cusp.issparse(x):
            # The array-API copy= argument is unsupported by cupyx sparse.
            kwargs.pop("copy", None)
            shape = tuple(int(size) for size in shape)
            if len(shape) > 2:
                # cupyx sparse supports at most two dimensions.
                return cupy.asarray(x.toarray()).reshape(shape)
            return x.reshape(shape, **kwargs)
        xp = self._array_namespace(x)
        return xp.reshape(x, shape, **kwargs)

    def eye(self, *args, **kwargs):
        # eye_array preserves sparse-array semantics; eye returns a spmatrix.
        kwargs.setdefault("format", "csr")
        return cusp.eye_array(*args, **kwargs)

    def concat(self, arrays, /, *, axis=0, **kwargs):
        if any(cusp.issparse(array) for array in arrays):
            axis = operator.index(axis)
            ndim = next(array.ndim for array in arrays if cusp.issparse(array))
            if not -ndim <= axis < ndim:
                raise ValueError(f"axis {axis} is out of bounds for {ndim}D input")
            axis %= ndim
            if axis not in (0, 1):
                raise ValueError(
                    f"axis {axis} is not supported for sparse concatenation"
                )
            # Normalize blocks so stacking a lone spmatrix still returns an array.
            blocks = [
                cusp.csr_array(array if cusp.issparse(array) else cupy.asarray(array))
                for array in arrays
            ]
            stack = cusp.vstack if axis == 0 else cusp.hstack
            return cusp.csr_array(stack(blocks, format="csr"))
        xp = self._array_namespace(*arrays)
        return xp.concat(arrays, axis=axis, **kwargs)

    @staticmethod
    def _keepdims(result, axis, keepdims, ndim):
        if not keepdims:
            return result
        result = cupy.asarray(result)
        if axis is None:
            return result.reshape((1,) * ndim)
        return compat_cp.expand_dims(result, axis=axis)

    def sum(self, x, /, *, axis=None, dtype=None, keepdims=False, **kwargs):
        if cusp.issparse(x):
            # cupyx sparse .sum takes no keepdims, so restore the axis by hand.
            result = x.sum(axis=axis, dtype=dtype)
            return self._keepdims(result, axis, keepdims, x.ndim)
        xp = self._array_namespace(x)
        return xp.sum(x, axis=axis, dtype=dtype, keepdims=keepdims, **kwargs)

    def _reduce_sparse(self, x, name, axis, keepdims):
        # cupyx implements max/min on the compressed formats only, and returns a
        # sparse vector for axis reductions. Densify it: cupyx sparse comparisons
        # broadcast against a dense operand but not against a sparse one.
        if isinstance(axis, tuple):
            dense = cupy.asarray(x.toarray())
            return getattr(compat_cp, name)(dense, axis=axis, keepdims=keepdims)
        result = getattr(x.tocsr(), name)(axis=axis)
        if cusp.issparse(result):
            result = cupy.asarray(result.toarray()).reshape(-1)
        return self._keepdims(result, axis, keepdims, x.ndim)

    def max(self, x, /, *, axis=None, keepdims=False, **kwargs):
        if cusp.issparse(x):
            return self._reduce_sparse(x, "max", axis, keepdims)
        xp = self._array_namespace(x)
        return xp.max(x, axis=axis, keepdims=keepdims, **kwargs)

    def min(self, x, /, *, axis=None, keepdims=False, **kwargs):
        if cusp.issparse(x):
            return self._reduce_sparse(x, "min", axis, keepdims)
        xp = self._array_namespace(x)
        return xp.min(x, axis=axis, keepdims=keepdims, **kwargs)

    def any(self, x, /, *, axis=None, keepdims=False, **kwargs):
        if cusp.issparse(x):
            if axis is None:
                result = cupy.asarray(x.count_nonzero() > 0)
                return self._keepdims(result, axis, keepdims, x.ndim)
            x = cupy.asarray(x.toarray())
        xp = self._array_namespace(x)
        return xp.any(x, axis=axis, keepdims=keepdims, **kwargs)

    def all(self, x, /, *, axis=None, keepdims=False, **kwargs):
        if cusp.issparse(x):
            if axis is None:
                result = cupy.asarray(x.count_nonzero() == math.prod(x.shape))
                return self._keepdims(result, axis, keepdims, x.ndim)
            x = cupy.asarray(x.toarray())
        xp = self._array_namespace(x)
        return xp.all(x, axis=axis, keepdims=keepdims, **kwargs)

    @staticmethod
    def _as_bool(x):
        if cusp.issparse(x):
            return x.astype(cupy.bool_)
        return cupy.asarray(x).astype(cupy.bool_)

    def logical_or(self, x1, x2, /, **kwargs):
        # Boolean max over the operands is the union of their nonzeros.
        if cusp.issparse(x1) or cusp.issparse(x2):
            left, right = self._as_bool(x1), self._as_bool(x2)
            return left.maximum(right) if cusp.issparse(left) else right.maximum(left)
        xp = self._array_namespace(x1, x2)
        return xp.logical_or(x1, x2, **kwargs)

    def logical_and(self, x1, x2, /, **kwargs):
        if cusp.issparse(x1) or cusp.issparse(x2):
            left, right = self._as_bool(x1), self._as_bool(x2)
            return left.minimum(right) if cusp.issparse(left) else right.minimum(left)
        xp = self._array_namespace(x1, x2)
        return xp.logical_and(x1, x2, **kwargs)

    def with_fill_value(self, array, value):
        return array

    def __getattr__(self, name):
        for module in self._modules:
            if hasattr(module, name):
                return getattr(module, name)

        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")


xp = CuPyFramework()
