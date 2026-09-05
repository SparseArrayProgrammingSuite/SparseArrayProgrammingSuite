from itertools import product

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla

import array_api_compat
import array_api_compat.numpy as compat_np
import sparse as sp
from binsparse import (
    CustomTensor,
    DenseLevel,
    DMATCMatrix,
    DMATRMatrix,
    DVECVector,
    ElementLevel,
)
from binsparse.conversions import from_numpy, from_sparse, to_numpy, to_sparse

from saps_framework import (
    Framework,
    einsum,
    normalize_unfold_args,
    unfold_output_shape,
)


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


def _value_is_zero(value) -> bool:
    array = np.asarray(value)
    return array.shape == () and array.item() == 0


def _axis_unfold_map(
    input_size,
    output_size,
    kernel_index,
    stride,
    dilation,
    pad_pair,
    dtype,
):
    padded_size = input_size + pad_pair[0] + pad_pair[1]
    padded_input = sp.eye(
        padded_size,
        input_size,
        k=-pad_pair[0],
        dtype=dtype,
        format="coo",
    )
    positions = np.arange(output_size) * stride + kernel_index * dilation
    window_positions = sp.eye(padded_size, dtype=dtype, format="coo")[positions, :]
    return window_positions @ padded_input


def _kron_all(arrays):
    result = arrays[0]
    for array in arrays[1:]:
        result = sp.kron(result, array)
    return result


def _unfold_block_coords(flat_rows, output_core_shape, kernel_index, output_rank):
    nnz = len(flat_rows)
    coords = np.empty((output_rank, nnz), dtype=np.intp)
    for axis, axis_coords in enumerate(np.unravel_index(flat_rows, output_core_shape)):
        coords[axis] = axis_coords
    for axis, kernel_axis_index in enumerate(
        kernel_index, start=len(output_core_shape)
    ):
        coords[axis].fill(kernel_axis_index)
    return coords


def _sparse_unfold_with_diagonals(
    array,
    kernel_shape,
    axes,
    strides,
    dilations,
    padding,
):
    input_shape = tuple(int(dim) for dim in array.shape)
    output_shape = unfold_output_shape(
        input_shape,
        kernel_shape,
        axes,
        strides,
        dilations,
        padding,
    )
    output_core_shape = output_shape[: array.ndim]
    axis_positions = {axis: i for i, axis in enumerate(axes)}
    flat_input = array.reshape((int(np.prod(input_shape)), 1))
    coords = []
    data = []

    for kernel_index in product(*(range(size) for size in kernel_shape)):
        axis_maps = []
        for axis, input_size in enumerate(input_shape):
            if axis in axis_positions:
                position = axis_positions[axis]
                axis_maps.append(
                    _axis_unfold_map(
                        input_size,
                        output_core_shape[axis],
                        kernel_index[position],
                        strides[position],
                        dilations[position],
                        padding[position],
                        array.dtype,
                    )
                )
            else:
                axis_maps.append(sp.eye(input_size, dtype=array.dtype, format="coo"))

        selected = (_kron_all(axis_maps) @ flat_input).asformat("coo")
        if selected.nnz == 0:
            continue
        coords.append(
            _unfold_block_coords(
                selected.coords[0],
                output_core_shape,
                kernel_index,
                len(output_shape),
            )
        )
        data.append(selected.data)

    if coords:
        output_coords = np.concatenate(coords, axis=1)
        output_data = np.concatenate(data)
    else:
        output_coords = np.empty((len(output_shape), 0), dtype=np.intp)
        output_data = np.asarray([], dtype=array.dtype)

    return sp.COO(output_coords, output_data, shape=output_shape)


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
                return to_sparse(array)

    def to_binsparse(self, array):
        if isinstance(array, sp.COO):
            if array.ndim == 0 or not self._fill_value_is_zero(array):
                return from_numpy(self._dense(array))
            return from_sparse(array)
        if isinstance(array, sp.SparseArray):
            if array.ndim == 0 or not self._fill_value_is_zero(array):
                return from_numpy(self._dense(array))
            return self.to_binsparse(array.tocoo())
        if isinstance(array, np.ndarray):
            return from_numpy(array)
        if np.isscalar(array):
            return from_numpy(np.asarray(array))
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
        if isinstance(x, sp.SparseArray) and _value_is_zero(fill_value):
            coo = x.asformat("coo")
            kernel_t = tuple(int(size) for size in kernel_shape)
            axes_t, strides_t, dilations_t, padding_t = normalize_unfold_args(
                coo.ndim,
                kernel_t,
                axes,
                strides,
                dilations,
                padding,
            )
            return _sparse_unfold_with_diagonals(
                coo,
                kernel_t,
                axes_t,
                strides_t,
                dilations_t,
                padding_t,
            )

        array = self._dense(x) if isinstance(x, sp.SparseArray) else x
        array = np.asarray(array)
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
            array = np.pad(
                array,
                pad_width,
                mode="constant",
                constant_values=fill_value,
            )

        windows = np.lib.stride_tricks.sliding_window_view(  # type: ignore[call-overload]
            array,
            effective_kernel,
            axis=axes_t,
        )
        slices: list[slice] = [slice(None)] * windows.ndim
        for axis, step in zip(axes_t, strides_t, strict=True):
            slices[axis] = slice(None, None, step)
        for window_axis, dilation in enumerate(dilations_t, start=array.ndim):
            slices[window_axis] = slice(None, None, dilation)
        windows = windows[tuple(slices)]
        if isinstance(x, sp.SparseArray):
            return sp.asarray(windows)
        return windows

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
