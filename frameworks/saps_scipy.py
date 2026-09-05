from itertools import product

import numpy as np
import scipy as sp
import scipy.linalg as la
import scipy.sparse as sps
import scipy.sparse.linalg as spla

import array_api_compat
import array_api_compat.numpy as compat_np
import sparse as pydata_sparse
from binsparse import (
    CustomTensor,
    DenseLevel,
    DMATCMatrix,
    DMATRMatrix,
    DVECVector,
    ElementLevel,
)
from binsparse.conversions import (
    from_numpy,
    from_scipy,
    from_sparse,
    to_numpy,
    to_scipy,
)

from saps_framework import (
    Framework,
    einsum,
    normalize_unfold_args,
    unfold_output_shape,
)


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


def _value_is_zero(value) -> bool:
    array = np.asarray(value)
    return array.shape == () and array.item() == 0


def _axis_unfold_map(
    input_size, output_size, kernel_index, stride, dilation, pad_pair, dtype
):
    padded_size = input_size + pad_pair[0] + pad_pair[1]
    padded_input = sps.eye(
        padded_size,
        input_size,
        k=-pad_pair[0],
        dtype=dtype,
        format="csr",
    )
    positions = np.arange(output_size) * stride + kernel_index * dilation
    window_positions = sps.eye(padded_size, dtype=dtype, format="csr")[positions, :]
    return window_positions @ padded_input


def _kron_all(matrices):
    result = matrices[0]
    for matrix in matrices[1:]:
        result = sps.kron(result, matrix, format="csr")
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
    matrix,
    kernel_shape,
    axes,
    strides,
    dilations,
    padding,
):
    input_shape = tuple(int(dim) for dim in matrix.shape)
    output_shape = unfold_output_shape(
        input_shape,
        kernel_shape,
        axes,
        strides,
        dilations,
        padding,
    )
    output_core_shape = output_shape[: matrix.ndim]
    axis_positions = {axis: i for i, axis in enumerate(axes)}
    flat_input = matrix.reshape((-1, 1))
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
                        matrix.dtype,
                    )
                )
            else:
                axis_maps.append(sps.eye(input_size, dtype=matrix.dtype, format="csr"))

        selected = (_kron_all(axis_maps) @ flat_input).tocoo()
        if selected.nnz == 0:
            continue
        coords.append(
            _unfold_block_coords(
                selected.row,
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
        output_data = np.asarray([], dtype=matrix.dtype)

    return pydata_sparse.COO(output_coords, output_data, shape=output_shape)


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
        if isinstance(array, pydata_sparse.SparseArray):
            return from_sparse(array.asformat("coo"))
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
        if sps.issparse(x) and _value_is_zero(fill_value):
            matrix = sps.coo_array(x)
            kernel_t = tuple(int(size) for size in kernel_shape)
            axes_t, strides_t, dilations_t, padding_t = normalize_unfold_args(
                matrix.ndim,
                kernel_t,
                axes,
                strides,
                dilations,
                padding,
            )
            return _sparse_unfold_with_diagonals(
                matrix,
                kernel_t,
                axes_t,
                strides_t,
                dilations_t,
                padding_t,
            )

        array = x.toarray() if sps.issparse(x) else x
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
        return windows[tuple(slices)]

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
