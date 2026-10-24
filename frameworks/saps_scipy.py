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
    normalize_pad_width,
    normalize_unfold_args,
    unfold_output_shape,
)

# Tests import frameworks as a package; the harness loads them from their folder.
try:
    from frameworks.saps_numpy import NumpyFramework
except ImportError:
    from saps_numpy import NumpyFramework  # type: ignore[no-redef]


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


def _axis_unfold_map(input_size, output_size, kernel_index, stride, dilation, dtype):
    positions = np.arange(output_size) * stride + kernel_index * dilation
    return sps.eye_array(input_size, dtype=dtype, format="csr")[positions]


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
):
    input_shape = tuple(int(dim) for dim in matrix.shape)
    output_shape = unfold_output_shape(
        input_shape,
        kernel_shape,
        axes,
        strides,
        dilations,
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


def _is_sparse(array):
    return sps.issparse(array) or isinstance(array, pydata_sparse.SparseArray)


class SciPyFramework(Framework):
    def __init__(self):
        self._modules = [sps, compat_np, sp, np]
        self._dense = NumpyFramework()

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
        if np.isscalar(array):
            return from_numpy(np.asarray(array))
        raise TypeError(f"Type {type(array)} is not a recognized SciPy/NumPy format.")

    def lazy(self, array):
        return array

    def compute(self, array):
        return array

    def einsum(self, prgm, **kwargs):
        if any(_is_sparse(arg) for arg in kwargs.values()):
            return einsum(self, prgm, **kwargs)
        return self._dense.einsum(prgm, **kwargs)

    # SciPy has no sparse convolution, so only dense inputs have a fast path.

    def conv(self, x, w, **kwargs):
        if _is_sparse(x) or _is_sparse(w):
            return super().conv(x, w, **kwargs)
        return self._dense.conv(x, w, **kwargs)

    def pool(self, x, kernel_shape, **kwargs):
        if _is_sparse(x):
            return super().pool(x, kernel_shape, **kwargs)
        return self._dense.pool(x, kernel_shape, **kwargs)

    def permute_dims(self, a, axes):
        if sps.issparse(a):
            return (
                a if tuple(axes) == tuple(range(a.ndim)) else a.transpose(tuple(axes))
            )
        return compat_np.permute_dims(a, axes)

    def expand_dims(self, a, axis):
        if sps.issparse(a):
            if not -a.ndim - 1 <= axis <= a.ndim:
                raise IndexError(f"axis {axis} is out of bounds for expand_dims")
            axis %= a.ndim + 1
            return a.reshape(a.shape[:axis] + (1,) + a.shape[axis:])
        return compat_np.expand_dims(a, axis=axis)

    def multiply(self, x1, x2):
        if sps.issparse(x1):
            return x1.multiply(x2)
        if sps.issparse(x2):
            return x2.multiply(x1)
        return compat_np.multiply(x1, x2)

    def concat(self, arrays, /, *, axis=0):
        if not any(sps.issparse(array) for array in arrays):
            return compat_np.concat(arrays, axis=axis)
        ndim = arrays[0].ndim
        axis %= ndim
        if ndim == 2:
            stack = sps.vstack if axis == 0 else sps.hstack
            return stack(arrays, format="coo")
        # scipy.sparse only stacks 2D arrays, so offset COO coordinates instead.
        arrays = [sps.coo_array(array) for array in arrays]
        coords = []
        offset = 0
        for array in arrays:
            shifted = list(array.coords)
            shifted[axis] = shifted[axis] + offset
            coords.append(np.stack(shifted))
            offset += array.shape[axis]
        shape = list(arrays[0].shape)
        shape[axis] = offset
        return sps.coo_array(
            (
                np.concatenate([array.data for array in arrays]),
                tuple(np.concatenate(coords, axis=1)),
            ),
            shape=tuple(shape),
        )

    # scipy.sparse can't pad, so sparse arrays fall back to concatenation.

    def pad_constant(self, x, pad_width, *, fill_value=0):
        if sps.issparse(x):
            return super().pad_constant(x, pad_width, fill_value=fill_value)
        pad_width = normalize_pad_width(np.ndim(x), pad_width)
        return np.pad(x, pad_width, mode="constant", constant_values=fill_value)

    def pad_reflect(self, x, pad_width):
        if sps.issparse(x):
            return super().pad_reflect(x, pad_width)
        return np.pad(x, normalize_pad_width(np.ndim(x), pad_width), mode="reflect")

    def pad_edge(self, x, pad_width):
        if sps.issparse(x):
            return super().pad_edge(x, pad_width)
        return np.pad(x, normalize_pad_width(np.ndim(x), pad_width), mode="edge")

    def pad_wrap(self, x, pad_width):
        if sps.issparse(x):
            return super().pad_wrap(x, pad_width)
        return np.pad(x, normalize_pad_width(np.ndim(x), pad_width), mode="wrap")

    def unfold(self, x, kernel_shape, *, axes=None, strides=None, dilations=None):
        kernel_t = tuple(int(size) for size in kernel_shape)
        if sps.issparse(x):
            matrix = sps.coo_array(x)
            axes_t, strides_t, dilations_t = normalize_unfold_args(
                matrix.ndim, kernel_t, axes, strides, dilations
            )
            return _sparse_unfold_with_diagonals(
                matrix, kernel_t, axes_t, strides_t, dilations_t
            )

        array = np.asarray(x)
        axes_t, strides_t, dilations_t = normalize_unfold_args(
            array.ndim,
            kernel_t,
            axes,
            strides,
            dilations,
        )
        effective_kernel = tuple(
            (kernel - 1) * dilation + 1
            for kernel, dilation in zip(kernel_t, dilations_t, strict=True)
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
            # scipy.sparse's @ (unlike the array-api namespace below) takes
            # no kwargs at all, so a caller-requested dtype has to be
            # applied as a cast afterward instead of steering the
            # contraction itself -- unlike the dense path, this can't
            # avoid computing in whatever wider dtype the contraction
            # naturally uses first.
            dtype = kwargs.pop("dtype", None)
            if kwargs:
                raise TypeError(
                    f"SciPyFramework.matmul doesn't support {sorted(kwargs)} "
                    "for sparse operands"
                )
            result = x1 @ x2
            return result if dtype is None else result.astype(dtype)
        xp = self._array_namespace(x1, x2)
        return xp.matmul(x1, x2, **kwargs)

    # NumPy ufuncs (np.add, np.subtract, ...) treat a scipy sparse operand as a
    # 0-d object scalar and broadcast it, exploding memory. Scipy's operators are
    # correctly overloaded for sparse, so route through them when an operand is
    # sparse and fall back to the array-api namespace for dense inputs.
    # `multiply` is defined above, using scipy's explicitly elementwise method.

    def add(self, x1, x2, /, **kwargs):
        if sps.issparse(x1) or sps.issparse(x2):
            return x1 + x2
        xp = self._array_namespace(x1, x2)
        return xp.add(x1, x2, **kwargs)

    def subtract(self, x1, x2, /, **kwargs):
        if sps.issparse(x1) or sps.issparse(x2):
            return x1 - x2
        xp = self._array_namespace(x1, x2)
        return xp.subtract(x1, x2, **kwargs)

    def divide(self, x1, x2, /, **kwargs):
        if sps.issparse(x1) or sps.issparse(x2):
            return x1 / x2
        xp = self._array_namespace(x1, x2)
        return xp.divide(x1, x2, **kwargs)

    def maximum(self, x1, x2, /, **kwargs):
        if sps.issparse(x1):
            return x1.maximum(x2)
        if sps.issparse(x2):
            return x2.maximum(x1)
        xp = self._array_namespace(x1, x2)
        return xp.maximum(x1, x2, **kwargs)

    def minimum(self, x1, x2, /, **kwargs):
        if sps.issparse(x1):
            return x1.minimum(x2)
        if sps.issparse(x2):
            return x2.minimum(x1)
        xp = self._array_namespace(x1, x2)
        return xp.minimum(x1, x2, **kwargs)

    def power(self, x1, x2, /, **kwargs):
        # Scipy sparse only supports a scalar exponent on a sparse base.
        if sps.issparse(x1) and np.isscalar(x2):
            return x1**x2
        xp = self._array_namespace(x1, x2)
        return xp.power(x1, x2, **kwargs)

    def replace(self, arr, old, new):
        if sps.issparse(arr):
            if old == 0 and new != 0:
                arr = arr.toarray()
            else:
                result = arr.tocoo(copy=True)
                result.sum_duplicates()
                values = result.data
                result.data = np.where(
                    np.isnan(values) if old != old else values == old, new, values
                )
                result.eliminate_zeros()
                return result.asformat(arr.format)
        if isinstance(arr, pydata_sparse.DOK):
            arr = arr.asformat("coo")
        xp = pydata_sparse if isinstance(arr, pydata_sparse.SparseArray) else np
        return xp.where(xp.isnan(arr) if old != old else arr == old, new, arr)

    def __getattr__(self, name):
        for module in self._modules:
            if hasattr(module, name):
                return getattr(module, name)

        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")


xp = SciPyFramework()
