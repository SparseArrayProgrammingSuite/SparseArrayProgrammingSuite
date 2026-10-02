import operator
import types
from itertools import product
from math import prod

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla

import array_api_compat
import array_api_compat.numpy as compat_np
import sparse as sp
from binsparse import (
    CSRMatrix,
    CustomTensor,
    DenseLevel,
    DMATCMatrix,
    DMATRMatrix,
    DVECVector,
    ElementLevel,
)
from binsparse.conversions import from_numpy, from_sparse, to_numpy, to_scipy, to_sparse

from saps_framework import (
    Framework,
    normalize_unfold_args,
    unfold_output_shape,
)
from saps_framework.einsum import Access, native_einsum, parse_einsum

_EINSUM_BLOCK_SIZE = 65_536
_EINSUM_DENSE_BYTES = 64 * 1024**2


class SmartSparseLinalg:
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
            return tuple(SmartSparseLinalg._wrap_result(item) for item in result)
        return result

    @staticmethod
    def solve(A, b):
        A = SmartSparseLinalg._scipy_sparse(A).tocsc()
        b = SmartSparseLinalg._scipy_sparse(b)
        x = spla.spsolve(A, b)
        return SmartSparseLinalg._wrap_result(x)

    @staticmethod
    def pinv(A, **kwargs):
        return np.linalg.pinv(SmartSparseLinalg._dense(A), **kwargs)

    @staticmethod
    def lstsq(A, b, **kwargs):
        return np.linalg.lstsq(
            SmartSparseLinalg._dense(A), SmartSparseLinalg._dense(b), **kwargs
        )

    @staticmethod
    def svd(A, full_matrices=False, k=None, **kwargs):
        # svds only finds k < min(A.shape) singular triplets, so anything but a
        # truncated SVD of a sparse matrix (a full reduced SVD, full_matrices,
        # or a matrix too thin to truncate) is computed densely instead.
        sparse_ok = (
            isinstance(A, sp.SparseArray)
            and not full_matrices
            and k is not None
            and k < min(A.shape)
        )
        if not sparse_ok:
            A = np.asarray(SmartSparseLinalg._dense(A))
            if A.dtype.kind not in "fc":
                A = A.astype(np.float64)
            U, S, Vt = np.linalg.svd(A, full_matrices=full_matrices, **kwargs)
            if k is not None:
                U = U[:, :k]
                S = S[:k]
                Vt = Vt[:k, :]
            return U, S, Vt

        A = SmartSparseLinalg._scipy_sparse(A)
        if A.dtype.kind not in "fc":
            A = A.astype(np.float64)
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


def _is_full_slice(index) -> bool:
    return (
        isinstance(index, slice)
        and index.start in (None, 0)
        and index.stop is None
        and index.step in (None, 1)
    )


class _MutableCOO(sp.COO):
    """COO arithmetic with indexed assignment through a temporary DOK builder."""

    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            key = tuple(
                SmartSparseKernels._dense(index)
                if isinstance(index, sp.SparseArray)
                else index
                for index in key
            )
        elif isinstance(key, sp.SparseArray):
            key = SmartSparseKernels._dense(key)
        if isinstance(value, sp.SparseArray):
            value = SmartSparseKernels._dense(value)
        # Scattering into an empty array with no repeated coordinates is
        # exactly building a COO from these coordinates -- construct it
        # directly (vectorized) instead of going through DOK, whose own
        # __setitem__ is a pure-Python loop over every index (see
        # sparse.numba_backend._dok.DOK._fancy_setitem), which dominates
        # runtime at realistic sizes regardless of how few entries are
        # actually being set. COO sums duplicate coordinates instead of
        # this class's (and DOK's) last-write-wins, so this only applies
        # once verified duplicate-free -- itself a cheap vectorized check,
        # unlike the Python loop it's replacing.
        if (
            self.nnz == 0
            and isinstance(key, tuple)
            and len(key) == self.ndim
            and all(isinstance(index, np.ndarray) and index.ndim == 1 for index in key)
        ):
            flat = np.ravel_multi_index(key, self.shape)
            if len(np.unique(flat)) == len(flat):
                coords = np.stack(key)
                data = np.broadcast_to(
                    np.asarray(value, dtype=self.dtype), (coords.shape[1],)
                )
                updated = sp.COO(
                    coords,
                    data,
                    shape=self.shape,
                    fill_value=self.fill_value,
                    has_duplicates=False,
                )
                self.coords = updated.coords
                self.data = updated.data
                self._cache = None
                return
        # A full slice on one axis and a scalar on the other (Q[:, i] = ...,
        # row-major equivalent) is a whole-row/column replace: filter out
        # that row/column's existing coordinates and append the new ones,
        # vectorized, instead of DOK's element-by-element Python loop. This
        # is the pattern iterative solvers hit rewriting one Krylov basis
        # column per step -- and it's a *replace*, so old entries at that
        # index must be dropped, not summed with the new ones the way COO
        # construction would.
        if self.ndim == 2 and isinstance(key, tuple) and len(key) == 2:
            if _is_full_slice(key[0]) and isinstance(key[1], (int, np.integer)):
                varying_axis, fixed_axis, fixed_index = 0, 1, int(key[1])
            elif isinstance(key[0], (int, np.integer)) and _is_full_slice(key[1]):
                varying_axis, fixed_axis, fixed_index = 1, 0, int(key[0])
            else:
                varying_axis = None
            if varying_axis is not None:
                value = np.broadcast_to(
                    np.asarray(value, dtype=self.dtype), (self.shape[varying_axis],)
                )
                keep = self.coords[fixed_axis] != fixed_index
                nonfill = np.flatnonzero(value != self.fill_value)
                new_coords = np.empty((2, len(nonfill)), dtype=self.coords.dtype)
                new_coords[varying_axis] = nonfill
                new_coords[fixed_axis] = fixed_index
                self.coords = np.concatenate([self.coords[:, keep], new_coords], axis=1)
                self.data = np.concatenate([self.data[keep], value[nonfill]])
                self._cache = None
                return
        builder = sp.DOK.from_coo(self)
        builder[key] = value
        updated = builder.to_coo()
        self.coords = updated.coords
        self.data = updated.data
        self._cache = None


class SmartSparseKernels(Framework):
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
    def _is_sparse(arg):
        # Sequence arguments (e.g. concat's array list) count too, or a mixed
        # list would dispatch to NumPy and reach pydata/sparse's concatenate
        # through __array_function__ with its dense members unconverted.
        if isinstance(arg, list | tuple):
            return any(isinstance(item, sp.SparseArray) for item in arg)
        return isinstance(arg, sp.SparseArray)

    @classmethod
    def _has_sparse_arg(cls, *args, **kwargs):
        return any(cls._is_sparse(arg) for arg in args) or any(
            cls._is_sparse(value) for value in kwargs.values()
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
        if isinstance(arg, list | tuple):
            return type(arg)(
                sp.asarray(item) if isinstance(item, np.ndarray) else item
                for item in arg
            )
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
            case CSRMatrix():  # also matches CSCMatrix, a CSRMatrix subclass
                # to_sparse only reads COO; go through SciPy (which reads
                # CSR/CSC/COO) and into GCXS instead of densifying via COO.
                return sp.GCXS.from_scipy_sparse(to_scipy(array))
            case _:
                return to_sparse(array)

    def to_binsparse(self, array):
        if isinstance(array, sp.COO):
            if array.ndim == 0:
                return from_numpy(self._dense(array))
            return from_sparse(array)
        if isinstance(array, sp.SparseArray):
            if array.ndim == 0:
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

    def to_dense(self, array):
        # A sparse array can carry a nonzero fill value (e.g. downstream of
        # a boolean mask that's mostly True) without being sparse in any
        # useful sense; densifying is then both cheap and the only way to
        # feed it to ordinary elementwise ops without tripping pydata/
        # sparse's mixed sparse-dense guard.
        if hasattr(array, "todense"):
            return array.todense()
        return array

    def sum(self, x, axis=None, **kwargs):
        # A reduced axis can still contain millions of elements. Preserve the
        # sparse result and its fill value; WrappedArray handles mixed operands.
        if isinstance(x, sp.SparseArray):
            return sp.sum(x, axis=axis, **kwargs)
        xp = self._array_namespace(x)
        return xp.sum(x, axis=axis, **kwargs)

    def where(self, condition, x, y):
        # A sparse boolean condition selecting from a dense x with a scalar
        # fill y (e.g. an LSH candidate mask picking real distances out of
        # a dense distance matrix, else infinity) never needs the dense
        # result materialized: gather x's values at condition's stored
        # (true, since fill_value is falsy) coordinates and hand them back
        # in a COO with fill_value=y -- O(nnz) instead of O(condition.size).
        # The generic dispatch below would instead sparsify x, since it
        # sparsifies every argument once any one of them is sparse.
        if (
            isinstance(condition, sp.SparseArray)
            and condition.ndim == 2
            and not condition.fill_value
            and not isinstance(x, sp.SparseArray)
            and getattr(x, "shape", None) == condition.shape
            and np.isscalar(y)
        ):
            coo = (
                condition
                if isinstance(condition, sp.COO)
                else condition.asformat("coo")
            )
            rows, cols = coo.coords
            x = np.asarray(x)
            return sp.COO(
                coo.coords,
                x[rows, cols],
                shape=condition.shape,
                fill_value=x.dtype.type(y),
            )
        if self._has_sparse_arg(condition, x, y):
            condition, x, y = (
                self._sparse_compatible_arg(arg) for arg in (condition, x, y)
            )
            return sp.where(condition, x, y)
        return compat_np.where(condition, x, y)

    def einsum(self, prgm, **kwargs):
        if all(not isinstance(value, sp.SparseArray) for value in kwargs.values()):
            xp = self._array_namespace(*kwargs.values())
            return native_einsum(xp, prgm, **kwargs)
        parsed = parse_einsum(prgm)
        result = self._blocked_einsum(parsed, kwargs)
        if result is not NotImplemented:
            return result
        kwargs = {
            key: self._sparse_compatible_arg(value) for key, value in kwargs.items()
        }
        result = parsed.run_native(sp, kwargs)
        return parsed.run(sp, kwargs) if result is NotImplemented else result

    def _blocked_einsum(self, parsed, kwargs):
        """Contract zero-fill sparse entries in bounded batches.

        A sparse operand must cover all reduced indices. Other indices enumerate
        output slices (for example, CP rank). Small factors and outputs may be
        dense; large factors/outputs and general expressions use the fallback.
        """
        contraction = parsed.native_contraction()
        if contraction is NotImplemented:
            return NotImplemented
        _, leaves, bitwise = contraction
        operands = [
            kwargs[leaf.tns] if isinstance(leaf, Access) else np.asarray(leaf.value)
            for leaf in leaves
        ]
        operands = [
            operand if isinstance(operand, sp.SparseArray) else np.asarray(operand)
            for operand in operands
        ]
        indices = [leaf.idxs if isinstance(leaf, Access) else [] for leaf in leaves]
        if any(not self._fill_value_is_zero(operand) for operand in operands):
            return NotImplemented
        boolean = all(operand.dtype.kind == "b" for operand in operands)
        logical = parsed.op in {"|", "or"}
        if (bitwise or logical) and not boolean:
            return NotImplemented
        sizes = {}
        for operand, term in zip(operands, indices, strict=True):
            for idx, size in zip(term, operand.shape, strict=True):
                if sizes.setdefault(idx, size) != size:
                    return NotImplemented
        reduced = set(sizes) - set(parsed.idxs)
        candidates = [
            i
            for i, operand in enumerate(operands)
            if isinstance(operand, sp.SparseArray)
            and reduced.issubset(indices[i])
            and len(indices[i]) == len(set(indices[i]))
        ]
        if not candidates or len(parsed.idxs) != len(set(parsed.idxs)):
            return NotImplemented
        anchor_index = min(candidates, key=lambda i: operands[i].nnz)
        dtype = np.result_type(*(operand.dtype for operand in operands))
        output_dtype = (
            np.dtype(bool) if logical else np.zeros((), dtype=dtype).sum().dtype
        )
        shape = tuple(sizes[idx] for idx in parsed.idxs)
        dense_bytes = prod(shape) * output_dtype.itemsize
        for i, operand in enumerate(operands):
            if i != anchor_index and isinstance(operand, sp.SparseArray):
                dense_bytes += prod(operand.shape) * operand.dtype.itemsize
        if dense_bytes > _EINSUM_DENSE_BYTES:
            return NotImplemented

        anchor = operands[anchor_index].asformat("coo")
        factors = [
            (term, self._dense(operand))
            for i, (operand, term) in enumerate(zip(operands, indices, strict=True))
            if i != anchor_index
        ]
        # Skipping implicit zeros is only valid when 0 * factor stays zero.
        if any(
            not np.isfinite(factor.flat[start : start + _EINSUM_BLOCK_SIZE]).all()
            for _, factor in factors
            for start in range(0, factor.size, _EINSUM_BLOCK_SIZE)
        ):
            return NotImplemented
        anchor_indices = indices[anchor_index]
        free = [idx for idx in parsed.idxs if idx not in anchor_indices]
        scatter = any(idx in anchor_indices for idx in parsed.idxs)
        result = np.zeros(shape, dtype=output_dtype)
        if anchor.nnz == 0:
            return result
        for free_coords in np.ndindex(*(sizes[idx] for idx in free)):
            positions = dict(zip(free, free_coords, strict=True))
            for start in range(0, anchor.nnz, _EINSUM_BLOCK_SIZE):
                block = slice(start, start + _EINSUM_BLOCK_SIZE)
                positions.update(
                    (idx, anchor.coords[axis, block])
                    for axis, idx in enumerate(anchor_indices)
                )
                values = anchor.data[block].astype(dtype, copy=True)
                for term, factor in factors:
                    values *= factor[tuple(positions[idx] for idx in term)]
                target = tuple(positions[idx] for idx in parsed.idxs)
                if scatter:
                    if logical:
                        np.logical_or.at(result, target, values)
                    else:
                        np.add.at(result, target, values)
                elif logical:
                    result[target] |= values.any()
                else:
                    result[target] += values.sum(dtype=output_dtype)
        return result

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
            # sparse.matmul (unlike the array-api namespace below) takes no
            # kwargs at all, so a caller-requested dtype has to be applied
            # as a cast afterward instead of steering the contraction
            # itself -- unlike the dense path, this can't avoid computing
            # in whatever wider dtype the contraction naturally uses first.
            dtype = kwargs.pop("dtype", None)
            if kwargs:
                raise TypeError(
                    f"SmartSparseKernels.matmul doesn't support {sorted(kwargs)} "
                    "for sparse operands"
                )
            if (
                isinstance(x1, sp.SparseArray)
                and isinstance(x2, sp.SparseArray)
                and x1.ndim == 2
                and x2.ndim == 2
            ):
                # Two genuinely sparse (as opposed to formally sparse, e.g.
                # LSH bucket indicators) 2D operands: go through SciPy's
                # CSR-CSR product instead of pydata/sparse's own `@`. That
                # path (sparse/_coo/core.py's linear_loc-based reshape/
                # tensordot machinery, flagged by its own "this self.size
                # enforces a 2**64 limit to array size" TODO) is built for
                # general N-D contractions, not tuned for the very wide,
                # very sparse 2D case -- SciPy's is.
                result = self._to_scipy_sparse(x1) @ self._to_scipy_sparse(x2)
                result = sp.GCXS.from_scipy_sparse(result.tocsr())
                return result if dtype is None else result.astype(dtype)
            if (
                x1.ndim == 2
                and x2.ndim <= 2
                and not (
                    isinstance(x1, sp.SparseArray) and isinstance(x2, sp.SparseArray)
                )
            ):
                # A 2D sparse matrix times a dense matrix or vector (or the
                # reverse): SciPy's sparse-dense kernels are far faster than
                # pydata/sparse's general `@`, and both produce a dense result.
                lhs, rhs = (
                    self._to_scipy_sparse(x) if isinstance(x, sp.SparseArray) else x
                    for x in (x1, x2)
                )
                result = np.asarray(lhs @ rhs)
                return result if dtype is None else result.astype(dtype)
            result = x1 @ x2
            return result if dtype is None else result.astype(dtype)
        xp = self._array_namespace(x1, x2)
        return xp.matmul(x1, x2, **kwargs)

    @staticmethod
    def _to_scipy_sparse(array):
        if hasattr(array, "to_scipy_sparse"):
            return array.to_scipy_sparse()
        return sps.coo_matrix(np.asarray(array))

    def zeros(self, shape, *args, **kwargs):
        # Vectors also serve as dense index and scalar buffers in the suite.
        if not isinstance(shape, tuple | list) or len(shape) < 2:
            return compat_np.zeros(shape, *args, **kwargs)
        return _MutableCOO(sp.zeros(shape, *args, **kwargs))

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
        dtype = kwargs.pop("dtype", None)
        return sp.eye(*args, dtype=float if dtype is None else dtype, **kwargs)

    def ones(self, *args, **kwargs):
        return compat_np.ones(*args, **kwargs)

    def expand_dims(self, a, axis):
        if isinstance(a, sp.SparseArray):
            return sp.expand_dims(a, axis=axis)
        xp = self._array_namespace(a)
        return xp.expand_dims(a, axis=axis)

    def stack(self, arrays, *, axis=0):
        if any(isinstance(array, sp.SparseArray) for array in arrays):
            return sp.stack([sp.asarray(array) for array in arrays], axis=axis)
        return compat_np.stack(arrays, axis=axis)

    def argsort(self, x, /, *args, **kwargs):
        if isinstance(x, sp.SparseArray):
            x = self._dense(x)
        xp = self._array_namespace(x)
        return xp.argsort(x, *args, **kwargs)

    def take(self, x, indices, /, *args, **kwargs):
        if isinstance(indices, sp.SparseArray):
            indices = self._dense(indices)
        if isinstance(x, sp.SparseArray):
            return sp.take(x, indices, *args, **kwargs)
        xp = self._array_namespace(x)
        return xp.take(x, indices, *args, **kwargs)

    def take_along_axis(self, x, indices, /, *, axis=-1):
        if isinstance(indices, sp.SparseArray):
            indices = self._dense(indices)
        if not isinstance(x, sp.SparseArray):
            xp = self._array_namespace(x)
            return xp.take_along_axis(x, indices, axis=axis)

        if not -x.ndim <= axis < x.ndim:
            raise IndexError("axis is out of bounds")
        axis %= x.ndim
        if indices.ndim != x.ndim:
            raise ValueError(
                "indices and input must have the same number of dimensions"
            )
        if indices.dtype.kind not in "iu":
            raise IndexError("indices must be integers")

        # COO accepts paired 1D index arrays. Broadcast only the output indices,
        # gather those entries, and restore the output shape without densifying x.
        indexers = []
        for dim, size in enumerate(x.shape):
            shape = [1] * x.ndim
            shape[dim] = size
            indexers.append(indices if dim == axis else np.arange(size).reshape(shape))
        indexers = np.broadcast_arrays(*indexers)
        result = x.asformat("coo")[tuple(index.ravel() for index in indexers)]
        return result.reshape(indexers[0].shape)

    def item(self, array):
        if isinstance(array, sp.SparseArray):
            return self._dense(array).item()
        return array.item()

    def replace(self, arr, old, new):
        if isinstance(arr, sp.DOK):
            arr = arr.asformat("coo")
        xp = sp if isinstance(arr, sp.SparseArray) else np
        return xp.where(xp.isnan(arr) if old != old else arr == old, new, arr)

    @property
    def linalg(self):
        return SmartSparseLinalg()

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


# Mixed sparse-dense policy for frameworks built on pydata/sparse.
#
# pydata/sparse refuses most operations mixing sparse and dense operands unless
# the result can stay sparse, and its contractions refuse nonzero fill values.
# The framework below intercepts those operations and densifies only when
# no sparse result can represent the answer, or when the backend can't operate on
# the sparse operand:
#
# - Elementwise: if applying the op to the sparse operands' fill values and the
#   dense operands gives a constant, the result has that fill and stays sparse.
#   Otherwise it is dense, and remembers the op applied to all operands' fill
#   values as its own fill value.
# - Contractions and linalg: sparse operands with a nonzero fill are densified.
# - Concatenation: dense members are stored sparsely under their own fill value
#   when every member agrees on it; otherwise everything is densified.
# - Indexing: sparse keys, and sparse values assigned into dense arrays, are
#   densified.
#
# Dense results of shape-only operations keep their input's fill value.

_ELEMENTWISE = {
    "abs",
    "acos",
    "acosh",
    "add",
    "asin",
    "asinh",
    "atan",
    "atan2",
    "atanh",
    "bitwise_and",
    "bitwise_invert",
    "bitwise_left_shift",
    "bitwise_or",
    "bitwise_right_shift",
    "bitwise_xor",
    "ceil",
    "clip",
    "conj",
    "copysign",
    "cos",
    "cosh",
    "divide",
    "equal",
    "exp",
    "expm1",
    "floor",
    "floor_divide",
    "greater",
    "greater_equal",
    "hypot",
    "imag",
    "isfinite",
    "isinf",
    "isnan",
    "less",
    "less_equal",
    "log",
    "log1p",
    "log2",
    "log10",
    "logaddexp",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "maximum",
    "minimum",
    "multiply",
    "negative",
    "nextafter",
    "not_equal",
    "positive",
    "pow",
    "power",
    "real",
    "reciprocal",
    "remainder",
    "replace",
    "round",
    "sign",
    "signbit",
    "sin",
    "sinh",
    "sqrt",
    "square",
    "subtract",
    "tan",
    "tanh",
    "trunc",
    "where",
}

_CONTRACTIONS = {
    "dot",
    "einsum",
    "inner",
    "kron",
    "matmul",
    "outer",
    "tensordot",
    "vecdot",
}

_CONCATENATIONS = {"concat", "concatenate", "stack"}

_FILL_PRESERVING = {
    "asarray",
    "broadcast_to",
    "copy",
    "expand_dims",
    "flatten",
    "flip",
    "getitem",
    "matrix_transpose",
    "moveaxis",
    "permute_dims",
    "ravel",
    "reshape",
    "roll",
    "squeeze",
    "swapaxes",
    "to_dense",
    "todense",
    "transpose",
}


def _is_sparse(raw) -> bool:
    return isinstance(raw, sp.SparseArray)


def _is_dense(raw) -> bool:
    return isinstance(raw, np.ndarray) and raw.ndim > 0


def _densify(raw):
    return raw.todense() if _is_sparse(raw) else raw


def _is_constant(array) -> bool:
    flat = np.asarray(array).ravel()
    if flat.size == 0:
        return True
    same = flat == flat[0]
    if flat.dtype.kind in "fc":
        same |= np.isnan(flat) & np.isnan(flat[0])
    return bool(np.all(same))


class WrappedArray:
    """An eager tensor whose every operation is redirected through its framework.

    ``array`` is the backend storage. ``fill_value`` is the value its implicit
    entries would take if it were stored sparsely: a sparse backing array's own
    fill value, or, for a dense one, whatever the framework says it was derived
    from (0 unless told otherwise). ``mod`` is the framework every operation is
    redirected to.
    """

    # NumPy defers to the reflected operators below for ``ndarray <op> self``.
    __array_ufunc__ = None

    def __init__(self, mod: "SmartSparseFramework", array, fill_value=0):
        self.mod = mod
        self.array = array
        self._fill_value = fill_value

    @property
    def fill_value(self):
        return getattr(self.array, "fill_value", self._fill_value)

    @property
    def shape(self):
        return self.array.shape

    @property
    def dtype(self):
        return self.array.dtype

    @property
    def ndim(self):
        return self.array.ndim

    @property
    def size(self):
        return self.array.size

    @property
    def T(self):
        return self.mod.permute_dims(self, tuple(reversed(range(self.ndim))))

    @property
    def mT(self):
        return self.mod.matrix_transpose(self)

    def __repr__(self):
        return f"WrappedArray({self.array!r}, fill_value={self.fill_value!r})"

    def __len__(self):
        return self.shape[0]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getattr__(self, name):
        # Private and dunder lookups (e.g. copy's __setstate__, or anything
        # before __init__ has run) must not recurse into the framework.
        if name.startswith("_") or name in ("array", "mod"):
            raise AttributeError(name)
        return self.mod.array_attribute(self, name)

    def __array__(self, dtype=None, copy=None):
        return self.mod.asnumpy(self, dtype=dtype)

    def __array_namespace__(self, *, api_version=None):
        if api_version not in {None, "2024.12"}:
            raise ValueError(f'"{api_version}" Array API version not supported.')
        return self.mod

    def __getitem__(self, key):
        return self.mod.getitem(self, key)

    def __setitem__(self, key, value):
        self.mod.setitem(self, key, value)

    def item(self):
        return self.mod.item(self)

    def __add__(self, other):
        return self.mod.add(self, other)

    def __radd__(self, other):
        return self.mod.add(other, self)

    def __sub__(self, other):
        return self.mod.subtract(self, other)

    def __rsub__(self, other):
        return self.mod.subtract(other, self)

    def __mul__(self, other):
        return self.mod.multiply(self, other)

    def __rmul__(self, other):
        return self.mod.multiply(other, self)

    def __abs__(self):
        return self.mod.abs(self)

    def __pos__(self):
        return self.mod.positive(self)

    def __neg__(self):
        return self.mod.negative(self)

    def __invert__(self):
        return self.mod.bitwise_invert(self)

    def __and__(self, other):
        return self.mod.bitwise_and(self, other)

    def __rand__(self, other):
        return self.mod.bitwise_and(other, self)

    def __lshift__(self, other):
        return self.mod.bitwise_left_shift(self, other)

    def __rlshift__(self, other):
        return self.mod.bitwise_left_shift(other, self)

    def __or__(self, other):
        return self.mod.bitwise_or(self, other)

    def __ror__(self, other):
        return self.mod.bitwise_or(other, self)

    def __rshift__(self, other):
        return self.mod.bitwise_right_shift(self, other)

    def __rrshift__(self, other):
        return self.mod.bitwise_right_shift(other, self)

    def __xor__(self, other):
        return self.mod.bitwise_xor(self, other)

    def __rxor__(self, other):
        return self.mod.bitwise_xor(other, self)

    def __truediv__(self, other):
        return self.mod.divide(self, other)

    def __rtruediv__(self, other):
        return self.mod.divide(other, self)

    def __floordiv__(self, other):
        return self.mod.floor_divide(self, other)

    def __rfloordiv__(self, other):
        return self.mod.floor_divide(other, self)

    def __mod__(self, other):
        return self.mod.remainder(self, other)

    def __rmod__(self, other):
        return self.mod.remainder(other, self)

    def __pow__(self, other):
        return self.mod.power(self, other)

    def __rpow__(self, other):
        return self.mod.power(other, self)

    def __matmul__(self, other):
        return self.mod.matmul(self, other)

    def __rmatmul__(self, other):
        return self.mod.matmul(other, self)

    def __sin__(self):
        return self.mod.sin(self)

    def __sinh__(self):
        return self.mod.sinh(self)

    def __cos__(self):
        return self.mod.cos(self)

    def __cosh__(self):
        return self.mod.cosh(self)

    def __tan__(self):
        return self.mod.tan(self)

    def __tanh__(self):
        return self.mod.tanh(self)

    def __asin__(self):
        return self.mod.asin(self)

    def __asinh__(self):
        return self.mod.asinh(self)

    def __acos__(self):
        return self.mod.acos(self)

    def __acosh__(self):
        return self.mod.acosh(self)

    def __atan__(self):
        return self.mod.atan(self)

    def __atanh__(self):
        return self.mod.atanh(self)

    def __atan2__(self, other):
        return self.mod.atan2(self, other)

    def __complex__(self):
        if self.ndim != 0:
            raise ValueError("Cannot convert non-scalar tensor to complex.")
        return complex(self.item())

    def __float__(self):
        if self.ndim != 0:
            raise ValueError("Cannot convert non-scalar tensor to float.")
        return float(self.item())

    def __int__(self):
        if self.ndim != 0:
            raise ValueError("Cannot convert non-scalar tensor to int.")
        return int(self.item())

    def __bool__(self):
        if self.ndim != 0:
            raise ValueError("Cannot convert non-scalar tensor to bool.")
        return bool(self.item())

    def __index__(self) -> int:
        if self.ndim != 0:
            raise ValueError("Cannot convert non-scalar tensor to index.")
        return operator.index(self.item())

    def __log__(self):
        return self.mod.log(self)

    def __log1p__(self):
        return self.mod.log1p(self)

    def __log2__(self):
        return self.mod.log2(self)

    def __log10__(self):
        return self.mod.log10(self)

    def __logaddexp__(self, other):
        return self.mod.logaddexp(self, other)

    def __logical_and__(self, other):
        return self.mod.logical_and(self, other)

    def __logical_or__(self, other):
        return self.mod.logical_or(self, other)

    def __logical_xor__(self, other):
        return self.mod.logical_xor(self, other)

    def __logical_not__(self):
        return self.mod.logical_not(self)

    def __lt__(self, other):
        return self.mod.less(self, other)

    def __le__(self, other):
        return self.mod.less_equal(self, other)

    def __gt__(self, other):
        return self.mod.greater(self, other)

    def __ge__(self, other):
        return self.mod.greater_equal(self, other)

    def __eq__(self, other):  # type: ignore[override]
        return self.mod.equal(self, other)

    def __ne__(self, other):  # type: ignore[override]
        return self.mod.not_equal(self, other)

    __hash__ = None  # type: ignore[assignment]


class WrappedNamespace:
    """A sub-namespace such as ``xp.linalg`` whose calls also go through ``call``."""

    def __init__(self, framework: "SmartSparseFramework", namespace, prefix: str):
        self.framework = framework
        self.namespace = namespace
        self.prefix = prefix

    def __getattr__(self, name):
        return self.framework._resolve(f"{self.prefix}.{name}", self.namespace, name)


class SmartSparseFramework(Framework):
    """SmartSparseKernels over WrappedArrays; every tensor operation goes through xp.

    ``kernels`` does the work on raw NumPy and pydata/sparse arrays. Every
    operation, whether called as ``xp.<name>(...)``, as an operator, or as an
    array method, reaches ``call(name, func, args, kwargs)``, which unwraps the
    arguments, densifies where a mixed sparse-dense operation needs it (see the
    policy above), runs ``func`` and wraps the result.
    """

    def __init__(self):
        self.kernels = SmartSparseKernels()

    def is_array(self, value) -> bool:
        return isinstance(value, np.ndarray | sp.SparseArray)

    @staticmethod
    def _fill(value):
        """An operand's fill value, or the operand itself if it isn't an array."""
        if isinstance(value, WrappedArray):
            return value.fill_value
        if _is_sparse(value):
            return value.fill_value
        if isinstance(value, np.ndarray) and value.ndim > 0:
            return value.dtype.type(0)
        return value

    def _operands(self, args, kwargs):
        values = list(args) + list(kwargs.values())
        return [self.unwrap(value) for value in values]

    def wrap(self, value, fill_value=0):
        if isinstance(value, WrappedArray):
            return value
        if isinstance(value, list):
            return [self.wrap(item, fill_value) for item in value]
        if isinstance(value, tuple):
            items = [self.wrap(item, fill_value) for item in value]
            return type(value)(*items) if hasattr(value, "_fields") else tuple(items)
        if self.is_array(value):
            return WrappedArray(self, value, fill_value)
        return value

    def unwrap(self, value):
        if isinstance(value, WrappedArray):
            return value.array
        if isinstance(value, list):
            return [self.unwrap(item) for item in value]
        if isinstance(value, tuple):
            items = [self.unwrap(item) for item in value]
            return type(value)(*items) if hasattr(value, "_fields") else tuple(items)
        if isinstance(value, dict):
            return {key: self.unwrap(item) for key, item in value.items()}
        return value

    def call(self, name, func, args, kwargs):
        op = name.rsplit(".", 1)[-1]
        if op in _ELEMENTWISE:
            return self._elementwise(op, func, args, kwargs)
        if op in _CONTRACTIONS or name.startswith("linalg."):
            return self._contraction(func, args, kwargs)
        if op in _CONCATENATIONS:
            return self._concatenate(func, args, kwargs)
        if op in ("getitem", "setitem"):
            return self._index(name, func, args)

        result = func(*self.unwrap(args), **self.unwrap(kwargs))
        fill = 0
        if args and isinstance(args[0], WrappedArray):
            if op in _FILL_PRESERVING:
                fill = args[0].fill_value
            elif op == "astype":
                dtype = args[1] if len(args) > 1 else kwargs["dtype"]
                with np.errstate(all="ignore"):
                    fill = np.asarray(args[0].fill_value).astype(dtype)[()]
        return self.wrap(result, fill)

    def _elementwise(self, op, func, args, kwargs):
        raw_args = self.unwrap(args)
        raw_kwargs = self.unwrap(kwargs)
        operands = self._operands(args, kwargs)

        def with_fills(dense):
            # Replace sparse operands by their fill values, and dense operands
            # by theirs too unless `dense` asks to keep them as arrays.
            def replace(value, raw):
                if _is_sparse(raw) or (not dense and _is_dense(raw)):
                    return self._fill(value)
                return raw

            return (
                [
                    replace(value, raw)
                    for value, raw in zip(args, raw_args, strict=True)
                ],
                {key: replace(kwargs[key], raw) for key, raw in raw_kwargs.items()},
            )

        with np.errstate(all="ignore"):
            fill_args, fill_kwargs = with_fills(dense=False)
            fill = np.asarray(func(*fill_args, **fill_kwargs))[()]

            if any(map(_is_sparse, operands)) and any(map(_is_dense, operands)):
                probe_args, probe_kwargs = with_fills(dense=True)
                if not _is_constant(func(*probe_args, **probe_kwargs)):
                    raw_args = [_densify(raw) for raw in raw_args]
                    raw_kwargs = {k: _densify(raw) for k, raw in raw_kwargs.items()}
                elif isinstance(
                    getattr(np, op, None), np.ufunc
                ) and np.broadcast_shapes(
                    *(raw.shape for raw in operands if _is_sparse(raw))
                ) == np.broadcast_shapes(
                    *(
                        raw.shape
                        for raw in operands
                        if _is_sparse(raw) or _is_dense(raw)
                    )
                ):
                    # Hand pydata/sparse the dense operand as is: it reads it
                    # only at the sparse coordinates. The kernels would store
                    # it as a full COO instead, and broadcasting that against
                    # a sparse matrix expands to every coordinate pair.
                    func = getattr(np, op)

        return self.wrap(func(*raw_args, **raw_kwargs), fill)

    def _contraction(self, func, args, kwargs):
        def prepare(raw):
            if _is_sparse(raw) and not np.all(np.asarray(raw.fill_value) == 0):
                return raw.todense()
            if isinstance(raw, list | tuple):
                return type(raw)(prepare(item) for item in raw)
            return raw

        raw_args = [prepare(raw) for raw in self.unwrap(args)]
        raw_kwargs = {key: prepare(raw) for key, raw in self.unwrap(kwargs).items()}
        return self.wrap(func(*raw_args, **raw_kwargs))

    def _concatenate(self, func, args, kwargs):
        (arrays, *rest) = args
        raws = self.unwrap(list(arrays))
        fills = [self._fill(array) for array in arrays]
        fill = fills[0] if fills and _is_constant(np.asarray(fills)) else None
        if any(map(_is_sparse, raws)):
            if fill is None:
                raws = [_densify(raw) for raw in raws]
            else:
                raws = [
                    raw if _is_sparse(raw) else sp.COO.from_numpy(raw, fill_value=fill)
                    for raw in raws
                ]
        fill = 0 if fill is None else fill
        return self.wrap(func(raws, *self.unwrap(rest), **self.unwrap(kwargs)), fill)

    def _index(self, name, func, args):
        (array, key, *value) = self.unwrap(args)
        if isinstance(key, tuple):
            key = tuple(_densify(index) for index in key)
        else:
            key = _densify(key)
        if name == "setitem":
            (value,) = value
            if not _is_sparse(array):
                value = _densify(value)
            func(array, key, value)
            return None
        return self.wrap(func(array, key), self._fill(args[0]))

    def _resolve(self, name, namespace, attr_name):
        attr = getattr(namespace, attr_name)
        if isinstance(attr, type) or not callable(attr):
            if isinstance(attr, types.ModuleType):
                return WrappedNamespace(self, attr, name)
            return attr

        def wrapped(*args, **kwargs):
            return self.call(name, attr, args, kwargs)

        wrapped.__name__ = attr_name
        return wrapped

    # Redirect targets for WrappedArray that aren't Array API functions.

    def getitem(self, array, key):
        return self.call("getitem", operator.getitem, (array, key), {})

    def setitem(self, array, key, value):
        self.call("setitem", operator.setitem, (array, key, value), {})

    def item(self, array):
        return self.call("item", self.kernels.item, (array,), {})

    def asnumpy(self, array, dtype=None):
        raw = self.unwrap(array)
        for method in ("todense", "toarray"):
            if hasattr(raw, method):
                raw = getattr(raw, method)()
                break
        return np.asarray(raw, dtype=dtype)

    def array_attribute(self, array, name):
        attr = getattr(array.array, name)
        if not callable(attr):
            return self.wrap(attr)

        def method(raw, *args, **kwargs):
            return getattr(raw, name)(*args, **kwargs)

        method.__name__ = name
        return lambda *args, **kwargs: self.call(name, method, (array, *args), kwargs)

    # Framework interface.

    def from_binsparse(self, array):
        return self.wrap(self.kernels.from_binsparse(array))

    def to_binsparse(self, array):
        return self.kernels.to_binsparse(self.unwrap(array))

    def lazy(self, array):
        return self.call("lazy", self.kernels.lazy, (array,), {})

    def compute(self, array):
        return self.call("compute", self.kernels.compute, (array,), {})

    def compile(self, func):
        return self.kernels.compile(func)

    def einsum(self, prgm, **kwargs):
        return self.call("einsum", self.kernels.einsum, (prgm,), kwargs)

    def unfold(self, x, kernel_shape, **kwargs):
        return self.call("unfold", self.kernels.unfold, (x, kernel_shape), kwargs)

    def replace(self, arr, old, new):
        return self.call("replace", self.kernels.replace, (arr, old, new), {})

    @property
    def linalg(self):
        return WrappedNamespace(self, self.kernels.linalg, "linalg")

    def __getattr__(self, name):
        if name.startswith("_") or name == "kernels":
            raise AttributeError(name)
        return self._resolve(name, self.kernels, name)


xp = SmartSparseFramework()
