"""SAPS adapter for galley-jl-python: Finch.jl tensors with the Galley scheduler.

Install with ``pip install galley-jl-python`` (import name ``galley_jl_python``;
the unrelated PyPI package ``galley`` is a Docker test tool). The adapter
converts inputs (storing sparse ones as row-major CSF) and outputs, translates
NumPy-style dtypes to galley's Julia dtypes, and forwards every operation to
galley's own namespace. It also spells a few standard operations galley has
under other names or can compose from its own (``vecdot``, ``transpose``,
``linalg.norm``, ``replace``); galley's tensors are read-only, so ``xp.at``
updates them with ``where`` (Framework.at).
Operations galley cannot do, such as ``concat`` and ``linalg.pinv``, are left
unsupported.
"""

import inspect
import types
from functools import wraps

import numpy as np

import galley_jl_python as galley
from binsparse import (
    CSCMatrix,
    CSRMatrix,
    CustomTensor,
    DenseLevel,
    DMATCMatrix,
    DMATRMatrix,
    DVECVector,
    ElementLevel,
)
from binsparse.conversions import from_numpy, from_scipy, to_numpy, to_sparse
from galley_jl_python.julia import jl

from saps_framework import Framework

# galley's dtypes are Julia DataTypes (galley_jl_python/dtypes.py), and its
# creation functions look dtypes up in a table keyed by them, so NumPy and
# builtin dtypes are translated before they reach galley.
_DTYPES = {
    np.dtype(np.bool_): galley.bool,
    np.dtype(np.int8): galley.int8,
    np.dtype(np.int16): galley.int16,
    np.dtype(np.int32): galley.int32,
    np.dtype(np.int64): galley.int64,
    np.dtype(np.uint8): galley.uint8,
    np.dtype(np.uint16): galley.uint16,
    np.dtype(np.uint32): galley.uint32,
    np.dtype(np.uint64): galley.uint64,
    np.dtype(np.float16): galley.float16,
    np.dtype(np.float32): galley.float32,
    np.dtype(np.float64): galley.float64,
    np.dtype(np.complex64): galley.complex64,
    np.dtype(np.complex128): galley.complex128,
}


def _galley_dtype(dtype):
    if dtype is None:
        return None
    try:
        np_dtype = np.dtype(dtype)
    except TypeError:
        return dtype  # Already one of galley's Julia dtypes.
    return _DTYPES[np_dtype]


def _translate_dtype(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if "dtype" in kwargs:
            kwargs["dtype"] = _galley_dtype(kwargs["dtype"])
        return func(*args, **kwargs)

    return wrapper


def _fill_value(array):
    """A Binsparse tensor's fill value; without an explicit fill it is zero."""
    return array.fill_value if array.fill is True else 0


def _julia_fill(dtype, fill_value):
    """fill_value as a scalar Julia reads as the type of NumPy's dtype.

    Finch.jl requires an isbits fill. Julia converts NumPy's numeric scalars to
    the matching Julia number, but not ``np.bool_``, so a Boolean fill is passed
    as a Python bool. (galley's own sparse constructors pass ``np.bool_``, which
    Finch.jl rejects with "Finch currently only supports isbits defaults".)
    """
    fill_value = np.asarray(fill_value).item()
    return bool(fill_value) if dtype == np.bool_ else dtype.type(fill_value)


def _element(values, fill_value):
    """Finch.jl's Element level over values, with a fill of the same Julia type."""
    values = np.ascontiguousarray(values)
    return jl.Element(_julia_fill(values.dtype, fill_value), values)


def _indices(indices):
    return jl.PlusOneVector(np.ascontiguousarray(indices))


def _compressed(array, *, row_major):
    """A CSR (row_major) or CSC matrix as Finch.jl's Dense(SparseList(Element)).

    Finch.jl stores tensors column-major, so a CSR matrix is the transpose of a
    CSC one, swizzled back, as galley's own CSR constructor builds it.
    """
    rows, cols = array.shape
    outer, inner = (rows, cols) if row_major else (cols, rows)
    level = jl.SparseList(
        _element(array.values, _fill_value(array)),
        inner,
        _indices(array.pointers_to_1),
        _indices(array.indices_1),
    )
    tensor = jl.Tensor(jl.Dense(level, outer))
    return galley.Tensor(jl.swizzle(tensor, *((2, 1) if row_major else (1, 2))))


def _coo(coo):
    """A PyData/Sparse COO array as Finch.jl's SparseCOO level, without
    densifying it, keeping its fill value.

    A row-major tensor is stored with its dimensions reversed, so its first
    axis is outermost; SparseCOO requires coordinates sorted outermost first.
    """
    coords, data = coo.coords, coo.data
    order = np.lexsort(coords[::-1])
    coords, data = coords[:, order], data[order]
    if coords.shape[1] > 1:
        new = np.any(coords[:, 1:] != coords[:, :-1], axis=0)
        if not new.all():
            starts = np.flatnonzero(np.concatenate(([True], new)))
            coords, data = coords[:, starts], np.add.reduceat(data, starts)
    ndim = coo.ndim
    level = jl.SparseCOO[ndim](
        _element(data, coo.fill_value),
        tuple(int(size) for size in coo.shape[::-1]),
        jl.Vector[jl.Int]([1, len(data) + 1]),
        tuple(_indices(axis) for axis in coords[::-1]),
    )
    return galley.Tensor(jl.swizzle(jl.Tensor(level), *range(ndim, 0, -1)))


def _csf(tensor, dtype, fill_value):
    """Finch.jl's copy of a sparse galley tensor into row-major CSF: a dense
    level for its first dimension over a SparseList for each later one, or one
    SparseList for a vector. dtype and fill_value are the tensor's NumPy element
    type and fill, which galley's own fill_value would widen to Python's."""
    levels = galley.Element(_julia_fill(dtype, fill_value))
    for _ in range(tensor.ndim - 1):
        levels = galley.SparseList(levels)
    levels = galley.Dense(levels) if tensor.ndim > 1 else galley.SparseList(levels)
    return tensor.to_storage(galley.Storage(levels))


def _norm(x, ord=None, axis=None, keepdims=False):
    """``linalg.norm`` of a vector, or the Frobenius norm of a whole array."""
    if axis is not None and x.ndim != 1:
        raise NotImplementedError("galley-jl-python has no norm along an axis.")
    if ord not in (None, 2, "fro") or (ord == 2 and x.ndim != 1):
        raise NotImplementedError(f"galley-jl-python has no {ord!r} norm.")
    return galley.linalg.vector_norm(x, keepdims=keepdims)


class GalleyFramework(Framework):
    """SAPS adapter for galley-jl-python with the Galley scheduler."""

    def __init__(self):
        # galley installs this scheduler on import; select it explicitly anyway.
        self._scheduler = galley.GalleyScheduler()
        galley.set_optimizer(self._scheduler)

    def from_binsparse(self, array):
        match array:
            case DVECVector() | DMATRMatrix() | DMATCMatrix():
                return galley.asarray(to_numpy(array))
            case CustomTensor(shape=(), transpose=None, level=ElementLevel()):
                return galley.asarray(to_numpy(array))
            case CustomTensor(
                shape=shape,
                transpose=None,
                level=DenseLevel(rank=rank, level=ElementLevel()),
            ) if rank == len(shape):
                return galley.asarray(to_numpy(array))
            # galley's sparse constructors go through SciPy, which drops
            # nonzero fill values, and only take 2-D COO, so the adapter builds
            # Finch.jl's levels itself. galley reads inputs in the layout they
            # arrive in, and Finch.jl can only scan COO, so Finch.jl copies
            # sparse inputs into row-major CSF, which CSR already is.
            case CSCMatrix():  # Before CSRMatrix, which it subclasses.
                tensor = _compressed(array, row_major=False)
                return _csf(tensor, array.values.dtype, _fill_value(array))
            case CSRMatrix():
                return _compressed(array, row_major=True)
            case _:
                coo = to_sparse(array)
                return _csf(_coo(coo), coo.data.dtype, coo.fill_value)

    def to_binsparse(self, array):
        if not isinstance(array, galley.Tensor):
            return from_numpy(np.asarray(array))
        array = self.compute(array)
        if array.ndim == 2:
            try:
                return from_scipy(array.to_scipy_sparse())
            except ValueError:
                pass  # Dense storage or a nonzero fill value.
        return from_numpy(np.asarray(array.todense()))

    def lazy(self, array):
        return galley.lazy(array)

    def compute(self, array):
        if isinstance(array, galley.Tensor):
            return galley.compute(array, opt=self._scheduler)
        return array

    def einsum(self, prgm, **kwargs):
        # galley's einop takes the same program syntax as SAPS.
        return galley.einop(prgm, **kwargs)

    def asarray(self, obj, /, **kwargs):
        # galley reads Python lists but not tuples.
        if isinstance(obj, tuple):
            obj = list(obj)
        return _translate_dtype(galley.asarray)(obj, **kwargs)

    def vecdot(self, x1, x2, /, *, axis=-1):
        if x1.dtype in (galley.complex64, galley.complex128):
            x1 = galley.conj(x1)
        return galley.sum(galley.multiply(x1, x2), axis=axis)

    def transpose(self, x, axes=None):
        if axes is None:
            axes = tuple(range(x.ndim - 1, -1, -1))
        return galley.permute_dims(x, tuple(axes))

    @property
    def linalg(self):
        return types.SimpleNamespace(
            **{name: getattr(galley.linalg, name) for name in galley.linalg.__all__},
            norm=_norm,
        )

    def where(self, condition, x1, x2, /):
        # The Array API allows Python scalars here; galley's where reads tensors.
        x1 = x1 if isinstance(x1, galley.Tensor) else galley.asarray(x1)
        x2 = x2 if isinstance(x2, galley.Tensor) else galley.asarray(x2)
        return galley.where(condition, x1, x2)

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
        raise NotImplementedError("galley-jl-python has no unfold.")

    def replace(self, arr, old, new):
        matches = galley.isnan(arr) if old != old else arr == old
        return self.where(matches, new, arr)

    def __getattr__(self, name):
        attr = getattr(galley, name)
        if inspect.isfunction(attr):
            return _translate_dtype(attr)
        return attr


xp = GalleyFramework()
