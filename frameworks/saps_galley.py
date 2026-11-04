"""SAPS adapter for galley-jl-python: Finch.jl tensors with the Galley scheduler.

Install with ``pip install galley-jl-python`` (import name ``galley_jl_python``;
the unrelated PyPI package ``galley`` is a Docker test tool). The adapter only
converts inputs and outputs, translates NumPy-style dtypes to galley's Julia
dtypes, and forwards every operation to galley's own namespace. Operations
galley does not provide, such as ``__setitem__`` and ``logical_not``, are left
unsupported.
"""

import inspect
from functools import wraps

import numpy as np

import galley_jl_python as galley
from binsparse import (
    COORMatrix,
    CSCMatrix,
    CSRMatrix,
    CustomTensor,
    DenseLevel,
    DMATCMatrix,
    DMATRMatrix,
    DVECVector,
    ElementLevel,
)
from binsparse.conversions import from_numpy, from_scipy, to_numpy, to_scipy, to_sparse

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
            case CSRMatrix() | CSCMatrix() | COORMatrix():
                # galley's Tensor constructor reads SciPy CSR, CSC and COO.
                return galley.asarray(to_scipy(array))
            case _:
                # galley has no sparse constructor for other ranks, so build a
                # NumPy array and let galley convert it to its COO format.
                coo = to_sparse(array)
                return galley.asarray(
                    coo.todense(), format="coo", fill_value=coo.fill_value
                )

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
        raise NotImplementedError("galley-jl-python has no replace.")

    def __getattr__(self, name):
        attr = getattr(galley, name)
        if inspect.isfunction(attr):
            return _translate_dtype(attr)
        return attr


xp = GalleyFramework()
