"""SAPS adapter for finch-tensor's Julia backend with the Galley scheduler.

Install finch-tensor (with its ``julia`` extra) from
https://github.com/finch-tensor/finch-tensor; ``COMPILE_JULIA_GALLEY`` is not in
the 0.7.0 release. Every eager finch operation computes with finch's default
scheduler, so the adapter makes Galley that default rather than steering
individual operations.
"""

import numpy as np

import binsparse as bsp
import finch as ft
from binsparse.conversions import from_numpy, to_scipy
from finch.algebra import as_fill
from finch.autoschedule import COMPILE_JULIA_GALLEY
from finch.compile_jl.interop import jl_tensor_to_python, tensor_to_jl
from finch.compile_jl.julia import jl

from saps_framework import Framework

# Eager operations (zeros, where, logical_or, sum, matmul, ...) call
# ft.compute with the thread-local default scheduler.
ft.set_default_scheduler(COMPILE_JULIA_GALLEY)


def _reformat(tensor, *, dense):
    """Copy a FiberTensor into CSR-like (or dense) levels with Finch.jl."""
    level = jl.Element(tensor.fill_value)
    for depth in range(tensor.ndim):
        sparse_level = not dense and depth == 0
        level = jl.SparseList(level) if sparse_level else jl.Dense(level)
    return jl_tensor_to_python(jl.Tensor(level, tensor_to_jl(tensor)))


# Binsparse formats SAPS's correctness checkers can read.
_READABLE = (
    bsp.CSRMatrix,
    bsp.CSCMatrix,
    bsp.COORMatrix,
    bsp.DMATRMatrix,
    bsp.DMATCMatrix,
    bsp.DVECVector,
)


def _canonical(tensor):
    """Parse a custom Binsparse descriptor as its standard format, if any."""
    container = bsp.InMemoryBinsparseContainer()
    tensor.serialize(container, alias=False)
    return bsp.BinsparseTensor.parse(container, alias=True)


class FinchFramework(Framework):
    def from_binsparse(self, array):
        try:
            return ft.from_binsparse(array)
        except NotImplementedError:
            # Finch has no transposed (CSC) Binsparse layout; read through
            # SciPy CSR, which finch.asarray imports natively.
            return ft.asarray(to_scipy(array).tocsr())

    def to_binsparse(self, array):
        if isinstance(array, ft.LazyTensor):
            array = self.compute(array)
        if hasattr(array, "raw_julia_obj") and array.ndim > 0:
            # Results Julia owns (JuliaOwnedTensor); its to_numpy assumes dense
            # storage, so read the Julia tensor's levels instead.
            array = jl_tensor_to_python(array.raw_julia_obj)
        if not isinstance(array, ft.FiberTensor):
            # Scalars and BufferizedNDArrays are dense.
            return from_numpy(np.asarray(ft.to_numpy(array)))
        try:
            tensor = _canonical(ft.to_binsparse(array))
        except NotImplementedError:
            # Galley may choose levels without a Binsparse equivalent (hash,
            # bytemap); Finch.jl copies them into a CSR-like layout.
            tensor = _canonical(ft.to_binsparse(_reformat(array, dense=False)))
        # SAPS's checkers read CSR/CSC/COO matrices with zero fill and dense
        # tensors. Other sparse matrices (e.g. DCSR) are copied to CSR by
        # Finch.jl; everything else is exported densely.
        zero_fill = not (tensor.fill is True and tensor.fill_value != 0)
        if zero_fill and isinstance(tensor, _READABLE):
            return tensor
        if zero_fill and array.ndim == 2:
            tensor = _canonical(ft.to_binsparse(_reformat(array, dense=False)))
            if isinstance(tensor, _READABLE):
                return tensor
        return _canonical(ft.to_binsparse(_reformat(array, dense=True)))

    def lazy(self, array):
        return ft.defer(array)

    def compute(self, array):
        return ft.compute(array, ctx=COMPILE_JULIA_GALLEY)

    def compile(self, func):
        # The default scheduler is thread-local; pin it wherever func runs.
        def compiled(*args, **kwargs):
            with ft.with_default_scheduler(COMPILE_JULIA_GALLEY):
                return func(*args, **kwargs)

        return compiled

    def einsum(self, prgm, **kwargs):
        inputs = {name: ft.defer(value) for name, value in kwargs.items()}
        return self.compute(ft.einop(prgm, **inputs))

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
        raise NotImplementedError("Finch has no unfold operation.")

    def replace(self, arr, old, new):
        matches = ft.isnan(arr) if old != old else ft.equal(arr, old)
        return ft.where(matches, new, arr)

    def with_fill_value(self, array, value):
        # Reinterpret missing entries, sharing the stored data.
        return array.with_fill(as_fill(value))

    def __getattr__(self, name):
        return getattr(ft, name)


xp = FinchFramework()
