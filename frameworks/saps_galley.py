"""SAPS adapter for galley-jl-python (Finch.jl driven by the Galley scheduler).

galley-jl-python is installed from https://github.com/finch-tensor/galley-jl-python
and imported as ``galley_jl_python``. Every eager operation is planned by the
Galley scheduler, which this module installs as Finch's global scheduler.

Benchmark functions are compiled with ``gl.jit``, which defers tensors and
computes them only at control flow, so galley fuses the operations in between.

Only operations galley (or Finch.jl) can run natively are provided. Where
galley's Python API has a gap that a native call fills, this module bridges it:
``_patch_tensor`` adds reflected operators, NumPy operands, and ``.T`` to
``galley_jl_python.Tensor``, and ``GalleyFramework`` adds namespace functions
such as ``maximum`` and ``vecdot``. Operations galley does not support (item
assignment, ``concat``, ``sort``, most of ``linalg``, ``unfold``, ...) raise
rather than fall back to a dense NumPy/SciPy implementation.
"""

import builtins
import inspect
import operator
import types
import warnings

import numpy as np
import scipy.sparse as sps

import galley_jl_python as gl
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
    to_sparse,
)
from galley_jl_python.dtypes import jl_to_np_dtype
from galley_jl_python.julia import jc, jl

from saps_framework import Framework

Tensor = gl.Tensor

_np_to_jl_dtype = {
    np.dtype(np_dtype): jl_dtype
    for jl_dtype, np_dtype in jl_to_np_dtype.items()
    if np_dtype is not None
}

# Builds a Finch COO tensor from 0-based coordinates listed in Python axis order.
_fsparse = jl.seval(
    "(coords, vals, shape, fv) -> Finch.fsparse("
    "(Vector{Int}(c) .+ 1 for c in coords)..., Vector(vals), Tuple(shape);"
    " fill_value=fv)"
)


def _np_dtype(dtype):
    """Map a Julia or NumPy dtype to the corresponding NumPy dtype."""
    if dtype in jl_to_np_dtype:
        return np.dtype(jl_to_np_dtype[dtype])
    return np.dtype(dtype)


def _jl_dtype(dtype):
    """Map a NumPy, Python, or Julia dtype to the Julia dtype galley expects."""
    if dtype is None or dtype in jl_to_np_dtype:
        return dtype
    return _np_to_jl_dtype[np.dtype(dtype)]


def _fill_value(array):
    return getattr(array, "fill_value", 0) if getattr(array, "fill", False) else 0


def _is_zero(value) -> bool:
    return builtins.bool(np.asarray(value) == 0)


def _to_tensor(value):
    """Coerce NumPy arrays, NumPy scalars, and SciPy matrices to galley tensors."""
    if isinstance(value, Tensor):
        return value
    if isinstance(value, np.ndarray | np.generic) or sps.issparse(value):
        return Tensor(np.asarray(value) if isinstance(value, np.generic) else value)
    return value


def _to_operand(value):
    """Prepare a binary-operator operand; Python scalars pass through."""
    if isinstance(value, np.generic):
        return value.item()
    return _to_tensor(value)


def _coo_parts(array):
    """Return 0-based COO coordinates (Python axis order) and values of a tensor."""
    storage = gl.Storage(
        gl.SparseCOO(array.ndim, gl.Element(array.fill_value)), order="F"
    )
    level = array.to_storage(storage)._obj.body.lvl
    nnz = builtins.int(level.ptr[1]) - 1
    coords = np.stack([np.asarray(idx)[:nnz] - 1 for idx in level.tbl])
    values = np.asarray(level.lvl.val)[:nnz]
    return coords, values


def _patch_tensor():
    def binary(op_name, reflected=False):
        forward = getattr(Tensor, op_name)

        def method(self, other):
            other = _to_operand(other)
            if reflected:
                if not isinstance(other, Tensor):
                    other = np.asarray(other, dtype=_np_dtype(self.dtype))
                    other = Tensor(other, fill_value=other.item())
                return forward(other, self)
            return forward(self, other)

        return method

    for op in (
        "add", "sub", "mul", "truediv", "floordiv", "mod", "pow", "matmul",
        "and", "or", "xor", "lshift", "rshift",
    ):  # fmt: skip
        name = f"__{op}__"
        setattr(Tensor, name, binary(name))
        setattr(Tensor, f"__r{op}__", binary(name, reflected=True))
    for op in ("lt", "le", "gt", "ge", "eq", "ne"):
        name = f"__{op}__"
        setattr(Tensor, name, binary(name))

    Tensor.__hash__ = object.__hash__
    Tensor.T = property(lambda self: self.permute_dims(tuple(range(self.ndim))[::-1]))
    Tensor.astype = lambda self, dtype, copy=True: gl.astype(
        self, _jl_dtype(dtype), copy=copy
    )
    # Under `gl.jit` tensors may be lazy; compute them where a value is needed.
    todense = Tensor.todense
    Tensor.todense = lambda self: todense(gl.compute(self))
    Tensor.item = lambda self: self.todense().item()


_patch_tensor()


class GalleyLinalg:
    def norm(self, x, ord=None, axis=None, keepdims=False):
        if axis is None and ord in (None, 2) and x.ndim < 2:
            return gl.linalg.vector_norm(x, keepdims=keepdims)
        if axis is None and ord in (None, "fro"):
            return gl.sqrt(gl.sum(x * x))
        raise NotImplementedError(
            f"galley supports only the flattened 2-norm; got ord={ord}, axis={axis}."
        )

    def vector_norm(self, x, *, axis=None, keepdims=False, ord=2):
        return self.norm(x, ord=ord, axis=axis, keepdims=keepdims)

    def __getattr__(self, name):
        return getattr(gl.linalg, name)


_DTYPE_NAMES = {
    "bool", "complex64", "complex128", "float16", "float32", "float64",
    "int_", "int8", "int16", "int32", "int64",
    "uint", "uint8", "uint16", "uint32", "uint64",
}  # fmt: skip


class GalleyFramework(Framework):
    def __init__(self, scheduler=None):
        self.scheduler = scheduler or gl.GalleyScheduler()
        gl.set_optimizer(self.scheduler)

    def compile(self, func):
        # The harness wraps the benchmark function in a closure
        # `benchmark(meta, *data_args)`, and `gl.jit` does not support `*args`,
        # so compile the wrapped function instead.
        closure = inspect.getclosurevars(func).nonlocals
        function = closure.get("function")
        if function is None:
            return func
        bound_args: tuple = (closure.get("xp", self),)
        if isinstance(function, types.MethodType):
            bound_args = (function.__self__, *bound_args)
            function = function.__func__
        try:
            jitted = gl.jit(function)
        except (OSError, ValueError, NotImplementedError) as e:
            warnings.warn(
                f"gl.jit could not compile {function.__qualname__}, so it runs "
                f"eagerly: {e}",
                stacklevel=2,
            )
            return func

        def compiled(meta, *data_args):
            return jitted(*bound_args, meta, *data_args)

        return compiled

    def from_binsparse(self, array):
        match array:
            case DVECVector() | DMATRMatrix() | DMATCMatrix():
                return Tensor(np.ascontiguousarray(to_numpy(array)))
            case CustomTensor(shape=(), transpose=None, level=ElementLevel()):
                return Tensor(np.asarray(to_numpy(array)))
            case CustomTensor(
                shape=shape,
                transpose=None,
                level=DenseLevel(rank=rank, level=ElementLevel()),
            ) if rank == len(shape):
                return Tensor(np.ascontiguousarray(to_numpy(array)))
        if len(array.shape) == 2 and _is_zero(_fill_value(array)):
            return Tensor(to_scipy(array).tocsr())
        coo = to_sparse(array)
        fill_value = coo.fill_value
        return Tensor(
            _fsparse(
                tuple(np.asarray(c, dtype=np.int64) for c in coo.coords),
                np.asarray(coo.data),
                coo.shape,
                coo.dtype.type(fill_value),
            )
        )

    def to_binsparse(self, array):
        if not isinstance(array, Tensor):
            return from_numpy(np.asarray(array))
        if not array.is_computed():
            array = gl.compute(array)
        if array.ndim == 0 or array._is_dense or not _is_zero(array.fill_value):
            return from_numpy(np.asarray(array.todense()))
        if array.ndim == 2:
            try:
                return from_scipy(array.to_scipy_sparse().tocoo())
            except ValueError:
                pass  # level format scipy can't read; use the generic COO path
        coords, values = _coo_parts(array)
        return from_sparse(pydata_sparse.COO(coords, values, shape=array.shape))

    def einsum(self, prgm, **kwargs):
        kwargs = {key: _to_tensor(value) for key, value in kwargs.items()}
        return gl.einop(prgm, **kwargs)

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
        raise NotImplementedError("galley-jl-python does not implement unfold.")

    def with_fill_value(self, array, value):
        # Relabel the background value in place of the stored one, like
        # pydata/sparse's fill_value assignment; stored entries are unchanged.
        array = gl.compute(_to_tensor(array))
        # The fill value is a type parameter, so this returns a new tensor.
        body = jl.Finch.set_fill_value_b(
            jl.deepcopy(array._obj.body), jc.convert(array.dtype, value)
        )
        return Tensor(jl.swizzle(body, *array._order))

    # Elementwise binary functions galley lacks, evaluated by Julia broadcast.
    def _broadcast(self, op, x1, x2):
        x1, x2 = _to_operand(x1), _to_operand(x2)
        if not isinstance(x1, Tensor):
            x1, x2 = x2, x1
        if not isinstance(x1, Tensor):
            return getattr(builtins, op)(x1, x2)
        return x1._elemwise_op(op, x2)

    def maximum(self, x1, x2, /):
        return self._broadcast("max", x1, x2)

    def minimum(self, x1, x2, /):
        return self._broadcast("min", x1, x2)

    def vecdot(self, x1, x2, /, *, axis=-1):
        return gl.sum(_to_tensor(x1) * _to_tensor(x2), axis=axis)

    def matmul(self, x1, x2, /):
        return operator.matmul(_to_tensor(x1), _to_tensor(x2))

    def where(self, condition, x1, x2, /):
        condition = _to_tensor(condition)
        dtype = np.result_type(
            *(
                _np_dtype(x.dtype) if isinstance(x, Tensor | np.ndarray) else type(x)
                for x in (x1, x2)
            )
        )

        def as_tensor(x):
            if isinstance(x, Tensor):
                return x
            return Tensor(np.broadcast_to(np.asarray(x, dtype=dtype), ()).copy())

        return gl.where(condition, as_tensor(_to_tensor(x1)), as_tensor(_to_tensor(x2)))

    def transpose(self, x, axes=None):
        x = _to_tensor(x)
        if axes is None:
            axes = tuple(range(x.ndim))[::-1]
        return gl.permute_dims(x, tuple(axes))

    def permute_dims(self, x, axes):
        return gl.permute_dims(_to_tensor(x), tuple(axes))

    def asarray(self, obj, /, *, dtype=None, copy=None, **kwargs):
        if isinstance(obj, list | tuple) or np.isscalar(obj):
            obj = np.asarray(obj, dtype=None if dtype is None else _np_dtype(dtype))
        return gl.asarray(_to_tensor(obj), dtype=_jl_dtype(dtype), copy=copy, **kwargs)

    def array(self, obj, /, *, dtype=None, **kwargs):
        return self.asarray(obj, dtype=dtype, **kwargs)

    def _creation(self, name, *args, dtype=None, **kwargs):
        return getattr(gl, name)(*args, dtype=_jl_dtype(dtype), **kwargs)

    def zeros(self, shape, *, dtype=None, **kwargs):
        return self._creation("zeros", shape, dtype=dtype, **kwargs)

    def ones(self, shape, *, dtype=None, **kwargs):
        return self._creation("ones", shape, dtype=dtype, **kwargs)

    def full(self, shape, fill_value, *, dtype=None, **kwargs):
        return self._creation("full", shape, fill_value, dtype=dtype, **kwargs)

    def eye(self, *args, dtype=None, **kwargs):
        return self._creation("eye", *args, dtype=dtype, **kwargs)

    def arange(self, *args, dtype=None, **kwargs):
        return self._creation("arange", *args, dtype=dtype, **kwargs)

    def astype(self, x, dtype, /, *, copy=True):
        return gl.astype(_to_tensor(x), _jl_dtype(dtype), copy=copy)

    def isdtype(self, dtype, kind):
        return np.isdtype(_np_dtype(dtype), kind)

    def iinfo(self, dtype):
        return np.iinfo(_np_dtype(dtype))

    def finfo(self, dtype):
        return np.finfo(_np_dtype(dtype))

    @property
    def linalg(self):
        return GalleyLinalg()

    def __getattr__(self, name):
        attr = getattr(gl, name, None)
        if attr is None:
            raise AttributeError(
                f"'{self.__class__.__name__}' has no attribute '{name}'"
            )
        if not callable(attr) or isinstance(attr, type) or name in _DTYPE_NAMES:
            return attr

        def wrapped(*args, **kwargs):
            args = tuple(_to_tensor(arg) for arg in args)
            kwargs = {key: _to_tensor(value) for key, value in kwargs.items()}
            if "dtype" in kwargs:
                kwargs["dtype"] = _jl_dtype(kwargs["dtype"])
            return attr(*args, **kwargs)

        return wrapped


xp = GalleyFramework()
