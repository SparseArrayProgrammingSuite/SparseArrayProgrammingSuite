"""SAPS adapter for python-graphblas (SuiteSparse:GraphBLAS).

A tensor is a GraphBLAS Vector, Matrix or Scalar plus a fill value: missing
entries mean fill_value, stored entries are actual values. GraphBLAS has no
notion of a fill value, so every Array API operation maps to the GraphBLAS
operation whose structural semantics agree with the operands' fills:

- Fills that annihilate the operator admit an intersection (ewise_mult, or a
  structural mask when only one operand annihilates).
- Fills that are the operator's identity admit a union (ewise_add).
- Any other fills use ewise_union with the fills as the missing-entry defaults.
- Contractions use mxm/mxv/vxm/inner over the semiring built from the
  reduction monoid and the elementwise operator, provided each operand's fill
  annihilates the multiply into the monoid's identity. Operands whose fills do
  not are first materialized as full GraphBLAS objects.

Annihilation is algebraic, so IEEE NaN/Inf stored values are out of scope.
GraphBLAS objects have at most two dimensions; higher-rank programs raise.
"""

import builtins
import operator
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sps

import graphblas as gb
import sparse
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
from graphblas import agg, binary, monoid, semiring, unary

from saps_framework import Framework
from saps_framework.einsum import (
    Access,
    Call,
    Literal,
    nary_ops,
    parse_einsum,
    reduction_ops,
    unary_ops,
)

# ---------------------------------------------------------------------- dtypes


def _np_dtype(dtype):
    if dtype is None:
        return None
    if isinstance(dtype, gb.dtypes.DataType):
        return np.dtype(dtype.np_type)
    return np.dtype(dtype)


def _gb_dtype(dtype):
    return gb.dtypes.lookup_dtype(_np_dtype(dtype))


def _cast(value, dtype):
    """Return value as a NumPy scalar of dtype."""
    with np.errstate(all="ignore"):
        return np.asarray(value).astype(dtype)[()]


def _sum_dtype(dtype):
    # NumPy's default accumulator: small integers and bools widen.
    if dtype.kind in "bi":
        return np.dtype(np.int64)
    if dtype.kind == "u":
        return np.dtype(np.uint64)
    return dtype


def _is_scalar(x):
    return isinstance(x, bool | int | float | complex | np.generic)


# ------------------------------------------------------------- operator tables

# Array API name -> (GraphBLAS binary operator, its counterpart for booleans).
_BINARY = {
    "add": ("plus", "lor"),
    "subtract": ("minus", None),
    "multiply": ("times", "land"),
    "divide": ("truediv", None),
    "floor_divide": ("floordiv", None),
    "remainder": ("numpy.remainder", None),
    "power": ("pow", None),
    "maximum": ("max", "lor"),
    "minimum": ("min", "land"),
    "atan2": ("atan2", None),
    "equal": ("eq", "eq"),
    "not_equal": ("ne", "ne"),
    "less": ("lt", "lt"),
    "less_equal": ("le", "le"),
    "greater": ("gt", "gt"),
    "greater_equal": ("ge", "ge"),
    "logical_and": ("land", "land"),
    "logical_or": ("lor", "lor"),
    "logical_xor": ("lxor", "lxor"),
    "bitwise_and": ("band", "land"),
    "bitwise_or": ("bor", "lor"),
    "bitwise_xor": ("bxor", "lxor"),
    "bitwise_left_shift": ("numpy.left_shift", None),
    "bitwise_right_shift": ("numpy.right_shift", None),
}
_NUMPY_BINARY = {"atan2": np.arctan2}
_COMPARISONS = {"equal", "not_equal", "less", "less_equal", "greater", "greater_equal"}
_LOGICAL = {"logical_and", "logical_or", "logical_xor"}

# Array API name -> GraphBLAS unary operator.
_UNARY = {
    "abs": "abs",
    "absolute": "abs",
    "negative": "ainv",
    "positive": "identity",
    "sqrt": "sqrt",
    "exp": "exp",
    "expm1": "expm1",
    "log": "log",
    "log1p": "log1p",
    "log2": "log2",
    "log10": "log10",
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "sinh": "sinh",
    "cosh": "cosh",
    "tanh": "tanh",
    "asin": "asin",
    "acos": "acos",
    "atan": "atan",
    "asinh": "asinh",
    "acosh": "acosh",
    "atanh": "atanh",
    "arcsin": "asin",
    "arccos": "acos",
    "arctan": "atan",
    "arcsinh": "asinh",
    "arccosh": "acosh",
    "arctanh": "atanh",
    "floor": "floor",
    "ceil": "ceil",
    "round": "round",
    "sign": "signum",
    "isfinite": "isfinite",
    "isinf": "isinf",
    "isnan": "isnan",
    "logical_not": "lnot",
    "bitwise_invert": "bnot",
}
_NUMPY_UNARY = {
    "asin": np.arcsin,
    "acos": np.arccos,
    "atan": np.arctan,
    "asinh": np.arcsinh,
    "acosh": np.arccosh,
    "atanh": np.arctanh,
    "bitwise_invert": np.invert,
}

_COMMUTATIVE = {
    "plus",
    "times",
    "max",
    "min",
    "land",
    "lor",
    "lxor",
    "band",
    "bor",
    "bxor",
    "eq",
    "ne",
    "any",
    "pair",
}
_REVERSED = {
    "minus": "rminus",
    "truediv": "rtruediv",
    "floordiv": "rfloordiv",
    "pow": "rpow",
    "lt": "gt",
    "gt": "lt",
    "le": "ge",
    "ge": "le",
}

# Array API reduction -> GraphBLAS monoid (and its counterpart for booleans).
_REDUCE = {
    "sum": ("plus", "plus"),
    "prod": ("times", "times"),
    "max": ("max", "lor"),
    "min": ("min", "land"),
    "any": ("lor", "lor"),
    "all": ("land", "land"),
    "bitwise_or": ("bor", "lor"),
}


def _max_value(dtype):
    if dtype.kind == "b":
        return True
    if dtype.kind in "iu":
        return np.iinfo(dtype).max
    return np.inf


def _min_value(dtype):
    if dtype.kind == "b":
        return False
    if dtype.kind in "iu":
        return np.iinfo(dtype).min
    return -np.inf


def _all_ones(dtype):
    return True if dtype.kind == "b" else _cast(-1, dtype)


def _identity(op, dtype):
    """The two-sided identity of a GraphBLAS monoid/operator, or None."""
    return {
        "plus": lambda: 0,
        "times": lambda: 1,
        "max": lambda: _min_value(dtype),
        "min": lambda: _max_value(dtype),
        "lor": lambda: False,
        "land": lambda: True,
        "lxor": lambda: False,
        "bor": lambda: 0,
        "band": lambda: _all_ones(dtype),
        "bxor": lambda: 0,
    }.get(op, lambda: None)()


def _annihilator(op, dtype):
    """The two-sided annihilator of a GraphBLAS operator, or None."""
    value = {
        "times": lambda: 0,
        "land": lambda: False,
        "lor": lambda: True,
        "max": lambda: _max_value(dtype),
        "min": lambda: _min_value(dtype),
        "band": lambda: 0,
        "bor": lambda: _all_ones(dtype),
    }.get(op, lambda: None)()
    if value is None and op == "plus" and dtype.kind in "fc":
        return np.inf  # -inf also annihilates; see _annihilates.
    return value


def _annihilates(op, dtype, fill):
    if op == "plus" and dtype.kind in "fc":
        return bool(np.isinf(fill))
    value = _annihilator(op, dtype)
    return value is not None and bool(fill == _cast(value, dtype))


def _gb_op(namespace, name):
    for part in name.split("."):
        namespace = getattr(namespace, part)
    return namespace


# ---------------------------------------------------------------------- tensor


@dataclass(eq=False)
class Tensor:
    """A GraphBLAS object whose missing entries mean fill_value."""

    data: gb.Vector | gb.Matrix | gb.Scalar
    fill_value: object = 0

    def __post_init__(self):
        self.fill_value = _cast(self.fill_value, self.dtype)

    @property
    def dtype(self):
        return np.dtype(self.data.dtype.np_type)

    @property
    def shape(self):
        return tuple(self.data.shape)

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return int(np.prod(self.shape, dtype=np.int64))

    @property
    def device(self):
        return "cpu"

    @property
    def T(self):
        return xp.permute_dims(self, tuple(reversed(range(self.ndim))))

    @property
    def mT(self):
        return xp.matrix_transpose(self)

    def item(self):
        if self.ndim != 0:
            raise ValueError("only size-1 tensors can be converted to scalars")
        value = self.data.value if self.data.nvals else self.fill_value
        return _cast(value, self.dtype).item()

    def __array_namespace__(self, *, api_version=None):
        return xp

    def __repr__(self):
        return f"Tensor(fill_value={self.fill_value!r},\n{self.data!r})"

    def __len__(self):
        if self.ndim == 0:
            raise TypeError("len() of a 0-d tensor")
        return self.shape[0]

    def __bool__(self):
        return builtins.bool(self.item())

    def __int__(self):
        return int(self.item())

    def __float__(self):
        return float(self.item())

    def __complex__(self):
        return complex(self.item())

    def __index__(self):
        return operator.index(self.item())

    def __getitem__(self, key):
        return xp._getitem(self, key)

    def __setitem__(self, key, value):
        xp._setitem(self, key, value)

    def __abs__(self):
        return xp.abs(self)

    def __neg__(self):
        return xp.negative(self)

    def __pos__(self):
        return xp.positive(self)

    def __invert__(self):
        return xp.bitwise_invert(self)

    def __matmul__(self, other):
        return xp.matmul(self, other)

    def __rmatmul__(self, other):
        return xp.matmul(other, self)


def _binary_dunder(name, reflected=False):
    if reflected:
        return lambda self, other: getattr(xp, name)(other, self)
    return lambda self, other: getattr(xp, name)(self, other)


for _name, _func in {
    "add": "add",
    "sub": "subtract",
    "mul": "multiply",
    "truediv": "divide",
    "floordiv": "floor_divide",
    "mod": "remainder",
    "pow": "power",
    "and": "bitwise_and",
    "or": "bitwise_or",
    "xor": "bitwise_xor",
    "lshift": "bitwise_left_shift",
    "rshift": "bitwise_right_shift",
}.items():
    setattr(Tensor, f"__{_name}__", _binary_dunder(_func))
    setattr(Tensor, f"__r{_name}__", _binary_dunder(_func, reflected=True))
for _name, _func in {
    "eq": "equal",
    "ne": "not_equal",
    "lt": "less",
    "le": "less_equal",
    "gt": "greater",
    "ge": "greater_equal",
}.items():
    setattr(Tensor, f"__{_name}__", _binary_dunder(_func))


# ------------------------------------------------------------------- framework


def _empty(dtype, shape):
    gb_dtype = _gb_dtype(dtype)
    if len(shape) == 0:
        return gb.Scalar(gb_dtype)
    if len(shape) == 1:
        return gb.Vector(gb_dtype, size=shape[0])
    if len(shape) == 2:
        return gb.Matrix(gb_dtype, nrows=shape[0], ncols=shape[1])
    raise NotImplementedError("GraphBLAS supports at most two dimensions.")


def _scalar_tensor(value, dtype=None):
    dtype = _np_dtype(dtype) or np.asarray(value).dtype
    return Tensor(gb.Scalar.from_value(_cast(value, dtype), dtype=_gb_dtype(dtype)))


def _from_dense(array):
    array = np.asarray(array)
    if array.ndim == 0:
        return _scalar_tensor(array[()], array.dtype)
    if array.ndim == 1:
        return Tensor(gb.Vector.from_dense(array), _cast(0, array.dtype))
    if array.ndim == 2:
        return Tensor(gb.Matrix.from_dense(array), _cast(0, array.dtype))
    raise NotImplementedError("GraphBLAS supports at most two dimensions.")


def _normalize_axes(axis, ndim):
    if axis is None:
        return tuple(range(ndim))
    if isinstance(axis, int):
        axis = (axis,)
    return tuple(sorted(operator.index(a) % ndim for a in axis))


class GraphBLASFramework(Framework):
    # ------------------------------------------------------------ conversion

    def from_binsparse(self, array):
        match array:
            case DVECVector() | DMATRMatrix() | DMATCMatrix():
                return _from_dense(to_numpy(array))
            case CustomTensor(shape=(), transpose=None, level=ElementLevel()):
                return _from_dense(to_numpy(array))
            case CustomTensor(
                shape=shape,
                transpose=None,
                level=DenseLevel(rank=rank, level=ElementLevel()),
            ) if rank == len(shape):
                return _from_dense(to_numpy(array))
        if len(array.shape) == 2:
            try:
                csr = sps.csr_array(to_scipy(array))
            except TypeError:
                pass
            else:
                csr.sum_duplicates()
                return Tensor(
                    gb.Matrix.from_csr(
                        csr.indptr, csr.indices, csr.data, ncols=csr.shape[1]
                    ),
                    _cast(0, csr.dtype),
                )
        coo = to_sparse(array).asformat("coo")
        if coo.ndim == 1:
            data = gb.Vector.from_coo(
                *coo.coords, coo.data, size=coo.shape[0], dtype=coo.dtype
            )
        elif coo.ndim == 2:
            data = gb.Matrix.from_coo(
                *coo.coords,
                coo.data,
                nrows=coo.shape[0],
                ncols=coo.shape[1],
                dtype=coo.dtype,
            )
        else:
            raise NotImplementedError("GraphBLAS supports at most two dimensions.")
        return Tensor(data, coo.fill_value)

    def to_binsparse(self, array):
        if not isinstance(array, Tensor):
            return from_numpy(np.asarray(array))
        if array.ndim == 0:
            return from_numpy(np.asarray(array.item(), dtype=array.dtype))
        full = array.data.nvals == array.size
        if full or array.fill_value != 0:
            return from_numpy(array.data.to_dense(fill_value=array.fill_value))
        if array.ndim == 1:
            indices, values = array.data.to_coo()
            return from_sparse(sparse.COO(indices[None, :], values, shape=array.shape))
        indptr, indices, values = array.data.to_csr()
        return from_scipy(sps.csr_array((values, indices, indptr), shape=array.shape))

    def lazy(self, array):
        return array

    def compute(self, array):
        return array

    def with_fill_value(self, array, value):
        # Reinterpret missing entries, sharing the stored data.
        return Tensor(array.data, value)

    # -------------------------------------------------------------- helpers

    def _tensor(self, x):
        if isinstance(x, Tensor):
            return x
        return self.asarray(x)

    def _full(self, x):
        """Materialize every missing entry of x as its fill value."""
        if x.ndim == 0 or x.data.nvals == x.size:
            return x
        data = x.data.dup()
        data(~data.S) << x.fill_value.item()
        return Tensor(data, x.fill_value)

    def _value(self, x):
        return x.item() if isinstance(x, Tensor) else x

    @staticmethod
    def _typed(namespace, name, dtype):
        return _gb_op(namespace, name)[_gb_dtype(dtype)]

    # ----------------------------------------------------- elementwise binary

    def _binary(self, name, x1, x2):
        a_scalar = _is_scalar(x1) or (isinstance(x1, Tensor) and x1.ndim == 0)
        b_scalar = _is_scalar(x2) or (isinstance(x2, Tensor) and x2.ndim == 0)
        a = x1 if isinstance(x1, Tensor) else x1
        b = x2 if isinstance(x2, Tensor) else x2
        dtype_a = a.dtype if isinstance(a, Tensor) else a
        dtype_b = b.dtype if isinstance(b, Tensor) else b
        # Python scalars are weakly typed, as in NumPy and the Array API.
        in_dtype = np.result_type(dtype_a, dtype_b)
        if name in _LOGICAL:
            in_dtype = np.dtype(bool)
        elif name == "divide" and in_dtype.kind in "biu":
            in_dtype = np.dtype(np.float64)
        np_func = _NUMPY_BINARY.get(name) or getattr(np, name)

        def scalar_op(u, v):
            with np.errstate(all="ignore"):
                return np_func(_cast(u, in_dtype), _cast(v, in_dtype))

        out_dtype = np.asarray(scalar_op(0, 1)).dtype
        if a_scalar and b_scalar:
            return _scalar_tensor(scalar_op(self._value(a), self._value(b)), out_dtype)

        gb_name, bool_name = _BINARY[name]
        if in_dtype.kind == "b":
            if bool_name is None:
                raise TypeError(f"{name} is not defined for boolean tensors")
            gb_name = bool_name
        op = self._typed(binary, gb_name, in_dtype)
        out_gb = _gb_dtype(out_dtype)

        if a_scalar or b_scalar:
            # Bind the scalar operand: GraphBLAS apply.
            if b_scalar:
                value = _cast(self._value(b), in_dtype).item()
                data = a.data.apply(op, right=value).new(dtype=out_gb)
                fill = scalar_op(a.fill_value, value)
            else:
                value = _cast(self._value(a), in_dtype).item()
                data = b.data.apply(op, left=value).new(dtype=out_gb)
                fill = scalar_op(value, b.fill_value)
            return Tensor(data, fill)

        if a.shape != b.shape:
            raise NotImplementedError(
                f"GraphBLAS has no broadcasting ({a.shape} vs {b.shape})."
            )
        fa, fb = _cast(a.fill_value, in_dtype), _cast(b.fill_value, in_dtype)
        fill = scalar_op(fa, fb)
        ann_a = _annihilates(gb_name, in_dtype, fa)
        ann_b = _annihilates(gb_name, in_dtype, fb)
        identity = _identity(gb_name, in_dtype)
        if ann_a and ann_b and fa == fb:
            data = a.data.ewise_mult(b.data, op).new(dtype=out_gb)
        elif (
            identity is not None
            and fa == _cast(identity, in_dtype)
            and fb == _cast(identity, in_dtype)
        ):
            data = a.data.ewise_add(b.data, op).new(dtype=out_gb)
        else:
            union = a.data.ewise_union(
                b.data, op, left_default=fa.item(), right_default=fb.item()
            )
            if ann_a != ann_b:
                # One annihilating operand bounds the result's structure.
                data = _empty(out_dtype, a.shape)
                data((a if ann_a else b).data.S) << union
            else:
                data = union.new(dtype=out_gb)
        return Tensor(data, fill)

    # ------------------------------------------------------ elementwise unary

    def _unary(self, name, x):
        x = self._tensor(x)
        np_func = _NUMPY_UNARY.get(name) or getattr(np, name)

        def scalar_op(u):
            with np.errstate(all="ignore"):
                return np_func(_cast(u, x.dtype))

        out_dtype = np.asarray(scalar_op(0)).dtype
        if name == "logical_not":
            in_dtype = np.dtype(bool)
            out_dtype = in_dtype
        elif name in {"isfinite", "isinf", "isnan"} and x.dtype.kind in "biu":
            in_dtype = np.dtype(np.float64)
        elif out_dtype.kind in "fc" and x.dtype.kind in "biu":
            in_dtype = out_dtype
        else:
            in_dtype = x.dtype
        gb_name = _UNARY[name]
        if name == "bitwise_invert" and in_dtype.kind == "b":
            gb_name = "lnot"
        if name == "abs" and in_dtype.kind == "b":
            gb_name = "identity"
        fill = scalar_op(x.fill_value)
        if x.ndim == 0:
            return _scalar_tensor(scalar_op(x.item()), out_dtype)
        op = self._typed(unary, gb_name, in_dtype)
        return Tensor(x.data.apply(op).new(dtype=_gb_dtype(out_dtype)), fill)

    # -------------------------------------------------------------- reduction

    def _reduce(self, name, x, axis=None, keepdims=False, dtype=None):
        x = self._tensor(x)
        axes = _normalize_axes(axis, x.ndim)
        if name in {"sum", "prod"}:
            out_dtype = _np_dtype(dtype) or _sum_dtype(x.dtype)
        elif name in {"any", "all"}:
            out_dtype = np.dtype(bool)
        else:
            out_dtype = x.dtype
        gb_name, bool_name = _REDUCE[name]
        if out_dtype.kind == "b":
            gb_name = bool_name
        if x.ndim == 0 or not axes:
            result = self.astype(x, out_dtype)
            if name in {"any", "all"}:
                result = self.astype(x, bool)
            return result
        mon = self._typed(monoid, gb_name, out_dtype)
        identity = _cast(_identity(gb_name, out_dtype), out_dtype)
        if _cast(x.fill_value, out_dtype) != identity:
            x = self._full(x)
        if len(axes) == x.ndim:
            if x.ndim == 1:
                scalar = x.data.reduce(mon).new()
            else:
                scalar = x.data.reduce_scalar(mon).new()
            value = scalar.value if scalar.nvals else identity
            result = _scalar_tensor(value, out_dtype)
            if keepdims:
                result = self.reshape(result, (1,) * x.ndim)
            return result
        if axes == (1,):
            data = x.data.reduce_rowwise(mon).new()
        else:
            data = x.data.reduce_columnwise(mon).new()
        result = Tensor(data, identity)
        if keepdims:
            result = self.expand_dims(result, axis=axes[0])
        return result

    def sum(self, x, /, *, axis=None, dtype=None, keepdims=False):
        return self._reduce("sum", x, axis, keepdims, dtype)

    def prod(self, x, /, *, axis=None, dtype=None, keepdims=False):
        return self._reduce("prod", x, axis, keepdims, dtype)

    def max(self, x, /, *, axis=None, keepdims=False):
        return self._reduce("max", x, axis, keepdims)

    def min(self, x, /, *, axis=None, keepdims=False):
        return self._reduce("min", x, axis, keepdims)

    def any(self, x, /, *, axis=None, keepdims=False):
        return self._reduce("any", x, axis, keepdims)

    def all(self, x, /, *, axis=None, keepdims=False):
        return self._reduce("all", x, axis, keepdims)

    def _arg_reduce(self, aggregator, identity, x, axis, keepdims):
        x = self._tensor(x)
        if x.fill_value != identity:
            x = self._full(x)
        if axis is None or x.ndim == 1:
            if x.ndim == 2:
                x = self.reshape(x, (-1,))
            result = _scalar_tensor(x.data.reduce(aggregator).new().value, np.int64)
        else:
            axis = operator.index(axis) % 2
            reduce = x.data.reduce_rowwise if axis == 1 else x.data.reduce_columnwise
            result = Tensor(reduce(aggregator).new(dtype=gb.dtypes.INT64), 0)
        if keepdims:
            result = (
                self.reshape(result, (1,) * x.ndim)
                if axis is None
                else (self.expand_dims(result, axis=axis))
            )
        return result

    def argmin(self, x, /, *, axis=None, keepdims=False):
        x = self._tensor(x)
        return self._arg_reduce(agg.ss.argmin, _max_value(x.dtype), x, axis, keepdims)

    def argmax(self, x, /, *, axis=None, keepdims=False):
        x = self._tensor(x)
        return self._arg_reduce(agg.ss.argmax, _min_value(x.dtype), x, axis, keepdims)

    # ------------------------------------------------------------ contraction

    def _semiring(self, monoid_name, op_name, in_dtype, out_dtype):
        typed = getattr(semiring, f"{monoid_name}_{op_name}", None)
        if typed is not None:
            try:
                return typed[_gb_dtype(in_dtype)]
            except KeyError:
                pass
        anonymous = semiring.register_anonymous(
            _gb_op(monoid, monoid_name), _gb_op(binary, op_name)
        )
        return anonymous[_gb_dtype(in_dtype)]

    def _contract(self, a, a_idxs, b, b_idxs, k, name, reduction, out_idxs):
        """Contract a[a_idxs] (name) b[b_idxs] over index k with a semiring."""
        in_dtype = np.result_type(a.dtype, b.dtype)
        if name in _LOGICAL:
            in_dtype = np.dtype(bool)
        gb_op, bool_op = _BINARY[name]
        if in_dtype.kind == "b":
            gb_op = bool_op
        if reduction in {"sum", "prod"}:
            out_dtype = _sum_dtype(
                np.asarray(
                    getattr(np, name)(_cast(0, in_dtype), _cast(1, in_dtype))
                ).dtype
            )
        elif reduction in {"any", "all"}:
            out_dtype = np.dtype(bool)
        else:
            out_dtype = np.asarray(
                getattr(np, name)(_cast(0, in_dtype), _cast(1, in_dtype))
            ).dtype
        mon_name, mon_bool = _REDUCE[reduction]
        if out_dtype.kind == "b":
            mon_name = mon_bool
        identity = _cast(_identity(mon_name, out_dtype), out_dtype)
        # Each operand's fill must annihilate the multiply into the identity.
        operands = []
        for t in (a, b):
            fill = _cast(t.fill_value, in_dtype)
            product = getattr(np, name)(fill, _cast(1, in_dtype))
            if not (
                _annihilates(gb_op, in_dtype, fill)
                and _cast(product, out_dtype) == identity
            ):
                t = self._full(t)
            operands.append(t)
        a, b = operands
        if in_dtype.kind == "b" and out_dtype.kind != "b":
            # Counting booleans: multiply as integers so sums don't saturate.
            gb_op = _BINARY[name][0]
            in_dtype = out_dtype
        sr = self._semiring(mon_name, gb_op, in_dtype, out_dtype)

        def oriented(t, idxs, k_first):
            # A matrix operand oriented with the contracted index as rows/cols.
            if (idxs[0] == k) == k_first:
                return t.data
            return t.data.T

        if len(a_idxs) == 1 and len(b_idxs) == 1:
            scalar = a.data.inner(b.data, sr).new()
            value = scalar.value if scalar.nvals else identity
            return _scalar_tensor(value, out_dtype), []
        if len(a_idxs) == 1:
            data = a.data.vxm(oriented(b, b_idxs, True), sr).new()
            return Tensor(data, identity), [i for i in b_idxs if i != k]
        if len(b_idxs) == 1:
            data = oriented(a, a_idxs, False).mxv(b.data, sr).new()
            return Tensor(data, identity), [i for i in a_idxs if i != k]
        left = oriented(a, a_idxs, False)
        right = oriented(b, b_idxs, True)
        i = next(idx for idx in a_idxs if idx != k)
        j = next(idx for idx in b_idxs if idx != k)
        if out_idxs == [j, i]:
            # (AB)^T = B^T A^T only for commutative multiplies; transpose after.
            data = left.mxm(right, sr).new().T.new()
            return Tensor(data, identity), [j, i]
        return Tensor(left.mxm(right, sr).new(), identity), [i, j]

    def matmul(self, x1, x2, /):
        a, b = self._tensor(x1), self._tensor(x2)
        if a.ndim == 0 or b.ndim == 0:
            raise ValueError("matmul does not accept 0-d operands")
        a_idxs = ["i", "k"] if a.ndim == 2 else ["k"]
        b_idxs = ["k", "j"] if b.ndim == 2 else ["k"]
        out = [i for i in a_idxs + b_idxs if i != "k"]
        reduction = "any" if np.result_type(a.dtype, b.dtype).kind == "b" else "sum"
        name = "logical_and" if reduction == "any" else "multiply"
        result, _ = self._contract(a, a_idxs, b, b_idxs, "k", name, reduction, out)
        return result

    def vecdot(self, x1, x2, /, *, axis=-1):
        a, b = self._tensor(x1), self._tensor(x2)
        if a.ndim == 1 and b.ndim == 1:
            return self.matmul(a, b)
        return self.sum(self.multiply(a, b), axis=axis)

    def tensordot(self, x1, x2, /, *, axes=2):
        raise NotImplementedError("GraphBLAS has no tensordot.")

    def einsum(self, prgm, **kwargs):
        ein = parse_einsum(prgm)
        reduction = reduction_ops[ein.op] if ein.op is not None else None
        if ein.op in {"|", "bitwise_or"}:
            reduction = "bitwise_or"
        expr = ein.arg
        if reduction in _REDUCE and isinstance(expr, Call) and len(expr.args) == 2:
            name = nary_ops[expr.func]
            left, right = expr.args
            if name in _BINARY and name not in _COMPARISONS:
                a, a_idxs = self._einsum_eval(left, kwargs)
                b, b_idxs = self._einsum_eval(right, kwargs)
                shared = set(a_idxs) & set(b_idxs)
                contracted = shared - set(ein.idxs)
                if (
                    a.ndim > 0
                    and b.ndim > 0
                    and len(contracted) == 1
                    and shared == contracted
                ):
                    (k,) = contracted
                    result, idxs = self._contract(
                        a, a_idxs, b, b_idxs, k, name, reduction, ein.idxs
                    )
                    return self._einsum_finish(result, idxs, ein.idxs, reduction)
                value, idxs = self._einsum_binary(name, a, a_idxs, b, b_idxs)
                return self._einsum_finish(value, idxs, ein.idxs, reduction)
        value, idxs = self._einsum_eval(expr, kwargs)
        return self._einsum_finish(value, idxs, ein.idxs, reduction)

    def _einsum_finish(self, value, idxs, out_idxs, reduction):
        if not set(out_idxs) <= set(idxs):
            raise NotImplementedError("GraphBLAS einsum cannot broadcast outputs.")
        extra = tuple(n for n, idx in enumerate(idxs) if idx not in out_idxs)
        if extra:
            if reduction is None:
                raise ValueError("Assignments must not drop indices.")
            value = self._reduce(reduction, value, axis=extra)
            idxs = [idx for idx in idxs if idx in out_idxs]
        if idxs != list(out_idxs):
            value = self.permute_dims(value, [idxs.index(idx) for idx in out_idxs])
        return value

    def _einsum_eval(self, expr, kwargs):
        match expr:
            case Access(tns=tns, idxs=idxs):
                if len(set(idxs)) != len(idxs):
                    raise NotImplementedError("GraphBLAS has no diagonal access.")
                return self._tensor(kwargs[tns]), list(idxs)
            case Literal(value=value):
                return self.asarray(value), []
            case Call(func=func, args=[arg]):
                value, idxs = self._einsum_eval(arg, kwargs)
                return getattr(self, unary_ops[func])(value), idxs
            case Call(func=func, args=[left, right]):
                a, a_idxs = self._einsum_eval(left, kwargs)
                b, b_idxs = self._einsum_eval(right, kwargs)
                return self._einsum_binary(nary_ops[func], a, a_idxs, b, b_idxs)
            case Call(func=func, args=[first, *rest]):
                value = first
                for arg in rest:
                    value = Call(func, [value, arg])
                return self._einsum_eval(value, kwargs)
        raise NotImplementedError(f"Unsupported einsum expression {expr}")

    def _einsum_binary(self, name, a, a_idxs, b, b_idxs):
        func = getattr(self, name)
        if not a_idxs or not b_idxs or set(a_idxs) == set(b_idxs):
            if a_idxs and b_idxs and a_idxs != b_idxs:
                b = self.permute_dims(b, [b_idxs.index(idx) for idx in a_idxs])
            return func(a, b), a_idxs or b_idxs
        if set(a_idxs).isdisjoint(b_idxs) and len(a_idxs) == len(b_idxs) == 1:
            return self._outer(name, a, b), a_idxs + b_idxs
        if set(b_idxs) < set(a_idxs) and len(b_idxs) == 1:
            return self._scale(name, a, a_idxs, b, b_idxs[0], vector_left=False)
        if set(a_idxs) < set(b_idxs) and len(a_idxs) == 1:
            return self._scale(name, b, b_idxs, a, a_idxs[0], vector_left=True)
        raise NotImplementedError(
            f"GraphBLAS cannot combine indices {a_idxs} and {b_idxs}."
        )

    def _intersection_operands(self, name, a, b, in_dtype, gb_op):
        """Materialize operands whose fills don't annihilate the operator."""
        result = []
        for t in (a, b):
            if not _annihilates(gb_op, in_dtype, _cast(t.fill_value, in_dtype)):
                t = self._full(t)
            result.append(t)
        return result

    def _binary_op(self, name, a, b):
        in_dtype = np.result_type(a.dtype, b.dtype)
        if name in _LOGICAL:
            in_dtype = np.dtype(bool)
        gb_name, bool_name = _BINARY[name]
        if in_dtype.kind == "b":
            gb_name = bool_name
        out_dtype = np.asarray(
            getattr(np, name)(_cast(0, in_dtype), _cast(1, in_dtype))
        ).dtype
        return in_dtype, out_dtype, gb_name

    def _outer(self, name, a, b):
        """c[i, j] = a[i] (op) b[j]: GraphBLAS outer product."""
        in_dtype, out_dtype, gb_name = self._binary_op(name, a, b)
        a, b = self._intersection_operands(name, a, b, in_dtype, gb_name)
        op = self._typed(binary, gb_name, in_dtype)
        fill = getattr(np, name)(
            _cast(a.fill_value, in_dtype), _cast(b.fill_value, in_dtype)
        )
        data = a.data.outer(b.data, op).new(dtype=_gb_dtype(out_dtype))
        return Tensor(data, fill)

    def _scale(self, name, m, m_idxs, v, v_idx, *, vector_left):
        """m[i, j] (op) v[i] or v[j]: GraphBLAS scaling by a diagonal matrix."""
        in_dtype, out_dtype, gb_name = self._binary_op(name, m, v)
        m, v = self._intersection_operands(name, m, v, in_dtype, gb_name)
        diag = v.data.diag()
        on_rows = m_idxs[0] == v_idx
        # diag(v) @ m computes op(v[i], m[i, j]); m @ diag(v) computes
        # op(m[i, j], v[j]). The other two orders need the reversed operator.
        if on_rows != vector_left:
            gb_name = gb_name if gb_name in _COMMUTATIVE else _REVERSED.get(gb_name)
            if gb_name is None:
                raise NotImplementedError(f"GraphBLAS has no reversed {name}.")
        sr = self._semiring("any", gb_name, in_dtype, out_dtype)
        data = diag.mxm(m.data, sr) if on_rows else m.data.mxm(diag, sr)
        fill = getattr(np, name)(
            *(
                (_cast(v.fill_value, in_dtype), _cast(m.fill_value, in_dtype))
                if vector_left
                else (_cast(m.fill_value, in_dtype), _cast(v.fill_value, in_dtype))
            )
        )
        return Tensor(data.new(dtype=_gb_dtype(out_dtype)), fill), list(m_idxs)

    # ---------------------------------------------------------------- creation

    def asarray(self, obj, /, *, dtype=None, device=None, copy=None):
        if isinstance(obj, Tensor):
            if dtype is not None and _np_dtype(dtype) != obj.dtype:
                return self.astype(obj, dtype)
            return Tensor(obj.data.dup(), obj.fill_value) if copy else obj
        array = np.asarray(obj, dtype=_np_dtype(dtype))
        return _from_dense(array)

    def _creation_dtype(self, dtype, fill):
        if dtype is not None:
            return _np_dtype(dtype)
        if isinstance(fill, builtins.bool):
            return np.dtype(bool)
        if isinstance(fill, int):
            return np.dtype(np.int64)
        if isinstance(fill, float):
            return np.dtype(np.float64)
        return np.asarray(fill).dtype

    def full(self, shape, fill_value, *, dtype=None, device=None):
        shape = (shape,) if isinstance(shape, int) else tuple(shape)
        fill_value = self._value(fill_value)
        dtype = self._creation_dtype(dtype, fill_value)
        if not shape:
            return _scalar_tensor(fill_value, dtype)
        return Tensor(_empty(dtype, shape), fill_value)

    def zeros(self, shape, *, dtype=None, device=None):
        return self.full(shape, 0, dtype=dtype or np.float64)

    def ones(self, shape, *, dtype=None, device=None):
        return self.full(shape, 1, dtype=dtype or np.float64)

    def empty(self, shape, *, dtype=None, device=None):
        return self.zeros(shape, dtype=dtype)

    def full_like(self, x, /, fill_value, *, dtype=None, device=None):
        return self.full(x.shape, fill_value, dtype=dtype or x.dtype)

    def zeros_like(self, x, /, *, dtype=None, device=None):
        return self.full(x.shape, 0, dtype=dtype or x.dtype)

    def ones_like(self, x, /, *, dtype=None, device=None):
        return self.full(x.shape, 1, dtype=dtype or x.dtype)

    def empty_like(self, x, /, *, dtype=None, device=None):
        return self.zeros_like(x, dtype=dtype)

    def eye(self, n_rows, n_cols=None, /, *, k=0, dtype=None, device=None):
        n_cols = n_rows if n_cols is None else n_cols
        dtype = _np_dtype(dtype) or np.dtype(np.float64)
        rows = np.arange(max(0, -k), min(n_rows, n_cols - k))
        data = gb.Matrix.from_coo(
            rows,
            rows + k,
            _cast(1, dtype).item(),
            nrows=n_rows,
            ncols=n_cols,
            dtype=_gb_dtype(dtype),
        )
        return Tensor(data, 0)

    def arange(self, start, /, stop=None, step=1, *, dtype=None, device=None):
        return _from_dense(np.arange(start, stop, step, dtype=_np_dtype(dtype)))

    def linspace(self, start, stop, /, num, *, dtype=None, device=None, endpoint=True):
        return _from_dense(
            np.linspace(start, stop, num, dtype=_np_dtype(dtype), endpoint=endpoint)
        )

    def astype(self, x, dtype, /, *, copy=True, device=None):
        x = self._tensor(x)
        dtype = _np_dtype(dtype)
        if x.ndim == 0:
            return _scalar_tensor(x.item(), dtype)
        return Tensor(x.data.dup(dtype=_gb_dtype(dtype)), _cast(x.fill_value, dtype))

    # ------------------------------------------------------------------ shape

    def permute_dims(self, x, /, axes):
        x = self._tensor(x)
        axes = tuple(operator.index(a) % x.ndim for a in axes) if x.ndim else ()
        if axes == tuple(range(x.ndim)):
            return x
        return Tensor(x.data.T.new(), x.fill_value)

    def matrix_transpose(self, x, /):
        return self.permute_dims(x, (1, 0))

    def transpose(self, x, /, axes=None):
        x = self._tensor(x)
        axes = tuple(reversed(range(x.ndim))) if axes is None else axes
        return self.permute_dims(x, axes)

    def reshape(self, x, /, shape, *, copy=None):
        x = self._tensor(x)
        shape = (shape,) if isinstance(shape, int) else tuple(shape)
        if -1 in shape:
            known = int(np.prod([s for s in shape if s != -1], dtype=np.int64))
            shape = tuple(x.size // known if s == -1 else s for s in shape)
        if int(np.prod(shape, dtype=np.int64)) != x.size:
            raise ValueError(f"cannot reshape {x.shape} into {shape}")
        if shape == x.shape:
            return x
        if len(shape) > 2:
            raise NotImplementedError("GraphBLAS supports at most two dimensions.")
        if x.ndim == 0:
            if len(shape) == 1:
                data = gb.Vector.from_coo([0], [x.item()], size=1, dtype=x.data.dtype)
            else:
                data = gb.Matrix.from_coo(
                    [0], [0], [x.item()], nrows=1, ncols=1, dtype=x.data.dtype
                )
            return Tensor(data, x.fill_value)
        if not shape:
            return _scalar_tensor(self._getitem(x, (0,) * x.ndim).item(), x.dtype)
        if len(shape) == 1:
            return Tensor(x.data.ss.flatten(), x.fill_value)
        return Tensor(x.data.ss.reshape(*shape), x.fill_value)

    def expand_dims(self, x, /, axis=0):
        x = self._tensor(x)
        axis = operator.index(axis) % (x.ndim + 1)
        shape = x.shape[:axis] + (1,) + x.shape[axis:]
        return self.reshape(x, shape)

    def squeeze(self, x, /, axis):
        x = self._tensor(x)
        axes = _normalize_axes(axis, x.ndim)
        shape = tuple(s for n, s in enumerate(x.shape) if n not in axes)
        return self.reshape(x, shape)

    def _match_fills(self, arrays):
        arrays = [self._tensor(a) for a in arrays]
        fill = arrays[0].fill_value
        return [a if a.fill_value == fill else self._full(a) for a in arrays], fill

    def concat(self, arrays, /, *, axis=0):
        arrays, fill = self._match_fills(arrays)
        if arrays[0].ndim == 1:
            return Tensor(gb.ss.concat([a.data for a in arrays]), fill)
        axis = operator.index(axis) % 2
        tiles = [[a.data] for a in arrays] if axis == 0 else [[a.data for a in arrays]]
        return Tensor(gb.ss.concat(tiles), fill)

    def stack(self, arrays, /, *, axis=0):
        arrays = [self.expand_dims(self._tensor(a), axis=axis) for a in arrays]
        return self.concat(arrays, axis=axis)

    # --------------------------------------------------------------- indexing

    def _key(self, x, key):
        if not isinstance(key, tuple):
            key = (key,)
        if any(k is Ellipsis for k in key):
            at = key.index(Ellipsis)
            fill = (slice(None),) * (x.ndim - len(key) + 1)
            key = key[:at] + fill + key[at + 1 :]
        key = key + (slice(None),) * (x.ndim - len(key))
        result = []
        for k, size in zip(key, x.shape, strict=True):
            if isinstance(k, Tensor):
                k = k.data.to_dense(fill_value=k.fill_value) if k.ndim else k.item()
            if isinstance(k, int | np.integer):
                k = operator.index(k)
                if not -size <= k < size:
                    raise IndexError(f"index {k} is out of bounds for size {size}")
                k %= size
            elif not isinstance(k, slice):
                k = np.asarray(k)
                if k.dtype.kind == "b":
                    raise NotImplementedError("GraphBLAS has no boolean indexing.")
                k = (k % size).tolist()
            result.append(k)
        return tuple(result)

    def _getitem(self, x, key):
        key = self._key(x, key)
        if x.ndim == 0:
            return x
        extracted = x.data[key if x.ndim > 1 else key[0]]
        if all(isinstance(k, int) for k in key):
            scalar = extracted.new()
            value = scalar.value if scalar.nvals else x.fill_value
            return _scalar_tensor(value, x.dtype)
        return Tensor(extracted.new(), x.fill_value)

    def _setitem(self, x, key, value):
        key = self._key(x, key)
        target = x.data[key if x.ndim > 1 else key[0]]
        if isinstance(value, Tensor) and value.ndim > 0:
            if value.fill_value != x.fill_value:
                value = self._full(value)
            target << value.data
        else:
            target << _cast(self._value(value), x.dtype).item()

    def take(self, x, indices, /, *, axis=None):
        x = self._tensor(x)
        if axis is None:
            if x.ndim != 1:
                x = self.reshape(x, (-1,))
            axis = 0
        key = [slice(None)] * x.ndim
        key[operator.index(axis) % x.ndim] = indices
        return self._getitem(x, tuple(key))

    # ---------------------------------------------------------------- where

    def where(self, condition, x1, x2, /):
        cond = self._tensor(condition)
        if cond.dtype.kind != "b":
            cond = self.astype(cond, bool)
        if cond.ndim == 0:
            return self.asarray(x1 if cond.item() else x2)
        if cond.fill_value:
            # Missing entries select x1: swap so the missing ones select x2.
            return self.where(self.logical_not(cond), x2, x1)
        dtype_1 = x1.dtype if isinstance(x1, Tensor) else x1
        dtype_2 = x2.dtype if isinstance(x2, Tensor) else x2
        dtype = np.result_type(dtype_1, dtype_2)
        mask = cond.data.V
        if isinstance(x2, Tensor) and x2.ndim > 0:
            out = x2.data.dup(dtype=_gb_dtype(dtype))
            fill = x2.fill_value
        else:
            out = _empty(dtype, cond.shape)
            fill = self._value(x2)
        if isinstance(x1, Tensor) and x1.ndim > 0:
            fill_1 = _cast(x1.fill_value, dtype)
            if fill_1 != _cast(fill, dtype):
                out(mask) << fill_1.item()
                selected = _empty(dtype, cond.shape)
                selected(mask) << x1.data
                out(selected.S) << selected
            else:
                out(mask) << x1.data
        else:
            out(mask) << _cast(self._value(x1), dtype).item()
        return Tensor(out, _cast(fill, dtype))

    def replace(self, arr, old, new):
        arr = self._tensor(arr)
        if arr.ndim == 0:
            value = arr.item()
            match = value != value if old != old else value == old
            return _scalar_tensor(new if match else value, arr.dtype)
        if old != old:
            hits = arr.data.apply(unary.isnan).new()
        else:
            hits = arr.data.apply(binary.eq, right=old).new()
        out = arr.data.dup()
        out(hits.V) << new
        fill = arr.fill_value
        matches = fill != fill if old != old else fill == old
        return Tensor(out, new if matches else fill)

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
        raise NotImplementedError("GraphBLAS has no unfold.")

    # ---------------------------------------------------------------- dtypes

    bool = np.bool_
    int8 = np.int8
    int16 = np.int16
    int32 = np.int32
    int64 = np.int64
    uint8 = np.uint8
    uint16 = np.uint16
    uint32 = np.uint32
    uint64 = np.uint64
    float32 = np.float32
    float64 = np.float64
    inf = np.inf
    nan = np.nan
    e = np.e
    pi = np.pi
    newaxis = None

    def iinfo(self, dtype):
        return np.iinfo(_np_dtype(dtype))

    def finfo(self, dtype):
        return np.finfo(_np_dtype(dtype))

    def isdtype(self, dtype, kind):
        import array_api_compat.numpy as compat_np

        return compat_np.isdtype(_np_dtype(dtype), kind)

    def result_type(self, *arrays_and_dtypes):
        return np.result_type(
            *(a.dtype if isinstance(a, Tensor) else a for a in arrays_and_dtypes)
        )

    def __getattr__(self, name):
        if name in _BINARY:
            return lambda x1, x2, /: self._binary(name, x1, x2)
        if name in _UNARY:
            return lambda x, /: self._unary(name, x)
        raise AttributeError(f"GraphBLAS has no '{name}' operation.")


xp = GraphBLASFramework()
