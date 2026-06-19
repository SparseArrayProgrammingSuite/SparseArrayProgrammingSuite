import operator
import os
import sys
from functools import wraps

import numpy as np

repo_root = os.environ.get("SAPS_REPO_ROOT")
if repo_root is not None and repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from frameworks.saps_sparse import PyDataSparseFramework
from saps_framework import Framework


_ELEMENTWISE_OPERATORS = {
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
    "cos",
    "cosh",
    "divide",
    "equal",
    "floor_divide",
    "greater",
    "greater_equal",
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
    "multiply",
    "negative",
    "not_equal",
    "positive",
    "power",
    "remainder",
    "sin",
    "sinh",
    "sqrt",
    "subtract",
    "tan",
    "tanh",
    "where",
}

_REDUCTION_OPERATORS = {
    "all",
    "any",
    "argmax",
    "argmin",
    "count_nonzero",
    "dot",
    "einsum",
    "max",
    "mean",
    "min",
    "prod",
    "std",
    "sum",
    "tensordot",
    "var",
}

_ARRAY_METHOD_FALLBACKS = {
    "astype",
    "flatten",
    "reshape",
    "squeeze",
    "sum",
    "transpose",
}


def _array_method_fallback(name):
    def fallback(array, *args, **kwargs):
        return getattr(array, name)(*args, **kwargs)

    return fallback


def _densify_for_numpy(value):
    if hasattr(value, "todense"):
        return value.todense()
    return value


def _numpy_fallback(name):
    func = getattr(np, name)

    def fallback(*args, **kwargs):
        args = tuple(_densify_for_numpy(arg) for arg in args)
        kwargs = {
            key: _densify_for_numpy(value) for key, value in kwargs.items()
        }
        return func(*args, **kwargs)

    return fallback


_NUMPY_FALLBACKS = {
    "arange",
    "argsort",
    "sort",
    "take",
}


class TaggedArray:
    __array_priority__ = 1000

    def __init__(
        self,
        array,
        framework: "TaggerFramework",
        elementwise_ops_since_reduction: int = 0,
    ):
        self.array = array
        self.framework = framework
        self.mod = framework
        self.elementwise_ops_since_reduction = elementwise_ops_since_reduction

    def __repr__(self):
        return repr(self.array)

    def __len__(self):
        return len(self.array)

    def __iter__(self):
        for item in self.array:
            yield self.framework._wrap(
                item,
                elementwise_ops_since_reduction=(
                    self.elementwise_ops_since_reduction
                ),
            )

    def __getitem__(self, key):
        self.framework._record_operation("array", "getitem", (self, key), {})
        result_lineage = self.framework._result_elementwise_count(
            "array", "getitem", (self, key), {}
        )
        return self.framework._wrap(
            self.array[key], elementwise_ops_since_reduction=result_lineage
        )

    def __setitem__(self, key, value):
        self.framework._record_operation("array", "setitem", (self, key, value), {})
        self.array[key] = self.framework._unwrap(value)

    def __array__(self, dtype=None):
        data = np.asarray(self.array)
        if dtype is not None:
            return data.astype(dtype)
        return data

    @property
    def ndim(self):
        return self.array.ndim

    def __getattr__(self, name):
        attr = getattr(self.array, name)
        if callable(attr):

            @wraps(attr)
            def wrapped(*args, **kwargs):
                self.framework._record_operation(
                    "array", name, (self, *args), kwargs
                )
                result_lineage = self.framework._result_elementwise_count(
                    "array", name, (self, *args), kwargs
                )
                args = self.framework._unwrap(args)
                kwargs = self.framework._unwrap_kwargs(kwargs)
                return self.framework._wrap(
                    attr(*args, **kwargs),
                    elementwise_ops_since_reduction=result_lineage,
                )

            return wrapped
        return self.framework._wrap(
            attr,
            elementwise_ops_since_reduction=(
                self.elementwise_ops_since_reduction
            ),
        )

    def _unary_operator(self, name, op):
        self.framework._record_operation("", name, (self,), {})
        result_lineage = self.framework._result_elementwise_count(
            "", name, (self,), {}
        )
        return self.framework._wrap(
            op(self.array),
            elementwise_ops_since_reduction=result_lineage,
        )

    def _binary_operator(self, name, op, left, right):
        self.framework._record_operation("", name, (left, right), {})
        result_lineage = self.framework._result_elementwise_count(
            "", name, (left, right), {}
        )
        return self.framework._wrap(
            op(self.framework._unwrap(left), self.framework._unwrap(right)),
            elementwise_ops_since_reduction=result_lineage,
        )

    def _unary_function(self, name):
        return self._unary_operator(name, getattr(np, name))

    def _binary_function(self, name, other):
        return self._binary_operator(name, getattr(np, name), self, other)

    def __add__(self, other):
        return self._binary_operator("add", operator.add, self, other)

    def __radd__(self, other):
        return self._binary_operator("add", operator.add, other, self)

    def __sub__(self, other):
        return self._binary_operator("subtract", operator.sub, self, other)

    def __rsub__(self, other):
        return self._binary_operator("subtract", operator.sub, other, self)

    def __mul__(self, other):
        return self._binary_operator("multiply", operator.mul, self, other)

    def __rmul__(self, other):
        return self._binary_operator("multiply", operator.mul, other, self)

    def __abs__(self):
        return self._unary_operator("abs", operator.abs)

    def __pos__(self):
        return self._unary_operator("positive", operator.pos)

    def __neg__(self):
        return self._unary_operator("negative", operator.neg)

    def __invert__(self):
        return self._unary_operator("bitwise_invert", operator.invert)

    def __and__(self, other):
        return self._binary_operator("bitwise_and", operator.and_, self, other)

    def __rand__(self, other):
        return self._binary_operator("bitwise_and", operator.and_, other, self)

    def __lshift__(self, other):
        return self._binary_operator(
            "bitwise_left_shift", operator.lshift, self, other
        )

    def __rlshift__(self, other):
        return self._binary_operator(
            "bitwise_left_shift", operator.lshift, other, self
        )

    def __or__(self, other):
        return self._binary_operator("bitwise_or", operator.or_, self, other)

    def __ror__(self, other):
        return self._binary_operator("bitwise_or", operator.or_, other, self)

    def __rshift__(self, other):
        return self._binary_operator(
            "bitwise_right_shift", operator.rshift, self, other
        )

    def __rrshift__(self, other):
        return self._binary_operator(
            "bitwise_right_shift", operator.rshift, other, self
        )

    def __xor__(self, other):
        return self._binary_operator("bitwise_xor", operator.xor, self, other)

    def __rxor__(self, other):
        return self._binary_operator("bitwise_xor", operator.xor, other, self)

    def __truediv__(self, other):
        return self._binary_operator("divide", operator.truediv, self, other)

    def __rtruediv__(self, other):
        return self._binary_operator("divide", operator.truediv, other, self)

    def __floordiv__(self, other):
        return self._binary_operator("floor_divide", operator.floordiv, self, other)

    def __rfloordiv__(self, other):
        return self._binary_operator("floor_divide", operator.floordiv, other, self)

    def __mod__(self, other):
        return self._binary_operator("remainder", operator.mod, self, other)

    def __rmod__(self, other):
        return self._binary_operator("remainder", operator.mod, other, self)

    def __pow__(self, other):
        return self._binary_operator("power", operator.pow, self, other)

    def __rpow__(self, other):
        return self._binary_operator("power", operator.pow, other, self)

    def __matmul__(self, other):
        return self._binary_operator("matmul", operator.matmul, self, other)

    def __rmatmul__(self, other):
        return self._binary_operator("matmul", operator.matmul, other, self)

    def __sin__(self):
        return self._unary_function("sin")

    def __sinh__(self):
        return self._unary_function("sinh")

    def __cos__(self):
        return self._unary_function("cos")

    def __cosh__(self):
        return self._unary_function("cosh")

    def __tan__(self):
        return self._unary_function("tan")

    def __tanh__(self):
        return self._unary_function("tanh")

    def __asin__(self):
        return self._unary_function("asin")

    def __asinh__(self):
        return self._unary_function("asinh")

    def __acos__(self):
        return self._unary_function("acos")

    def __acosh__(self):
        return self._unary_function("acosh")

    def __atan__(self):
        return self._unary_function("atan")

    def __atanh__(self):
        return self._unary_function("atanh")

    def __atan2__(self, other):
        return self._binary_function("atan2", other)

    def _scalar_value(self):
        value = self.array
        if hasattr(value, "todense"):
            value = value.todense()
        if hasattr(value, "item"):
            return value.item()
        return value

    def __complex__(self):
        if self.ndim != 0:
            raise ValueError("Cannot convert non-scalar tensor to complex.")
        return complex(self._scalar_value())

    def __float__(self):
        if self.ndim != 0:
            raise ValueError("Cannot convert non-scalar tensor to float.")
        return float(self._scalar_value())

    def __int__(self):
        if self.ndim != 0:
            raise ValueError("Cannot convert non-scalar tensor to int.")
        return int(self._scalar_value())

    def __bool__(self):
        if self.ndim != 0:
            raise ValueError("Cannot convert non-scalar tensor to bool.")
        return bool(self._scalar_value())

    def __index__(self) -> int:
        if self.ndim != 0:
            raise ValueError("Cannot convert non-scalar tensor to index.")
        return operator.index(self.__int__())

    def __log__(self):
        return self._unary_function("log")

    def __log1p__(self):
        return self._unary_function("log1p")

    def __log2__(self):
        return self._unary_function("log2")

    def __log10__(self):
        return self._unary_function("log10")

    def __logaddexp__(self, other):
        return self._binary_function("logaddexp", other)

    def __logical_and__(self, other):
        return self._binary_function("logical_and", other)

    def __logical_or__(self, other):
        return self._binary_function("logical_or", other)

    def __logical_xor__(self, other):
        return self._binary_function("logical_xor", other)

    def __logical_not__(self):
        return self._unary_function("logical_not")

    def __lt__(self, other):
        return self._binary_operator("less", operator.lt, self, other)

    def __le__(self, other):
        return self._binary_operator("less_equal", operator.le, self, other)

    def __gt__(self, other):
        return self._binary_operator("greater", operator.gt, self, other)

    def __ge__(self, other):
        return self._binary_operator("greater_equal", operator.ge, self, other)

    def __eq__(self, other):
        return self._binary_operator("equal", operator.eq, self, other)

    def __ne__(self, other):
        return self._binary_operator("not_equal", operator.ne, self, other)


class TaggedLinalg:
    def __init__(self, linalg, framework: "TaggerFramework", namespace: str):
        self.linalg = linalg
        self.framework = framework
        self.namespace = namespace

    def __getattr__(self, name):
        attr = getattr(self.linalg, name)
        if callable(attr):

            @wraps(attr)
            def wrapped(*args, **kwargs):
                self.framework._record_operation(
                    self.namespace, name, args, kwargs
                )
                result_lineage = self.framework._result_elementwise_count(
                    self.namespace, name, args, kwargs
                )
                args = self.framework._unwrap(args)
                kwargs = self.framework._unwrap_kwargs(kwargs)
                return self.framework._wrap(
                    attr(*args, **kwargs),
                    elementwise_ops_since_reduction=result_lineage,
                )

            return wrapped
        return attr


class TaggerFramework(Framework):
    def __init__(self, wrapped: Framework | None = None):
        self.wrapped = wrapped or PyDataSparseFramework()
        self.stats = {
            "operators": {},
            "operator_arg_counts": {},
            "operator_operand_stats": {},
            "tensors": [],
        }

    def reset_stats(self):
        self.stats["operators"].clear()
        self.stats["operator_arg_counts"].clear()
        self.stats["operator_operand_stats"].clear()
        self.stats["tensors"].clear()

    def _record_operation(self, namespace, name, args, kwargs):
        key = f"{namespace}.{name}" if namespace else name
        self.stats["operators"][key] = self.stats["operators"].get(key, 0) + 1
        self.stats["operator_arg_counts"].setdefault(key, []).append(
            len(args) + len(kwargs)
        )
        self.stats["operator_operand_stats"].setdefault(key, []).append(
            self._operand_stats(args, kwargs)
        )

    def _tensor_stats(self, array, elementwise_ops_since_reduction=0):
        shape = getattr(array, "shape", None)
        if shape is not None:
            shape = tuple(int(dim) for dim in shape)

        nnz = getattr(array, "nnz", None)
        if nnz is None and hasattr(array, "data"):
            data = getattr(array, "data")
            if hasattr(data, "size"):
                nnz = int(data.size)
        if nnz is None and hasattr(array, "size"):
            nnz = int(array.size)
        elif nnz is not None:
            nnz = int(nnz)

        size = getattr(array, "size", None)
        if size is None and shape is not None:
            size = int(np.prod(shape, dtype=np.int64))
        elif size is not None:
            size = int(size)

        sparsity = None
        sparsity_factor = None
        if size not in (None, 0) and nnz is not None:
            sparsity = nnz / size
            sparsity_factor = size / nnz if nnz != 0 else None

        fill_value = getattr(array, "fill_value", None)
        if hasattr(fill_value, "item"):
            fill_value = fill_value.item()

        return {
            "ndim": int(getattr(array, "ndim")),
            "shape": shape,
            "size": size,
            "nnz": nnz,
            "sparsity": sparsity,
            "sparsity_factor": sparsity_factor,
            "fill_value": fill_value,
            "elementwise_ops_since_reduction": elementwise_ops_since_reduction,
        }

    def _record_tensor(self, array, elementwise_ops_since_reduction):
        self.stats["tensors"].append(
            self._tensor_stats(array, elementwise_ops_since_reduction)
        )

    def _operand_stats(self, args, kwargs):
        operands = []
        for value in list(args) + list(kwargs.values()):
            operands.extend(self._collect_operand_stats(value))
        return operands

    def _collect_operand_stats(self, value):
        if isinstance(value, TaggedArray):
            return [
                self._tensor_stats(
                    value.array, value.elementwise_ops_since_reduction
                )
            ]
        if isinstance(value, list) or isinstance(value, tuple):
            operands = []
            for item in value:
                operands.extend(self._collect_operand_stats(item))
            return operands
        if hasattr(value, "ndim"):
            return [self._tensor_stats(value)]
        return []

    def _operand_elementwise_counts(self, args, kwargs):
        counts = []
        for value in list(args) + list(kwargs.values()):
            counts.extend(self._collect_elementwise_counts(value))
        return counts

    def _collect_elementwise_counts(self, value):
        if isinstance(value, TaggedArray):
            return [value.elementwise_ops_since_reduction]
        if isinstance(value, list) or isinstance(value, tuple):
            counts = []
            for item in value:
                counts.extend(self._collect_elementwise_counts(item))
            return counts
        return []

    def _operation_kind(self, namespace, name):
        if namespace == "linalg" or name in _REDUCTION_OPERATORS:
            return "reduction"
        if name == "matmul":
            return "reduction"
        if name in _ELEMENTWISE_OPERATORS:
            return "elementwise"
        return "preserve"

    def _result_elementwise_count(self, namespace, name, args, kwargs):
        counts = self._operand_elementwise_counts(args, kwargs)
        operand_count = max(counts, default=0)
        kind = self._operation_kind(namespace, name)
        if kind == "reduction":
            return 0
        if kind == "elementwise":
            return operand_count + 1
        return operand_count

    def _wrap(self, value, elementwise_ops_since_reduction=0):
        if isinstance(value, TaggedArray):
            return value
        if isinstance(value, list):
            return [
                self._wrap(
                    item,
                    elementwise_ops_since_reduction=elementwise_ops_since_reduction,
                )
                for item in value
            ]
        if isinstance(value, tuple):
            return tuple(
                self._wrap(
                    item,
                    elementwise_ops_since_reduction=elementwise_ops_since_reduction,
                )
                for item in value
            )
        if isinstance(value, dict):
            return value
        if hasattr(value, "ndim"):
            self._record_tensor(value, elementwise_ops_since_reduction)
            return TaggedArray(value, self, elementwise_ops_since_reduction)
        return value

    def _unwrap(self, value):
        if isinstance(value, TaggedArray):
            return value.array
        if isinstance(value, list):
            return [self._unwrap(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._unwrap(item) for item in value)
        return value

    def _unwrap_kwargs(self, kwargs):
        return {key: self._unwrap(value) for key, value in kwargs.items()}

    def from_binsparse(self, array):
        self._record_operation("", "from_binsparse", (array,), {})
        return self._wrap(self.wrapped.from_binsparse(array))

    def to_binsparse(self, array):
        self._record_operation("", "to_binsparse", (array,), {})
        return self.wrapped.to_binsparse(self._unwrap(array))

    def lazy(self, array):
        self._record_operation("", "lazy", (array,), {})
        result_lineage = self._result_elementwise_count("", "lazy", (array,), {})
        return self._wrap(
            self.wrapped.lazy(self._unwrap(array)),
            elementwise_ops_since_reduction=result_lineage,
        )

    def compute(self, array):
        self._record_operation("", "compute", (array,), {})
        result_lineage = self._result_elementwise_count("", "compute", (array,), {})
        return self._wrap(
            self.wrapped.compute(self._unwrap(array)),
            elementwise_ops_since_reduction=result_lineage,
        )

    def compile(self, func):
        self._record_operation("", "compile", (func,), {})
        return self.wrapped.compile(func)

    def einsum(self, prgm, **kwargs):
        self._record_operation("", "einsum", (prgm,), kwargs)
        result_lineage = self._result_elementwise_count("", "einsum", (), kwargs)
        kwargs = self._unwrap_kwargs(kwargs)
        asarray = getattr(self.wrapped, "asarray", None)
        if asarray is not None:
            kwargs = {
                key: asarray(value) if hasattr(value, "ndim") else value
                for key, value in kwargs.items()
            }
        return self._wrap(
            self.wrapped.einsum(prgm, **kwargs),
            elementwise_ops_since_reduction=result_lineage,
        )

    def with_fill_value(self, array, value):
        self._record_operation("", "with_fill_value", (array, value), {})
        result_lineage = self._result_elementwise_count(
            "", "with_fill_value", (array, value), {}
        )
        array = self._unwrap(array)
        return self._wrap(
            self.wrapped.with_fill_value(array, value),
            elementwise_ops_since_reduction=result_lineage,
        )

    @property
    def linalg(self):
        return TaggedLinalg(self.wrapped.linalg, self, "linalg")

    def __getattr__(self, name):
        if name in _NUMPY_FALLBACKS:
            attr = _numpy_fallback(name)
        else:
            try:
                attr = getattr(self.wrapped, name)
            except AttributeError:
                if name in _ARRAY_METHOD_FALLBACKS:
                    attr = _array_method_fallback(name)
                else:
                    raise
        if isinstance(attr, type):
            return attr
        if name in {"fft", "linalg"}:
            return TaggedLinalg(attr, self, name)
        if callable(attr):

            @wraps(attr)
            def wrapped(*args, **kwargs):
                self._record_operation("", name, args, kwargs)
                result_lineage = self._result_elementwise_count(
                    "", name, args, kwargs
                )
                args = self._unwrap(args)
                kwargs = self._unwrap_kwargs(kwargs)
                return self._wrap(
                    attr(*args, **kwargs),
                    elementwise_ops_since_reduction=result_lineage,
                )

            return wrapped
        return attr


xp = TaggerFramework()
