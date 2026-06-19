import operator
from functools import wraps

import numpy as np

from frameworks.saps_sparse import PyDataSparseFramework
from saps_framework import Framework


class TaggedArray:
    __array_priority__ = 1000

    def __init__(self, array, framework: "TaggerFramework"):
        self.array = array
        self.framework = framework
        self.mod = framework

    def __repr__(self):
        return repr(self.array)

    def __len__(self):
        return len(self.array)

    def __iter__(self):
        for item in self.array:
            yield self.framework._wrap(item)

    def __getitem__(self, key):
        self.framework._record_operation("array", "getitem", (self, key), {})
        return self.framework._wrap(self.array[key])

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
                self.framework._record_operation("array", name, args, kwargs)
                args = self.framework._unwrap(args)
                kwargs = self.framework._unwrap_kwargs(kwargs)
                return self.framework._wrap(attr(*args, **kwargs))

            return wrapped
        return self.framework._wrap(attr)

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
        return complex(self[()])

    def __float__(self):
        if self.ndim != 0:
            raise ValueError("Cannot convert non-scalar tensor to float.")
        return float(self[()])

    def __int__(self):
        if self.ndim != 0:
            raise ValueError("Cannot convert non-scalar tensor to int.")
        return int(self[()])

    def __bool__(self):
        if self.ndim != 0:
            raise ValueError("Cannot convert non-scalar tensor to bool.")
        return bool(self[()])

    def __index__(self) -> int:
        if self.ndim != 0:
            raise ValueError("Cannot convert non-scalar tensor to index.")
        return operator.index(self.__int__())

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

    def __eq__(self, other):
        return self.mod.equal(self, other)

    def __ne__(self, other):
        return self.mod.not_equal(self, other)


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
                args = self.framework._unwrap(args)
                kwargs = self.framework._unwrap_kwargs(kwargs)
                return self.framework._wrap(attr(*args, **kwargs))

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

    def _tensor_stats(self, array):
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
        }

    def _record_tensor(self, array):
        self.stats["tensors"].append(self._tensor_stats(array))

    def _operand_stats(self, args, kwargs):
        operands = []
        for value in list(args) + list(kwargs.values()):
            operands.extend(self._collect_operand_stats(value))
        return operands

    def _collect_operand_stats(self, value):
        if isinstance(value, TaggedArray):
            return [self._tensor_stats(value.array)]
        if isinstance(value, list) or isinstance(value, tuple):
            operands = []
            for item in value:
                operands.extend(self._collect_operand_stats(item))
            return operands
        if hasattr(value, "ndim"):
            return [self._tensor_stats(value)]
        return []

    def _wrap(self, value):
        if isinstance(value, TaggedArray):
            return value
        if isinstance(value, list):
            return [self._wrap(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._wrap(item) for item in value)
        if isinstance(value, dict):
            return value
        if hasattr(value, "ndim"):
            self._record_tensor(value)
            return TaggedArray(value, self)
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
        return self._wrap(self.wrapped.lazy(self._unwrap(array)))

    def compute(self, array):
        self._record_operation("", "compute", (array,), {})
        return self._wrap(self.wrapped.compute(self._unwrap(array)))

    def compile(self, func):
        self._record_operation("", "compile", (func,), {})
        return self.wrapped.compile(func)

    def einsum(self, prgm, **kwargs):
        self._record_operation("", "einsum", (prgm,), kwargs)
        kwargs = self._unwrap_kwargs(kwargs)
        return self._wrap(self.wrapped.einsum(prgm, **kwargs))

    def with_fill_value(self, array, value):
        self._record_operation("", "with_fill_value", (array, value), {})
        array = self._unwrap(array)
        return self._wrap(self.wrapped.with_fill_value(array, value))

    @property
    def linalg(self):
        return TaggedLinalg(self.wrapped.linalg, self, "linalg")

    def __getattr__(self, name):
        attr = getattr(self.wrapped, name)
        if isinstance(attr, type):
            return attr
        if name in {"fft", "linalg"}:
            return TaggedLinalg(attr, self, name)
        if callable(attr):

            @wraps(attr)
            def wrapped(*args, **kwargs):
                self._record_operation("", name, args, kwargs)
                args = self._unwrap(args)
                kwargs = self._unwrap_kwargs(kwargs)
                return self._wrap(attr(*args, **kwargs))

            return wrapped
        return attr


xp = TaggerFramework()
