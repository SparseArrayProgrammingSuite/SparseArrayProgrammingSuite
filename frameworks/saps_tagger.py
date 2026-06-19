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
        return self.framework._wrap(self.array[key])

    def __setitem__(self, key, value):
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
    def __init__(self, linalg, framework: "TaggerFramework"):
        self.linalg = linalg
        self.framework = framework

    def __getattr__(self, name):
        attr = getattr(self.linalg, name)
        if callable(attr):

            @wraps(attr)
            def wrapped(*args, **kwargs):
                args = self.framework._unwrap(args)
                kwargs = self.framework._unwrap_kwargs(kwargs)
                return self.framework._wrap(attr(*args, **kwargs))

            return wrapped
        return attr


class TaggerFramework(Framework):
    def __init__(self, wrapped: Framework | None = None):
        self.wrapped = wrapped or PyDataSparseFramework()

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
        return self._wrap(self.wrapped.from_binsparse(array))

    def to_binsparse(self, array):
        return self.wrapped.to_binsparse(self._unwrap(array))

    def lazy(self, array):
        return self._wrap(self.wrapped.lazy(self._unwrap(array)))

    def compute(self, array):
        return self._wrap(self.wrapped.compute(self._unwrap(array)))

    def compile(self, func):
        return self.wrapped.compile(func)

    def einsum(self, prgm, **kwargs):
        kwargs = self._unwrap_kwargs(kwargs)
        return self._wrap(self.wrapped.einsum(prgm, **kwargs))

    def with_fill_value(self, array, value):
        array = self._unwrap(array)
        return self._wrap(self.wrapped.with_fill_value(array, value))

    @property
    def linalg(self):
        return TaggedLinalg(self.wrapped.linalg, self)

    def __getattr__(self, name):
        attr = getattr(self.wrapped, name)
        if isinstance(attr, type):
            return attr
        if callable(attr):

            @wraps(attr)
            def wrapped(*args, **kwargs):
                args = self._unwrap(args)
                kwargs = self._unwrap_kwargs(kwargs)
                return self._wrap(attr(*args, **kwargs))

            return wrapped
        return attr


xp = TaggerFramework()
