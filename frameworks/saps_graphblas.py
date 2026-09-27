"""Slide-sized GraphBLAS demo: equal-shaped vectors/matrices, finite stored values.

Missing entries mean fill_value; stored entries are actual values, not deltas.
This uses algebraic annihilation (so IEEE NaN/Inf stored values are out of scope).
Install python-graphblas, then run: PYTHONPATH=src python frameworks/saps_graphblas.py
"""

from dataclasses import dataclass
from math import inf

import graphblas as gb
import sparse as sp
from binsparse.conversions import from_numpy, to_numpy, to_sparse

from saps_framework import Framework


@dataclass
class Tensor:
    data: gb.Vector | gb.Matrix
    fill_value: float = 0


    def __array_namespace__(self, *, api_version=None):
        # https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__array_namespace__.html#array_api.array.__array_namespace__
        if api_version is None:
            api_version = "2024.12"

        if api_version not in {"2024.12"}:
            raise ValueError(f'"{api_version}" Array API version not supported.')

        return sys.modules["finch"]

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
        return self.mod.truediv(self, other)

    def __rtruediv__(self, other):
        return self.mod.truediv(other, self)

    def __floordiv__(self, other):
        return self.mod.floor_divide(self, other)

    def __rfloordiv__(self, other):
        return self.mod.floor_divide(other, self)

    def __mod__(self, other):
        return self.mod.mod(self, other)

    def __rmod__(self, other):
        return self.mod.mod(other, self)

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
        """
        Converts a zero-dimensional array to a Python `complex` object.
        """
        if self.ndim != 0:
            raise ValueError("Cannot convert non-scalar tensor to complex.")
        # dispatch to the scalar value's `__complex__` method
        return complex(self.item())

    def __float__(self):
        """
        Converts a zero-dimensional array to a Python `float` object.
        """
        if self.ndim != 0:
            raise ValueError("Cannot convert non-scalar tensor to float.")
        # dispatch to the scalar value's `__float__` method
        return float(self.item())

    def __int__(self):
        """
        Converts a zero-dimensional array to a Python `int` object.
        """
        if self.ndim != 0:
            raise ValueError("Cannot convert non-scalar tensor to int.")
        # dispatch to the scalar value's `__int__` method
        return int(self.item())

    def __bool__(self):
        """
        Converts a zero-dimensional array to a Python `bool` object.
        """
        if self.ndim != 0:
            raise ValueError("Cannot convert non-scalar tensor to bool.")
        # dispatch to the scalar value's `__bool__` method
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

    def __eq__(self, other):
        return self.mod.equal(self, other)

    def __ne__(self, other):
        return self.mod.not_equal(self, other)


class GraphBLASFramework(Framework):
    def with_fill_value(self, array, value):
        # Reinterpret missing entries, sharing the stored data.
        return Tensor(array.data, value)

    def add(self, a, b):
        fa, fb = a.fill_value, b.fill_value
        if fa == fb and fa in (inf, -inf):  # Infinity annihilates addition.
            result = a.data.ewise_mult(b.data, gb.binary.plus)
        else:
            result = a.data.ewise_union(b.data, gb.binary.plus, fa, fb)
        return Tensor(result.new(), fa + fb)

    def multiply(self, a, b):
        fa, fb = a.fill_value, b.fill_value
        if fa == fb == 0:  # Zero annihilates multiplication.
            result = a.data.ewise_mult(b.data, gb.binary.times)
        else:
            result = a.data.ewise_union(b.data, gb.binary.times, fa, fb)
        return Tensor(result.new(), fa * fb)

    def from_binsparse(self, array):
        try:
            coo = to_sparse(array).asformat("coo")
        except TypeError:  # Binsparse's dense formats use a separate converter.
            coo = sp.COO.from_numpy(to_numpy(array))
        if coo.ndim == 1:
            data = gb.Vector.from_coo(*coo.coords, coo.data, size=coo.shape[0])
        elif coo.ndim == 2:
            data = gb.Matrix.from_coo(
                *coo.coords, coo.data, nrows=coo.shape[0], ncols=coo.shape[1]
            )
        else:
            raise NotImplementedError("This demo supports only vectors and matrices.")
        return Tensor(data, coo.fill_value)

    mul = multiply

    def todense(self, array):
        return array.data.to_dense(fill_value=array.fill_value)


    

    def to_binsparse(self, array):
        # Dense export keeps arbitrary fills simple in this proof of concept.
        return from_numpy(self.todense(array))

    def lazy(self, array):
        return array

    def compute(self, array):
        return array

    def einsum(self, prgm, **kwargs):
        raise NotImplementedError("This demo only implements elementwise add and mul.")

    def __getattr__(self, name):
        raise AttributeError(name)


xp = GraphBLASFramework()


if __name__ == "__main__":
    a = Tensor(gb.Vector.from_coo([0, 1], [2.0, 3.0], size=4))
    b = Tensor(gb.Vector.from_coo([1, 2], [4.0, 5.0], size=4))
    print("add, fill 0:  ", xp.todense(xp.add(a, b)))  # [2, 7, 5, 0]: union
    # Intersection: [0, 12, 0, 0]
    print("mul, fill 0:  ", xp.todense(xp.multiply(a, b)))
    a, b = xp.with_fill_value(a, 1), xp.with_fill_value(b, 1)
    print("mul, fill 1:  ", xp.todense(xp.multiply(a, b)))  # [2, 12, 5, 1]: union
    a, b = xp.with_fill_value(a, inf), xp.with_fill_value(b, inf)
    # Intersection: [inf, 7, inf, inf]
    print("add, fill inf:", xp.todense(xp.add(a, b)))
