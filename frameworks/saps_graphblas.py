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
