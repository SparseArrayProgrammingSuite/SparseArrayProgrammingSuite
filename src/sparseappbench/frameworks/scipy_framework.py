import scipy as sp

from ..binsparse_format import BinsparseFormat
from .abstract_framework import AbstractFramework


class SciPyFramework(AbstractFramework):
    def __init__(self):
        pass

    def from_benchmark(self, array):
        if array.data["format"] == "dense":
            return array.data["values"].reshape(array.data["shape"])
        if array.data["format"] == "COO":
            indices = []
            idx_dim = 0
            while "indices_" + str(idx_dim) in array.data:
                indices.append(array.data["indices_" + str(idx_dim)])
                idx_dim += 1
            return sp.coo_array(
                (array.data["values"], tuple(indices)), shape=array.data["shape"]
            )
        raise ValueError(f"Unsupported format: {array.data['format']}")

    def to_benchmark(self, array):
        return BinsparseFormat.from_scipy(array)

    def lazy(self, array):
        return array

    def compute(self, array):
        return array

    def einsum(self, prgm, **kwargs):
        pass

    def with_fill_value(self, array, value):
        return array

    def __getattr__(self, name):
        return getattr(sp, name)
