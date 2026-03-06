import scipy as sp

from ..binsparse_format import BinsparseFormat
from .abstract_framework import AbstractFramework


class NumpyFramework(AbstractFramework):
    def __init__(self):
        pass

    def from_benchmark(self, array):
        pass

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
