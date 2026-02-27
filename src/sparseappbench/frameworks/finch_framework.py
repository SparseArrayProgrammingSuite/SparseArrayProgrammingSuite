import finch

from ..binsparse_format import BinsparseFormat
from .abstract_framework import AbstractFramework


class FinchFramework(AbstractFramework):
    def __init__(self):
        pass

    def from_benchmark(self, array):
        pass

    def to_benchmark(self, array):
        return BinsparseFormat.from_finch(array)

    def lazy(self, array):
        return array

    def compute(self, array):
        return array

    def einsum(self, prgm, **kwargs):
        return finch.einsum(prgm, **kwargs)

    def __getattr__(self, name):
        return getattr(finch, name)
