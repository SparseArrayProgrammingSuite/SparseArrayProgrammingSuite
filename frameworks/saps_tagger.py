from frameworks.saps_sparse import PyDataSparseFramework
from saps_framework import Framework


class TaggerFramework(Framework):
    def __init__(self, wrapped: Framework | None = None):
        self.wrapped = wrapped or PyDataSparseFramework()

    def from_binsparse(self, array):
        return self.wrapped.from_binsparse(array)

    def to_binsparse(self, array):
        return self.wrapped.to_binsparse(array)

    def lazy(self, array):
        return self.wrapped.lazy(array)

    def compute(self, array):
        return self.wrapped.compute(array)

    def compile(self, func):
        return self.wrapped.compile(func)

    def einsum(self, prgm, **kwargs):
        return self.wrapped.einsum(prgm, **kwargs)

    def with_fill_value(self, array, value):
        return self.wrapped.with_fill_value(array, value)

    def __getattr__(self, name):
        return getattr(self.wrapped, name)


xp = TaggerFramework()
