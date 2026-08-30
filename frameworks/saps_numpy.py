import numpy as np

from binsparse.conversions import from_numpy, to_numpy, to_sparse

from saps_framework import Framework, einsum


class NumpyFramework(Framework):
    def __init__(self):
        pass

    def from_binsparse(self, array):
        try:
            return to_numpy(array)
        except TypeError:
            return np.asarray(to_sparse(array).todense())

    def to_binsparse(self, array):
        return from_numpy(np.asarray(array))

    def lazy(self, array):
        return array

    def compute(self, array):
        return array

    def einsum(self, prgm, **kwargs):
        return einsum(np, prgm, **kwargs)

    def with_fill_value(self, array, value):
        return array

    def __getattr__(self, name):
        return getattr(np, name)


xp = NumpyFramework()
