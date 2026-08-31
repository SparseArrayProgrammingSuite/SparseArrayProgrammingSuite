"""Small SAPS integration helpers for the reference ``binsparse`` package."""

from binsparse import BinsparseTensor
from binsparse.conversions import to_numpy, to_scipy, to_sparse


def tensor_data(tensor: BinsparseTensor):
    for convert in (to_scipy, to_sparse, to_numpy):
        try:
            return convert(tensor)
        except TypeError:
            pass
    raise TypeError(f"Cannot convert {type(tensor).__name__} to a supported array")


def binsparse_equal(left: BinsparseTensor, right: BinsparseTensor) -> bool:
    left_data, right_data = tensor_data(left), tensor_data(right)
    return left_data.shape == right_data.shape and bool(all(left_data == right_data))
