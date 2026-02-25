from ..binsparse_format import BinsparseFormat
from .abstract_framework import AbstractFramework
from .numpy_framework import NumpyFramework


def unwrap(x):
    if isinstance(x, CheckerTensor):
        return x.array
    return x


class CheckerTensor:
    def __init__(self, xp, array):
        self.xp = xp
        self.array = unwrap(array)

    def __add__(self, other):
        return self.xp.add(self, other)

    def __radd__(self, other):
        return self.xp.add(other, self)

    def __sub__(self, other):
        return self.xp.subtract(self, other)

    def __rsub__(self, other):
        return self.xp.subtract(other, self)

    def __mul__(self, other):
        return self.xp.multiply(self, other)

    def __rmul__(self, other):
        return self.xp.multiply(other, self)

    def __abs__(self):
        return self.xp.abs(self)

    def __pos__(self):
        return self.xp.positive(self)

    def __neg__(self):
        return self.xp.negative(self)

    def __invert__(self):
        return self.xp.bitwise_inverse(self)

    def __and__(self, other):
        return self.xp.bitwise_and(self, other)

    def __rand__(self, other):
        return self.xp.bitwise_and(other, self)

    def __lshift__(self, other):
        return self.xp.bitwise_left_shift(self, other)

    def __rlshift__(self, other):
        return self.xp.bitwise_left_shift(other, self)

    def __or__(self, other):
        return self.xp.bitwise_or(self, other)

    def __ror__(self, other):
        return self.xp.bitwise_or(other, self)

    def __rshift__(self, other):
        return self.xp.bitwise_right_shift(self, other)

    def __rrshift__(self, other):
        return self.xp.bitwise_right_shift(other, self)

    def __xor__(self, other):
        return self.xp.bitwise_xor(self, other)

    def __rxor__(self, other):
        return self.xp.bitwise_xor(other, self)

    def __truediv__(self, other):
        return self.xp.true_divide(self, other)

    def __rtruediv__(self, other):
        return self.xp.true_divide(other, self)

    def __floordiv__(self, other):
        return self.xp.floordiv(self, other)

    def __rfloordiv__(self, other):
        return self.xp.floordiv(other, self)

    def __mod__(self, other):
        return self.xp.mod(self, other)

    def __rmod__(self, other):
        return self.xp.mod(other, self)

    def __pow__(self, other):
        return self.xp.pow(self, other)

    def __rpow__(self, other):
        return self.xp.pow(other, self)

    def __matmul__(self, other):
        return self.xp.matmul(self, other)

    def __rmatmul__(self, other):
        return self.xp.matmul(other, self)

    def __sin__(self):
        return self.xp.sin(self)

    def __sinh__(self):
        return self.xp.sinh(self)

    def __cos__(self):
        return self.xp.cos(self)

    def __cosh__(self):
        return self.xp.cosh(self)

    def __tan__(self):
        return self.xp.tan(self)

    def __tanh__(self):
        return self.xp.tanh(self)

    def __asin__(self):
        return self.xp.asin(self)

    def __asinh__(self):
        return self.xp.asinh(self)

    def __acos__(self):
        return self.xp.acos(self)

    def __acosh__(self):
        return self.xp.acosh(self)

    def __atan__(self):
        return self.xp.atan(self)

    def __atanh__(self):
        return self.xp.atanh(self)

    def __atan2__(self, other):
        return self.xp.atan2(self, other)

    def __complex__(self):
        """
        Converts a zero-dimensional array to a Python `complex` object.
        """
        if self.ndim != 0:
            raise ValueError("Cannot convert non-scalar tensor to complex.")
        # dispatch to the scalar value's `__complex__` method
        return complex(self.xp.compute(self)[()])

    def __float__(self):
        """
        Converts a zero-dimensional array to a Python `float` object.
        """
        if self.ndim != 0:
            raise ValueError("Cannot convert non-scalar tensor to float.")
        # dispatch to the scalar value's `__float__` method
        return float(self.xp.compute(self)[()])

    def __int__(self):
        """
        Converts a zero-dimensional array to a Python `int` object.
        """
        if self.ndim != 0:
            raise ValueError("Cannot convert non-scalar tensor to int.")
        # dispatch to the scalar value's `__int__` method
        return int(self.xp.compute(self)[()])

    def __bool__(self):
        """
        Converts a zero-dimensional array to a Python `bool` object.
        """
        if self.ndim != 0:
            raise ValueError("Cannot convert non-scalar tensor to bool.")
        # dispatch to the scalar value's `__bool__` method
        return bool(self.xp.compute(self)[()])

    def __log__(self):
        return self.xp.log(self)

    def __log1p__(self):
        return self.xp.log1p(self)

    def __log2__(self):
        return self.xp.log2(self)

    def __log10__(self):
        return self.xp.log10(self)

    def __logaddexp__(self, other):
        return self.xp.logaddexp(self, other)

    def __logical_and__(self, other):
        return self.xp.logical_and(self, other)

    def __logical_or__(self, other):
        return self.xp.logical_or(self, other)

    def __logical_xor__(self, other):
        return self.xp.logical_xor(self, other)

    def __logical_not__(self):
        return self.xp.logical_not(self)

    def __lt__(self, other):
        return self.xp.less(self, other)

    def __le__(self, other):
        return self.xp.less_equal(self, other)

    def __gt__(self, other):
        return self.xp.greater(self, other)

    def __ge__(self, other):
        return self.xp.greater_equal(self, other)

    def __eq__(self, other):
        return self.xp.equal(self, other)

    def __ne__(self, other):
        return self.xp.not_equal(self, other)

    def __getattr__(self, name):
        return getattr(self.array, name)


class LazyCheckerTensor(CheckerTensor):
    def __getitem__(self, key):
        raise AssertionError(
            "Lazy Tensors should not be indexed directly; they must be computed first!"
        )

    def __setitem__(self, key, value):
        raise AssertionError(
            "Lazy Tensors should not be modified directly; they must be computed first!"
        )

    def __complex__(self):
        """
        Converts a zero-dimensional array to a Python `complex` object.
        """
        raise ValueError("Cannot convert lazy tensor to complex.")

    def __float__(self):
        """
        Converts a zero-dimensional array to a Python `float` object.
        """
        raise ValueError("Cannot convert lazy tensor to float.")

    def __int__(self):
        """
        Converts a zero-dimensional array to a Python `int` object.
        """
        raise ValueError("Cannot convert lazy tensor to int.")

    def __bool__(self):
        """
        Converts a zero-dimensional array to a Python `bool` object.
        """
        raise ValueError("Cannot convert lazy tensor to bool.")


class EagerCheckerTensor(CheckerTensor):
    def __getitem__(self, key):
        return self.array.__getitem__(key)

    """
    Though we don't want to run computations on eager tensors, we probably want
    to allow in-place updates. This is sticky.
    """

    def __setitem__(self, key, value):
        return self.array.__setitem__(key, value)


class CheckerOperator:
    def __init__(self, xp, operator):
        self.xp = xp
        self.operator = operator

    def __call__(self, *args, **kwargs):
        if any(isinstance(arg, LazyCheckerTensor) for arg in args) or any(
            isinstance(kwarg, LazyCheckerTensor) for kwarg in kwargs.values()
        ):
            args = [unwrap(arg) for arg in args]
            kwargs = {k: unwrap(v) for k, v in kwargs.items()}
            return LazyCheckerTensor(self.xp, self.operator(*args, **kwargs))
        args = [unwrap(arg) for arg in args]
        kwargs = {k: unwrap(v) for k, v in kwargs.items()}
        return EagerCheckerTensor(self.xp, self.operator(*args, **kwargs))


class CheckerLinalg:
    """
    Allows use of linalg operations without associating new tensors
    with the linalg checker framework.
    """

    def __init__(self, framework, linalg):
        self.framework = framework
        self.linalg = linalg

    def __getattr__(self, name):
        return CheckerOperator(self.framework, getattr(self.linalg, name))


class CheckerFramework(AbstractFramework):
    """
    This framework uses numpy to perform operations, but checks that lazy and compute
    are used correctly in a benchmark function.
    """

    def __init__(self, xp=None):
        if xp is None:
            xp = NumpyFramework()
        self.xp = xp

    def from_benchmark(self, array):
        if array.data["format"] == "dense":
            return EagerCheckerTensor(
                self, self.xp.array(array.data["values"]).reshape(array.data["shape"])
            )
        if array.data["format"] == "COO":
            indices = []
            idx_dim = 0
            while "indices_" + str(idx_dim) in array.data:
                indices.append(array.data["indices_" + str(idx_dim)])
                idx_dim += 1
            V = array.data["values"]
            shape = array.data["shape"]
            data = self.xp.zeros(shape, dtype=V.dtype)
            data[tuple(indices)] = V
            return EagerCheckerTensor(self, data)
        raise ValueError("Unsupported format: " + array.data["format"])

    def to_benchmark(self, array: CheckerTensor):
        if isinstance(array, LazyCheckerTensor):
            raise AssertionError(
                "Lazy Tensors should always be computed before being converted to"
                " benchmark format!"
            )
        return BinsparseFormat.from_numpy(unwrap(array))

    def lazy(self, array: CheckerTensor | tuple[CheckerTensor, ...]):
        if isinstance(array, tuple):
            return tuple(LazyCheckerTensor(self, arr) for arr in array)
        return LazyCheckerTensor(self, array)

    def compute(self, array: CheckerTensor | tuple[CheckerTensor, ...]):
        if isinstance(array, tuple):
            return tuple(EagerCheckerTensor(self, arr) for arr in array)
        return EagerCheckerTensor(self, array)

    def einsum(self, prgm, **kwargs):
        return CheckerOperator(self, self.xp.einsum)(prgm, **kwargs)

    def with_fill_value(self, array, value):
        return CheckerOperator(self, self.xp.with_fill_value)(array, value)

    def __getattr__(self, name):
        if name == "linalg":
            return CheckerLinalg(self, self.xp.linalg)
        return CheckerOperator(self, getattr(self.xp, name))
