"""
Midpoint-radius bounds on float64 einsum kernels, for computing reference bounds.

A Ball holds float64 values and absolute error bounds such that every float64
evaluation of the same arithmetic on inputs within the input balls lands within
error of value, whatever order each einsum sums its terms in. Run an einsum
kernel with `einsum_bound` as its `einsum` and Balls as its inputs to bound it.
Only the operations einsum kernels such as CCSD use are defined.
"""

import math

import numpy as np

from saps.util.error_bounds import summation_error_bound
from saps_framework.einsum import Access, native_einsum, parse_einsum

_EPS = float(np.finfo(np.float64).eps)


def _up(x):
    return np.nextafter(x, np.inf)


def _ball(x):
    return x if isinstance(x, Ball) else Ball(x)


def _rounded(value, propagated, steps):
    """Add the rounding of one elementwise operation to the propagated error.

    Eps, twice the unit roundoff, covers both the framework's rounding and ours,
    while the factor on `propagated` covers the `steps` roundings computing it.
    """
    error = (1 + steps * _EPS) * propagated + _EPS * (np.abs(value) + propagated)
    return Ball(value, _up(error))


class Ball:
    # NumPy operands defer to the reflected operators below.
    __array_ufunc__ = None

    def __init__(self, value, error=0.0):
        self.value = np.asarray(value, dtype=np.float64)
        self.error = np.asarray(error, dtype=np.float64)

    def __neg__(self):
        return Ball(-self.value, self.error)

    def __add__(self, other):
        other = _ball(other)
        return _rounded(self.value + other.value, self.error + other.error, 1)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + -_ball(other)

    def __rsub__(self, other):
        return _ball(other) + -self

    def __mul__(self, scalar):
        return _rounded(scalar * self.value, abs(scalar) * self.error, 1)

    def __rmul__(self, scalar):
        return self * scalar

    def __truediv__(self, other):
        other = _ball(other)
        margin = np.abs(other.value) - other.error
        assert np.all(margin > 0), "Divisor may be zero"
        value = self.value / other.value
        propagated = (self.error + np.abs(value) * other.error) / margin
        return _rounded(value, propagated, 4)


def einsum_bound(prgm, **operands):
    """Bound an einsum of Balls that permutes one operand or sums products."""
    parsed = parse_einsum(prgm)
    value = native_einsum(np, prgm, **{n: b.value for n, b in operands.items()})
    if parsed.op is None:
        assert isinstance(parsed.arg, Access), "Assignments may only permute"
        errors = {name: ball.error for name, ball in operands.items()}
        return Ball(value, native_einsum(np, prgm, **errors))

    contraction = parsed.native_contraction()
    assert contraction is not NotImplemented, "Einsums may only sum products"
    _, leaves, _ = contraction
    sizes = {
        index: size
        for leaf in leaves
        if isinstance(leaf, Access)
        for index, size in zip(leaf.idxs, operands[leaf.tns].value.shape, strict=True)
    }
    terms = math.prod(size for index, size in sizes.items() if index not in parsed.idxs)
    # Products within a ball change by at most sum(prod(|x| + e) - prod(|x|)).
    widened = {n: np.abs(b.value) + b.error for n, b in operands.items()}
    magnitude = np.abs(native_einsum(np, prgm, **widened))
    exact = {n: np.abs(b.value) for n, b in operands.items()}
    propagated = magnitude - np.abs(native_einsum(np, prgm, **exact))
    # Each term takes len(leaves) - 1 products, then the terms sum in any order.
    # Six such bounds cover the framework's rounding, ours, and this bound's.
    steps = terms + len(leaves) - 1
    rounding = summation_error_bound(np, steps, magnitude, np.float64)
    return Ball(value, _up(propagated + 6 * rounding))
