"""
Interval arithmetic on float64 NumPy arrays, for bounding reference outputs.

An Interval holds arrays lo <= hi. Each operation rounds NumPy's
round-to-nearest endpoints outward by one ulp, so the result encloses the exact
result for every input in the operands. Rounding is monotone, so this also
encloses every float64 evaluation of the same expression, including ones that
fuse a multiply and an add. Sums are widened to hold in any order. Only the
operations the particle simulation reference uses are defined.
"""

from functools import reduce

import numpy as np

from saps.util.error_bounds import summation_error_bound


def _down(x):
    return np.nextafter(x, -np.inf)


def _up(x):
    return np.nextafter(x, np.inf)


def _interval(x):
    return x if isinstance(x, Interval) else Interval(x)


def _rounded_hull(values):
    return Interval(_down(reduce(np.minimum, values)), _up(reduce(np.maximum, values)))


class Interval:
    # NumPy operands defer to the reflected operators below instead of
    # broadcasting over Intervals as objects.
    __array_ufunc__ = None

    def __init__(self, lo, hi=None):
        self.lo = np.asarray(lo, dtype=np.float64)
        self.hi = self.lo if hi is None else np.asarray(hi, dtype=np.float64)

    def __getitem__(self, index):
        return Interval(self.lo[index], self.hi[index])

    def __add__(self, other):
        other = _interval(other)
        return Interval(_down(self.lo + other.lo), _up(self.hi + other.hi))

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        other = _interval(other)
        return Interval(_down(self.lo - other.hi), _up(self.hi - other.lo))

    def __rsub__(self, other):
        return _interval(other) - self

    def __mul__(self, other):
        other = _interval(other)
        return _rounded_hull(
            [
                self.lo * other.lo,
                self.lo * other.hi,
                self.hi * other.lo,
                self.hi * other.hi,
            ]
        )

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = _interval(other)
        assert np.all((other.lo > 0) | (other.hi < 0)), "Divisor may be zero"
        return _rounded_hull(
            [
                self.lo / other.lo,
                self.lo / other.hi,
                self.hi / other.lo,
                self.hi / other.hi,
            ]
        )

    def __rtruediv__(self, other):
        return _interval(other) / self

    def sqrt(self):
        return Interval(_down(np.sqrt(self.lo)), _up(np.sqrt(self.hi)))

    def round(self):
        return Interval(np.round(self.lo), np.round(self.hi))

    def sum_by(self, index, size, terms):
        """Bound the sum of each index's values in any order of up to `terms`
        terms, since frameworks may reorder sums or add explicit zeros.
        """
        magnitude = np.bincount(index, np.maximum(-self.lo, self.hi), size)
        # One bound covers the framework's rounding and one covers ours.
        slack = _up(2 * summation_error_bound(np, terms, magnitude, np.float64))
        return Interval(
            _down(np.bincount(index, self.lo, size) - slack),
            _up(np.bincount(index, self.hi, size) + slack),
        )


def maximum(a, b):
    a, b = _interval(a), _interval(b)
    return Interval(np.maximum(a.lo, b.lo), np.maximum(a.hi, b.hi))


def minimum(a, b):
    a, b = _interval(a), _interval(b)
    return Interval(np.minimum(a.lo, b.lo), np.minimum(a.hi, b.hi))
