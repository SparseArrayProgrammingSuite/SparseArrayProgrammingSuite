import operator
from fractions import Fraction

import pytest

import numpy as np

from saps.util.interval import Interval, maximum, minimum


def exact(x):
    return Fraction(float(x))


def assert_encloses(interval, values):
    for lo, value, hi in zip(interval.lo, values, interval.hi, strict=True):
        assert exact(lo) <= value <= exact(hi)


rng = np.random.default_rng(0)
a = rng.uniform(-3, 3, 64)
b = rng.uniform(0.5, 3, 64)
c = rng.uniform(-3, 3, 64)


@pytest.mark.parametrize(
    "op", [operator.add, operator.sub, operator.mul, operator.truediv]
)
def test_arithmetic_encloses_exact_results(op):
    expected = [op(exact(x), exact(y)) for x, y in zip(a, b, strict=True)]
    assert_encloses(op(Interval(a), Interval(b)), expected)
    assert_encloses(op(Interval(a), b), expected)
    assert_encloses(op(a, Interval(b)), expected)


def test_wide_operands_enclose_every_corner():
    x = Interval(a, a + 0.5)
    y = Interval(-b, b)
    product = x * y
    quotient = x / Interval(b, b + 1)
    for corner_x in (x.lo, x.hi):
        for corner_y in (y.lo, y.hi):
            assert_encloses(
                product, map(operator.mul, map(exact, corner_x), map(exact, corner_y))
            )
        for corner_y in (b, b + 1):
            assert_encloses(
                quotient,
                map(operator.truediv, map(exact, corner_x), map(exact, corner_y)),
            )


def test_sqrt_encloses_exact_root():
    root = Interval(b).sqrt()
    for lo, value, hi in zip(root.lo, b, root.hi, strict=True):
        assert exact(lo) ** 2 <= exact(value) <= exact(hi) ** 2


def test_bounds_cover_fused_multiply_add():
    fused = [
        float(exact(x) * exact(y) + exact(z)) for x, y, z in zip(a, b, c, strict=True)
    ]
    assert_encloses(Interval(a) * Interval(b) + Interval(c), map(exact, fused))


def test_division_rejects_divisors_that_may_be_zero():
    with pytest.raises(AssertionError):
        Interval(a) / Interval(-b, b)


def test_grouped_sum_covers_every_order():
    terms = rng.uniform(-1, 1, 64) * 10.0 ** rng.integers(-8, 8, 64)
    group = rng.integers(0, 3, 64)
    total = Interval(terms).sum_by(group, 4, len(terms))
    assert total.lo[3] <= 0 <= total.hi[3]
    for g in range(3):
        members = terms[group == g]
        assert exact(total.lo[g]) <= sum(map(exact, members)) <= exact(total.hi[g])
        for _ in range(100):
            running = 0.0
            for term in rng.permutation(members):
                running += term
            assert total.lo[g] <= running <= total.hi[g]


def test_minimum_maximum_round_are_exact():
    x = Interval([-2.0, -1.0, 1.0], [-1.0, 2.0, 3.0])
    clipped = minimum(maximum(x, 0.0), 1.5)
    np.testing.assert_array_equal(clipped.lo, [0.0, 0.0, 1.0])
    np.testing.assert_array_equal(clipped.hi, [0.0, 1.5, 1.5])
    rounded = Interval([-0.7, 0.2, 2.4], [-0.6, 0.6, 2.6]).round()
    np.testing.assert_array_equal(rounded.lo, [-1.0, 0.0, 2.0])
    np.testing.assert_array_equal(rounded.hi, [-1.0, 1.0, 3.0])
