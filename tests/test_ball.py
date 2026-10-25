from fractions import Fraction

import pytest

import numpy as np

from saps.util.ball import Ball, einsum_bound

rng = np.random.default_rng(0)
A = rng.uniform(-1, 1, (3, 50)) * 10.0 ** rng.integers(-6, 6, (3, 50))
B = rng.uniform(-1, 1, 50)


def test_einsum_bound_covers_exact_sums_and_every_order():
    result = einsum_bound("C[i] += 0.5 * A[i,k] * B[k]", A=Ball(A), B=Ball(B))
    for row in range(3):
        terms = [
            Fraction(0.5) * Fraction(a) * Fraction(b)
            for a, b in zip(A[row], B, strict=True)
        ]
        assert abs(sum(terms) - Fraction(result.value[row])) <= result.error[row]
        for _ in range(50):
            running = 0.0
            for k in rng.permutation(50):
                running += 0.5 * A[row, k] * B[k]
            assert abs(running - result.value[row]) <= result.error[row]


def test_einsum_bound_covers_inputs_anywhere_in_their_balls():
    errors = 1e-9 * np.abs(A)
    result = einsum_bound("C[i] += A[i,k] * B[k]", A=Ball(A, errors), B=Ball(B, 1e-12))
    for _ in range(50):
        perturbed = A + rng.uniform(-1, 1, A.shape) * errors
        nudged = B + rng.uniform(-1, 1, B.shape) * 1e-12
        assert np.all(np.abs(perturbed @ nudged - result.value) <= result.error)


def test_permutations_are_exact():
    errors = np.abs(A) * 1e-3
    result = einsum_bound("T[k,i] = A[i,k]", A=Ball(A, errors))
    np.testing.assert_array_equal(result.value, A.T)
    np.testing.assert_array_equal(result.error, errors.T)


def test_elementwise_operations_cover_inputs_anywhere_in_their_balls():
    x = Ball(A[0], np.abs(A[0]) * 1e-10)
    y = Ball(B + 2, 1e-10)
    results = {
        "sum": (x + y, lambda a, b: a + b),
        "difference": (x - y, lambda a, b: a - b),
        "scaled": (2 * x - 3 * y, lambda a, b: 2 * a - 3 * b),
        "negated": (-x, lambda a, b: -a),
        "quotient": (x / y, lambda a, b: a / b),
    }
    for _ in range(50):
        a = x.value + rng.uniform(-1, 1, x.value.shape) * x.error
        b = y.value + rng.uniform(-1, 1, y.value.shape) * y.error
        for name, (bound, evaluate) in results.items():
            assert np.all(np.abs(evaluate(a, b) - bound.value) <= bound.error), name


def test_division_rejects_divisors_that_may_be_zero():
    with pytest.raises(AssertionError):
        Ball(A[0]) / Ball(B, np.abs(B) * 2)
