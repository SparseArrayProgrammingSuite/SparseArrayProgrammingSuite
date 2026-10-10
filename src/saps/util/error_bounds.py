"""
A function to calculate the floating-point summation error bound for a given input.
Uses eq(2.6) of "The Accuracy of Floating Point Summation," by Nicholas J. Higham,
SIAM Journal of Scientific Computing, Vol. 14, No. 4, pp. 783-799, July 1993.
Here, n is the number of terms in the summation, abs_sum is the sum of the absolute
values of the terms, and dtype is the data type of the floating-point numbers.
"""

def _gamma(xp, n, dtype):
    if n < 0:
        raise ValueError("The number of rounding steps must be nonnegative")
    if xp.isdtype(dtype, ("bool", "integral")):
        return 0.0
    if xp.isdtype(dtype, "complex floating"):
        # Allow for the real operations in a complex product.
        n *= 4
    # Using eps rather than eps / 2 also covers reference and absolute-sum rounding.
    eps = xp.finfo(dtype).eps
    if n * eps >= 1:
        raise ValueError("Too many rounding steps for a finite error bound")
    return n * eps / (1 - n * eps)


def summation_error_bound(xp, n, abs_sum, dtype):
    if n < 0:
        raise ValueError("The number of terms must be nonnegative")
    return _gamma(xp, max(n - 1, 0), dtype) * abs_sum


def operation_error_bound(xp, x, dtype):
    return _gamma(xp, 1, dtype) * xp.abs(xp.asarray(x))
