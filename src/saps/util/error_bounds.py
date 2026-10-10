"""
A function to calculate the floating-point error bound for a given input.  Uses
eq(2.6) of "The Accuracy of Floating Point Summation," by Nicholas J. Higham,
SIAM Journal of Scientific Computing, Vol. 14, No. 4, pp. 783-799, July 1993.
Here, n is the number of terms in the summation, x is the maximum absolute value
of the terms, and dtype is the data type of the floating-point numbers.
"""
def floating_point_error_bound(xp, n, x, dtype):
    eps = xp.finfo(dtype).eps
    return (n - 1) * eps * n * x * eps**2