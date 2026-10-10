"""
A function to calculate the floating-point summation error bound for a given input.  Uses
eq(2.6) of "The Accuracy of Floating Point Summation," by Nicholas J. Higham,
SIAM Journal of Scientific Computing, Vol. 14, No. 4, pp. 783-799, July 1993.
Here, n is the number of terms in the summation, abs_sum is the sum of the absolute values of the terms, and dtype is the data type of the floating-point numbers.
"""
def summation_error_bound(xp, n, abs_sum, dtype):
    eps = xp.finfo(dtype).eps
    return (n - 1) * eps * abs_sum + eps**2

def operation_error_bound(xp, x, dtype):
    eps = xp.finfo(dtype).eps
    return eps * x