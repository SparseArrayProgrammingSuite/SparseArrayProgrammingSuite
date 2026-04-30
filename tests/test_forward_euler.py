import numpy as np
import pytest
from scipy.integrate import solve_ivp
from sparseappbench.benchmarks.forward_euler import forward_euler

def limit(a, N):
    return a % N

# y0 = init_brusselator_2d()
def init_brusselator_2d(n):
    u = [0.0] * (n * n * 2)
    for i in range(n):
        for j in range(n):
            fi = i / (n - 1) if n > 1 else 0.0
            fj = j / (n - 1) if n > 1 else 0.0
            u[(i * n + j) * 2]     = float(np.real(22 * (fj * (1 - fj)) ** 1.5))
            u[(i * n + j) * 2 + 1] = float(np.real(27 * (fi * (1 - fi)) ** 1.5))
    return u

def compute_C(n, alpha, b):
    size = n * n * 2
    C = np.zeros((size, size))

    for i in range(n):
        for j in range(n):
            u_idx = (i * n + j) * 2
            v_idx = u_idx + 1

            ip1, im1, jp1, jm1 = (
                limit(i + 1, n),
                limit(i - 1, n),
                limit(j + 1, n),
                limit(j - 1, n),
            )

            for ni, nj in [(ip1, j), (im1, j), (i, jp1), (i, jm1)]:
                C[u_idx][(ni * n + nj) * 2] += alpha
                C[v_idx][(ni * n + nj) * 2 + 1] += alpha

            C[u_idx][u_idx] -= (4 * alpha + (b + 1))
            C[v_idx][v_idx] -= 4 * alpha
            C[v_idx][u_idx] += b

    return C

def compute_brusselator_cb(n):
    size = n * n * 2
    brusselator_cb = [0.0] * size
    for i in range(n):
        for j in range(n):
            x = i / (n - 1)
            y = j / (n - 1)
            if (x - 0.3) ** 2 + (y - 0.6) ** 2 <= 0.1**2:
                brusselator_cb[(i * n + j) * 2] = 5
    return brusselator_cb


def brusselator_dydx(t, u_vec, C, brusselator_cb, n, a, b, alpha):
    size = n * n * 2
    
    u_arr = np.array(u_vec, dtype=float)
    
    # Linear part
    lin = C @ u_arr
    lin[0::2] += a

    if t >= 1.1:
        lin += brusselator_cb

    # Non-linear part: u^2 * v
    u_vals = u_arr[0::2]
    v_vals = u_arr[1::2]
    uv2 = u_vals**2 * v_vals
    
    non_lin = np.zeros(size, dtype=float)
    non_lin[0::2] = uv2
    non_lin[1::2] = -uv2
    
    return (lin + non_lin).tolist()

def dg_forward_euler_bruss(n, a, b, alpha, t_max, y0, step):
    """Data Generator for Forward Euler with Brusselator."""
    size = n * n * 2
    C = compute_C(n, alpha, b)
    brusselator_cb = compute_brusselator_cb(n)


    def dydx(t, u_vec):
        return brusselator_dydx(t, u_vec, C, brusselator_cb, n, a, b, alpha)


    return (np, dydx, (0, t_max), y0, step)

C = compute_C(4, 0.01, 1.0)
brusselator_cb = compute_brusselator_cb(4)

def dydx_brusselator(t, u_vec):
    return brusselator_dydx(t, u_vec, C, brusselator_cb, 4, 3.4, 1.0, 0.01)

@pytest.mark.parametrize(
    "dydt, t_span, y0, step, tolerance",
    [
        (dydx_brusselator, (0, 1), init_brusselator_2d(4), 0.01, 0.5),
    ],
)
def test_euler_forward(dydt, t_span, y0, step, tolerance):
    """Test function for Forward Euler."""
    (time, y_euler) = forward_euler(np, dydt, t_span, y0, step)
    y_euler = np.array(y_euler).real

    # Internally solve_ivp does not use fixed step sizes, unlike forward_euler
    actual = solve_ivp(dydt, t_span, y0, t_eval=time)
    actual_vals = actual.y.T.real

    error = np.max(np.abs(y_euler - actual_vals))
    assert error < tolerance, f"Exceeds error tolerance: {error}"
