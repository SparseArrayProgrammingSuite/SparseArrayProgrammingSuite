import pytest

import numpy as np
from scipy.integrate import solve_ivp


from sparseappbench.benchmarks.ode import (
    brusselator_dydx,
    forward_euler,
    init_brusselator_2d,
    
)

def dydx_brusselator(t, u_vec):
    return brusselator_dydx(t, u_vec, 4, 3.4, 1.0, 0.01)

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
