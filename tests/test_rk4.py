import pytest
import numpy as np
from scipy.integrate import solve_ivp
from sparseappbench.benchmarks.rk4 import rk4
from sparseappbench.benchmarks.circuitsim import rc, rlc, lotka_volterra, step_input

@pytest.mark.parametrize(
    "dydt, t_span, y0, step, tolerance, second_order",
    [
        (dVdt_rc, (0, 5), [0], 0.000001, 0.05, False),
        (dVdt_rlc, (0, 10e-3), [0, 0], 0.0000001, 0.05, True),
        (dydt_lv, (0, 100), [40, 9], 0.00001, 0.5, False),
    ],
)
def test_rk4(dydt, t_span, y0, step, tolerance, second_order):
    """Test function for Runge-Kutta 4th order."""
    (time, y_euler) = rk4(np, dydt, t_span, y0, step)
    y_euler = np.array(y_euler)

    # Internally solve_ivp does not use fixed step sizes, unlike forward_euler
    actual = solve_ivp(dydt, t_span, y0, t_eval=time)
    print(y_euler)
    print(actual.y.T)

    if second_order:
        error = np.max(np.abs(y_euler[:, 0] - actual.y[0].T))
    else:
        error = np.max(np.abs(y_euler - actual.y.T))
    assert error < tolerance, f"Exceeds error tolerance: {error}"
