import pytest

import numpy as np
from scipy.integrate import solve_ivp


from sparseappbench.benchmarks.ode import (
    ForwardEuler,
    BackwardEuler,
    RK4,
    BrusselatorGenerator,
)


def test_forward_euler_brusselator():
    """Test Forward Euler with Brusselator."""
    benchmark = ForwardEuler()
    generator = BrusselatorGenerator()
    dataset = generator.datasets[0]  # brusselator_4
    data, meta = generator.generate(dataset)
    
    dydx, span, y0, step = data
    
    time, y_fe = benchmark.benchmark(data, meta)
    y_fe = np.array(y_fe).real

    # Reference solution
    actual = solve_ivp(dydx, span, y0, t_eval=time)
    
    error = np.max(np.abs(y_fe - actual.y.T))
    assert error < 0.5, f"Exceeds error tolerance: {error}"


def test_backward_euler_brusselator():
    """Test Backward Euler with Brusselator."""
    benchmark = BackwardEuler()
    generator = BrusselatorGenerator()
    dataset = generator.datasets[0]  # brusselator_4
    data, meta = generator.generate(dataset)
    
    dydx, span, y0, step = data
    
    time, y_be = benchmark.benchmark(data, meta)
    y_be = np.array(y_be).real

    # Reference solution
    actual = solve_ivp(dydx, span, y0, t_eval=time)
    
    error = np.max(np.abs(y_be - actual.y.T))
    assert error < 0.5, f"Exceeds error tolerance: {error}"


def test_rk4_brusselator():
    """Test RK4 with Brusselator."""
    benchmark = RK4()
    generator = BrusselatorGenerator()
    dataset = generator.datasets[0]  # brusselator_4
    data, meta = generator.generate(dataset)
    
    dydx, span, y0, step = data
    
    time, y_rk4 = benchmark.benchmark(data, meta)
    y_rk4 = np.array(y_rk4).real

    # Reference solution
    actual = solve_ivp(dydx, span, y0, t_eval=time)
    
    error = np.max(np.abs(y_rk4 - actual.y.T))
    assert error < 0.5, f"Exceeds error tolerance: {error}"
