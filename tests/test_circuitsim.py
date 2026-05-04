import pytest

import numpy as np
from scipy.integrate import solve_ivp

from sparseappbench.benchmarks.circuitsim import ForwardEuler, RCGenerator, RLCGenerator, LotkaVolterraGenerator


def test_euler_forward_rc():
    """Test Forward Euler with RC circuit."""
    benchmark = ForwardEuler()
    generator = RCGenerator()
    dataset = generator.datasets[0]  # rc_small
    data, meta = generator.generate(dataset)
    
    dydx, span, y0, step = data
    
    time, y_euler = benchmark.benchmark(data, meta)
    y_euler = np.array(y_euler)
    
    # Reference solution
    actual = solve_ivp(dydx, span, y0, t_eval=time)
    
    error = np.max(np.abs(y_euler - actual.y.T))
    assert error < 0.05, f"Exceeds error tolerance: {error}"


def test_euler_forward_rlc():
    """Test Forward Euler with RLC circuit."""
    benchmark = ForwardEuler()
    generator = RLCGenerator()
    dataset = generator.datasets[0]  # rlc_small
    data, meta = generator.generate(dataset)
    
    dydx, span, y0, step = data
    
    time, y_euler = benchmark.benchmark(data, meta)
    y_euler = np.array(y_euler)
    
    # Reference solution
    actual = solve_ivp(dydx, span, y0, t_eval=time)
    
    error = np.max(np.abs(y_euler[:, 0] - actual.y[0].T))
    assert error < 0.05, f"Exceeds error tolerance: {error}"


def test_euler_forward_lotka_volterra():
    """Test Forward Euler with Lotka-Volterra equations."""
    benchmark = ForwardEuler()
    generator = LotkaVolterraGenerator()
    dataset = generator.datasets[0]  # lotka_volterra_small
    data, meta = generator.generate(dataset)
    
    dydx, span, y0, step = data
    
    time, y_euler = benchmark.benchmark(data, meta)
    y_euler = np.array(y_euler)
    
    # Reference solution
    actual = solve_ivp(dydx, span, y0, t_eval=time)
    
    error = np.max(np.abs(y_euler - actual.y.T))
    assert error < 10.0, f"Exceeds error tolerance: {error}"
