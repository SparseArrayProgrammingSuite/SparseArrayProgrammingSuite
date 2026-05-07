import pytest

import numpy as np
from scipy.integrate import solve_ivp

from saps.benchmarks.ode import ForwardEuler, BackwardEuler, RK4, RCGenerator, RLCGenerator, LotkaVolterraGenerator, BrusselatorGenerator


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


def test_backward_euler_rc():
    """Test Backward Euler with RC circuit."""
    benchmark = BackwardEuler()
    generator = RCGenerator()
    dataset = generator.datasets[0]  # rc_small
    data, meta = generator.generate(dataset)
    
    dydx, span, y0, step = data
    
    time, y_be = benchmark.benchmark(data, meta)
    y_be = np.array(y_be)
    
    # Reference solution
    actual = solve_ivp(dydx, span, y0, t_eval=time)
    
    error = np.max(np.abs(y_be - actual.y.T))
    assert error < 0.05, f"Exceeds error tolerance: {error}"


def test_backward_euler_rlc():
    """Test Backward Euler with RLC circuit."""
    benchmark = BackwardEuler()
    generator = RLCGenerator()
    dataset = generator.datasets[0]  # rlc_small
    data, meta = generator.generate(dataset)
    
    dydx, span, y0, step = data
    
    time, y_be = benchmark.benchmark(data, meta)
    y_be = np.array(y_be)
    
    # Reference solution
    actual = solve_ivp(dydx, span, y0, t_eval=time)
    
    error = np.max(np.abs(y_be[:, 0] - actual.y[0].T))
    assert error < 0.05, f"Exceeds error tolerance: {error}"


def test_backward_euler_lotka_volterra():
    """Test Backward Euler with Lotka-Volterra equations."""
    benchmark = BackwardEuler()
    generator = LotkaVolterraGenerator()
    dataset = generator.datasets[0]  # lotka_volterra_small
    data, meta = generator.generate(dataset)
    
    dydx, span, y0, step = data
    
    time, y_be = benchmark.benchmark(data, meta)
    y_be = np.array(y_be)
    
    # Reference solution
    actual = solve_ivp(dydx, span, y0, t_eval=time)
    
    error = np.max(np.abs(y_be - actual.y.T))
    assert error < 10.0, f"Exceeds error tolerance: {error}"


def test_rk4_rc():
    """Test RK4 with RC circuit."""
    benchmark = RK4()
    generator = RCGenerator()
    dataset = generator.datasets[0]  # rc_small
    data, meta = generator.generate(dataset)
    
    dydx, span, y0, step = data
    
    time, y_rk4 = benchmark.benchmark(data, meta)
    y_rk4 = np.array(y_rk4)
    
    # Reference solution
    actual = solve_ivp(dydx, span, y0, t_eval=time)
    
    error = np.max(np.abs(y_rk4 - actual.y.T))
    assert error < 0.05, f"Exceeds error tolerance: {error}"


def test_rk4_rlc():
    """Test RK4 with RLC circuit."""
    benchmark = RK4()
    generator = RLCGenerator()
    dataset = generator.datasets[0]  # rlc_small
    data, meta = generator.generate(dataset)
    
    dydx, span, y0, step = data
    
    time, y_rk4 = benchmark.benchmark(data, meta)
    y_rk4 = np.array(y_rk4)
    
    # Reference solution
    actual = solve_ivp(dydx, span, y0, t_eval=time)
    
    error = np.max(np.abs(y_rk4[:, 0] - actual.y[0].T))
    assert error < 0.05, f"Exceeds error tolerance: {error}"


def test_rk4_lotka_volterra():
    """Test RK4 with Lotka-Volterra equations."""
    benchmark = RK4()
    generator = LotkaVolterraGenerator()
    dataset = generator.datasets[0]  # lotka_volterra_small
    data, meta = generator.generate(dataset)
    
    dydx, span, y0, step = data
    
    time, y_rk4 = benchmark.benchmark(data, meta)
    y_rk4 = np.array(y_rk4)
    
    # Reference solution
    actual = solve_ivp(dydx, span, y0, t_eval=time)
    
    error = np.max(np.abs(y_rk4 - actual.y.T))
    assert error < 10.0, f"Exceeds error tolerance: {error}"


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
