import pytest

import numpy as np
from scipy.integrate import solve_ivp

from saps.benchmarks.ode import (
    RCRK4,
    RLCRK4,
    BrusselatorBackwardEuler,
    BrusselatorForwardEuler,
    BrusselatorRK4,
    LotkaVolterraBackwardEuler,
    LotkaVolterraForwardEuler,
    LotkaVolterraRK4,
    RCBackwardEuler,
    RCForwardEuler,
    RLCBackwardEuler,
    RLCForwardEuler,
)


def _run(bench_cls):
    bench = bench_cls()
    generator = bench.generators[0]
    dataset = generator.datasets[0]
    problem = generator.generate(dataset)
    data = problem.inputs
    meta = problem.meta
    time, y_out = bench.benchmark(data, meta)
    rhs = lambda t, y: bench._dydt(t, list(y), meta)  # noqa: E731
    actual = solve_ivp(rhs, meta["span"], meta["y0"], t_eval=time)
    return np.array(y_out), actual.y.T


@pytest.mark.parametrize("bench_cls", [RCForwardEuler, RCBackwardEuler, RCRK4])
def test_rc(bench_cls):
    y, ref = _run(bench_cls)
    error = np.max(np.abs(y - ref))
    assert error < 0.05, f"{bench_cls.__name__}: error {error} exceeds tolerance"


@pytest.mark.parametrize("bench_cls", [RLCForwardEuler, RLCBackwardEuler, RLCRK4])
def test_rlc(bench_cls):
    y, ref = _run(bench_cls)
    error = np.max(np.abs(y[:, 0] - ref[:, 0]))
    assert error < 0.05, f"{bench_cls.__name__}: error {error} exceeds tolerance"


@pytest.mark.parametrize(
    "bench_cls",
    [LotkaVolterraForwardEuler, LotkaVolterraBackwardEuler, LotkaVolterraRK4],
)
def test_lotka_volterra(bench_cls):
    y, ref = _run(bench_cls)
    error = np.max(np.abs(y - ref))
    assert error < 10.0, f"{bench_cls.__name__}: error {error} exceeds tolerance"


@pytest.mark.parametrize(
    "bench_cls", [BrusselatorForwardEuler, BrusselatorBackwardEuler, BrusselatorRK4]
)
def test_brusselator(bench_cls):
    y, ref = _run(bench_cls)
    y = y.real
    error = np.max(np.abs(y - ref))
    assert error < 0.5, f"{bench_cls.__name__}: error {error} exceeds tolerance"
