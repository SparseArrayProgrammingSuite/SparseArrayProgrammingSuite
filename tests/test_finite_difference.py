import pytest

import saps.benchmarks.Finite_Difference as fd
from frameworks.saps_numpy import NumpyFramework
from saps.benchmarks.Finite_Difference import (
    BuckleyLeverettFiniteDifferenceBenchmark,
    BurgersFiniteDifferenceBenchmark,
    LinearAdvectionFiniteDifferenceBenchmark,
)
from saps_framework import BinsparseFormat


def _burgers_ref(u):
    return 0.5 * u * u


def _buckley_leverett_ref(u):
    sq = u * u
    return sq / (sq + (0.25 * (1 - u) * (1 - u)))


def _linear_advection_ref(c):
    def flux(u):
        return c * u

    return flux


def generate_fd_triplet(xp, N):
    gen = fd.FiniteDifferenceGenerator()
    ds = fd.FiniteDifferenceDataset(
        name=f"test_trip_{N}",
        pretty_name="test_trip",
        tags=[],
        Nx=N,
        dx=0.1,
        Nt=20,
        dt=0.01,
    )
    prev_xp = getattr(fd, "xp", None)
    fd.xp = xp
    try:
        data = gen.generate(ds).inputs
    finally:
        fd.xp = prev_xp

    u = xp.from_binsparse(data[0])
    return u, data[1], data[2]


def lax_friedrichs_solver_matrix_general(
    xp, bench, u0_bench, matrix_bench, difference_bench, timesteps, dt, dx
):
    def ensure_array(a):
        return xp.from_binsparse(a) if isinstance(a, BinsparseFormat) else a

    data = (
        ensure_array(u0_bench),
        ensure_array(matrix_bench),
        ensure_array(difference_bench),
    )
    meta = {"timesteps": timesteps, "dt": dt, "dx": dx}

    prev_xp = getattr(fd, "xp", None)
    fd.xp = xp
    try:
        out = bench.benchmark(list(data), meta)
    finally:
        fd.xp = prev_xp
    return out[0]


def lax_friedrichs_solver(xp, u0_bench, dt, dx, flux, timesteps):
    u_0 = u0_bench
    Nt = timesteps + 1
    u = xp.zeros((Nt, int(u_0.shape[0])))
    u[0] = u_0
    alpha = dt / (2 * dx)
    for n in range(Nt - 1):
        u_n = u[n]
        u_next_spatial = xp.roll(u_n, -1)
        u_prev_spatial = xp.roll(u_n, 1)
        u_next = 0.5 * (u_next_spatial + u_prev_spatial) - alpha * (
            flux(u_next_spatial) - flux(u_prev_spatial)
        )
        u[n + 1] = u_next
    return u


@pytest.fixture
def xp():
    return NumpyFramework()


@pytest.mark.parametrize(
    "c,dx,dt",
    [
        (0.9, 1, 1),
        (2, 0.5, 0.2),
    ],
)
def test_linear_advection_cfl_check(xp, c, dx, dt):
    N = 200
    timesteps = 20

    u0, matrix, dif = generate_fd_triplet(xp, N)

    bench = LinearAdvectionFiniteDifferenceBenchmark()
    bench.C = c  # override per-test advection speed

    result = lax_friedrichs_solver_matrix_general(
        xp=xp,
        bench=bench,
        u0_bench=u0,
        matrix_bench=matrix,
        difference_bench=dif,
        timesteps=timesteps,
        dt=dt,
        dx=dx,
    )

    cfl = (c * dt) / dx
    norm_initial = xp.linalg.norm(u0)
    norm_final = xp.linalg.norm(result[-1])
    growth_ratio = norm_final / norm_initial

    if cfl <= 1:
        assert growth_ratio <= 1.01


@pytest.mark.parametrize(
    "dx,dt,bench_cls,ref_flux",
    [
        (0.01, 0.0025, BuckleyLeverettFiniteDifferenceBenchmark, _buckley_leverett_ref),
        (0.01, 0.0025, BurgersFiniteDifferenceBenchmark, _burgers_ref),
    ],
)
def test_mass_conservation_nonlinear_flux(xp, dx, dt, bench_cls, ref_flux):
    N = 200
    timesteps = 20

    u0, matrix, dif = generate_fd_triplet(xp, N)

    result = lax_friedrichs_solver_matrix_general(
        xp=xp,
        bench=bench_cls(),
        u0_bench=u0,
        matrix_bench=matrix,
        difference_bench=dif,
        timesteps=timesteps,
        dt=dt,
        dx=dx,
    )

    initial_mass = xp.sum(result[0])
    final_mass = xp.sum(result[-1])
    assert xp.abs(final_mass - initial_mass) <= 1e-6


@pytest.mark.parametrize(
    "dx,dt,bench_cls,ref_flux",
    [
        (1, 1, BurgersFiniteDifferenceBenchmark, _burgers_ref),
        (0.5, 0.2, BurgersFiniteDifferenceBenchmark, _burgers_ref),
        (1, 1, BuckleyLeverettFiniteDifferenceBenchmark, _buckley_leverett_ref),
        (0.5, 0.2, BuckleyLeverettFiniteDifferenceBenchmark, _buckley_leverett_ref),
    ],
)
def test_nonlinear_matrix_stencil_check(xp, dx, dt, bench_cls, ref_flux):
    Nx = 200
    timesteps = 20

    u0, matrix, dif = generate_fd_triplet(xp, Nx)

    result_matrix = lax_friedrichs_solver_matrix_general(
        xp=xp,
        bench=bench_cls(),
        u0_bench=u0,
        matrix_bench=matrix,
        difference_bench=dif,
        timesteps=timesteps,
        dt=dt,
        dx=dx,
    )

    result_iter = lax_friedrichs_solver(
        xp=xp,
        u0_bench=u0,
        dt=dt,
        dx=dx,
        flux=ref_flux,
        timesteps=timesteps,
    )

    assert xp.linalg.norm(result_iter - result_matrix) <= 1e-6
