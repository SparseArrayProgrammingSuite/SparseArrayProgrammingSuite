import pytest
import numpy as np
import saps.benchmarks.Finite_Difference as fd
from saps_framework import BinsparseFormat
from saps.benchmarks.Finite_Difference import (
    buckley_leverett_flux,
    burgers_flux,
    linear_advection_flux,
)
from frameworks.saps_numpy import NumpyFramework

def generate_fd_triplet(xp, N, flux=fd.burgers_flux):
    gen = fd.FiniteDifferenceGenerator()
    ds = fd.FiniteDifferenceDataset(
        name=f"test_trip_{N}",
        pretty_name="test_trip",
        tags=[],
        Nx=N,
        dx=0.1,
        Nt=20,
        dt=0.01,
        flux=flux,
    )
    prev_xp = getattr(fd, "xp", None)
    fd.xp = xp
    try:
        data, meta = gen.generate(ds)
    finally:
        fd.xp = prev_xp

        u = xp.from_binsparse(data[0])
    return u, data[1], data[2]


def lax_friedrichs_solver_matrix_general(
    xp, u0_bench, matrix_bench, difference_bench, timesteps, flux, dt, dx
):
    # Use the FiniteDifferenceBenchmark implementation to run the matrix-based solver
    from saps_framework import BinsparseFormat

    def ensure_array(a):
        return xp.from_binsparse(a) if isinstance(a, BinsparseFormat) else a

    data = (ensure_array(u0_bench), ensure_array(matrix_bench), ensure_array(difference_bench))
    meta = {"flux": flux, "timesteps": timesteps, "dt": dt, "dx": dx}

    prev_xp = getattr(fd, "xp", None)
    fd.xp = xp
    try:
        out = fd.FiniteDifferenceBenchmark().benchmark(list(data), meta)
    finally:
        fd.xp = prev_xp
    return out[0]


def lax_friedrichs_solver(xp, u0_bench, dt, dx, flux, timesteps):
    u_0 = u0_bench

    Nt = timesteps + 1

    # Intializes the space-time grid
    u = xp.zeros((Nt, int(u_0.shape[0])))

    u[0] = u_0

    alpha = dt / (2 * dx)
    for n in range(Nt - 1):
        u_n = u[n]
        # Vector equivalent of doing
        # u[t+1][x] = 0.5(u[t][n+1] - u[t][n-1]) -  alpha (flux(u[t][n+1])
        # - flux(u[t][n-1]))
        # Naturally incorporates periodic BC.
        u_next_spatial = xp.roll(u_n, -1)  # u[i +1]
        u_prev_spatial = xp.roll(u_n, 1)  # u[i -1]
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

    # generator already returns initial condition, matrix, and difference
    u0, matrix, dif = generate_fd_triplet(xp, N, flux=linear_advection_flux(c))

    flux = linear_advection_flux(c)

    result_bench = lax_friedrichs_solver_matrix_general(
        xp=xp,
        u0_bench=u0,
        matrix_bench=matrix,
        difference_bench=dif,
        timesteps=timesteps,
        flux=flux,
        dt=dt,
        dx=dx,
    )

    result = result_bench
    cfl = (c * dt) / dx

    norm_initial = xp.linalg.norm(u0)
    norm_final = xp.linalg.norm(result[-1])
    growth_ratio = norm_final / norm_initial

    # For Linear Advection: We should show that the soultion does not blow up.
    # This depends on the CFL condition.
    if cfl <= 1:
        assert growth_ratio <= 1.01


# These numbers for dx and dt were determined
# to be safe to pass CFL test for the two fluxes.
@pytest.mark.parametrize(
    "dx,dt,flux",
    [
        (0.01, 0.0025, buckley_leverett_flux),
        (0.01, 0.0025, burgers_flux),
    ],
)
# "mass" just means conservation of mass. Because of Periodic BC
# The integral (sum discrete) of u should remain constant
# tests mass conservation using the matrix calculation.
def test_mass_conservation_nonlinear_flux(xp, dx, dt, flux):
    N = 200
    timesteps = 20

    u0, matrix, dif = generate_fd_triplet(xp, N, flux=flux)

    result_bench = lax_friedrichs_solver_matrix_general(
        xp=xp,
        u0_bench=u0,
        matrix_bench=matrix,
        difference_bench=dif,
        timesteps=timesteps,
        flux=flux,
        dt=dt,
        dx=dx,
    )

    result_bench_inter = result_bench

    inital_mass = xp.sum(result_bench_inter[0])
    final_mass = xp.sum(result_bench_inter[-1])

    assert xp.abs(final_mass - inital_mass) <= 1e-6


# I made an iterative stencil to test the matrix method against.
@pytest.mark.parametrize(
    "dx,dt, flux",
    [
        (1, 1, burgers_flux),
        (0.5, 0.2, burgers_flux),
        (1, 1, buckley_leverett_flux),
        (0.5, 0.2, buckley_leverett_flux),
    ],
)
def test_nonlinear_matrix_stencil_check(xp, dx, dt, flux):
    Nx = 200
    timesteps = 20

    u0, matrix, dif = generate_fd_triplet(xp, Nx, flux=flux)

    result_bench_matrix = lax_friedrichs_solver_matrix_general(
        xp=xp,
        u0_bench=u0,
        matrix_bench=matrix,
        difference_bench=dif,
        timesteps=timesteps,
        flux=flux,
        dt=dt,
        dx=dx,
    )

    result_bench_interative = lax_friedrichs_solver(
        xp=xp,
        u0_bench=u0,
        dt=dt,
        dx=dx,
        flux=flux,
        timesteps=timesteps,
    )
    result_matrix = result_bench_matrix
    result_bench_inter = result_bench_interative

    assert xp.linalg.norm(result_bench_inter - result_matrix) <= 1e-6
