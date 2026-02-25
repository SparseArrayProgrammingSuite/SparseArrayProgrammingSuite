import pytest

from sparseappbench.benchmarks.Finite_Difference import (
    buckley_leverett_flux,
    burgers_flux,
    lax_freidrichs_data_generator,
    lax_freidrichs_matrix,
    lax_friedrichs_solver,
    lax_friedrichs_solver_matrix,
    linear_advection_flux,
)
from sparseappbench.frameworks.numpy_framework import NumpyFramework


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

    u0 = lax_freidrichs_data_generator(xp, N, density=0.05)
    matrix = lax_freidrichs_matrix(xp, N, dx, dt, c)
    result_bench = lax_friedrichs_solver_matrix(
        xp=xp, u0_bench=u0, matrix_bench=matrix, timesteps=timesteps
    )

    result = xp.from_benchmark(result_bench)
    cfl = (c * dt) / dx

    norm_initial = xp.linalg.norm(u0)
    norm_final = xp.linalg.norm(result)
    growth_ratio = norm_initial / norm_final

    # For Linear Advection: We should show that the soultion does not blow up.
    # This depends on the CFL condition.
    if cfl <= 1:
        assert growth_ratio <= 1.01


@pytest.mark.parametrize(
    "c,dx,dt",
    [
        (0.9, 1, 1),
        (2, 0.5, 0.2),
    ],
)
def test_linear_advection_stencil_check(xp, c, dx, dt):
    N = 200
    timesteps = 20

    u0 = lax_freidrichs_data_generator(xp, N, density=0.05)
    matrix = lax_freidrichs_matrix(xp, N, dx, dt, c)
    result_bench_matrix = lax_friedrichs_solver_matrix(
        xp=xp, u0_bench=u0, matrix_bench=matrix, timesteps=timesteps
    )
    flux = linear_advection_flux(c)
    result_bench_interative = lax_friedrichs_solver(
        xp=xp,
        u0_bench=u0,
        dt=dt,
        dx=dx,
        flux=flux,
        timesteps=timesteps,
    )
    result_matrix = xp.from_benchmark(result_bench_matrix)
    result_bench_inter = xp.from_benchmark(result_bench_interative)

    assert xp.linalg.norm(result_bench_inter - result_matrix) <= 1e-6


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
def test_mass_conservation_nonlinear_flux(xp, dx, dt, flux):
    N = 200
    timesteps = 20

    u0 = lax_freidrichs_data_generator(xp, N, density=0.05)

    result_bench_interative = lax_friedrichs_solver(
        xp=xp,
        u0_bench=u0,
        dt=dt,
        dx=dx,
        flux=flux,
        timesteps=timesteps,
    )
    result_bench_inter = xp.from_benchmark(result_bench_interative)

    inital_mass = xp.sum(result_bench_inter[0])
    final_mass = xp.sum(result_bench_inter[-1])

    assert (final_mass - inital_mass) <= 1e-6
