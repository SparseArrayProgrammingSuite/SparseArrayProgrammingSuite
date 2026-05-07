import pytest

import numpy as np

import saps.benchmarks.Finite_Difference_2D as fd2d
from saps.benchmarks.Finite_Difference_2D import (
    aniso_burgers_flux_2D,
    buckley_leverett_flux_2D,
    linear_advection_flux_2D,
)
from frameworks.saps_numpy import NumpyFramework


@pytest.fixture
def xp():
    return NumpyFramework()


def generate_fd2d_triplet(xp, x_spatial, y_spatial, flux):
    gen = fd2d.FiniteDifference2DGenerator()
    ds = fd2d.FiniteDifference2DDataset(
        name=f"test_{x_spatial}x{y_spatial}",
        pretty_name="test",
        tags=[],
        Nx=x_spatial,
        dx=0.1,
        Ny=y_spatial,
        dy=0.1,
        Nt=20,
        dt=0.01,
        flux=flux,
    )

    prev_xp = getattr(fd2d, "xp", None)
    fd2d.xp = xp
    try:
        data, meta = gen.generate(ds)
    finally:
        fd2d.xp = prev_xp

    u = xp.from_binsparse(data[0])
    matrix = xp.from_binsparse(data[1])
    diff_x = xp.from_binsparse(data[2])
    diff_y = xp.from_binsparse(data[3])
    return u, matrix, diff_x, diff_y


def lax_friedrichs_solver_matrix_2d(
    xp,
    u0_bench,
    matrix_bench,
    diff_x_bench,
    diff_y_bench,
    timesteps,
    flux_x,
    flux_y,
    dt,
    dx,
    dy,
):
    data = [u0_bench, matrix_bench, diff_x_bench, diff_y_bench]
    meta = {
        "flux_x": flux_x,
        "flux_y": flux_y,
        "timesteps": timesteps,
        "dt": dt,
        "dx": dx,
        "dy": dy,
    }

    prev_xp = getattr(fd2d, "xp", None)
    fd2d.xp = xp
    try:
        return fd2d.FiniteDifference2DBenchmark().benchmark(data, meta)[0]
    finally:
        fd2d.xp = prev_xp


@pytest.mark.parametrize(
    "cx,cy,dx,dt,dy",
    [
        (0.9, 0.9, 1, 1, 1),
        (2, 2, 0.5, 0.2, 0.5),
    ],
)
def test_linear_advection_cfl_check(xp, cx, cy, dx, dt, dy):
    x_spatial = 10
    y_spatial = 10
    timesteps = 20

    fl_x, fl_y = linear_advection_flux_2D(cx, cy)

    u0, matrix, dif_x, dif_y = generate_fd2d_triplet(xp, x_spatial, y_spatial, (fl_x, fl_y))

    result_bench = lax_friedrichs_solver_matrix_2d(
        xp=xp,
        u0_bench=u0,
        matrix_bench=matrix,
        diff_x_bench=dif_x,
        diff_y_bench=dif_y,
        timesteps=timesteps,
        flux_x=fl_x,
        flux_y=fl_y,
        dt=dt,
        dx=dx,
        dy=dy,
    )

    result = xp.from_binsparse(result_bench)
    cfl_x = (cx * dt) / dx
    cfl_y = (cy * dt) / dy

    cfl = cfl_x + cfl_y

    norm_initial = xp.linalg.norm(u0)
    norm_final = xp.linalg.norm(result[-1])
    growth_ratio = norm_final / norm_initial

    # For Linear Advection: We should show that the soultion does not blow up.
    # This depends on the CFL condition.
    if cfl <= 1:
        assert growth_ratio <= 1.01


@pytest.mark.parametrize(
    "dx,dy,dt,flux",
    [
        (0.01, 0.05, 0.0025, buckley_leverett_flux_2D),
        (0.01, 0.05, 0.0025, aniso_burgers_flux_2D),
    ],
)
# tests the nonlinear flux to see if the values blow up or if they stay stable.
# For the burgers, I picked arbitrary value that it stays within.
def test_nonlinear_flux(xp, dx, dy, dt, flux):
    x_spatial = 10
    y_spatial = 10
    timesteps = 20

    flux_x, flux_y = flux()

    u0, matrix, dif_x, dif_y = generate_fd2d_triplet(xp, x_spatial, y_spatial, (flux_x, flux_y))

    result_bench = lax_friedrichs_solver_matrix_2d(
        xp=xp,
        u0_bench=u0,
        matrix_bench=matrix,
        diff_x_bench=dif_x,
        diff_y_bench=dif_y,
        timesteps=timesteps,
        flux_x=flux_x,
        flux_y=flux_y,
        dt=dt,
        dx=dx,
        dy=dy,
    )

    result = xp.from_binsparse(result_bench)

    # Checking to see if values are finite
    assert xp.all(xp.isfinite(result))
    assert xp.max(result) <= 5  # just a arbitrary number
    assert xp.min(result) >= -5

    # Buckley_leverett should stay between 0 and 1.
    if flux == buckley_leverett_flux_2D:
        assert xp.max(result) <= 1
        assert xp.min(result) >= 0


# For this case, we are seeing if the matrix multiplication works.
# So if we have only 1 position and update it once, then only its neighbours
# are nonzero. This case only is possible to test using nonlinear advection.
@pytest.mark.parametrize(
    "dx,dy,dt,",
    [
        (1, 1, 0.1),
        (0.5, 0.5, 0.05),
        (1, 1, 0.5),
    ],
)
def test__linear_adv_sparse_stencil_check(xp, dx, dy, dt):
    x_spatial = 10
    y_spatial = 10
    timesteps = 1

    u0 = xp.zeros(x_spatial * y_spatial)
    center = (y_spatial // 2) * x_spatial + (x_spatial // 2)
    u0[center] = 1

    fl_x, fl_y = linear_advection_flux_2D(1.0, 1.0)

    _, matrix, dif_x, dif_y = generate_fd2d_triplet(xp, x_spatial, y_spatial, (fl_x, fl_y))

    result_bench = lax_friedrichs_solver_matrix_2d(
        xp=xp,
        u0_bench=u0,
        matrix_bench=matrix,
        diff_x_bench=dif_x,
        diff_y_bench=dif_y,
        timesteps=timesteps,
        flux_x=fl_x,
        flux_y=fl_y,
        dt=dt,
        dx=dx,
        dy=dy,
    )

    result = xp.from_binsparse(result_bench)

    final_results = result[-1]

    theory_non_zero_points = [
        center,
        center - 1,
        center + 1,
        center - x_spatial,
        center + x_spatial,
    ]

    actual_non_zero_points = xp.nonzero(final_results)[0].tolist()
    for id in actual_non_zero_points:
        assert id in theory_non_zero_points
