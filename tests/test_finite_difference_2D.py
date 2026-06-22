import pytest

import saps.benchmarks.Finite_Difference_2D as fd2d
from frameworks.saps_numpy import NumpyFramework
from saps.benchmarks.Finite_Difference_2D import (
    BuckleyLeverettFiniteDifference2DBenchmark,
    BurgersFiniteDifference2DBenchmark,
    LinearAdvectionFiniteDifference2DBenchmark,
)


@pytest.fixture
def xp():
    return NumpyFramework()


def generate_fd2d_triplet(xp, x_spatial, y_spatial):
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
    )

    prev_xp = getattr(fd2d, "xp", None)
    fd2d.xp = xp
    try:
        data = gen.generate(ds).inputs
    finally:
        fd2d.xp = prev_xp

    u = xp.from_binsparse(data[0])
    matrix = xp.from_binsparse(data[1])
    diff_x = xp.from_binsparse(data[2])
    diff_y = xp.from_binsparse(data[3])
    return u, matrix, diff_x, diff_y


def lax_friedrichs_solver_matrix_2d(
    xp, bench, u0_bench, matrix_bench, diff_x_bench, diff_y_bench, timesteps, dt, dx, dy
):
    data = [u0_bench, matrix_bench, diff_x_bench, diff_y_bench]
    meta = {"timesteps": timesteps, "dt": dt, "dx": dx, "dy": dy}

    prev_xp = getattr(fd2d, "xp", None)
    fd2d.xp = xp
    try:
        return bench.benchmark(data, meta)[0]
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

    u0, matrix, dif_x, dif_y = generate_fd2d_triplet(xp, x_spatial, y_spatial)

    bench = LinearAdvectionFiniteDifference2DBenchmark()
    bench.CX = cx
    bench.CY = cy

    result = lax_friedrichs_solver_matrix_2d(
        xp=xp,
        bench=bench,
        u0_bench=u0,
        matrix_bench=matrix,
        diff_x_bench=dif_x,
        diff_y_bench=dif_y,
        timesteps=timesteps,
        dt=dt,
        dx=dx,
        dy=dy,
    )

    cfl = (cx * dt) / dx + (cy * dt) / dy
    norm_initial = xp.linalg.norm(u0)
    norm_final = xp.linalg.norm(result[-1])
    growth_ratio = norm_final / norm_initial

    if cfl <= 1:
        assert growth_ratio <= 1.01


@pytest.mark.parametrize(
    "dx,dy,dt,bench_cls",
    [
        (0.01, 0.05, 0.0025, BuckleyLeverettFiniteDifference2DBenchmark),
        (0.01, 0.05, 0.0025, BurgersFiniteDifference2DBenchmark),
    ],
)
def test_nonlinear_flux(xp, dx, dy, dt, bench_cls):
    x_spatial = 10
    y_spatial = 10
    timesteps = 20

    u0, matrix, dif_x, dif_y = generate_fd2d_triplet(xp, x_spatial, y_spatial)

    result = lax_friedrichs_solver_matrix_2d(
        xp=xp,
        bench=bench_cls(),
        u0_bench=u0,
        matrix_bench=matrix,
        diff_x_bench=dif_x,
        diff_y_bench=dif_y,
        timesteps=timesteps,
        dt=dt,
        dx=dx,
        dy=dy,
    )

    assert xp.all(xp.isfinite(result))
    assert xp.max(result) <= 5
    assert xp.min(result) >= -5

    if bench_cls is BuckleyLeverettFiniteDifference2DBenchmark:
        assert xp.max(result) <= 1
        assert xp.min(result) >= 0


@pytest.mark.parametrize(
    "dx,dy,dt",
    [
        (1, 1, 0.1),
        (0.5, 0.5, 0.05),
        (1, 1, 0.5),
    ],
)
def test_linear_adv_sparse_stencil_check(xp, dx, dy, dt):
    x_spatial = 10
    y_spatial = 10
    timesteps = 1

    u0 = xp.zeros(x_spatial * y_spatial)
    center = (y_spatial // 2) * x_spatial + (x_spatial // 2)
    u0[center] = 1

    _, matrix, dif_x, dif_y = generate_fd2d_triplet(xp, x_spatial, y_spatial)

    bench = LinearAdvectionFiniteDifference2DBenchmark()
    bench.CX = 1.0
    bench.CY = 1.0

    result = lax_friedrichs_solver_matrix_2d(
        xp=xp,
        bench=bench,
        u0_bench=u0,
        matrix_bench=matrix,
        diff_x_bench=dif_x,
        diff_y_bench=dif_y,
        timesteps=timesteps,
        dt=dt,
        dx=dx,
        dy=dy,
    )

    final_results = result[-1]
    theory_non_zero_points = [
        center,
        center - 1,
        center + 1,
        center - x_spatial,
        center + x_spatial,
    ]
    actual_non_zero_points = xp.nonzero(final_results)[0].tolist()
    for idx in actual_non_zero_points:
        assert idx in theory_non_zero_points
