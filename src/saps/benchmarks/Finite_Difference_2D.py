import numpy as np

import saps
from saps.benchmark import (
    Author,
    Benchmark,
    Contributor,
    DataInstance,
    Dataset,
    Generator,
    Ref,
)

xp = saps.xp


# This matrix formula assume Dirichlet BC instead of Periodic BC.
def _lax_freidrichs_matrix_no_flux_2D(number_spatial_x, number_spatial_y):
    N = number_spatial_x * number_spatial_y
    matrix = np.zeros((N, N))
    for i in range(N):
        x = i % number_spatial_x
        y = i // number_spatial_x
        if x > 0:
            matrix[i, i - 1] = 0.25
        if x < number_spatial_x - 1:
            matrix[i, i + 1] = 0.25

        if y > 0:
            matrix[i, i - number_spatial_x] = 0.25
        if y < number_spatial_y - 1:
            matrix[i, i + number_spatial_x] = 0.25

    return matrix


def _difference_matrix_x_direction(number_spatial_x, number_spatial_y):
    N = number_spatial_x * number_spatial_y
    dif_x_matrix = np.zeros((N, N))
    for i in range(N):
        x = i % number_spatial_x
        if x > 0:
            dif_x_matrix[i, i - 1] = -1
        if x < number_spatial_x - 1:
            dif_x_matrix[i, i + 1] = +1

    return dif_x_matrix


def _difference_matrix_y_direction(number_spatial_x, number_spatial_y):
    N = number_spatial_x * number_spatial_y
    dif_y_matrix = np.zeros((N, N))
    for i in range(N):
        y = i // number_spatial_x
        if y > 0:
            dif_y_matrix[i, i - number_spatial_x] = -1
        if y < number_spatial_y - 1:
            dif_y_matrix[i, i + number_spatial_x] = +1

    return dif_y_matrix


class FiniteDifference2DDataset(Dataset):
    def __init__(self, name, pretty_name, suites, Nx, dx, Ny, dy, Nt, dt):
        self._name = name
        self._pretty_name = pretty_name
        self._suites = suites
        self.Nx = Nx
        self.dx = dx
        self.Ny = Ny
        self.dy = dy
        self.Nt = Nt
        self.dt = dt

    @property
    def name(self) -> str:
        return self._name

    @property
    def pretty_name(self) -> str:
        return self._pretty_name

    @property
    def description(self) -> str:
        return (
            f"{self.pretty_name}: Nx = {self.Nx}, dx = {self.dx}, "
            f"Ny = {self.Ny}, dy = {self.dy}, dt = {self.dt}."
        )

    @property
    def suites(self) -> list[str]:
        return self._suites

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"


# BEGIN COPIED TEST FILE: tests/test_finite_difference_2D.py
# import pytest
#
# import saps.benchmarks.Finite_Difference_2D as fd2d
# from frameworks.saps_numpy import NumpyFramework
# from saps.benchmarks.Finite_Difference_2D import (
#     BuckleyLeverettFiniteDifference2DBenchmark,
#     BurgersFiniteDifference2DBenchmark,
#     LinearAdvectionFiniteDifference2DBenchmark,
# )
#
#
# @pytest.fixture
# def xp():
#     return NumpyFramework()
#
#
# def generate_fd2d_triplet(xp, x_spatial, y_spatial):
#     gen = fd2d.FiniteDifference2DGenerator()
#     ds = fd2d.FiniteDifference2DDataset(
#         name=f"test_{x_spatial}x{y_spatial}",
#         pretty_name="test",
#         tags=[],
#         Nx=x_spatial,
#         dx=0.1,
#         Ny=y_spatial,
#         dy=0.1,
#         Nt=20,
#         dt=0.01,
#     )
#
#     prev_xp = getattr(fd2d, "xp", None)
#     fd2d.xp = xp
#     try:
#         data = gen.generate(ds).inputs
#     finally:
#         fd2d.xp = prev_xp
#
#     u = xp.from_binsparse(data[0])
#     matrix = xp.from_binsparse(data[1])
#     diff_x = xp.from_binsparse(data[2])
#     diff_y = xp.from_binsparse(data[3])
#     return u, matrix, diff_x, diff_y
#
#
# def lax_friedrichs_solver_matrix_2d(
#     xp, bench, u0_bench, matrix_bench, diff_x_bench, diff_y_bench, timesteps, dt, dx, dy  # noqa: E501
# ):
#     data = [u0_bench, matrix_bench, diff_x_bench, diff_y_bench]
#     meta = {"timesteps": timesteps, "dt": dt, "dx": dx, "dy": dy}
#
#     prev_xp = getattr(fd2d, "xp", None)
#     fd2d.xp = xp
#     try:
#         return bench.benchmark(data, meta)[0]
#     finally:
#         fd2d.xp = prev_xp
#
#
# @pytest.mark.parametrize(
#     "cx,cy,dx,dt,dy",
#     [
#         (0.9, 0.9, 1, 1, 1),
#         (2, 2, 0.5, 0.2, 0.5),
#     ],
# )
# def test_linear_advection_cfl_check(xp, cx, cy, dx, dt, dy):
#     x_spatial = 10
#     y_spatial = 10
#     timesteps = 20
#
#     u0, matrix, dif_x, dif_y = generate_fd2d_triplet(xp, x_spatial, y_spatial)
#
#     bench = LinearAdvectionFiniteDifference2DBenchmark()
#     bench.CX = cx
#     bench.CY = cy
#
#     result = lax_friedrichs_solver_matrix_2d(
#         xp=xp,
#         bench=bench,
#         u0_bench=u0,
#         matrix_bench=matrix,
#         diff_x_bench=dif_x,
#         diff_y_bench=dif_y,
#         timesteps=timesteps,
#         dt=dt,
#         dx=dx,
#         dy=dy,
#     )
#
#     cfl = (cx * dt) / dx + (cy * dt) / dy
#     norm_initial = xp.linalg.norm(u0)
#     norm_final = xp.linalg.norm(result[-1])
#     growth_ratio = norm_final / norm_initial
#
#     if cfl <= 1:
#         assert growth_ratio <= 1.01
#
#
# @pytest.mark.parametrize(
#     "dx,dy,dt,bench_cls",
#     [
#         (0.01, 0.05, 0.0025, BuckleyLeverettFiniteDifference2DBenchmark),
#         (0.01, 0.05, 0.0025, BurgersFiniteDifference2DBenchmark),
#     ],
# )
# def test_nonlinear_flux(xp, dx, dy, dt, bench_cls):
#     x_spatial = 10
#     y_spatial = 10
#     timesteps = 20
#
#     u0, matrix, dif_x, dif_y = generate_fd2d_triplet(xp, x_spatial, y_spatial)
#
#     result = lax_friedrichs_solver_matrix_2d(
#         xp=xp,
#         bench=bench_cls(),
#         u0_bench=u0,
#         matrix_bench=matrix,
#         diff_x_bench=dif_x,
#         diff_y_bench=dif_y,
#         timesteps=timesteps,
#         dt=dt,
#         dx=dx,
#         dy=dy,
#     )
#
#     assert xp.all(xp.isfinite(result))
#     assert xp.max(result) <= 5
#     assert xp.min(result) >= -5
#
#     if bench_cls is BuckleyLeverettFiniteDifference2DBenchmark:
#         assert xp.max(result) <= 1
#         assert xp.min(result) >= 0
#
#
# @pytest.mark.parametrize(
#     "dx,dy,dt",
#     [
#         (1, 1, 0.1),
#         (0.5, 0.5, 0.05),
#         (1, 1, 0.5),
#     ],
# )
# def test_linear_adv_sparse_stencil_check(xp, dx, dy, dt):
#     x_spatial = 10
#     y_spatial = 10
#     timesteps = 1
#
#     u0 = xp.zeros(x_spatial * y_spatial)
#     center = (y_spatial // 2) * x_spatial + (x_spatial // 2)
#     u0[center] = 1
#
#     _, matrix, dif_x, dif_y = generate_fd2d_triplet(xp, x_spatial, y_spatial)
#
#     bench = LinearAdvectionFiniteDifference2DBenchmark()
#     bench.CX = 1.0
#     bench.CY = 1.0
#
#     result = lax_friedrichs_solver_matrix_2d(
#         xp=xp,
#         bench=bench,
#         u0_bench=u0,
#         matrix_bench=matrix,
#         diff_x_bench=dif_x,
#         diff_y_bench=dif_y,
#         timesteps=timesteps,
#         dt=dt,
#         dx=dx,
#         dy=dy,
#     )
#
#     final_results = result[-1]
#     theory_non_zero_points = [
#         center,
#         center - 1,
#         center + 1,
#         center - x_spatial,
#         center + x_spatial,
#     ]
#     actual_non_zero_points = xp.nonzero(final_results)[0].tolist()
#     for idx in actual_non_zero_points:
#         assert idx in theory_non_zero_points
# END COPIED TEST FILE: tests/test_finite_difference_2D.py

class FiniteDifference2DGenerator(Generator[FiniteDifference2DDataset]):
    @property
    def name(self) -> str:
        return "finite_difference_inputs_2d"

    @property
    def pretty_name(self) -> str:
        return "Finite Difference Data Generator"

    @property
    def description(self) -> str:
        return (
            "The purpose of this is to analyze the importance of numerical methods for"
            " PDEs, and applications sparse array theory into these method, through the"
            " form of benchmarks. This paticular benchmark analyzes the use of the"
            " Lax–Friedrichs method for solving nonlinear hyberbolic PDEs, with"
            " numerical stability and accuracy not seen in FTCS. This benchmark will"
            " run a simulation using both Lax–Friedrichs and analyze core concepts such"
            " as numerical stability, conservation law consistency, etc."
        )

    @property
    def suites(self) -> list[str]:
        return []

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def authors(self) -> list[Contributor]:
        return [Contributor("Vilohith Gokarakonda", "vgokarakonda3@gatech.edu")]

    @property
    def references(self) -> list[Ref]:
        return [
            Ref(
                title=(
                    "Synthesizing Sound and Precise Abstract Transformers"
                    " for Nonlinear Hyperbolic PDE Solvers."
                ),
                authors=[
                    Author("Jacob Laurel"),
                    Author("Ignacio Laguna"),
                    Author("Jan Hückelheim"),
                ],
                journal="Proceedings of the ACM on Programming Languages",
                publisher="Association for Computing Machinery (ACM)",
                volume="9",
                number="OOPSLA2",
                pages="1063-1091",
                year=2025,
                url="https://doi.org/10.1145/3763088",
                doi="10.1145/3763088",
            )
        ]

    @property
    def ai_disclosure(self) -> str:
        return (
            "No generative AI was used to construct the benchmark function itself."
            " Generative AI might have been used to construct tests. This statement was"
            " written by hand."
        )

    @property
    def motivation(self) -> str:
        return (
            "For linear advection, updates are done using a sparse matrix"
            " representation, to updates the spatial coordinates for time t."
        )

    @property
    def datasets(self) -> list[FiniteDifference2DDataset]:
        return [
            FiniteDifference2DDataset(
                name="default",
                pretty_name="Default",
                suites=[],
                Nx=100,
                dx=0.1,
                Ny=100,
                dy=0.1,
                Nt=100,
                dt=0.01,
            ),
        ]

    def generate(self, dataset: FiniteDifference2DDataset):
        # Produce a gentle, sparse initial condition (small amplitudes)
        density = 0.05
        u_0 = np.zeros(dataset.Nx * dataset.Ny, dtype=float)
        k = max(1, int(dataset.Nx * dataset.Ny * density))
        rng = np.random.default_rng()
        idx = rng.choice(dataset.Nx, size=k, replace=False)
        # small random amplitudes to avoid nonlinear overflow
        u_0[idx] = rng.random(k) * 0.5
        # a modest central pulse (order 1), previously was 10 which caused instability
        u_0[dataset.Nx // 2 * dataset.Ny + dataset.Ny // 2] = max(
            u_0[dataset.Nx // 2 * dataset.Ny + dataset.Ny // 2], 1.0
        )

        diff_x = _difference_matrix_x_direction(dataset.Nx, dataset.Ny)
        diff_y = _difference_matrix_y_direction(dataset.Nx, dataset.Ny)
        matrix = _lax_freidrichs_matrix_no_flux_2D(dataset.Nx, dataset.Ny)

        data = (
            xp.to_binsparse(u_0),
            xp.to_binsparse(matrix),
            xp.to_binsparse(diff_x),
            xp.to_binsparse(diff_y),
        )

        meta = {
            "timesteps": dataset.Nt,
            "dt": dataset.dt,
            "dx": dataset.dx,
            "dy": dataset.dy,
        }
        return DataInstance(inputs=data, meta=meta)


class _FiniteDifference2DBenchmarkBase(Benchmark):
    @property
    def suites(self) -> list[str]:
        return []

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def authors(self) -> list[Contributor]:
        return [Contributor("Vilohith Gokarakonda", "vgokarakonda3@gatech.edu")]

    @property
    def references(self) -> list[Ref]:
        return [
            Ref(
                title=(
                    "Synthesizing Sound and Precise Abstract Transformers"
                    " for Nonlinear Hyperbolic PDE Solvers."
                ),
                authors=[
                    Author("Jacob Laurel"),
                    Author("Ignacio Laguna"),
                    Author("Jan Hückelheim"),
                ],
                journal="Proceedings of the ACM on Programming Languages",
                publisher="Association for Computing Machinery (ACM)",
                volume="9",
                number="OOPSLA2",
                pages="1063-1091",
                year=2025,
                url="https://doi.org/10.1145/3763088",
                doi="10.1145/3763088",
            )
        ]

    @property
    def ai_disclosure(self) -> str:
        return (
            "No generative AI was used to construct the benchmark function itself."
            " Generative AI might have been used to construct tests. This statement was"
            " written by hand."
        )

    @property
    def motivation(self) -> str:
        return (
            "Updates are done using a matrix representation, to updates"
            " the spatial coordinates for time t."
        )

    @property
    def description(self) -> str:
        return (
            "The purpose of this is to analyze the importance of numerical methods for"
            " PDEs, and applications sparse array theory into these method, through the"
            " form of benchmarks. This paticular benchmark analyzes the use of the"
            " Lax–Friedrichs method for solving nonlinear hyberbolic PDEs, with"
            " numerical stability and accuracy not seen in FTCS. This benchmark will"
            " run a simulation using both Lax–Friedrichs and analyze core concepts such"
            " as numerical stability, conservation law consistency, etc."
        )

    @property
    def generators(self):
        return [FiniteDifference2DGenerator()]


class BurgersFiniteDifference2DBenchmark(_FiniteDifference2DBenchmarkBase):
    @property
    def name(self) -> str:
        return "burgers_finite_difference_2d"

    @property
    def pretty_name(self) -> str:
        return "2D Finite Difference (Burgers flux)"

    def benchmark(self, data: list, meta: dict):
        u_0, matrix, diff_x, diff_y = data
        timesteps = meta["timesteps"]
        dt = meta["dt"]
        dx = meta["dx"]
        dy = meta["dy"]

        Nt = timesteps + 1
        u = xp.zeros((Nt, u_0.shape[0]))
        u[0] = u_0

        alpha = dt / (2 * dx)
        beta = dt / (2 * dy)

        for n in range(Nt - 1):
            u_n = u[n]
            fl_x = 0.5 * u_n * u_n
            fl_y = (1 / 3) * u_n * u_n
            u_next = matrix @ u_n - alpha * (diff_x @ fl_x) - beta * (diff_y @ fl_y)
            u[n + 1] = u_next

        return [u]


class BuckleyLeverettFiniteDifference2DBenchmark(_FiniteDifference2DBenchmarkBase):
    @property
    def name(self) -> str:
        return "buckley_leverett_finite_difference_2d"

    @property
    def pretty_name(self) -> str:
        return "2D Finite Difference (Buckley-Leverett flux)"

    def benchmark(self, data: list, meta: dict):
        u_0, matrix, diff_x, diff_y = data
        timesteps = meta["timesteps"]
        dt = meta["dt"]
        dx = meta["dx"]
        dy = meta["dy"]

        Nt = timesteps + 1
        u = xp.zeros((Nt, u_0.shape[0]))
        u[0] = u_0

        alpha = dt / (2 * dx)
        beta = dt / (2 * dy)

        for n in range(Nt - 1):
            u_n = u[n]
            sq = u_n * u_n
            denom = sq + (0.25 * (1 - u_n) * (1 - u_n))
            fl_x = sq / denom
            fl_y = sq / denom
            u_next = matrix @ u_n - alpha * (diff_x @ fl_x) - beta * (diff_y @ fl_y)
            u[n + 1] = u_next

        return [u]


class LinearAdvectionFiniteDifference2DBenchmark(_FiniteDifference2DBenchmarkBase):
    CX = 0.9
    CY = 0.9

    @property
    def name(self) -> str:
        return "linear_advection_finite_difference_2d"

    @property
    def pretty_name(self) -> str:
        return "2D Finite Difference (Linear Advection flux)"

    def benchmark(self, data: list, meta: dict):
        u_0, matrix, diff_x, diff_y = data
        timesteps = meta["timesteps"]
        dt = meta["dt"]
        dx = meta["dx"]
        dy = meta["dy"]

        Nt = timesteps + 1
        u = xp.zeros((Nt, u_0.shape[0]))
        u[0] = u_0

        alpha = dt / (2 * dx)
        beta = dt / (2 * dy)

        for n in range(Nt - 1):
            u_n = u[n]
            fl_x = self.CX * u_n
            fl_y = self.CY * u_n
            u_next = matrix @ u_n - alpha * (diff_x @ fl_x) - beta * (diff_y @ fl_y)
            u[n + 1] = u_next

        return [u]
