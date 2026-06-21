import numpy as np

import saps
from saps.benchmark import (
    Author,
    Benchmark,
    Contributor,
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
                    Author("Laurel, J."),
                    Author("Laguna, I."),
                    Author("Hückelheim, J."),
                ],
                journal="Proceedings of the ACM on Programming Languages",
                volume="9",
                number="OOPSLA2",
                pages="1063–1091",
                year=2025,
                url="https://doi.org/10.1145/3763088",
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
        return data, meta


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
                    Author("Laurel, J."),
                    Author("Laguna, I."),
                    Author("Hückelheim, J."),
                ],
                journal="Proceedings of the ACM on Programming Languages",
                volume="9",
                number="OOPSLA2",
                pages="1063–1091",
                year=2025,
                url="https://doi.org/10.1145/3763088",
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
