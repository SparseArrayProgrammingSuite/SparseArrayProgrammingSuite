import numpy as np

import sparse as pydata_sparse

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
from saps_framework import BinsparseFormat

xp = saps.xp


def _from_binsparse(array):
    if array.data["format"] == "dense":
        return array.data["values"].reshape(array.data["shape"])
    if array.data["format"] == "COO":
        shape = array.data["shape"]
        coords = np.array(
            [array.data[f"indices_{dim}"] for dim in range(len(shape))]
        )
        return pydata_sparse.COO(coords, array.data["values"], shape=shape).todense()
    raise ValueError(f"Unsupported format: {array.data['format']}")


def _to_binsparse(array):
    if isinstance(array, BinsparseFormat):
        return array
    if isinstance(array, pydata_sparse.SparseArray):
        coo = array.to_coo()
        return BinsparseFormat.from_coo(tuple(coo.coords), coo.data, coo.shape)
    return BinsparseFormat.from_numpy(np.asarray(array))


# This matrix formula assume Dirichlet BC instead of Periodic BC.
def _lax_freidrichs_matrix_no_flux_2D(number_spatial_x, number_spatial_y):
    N = number_spatial_x * number_spatial_y
    matrix = pydata_sparse.DOK((N, N), dtype=float)
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
    dif_x_matrix = pydata_sparse.DOK((N, N), dtype=float)
    for i in range(N):
        x = i % number_spatial_x
        if x > 0:
            dif_x_matrix[i, i - 1] = -1
        if x < number_spatial_x - 1:
            dif_x_matrix[i, i + 1] = +1

    return dif_x_matrix


def _difference_matrix_y_direction(number_spatial_x, number_spatial_y):
    N = number_spatial_x * number_spatial_y
    dif_y_matrix = pydata_sparse.DOK((N, N), dtype=float)
    for i in range(N):
        y = i // number_spatial_x
        if y > 0:
            dif_y_matrix[i, i - number_spatial_x] = -1
        if y < number_spatial_y - 1:
            dif_y_matrix[i, i + number_spatial_x] = +1

    return dif_y_matrix


class FiniteDifference2DDataset(Dataset):
    def __init__(
        self,
        name,
        pretty_name,
        suites,
        Nx,
        dx,
        Ny,
        dy,
        Nt,
        dt,
    ):
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
    def __init__(self, flux_x=None, flux_y=None):
        self._flux_x = flux_x
        self._flux_y = flux_y

    def flux_x(self, u):
        if self._flux_x is None:
            raise ValueError("FiniteDifference2DGenerator requires flux_x for checks")
        return self._flux_x(u)

    def flux_y(self, u):
        if self._flux_y is None:
            raise ValueError("FiniteDifference2DGenerator requires flux_y for checks")
        return self._flux_y(u)

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
        rng = np.random.default_rng(0)
        idx = rng.choice(dataset.Nx * dataset.Ny, size=k, replace=False)
        # small random amplitudes to avoid nonlinear overflow
        u_0[idx] = rng.random(k) * 0.5
        # a modest central pulse, previously was 10 which caused instability
        center = (dataset.Ny // 2) * dataset.Nx + (dataset.Nx // 2)
        u_0[center] = max(u_0[center], 1.0)

        inputs = [
            u_0,
            _lax_freidrichs_matrix_no_flux_2D(dataset.Nx, dataset.Ny),
            _difference_matrix_x_direction(dataset.Nx, dataset.Ny),
            _difference_matrix_y_direction(dataset.Nx, dataset.Ny),
        ]

        data = [_to_binsparse(item) for item in inputs]

        meta = {
            "timesteps": dataset.Nt,
            "dt": dataset.dt,
            "dx": dataset.dx,
            "dy": dataset.dy,
        }
        return DataInstance(
            inputs=data,
            meta=meta,
        )


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
        return [
            FiniteDifference2DGenerator(flux_x=self.flux_x, flux_y=self.flux_y),
        ]

    def flux_x(self, u):
        raise NotImplementedError

    def flux_y(self, u):
        raise NotImplementedError

    def check(self, param):
        super().check(param)
        result = _from_binsparse(self._output[0])
        u0 = _from_binsparse(self._input[0])
        dt = self._meta["dt"]
        dx = self._meta["dx"]
        dy = self._meta["dy"]
        Nx = param.dataset.Nx
        Ny = param.dataset.Ny

        assert np.allclose(result[0], u0, rtol=1e-12, atol=1e-12)

        time_derivative = np.diff(result, axis=0) / dt
        for timestep in range(time_derivative.shape[0]):
            u_n = result[timestep]
            u_grid = u_n.reshape(Ny, Nx)
            flux_x = param.generator.flux_x(u_n).reshape(Ny, Nx)
            flux_y = param.generator.flux_y(u_n).reshape(Ny, Nx)

            neighbor_average = np.zeros_like(u_grid)
            neighbor_average[:, 1:] += 0.25 * u_grid[:, :-1]
            neighbor_average[:, :-1] += 0.25 * u_grid[:, 1:]
            neighbor_average[1:, :] += 0.25 * u_grid[:-1, :]
            neighbor_average[:-1, :] += 0.25 * u_grid[1:, :]

            flux_difference_x = np.zeros_like(u_grid)
            flux_difference_x[:, 1:] -= flux_x[:, :-1]
            flux_difference_x[:, :-1] += flux_x[:, 1:]

            flux_difference_y = np.zeros_like(u_grid)
            flux_difference_y[1:, :] -= flux_y[:-1, :]
            flux_difference_y[:-1, :] += flux_y[1:, :]

            flux_derivative = (
                flux_difference_x / (2 * dx) + flux_difference_y / (2 * dy)
            )
            smoothing_derivative = (neighbor_average - u_grid) / dt
            assert np.allclose(
                time_derivative[timestep],
                (smoothing_derivative - flux_derivative).ravel(),
                rtol=1e-12,
                atol=1e-12,
            ), f"{param.dataset.name} has an inconsistent discrete derivative"


class BurgersFiniteDifference2DBenchmark(_FiniteDifference2DBenchmarkBase):
    @property
    def name(self) -> str:
        return "burgers_finite_difference_2d"

    @property
    def pretty_name(self) -> str:
        return "2D Finite Difference (Burgers flux)"

    def flux_x(self, u):
        return 0.5 * u * u

    def flux_y(self, u):
        return (1 / 3) * u * u

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
            fl_x = self.flux_x(u_n)
            fl_y = self.flux_y(u_n)
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

    def flux_x(self, u):
        sq = u * u
        denom = sq + (0.25 * (1 - u) * (1 - u))
        return sq / denom

    def flux_y(self, u):
        return self.flux_x(u)

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
            fl_x = self.flux_x(u_n)
            fl_y = self.flux_y(u_n)
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

    def flux_x(self, u):
        return self.CX * u

    def flux_y(self, u):
        return self.CY * u

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
            fl_x = self.flux_x(u_n)
            fl_y = self.flux_y(u_n)
            u_next = matrix @ u_n - alpha * (diff_x @ fl_x) - beta * (diff_y @ fl_y)
            u[n + 1] = u_next

        return [u]
