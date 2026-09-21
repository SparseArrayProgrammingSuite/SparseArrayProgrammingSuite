import numpy as np

import sparse as pydata_sparse
from binsparse import BinsparseTensor
from binsparse.conversions import from_numpy, from_sparse, to_numpy, to_sparse

from saps.benchmark import (
    Author,
    Benchmark,
    Contributor,
    DataInstance,
    Dataset,
    Generator,
    Ref,
)


def _from_binsparse(array):
    try:
        return to_numpy(array)
    except TypeError:
        return to_sparse(array).todense()


def _to_binsparse(array):
    if isinstance(array, BinsparseTensor):
        return array
    if isinstance(array, pydata_sparse.SparseArray):
        return from_sparse(array.asformat("coo"))
    return from_numpy(np.asarray(array))


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


#: Flux functions, keyed by name. Functions can't be serialized in
#: benchmark metadata, so the generator stores this keyword instead and
#: both the generator and the benchmark look the functions up by name.
_FLUX_PRETTY_NAMES = {
    "burgers": "Burgers",
    "buckley_leverett": "Buckley-Leverett",
    "linear_advection": "Linear Advection",
}

_LINEAR_ADVECTION_CX = 0.9
_LINEAR_ADVECTION_CY = 0.9


def _burgers_flux_x(u):
    return 0.5 * u * u


def _burgers_flux_y(u):
    return (1 / 3) * u * u


def _buckley_leverett_flux(u):
    sq = u * u
    return sq / (sq + 0.25 * (1 - u) * (1 - u))


def _linear_advection_flux_x(u):
    return _LINEAR_ADVECTION_CX * u


def _linear_advection_flux_y(u):
    return _LINEAR_ADVECTION_CY * u


def _resolve_flux(flux_name):
    match flux_name:
        case "burgers":
            return _burgers_flux_x, _burgers_flux_y
        case "buckley_leverett":
            return _buckley_leverett_flux, _buckley_leverett_flux
        case "linear_advection":
            return _linear_advection_flux_x, _linear_advection_flux_y
        case _:
            raise NotImplementedError(f"Unknown flux_name: {flux_name!r}")


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
    def __init__(self, flux_name):
        self.flux_name = flux_name

    @property
    def name(self) -> str:
        return f"finite_difference_inputs_2d_{self.flux_name}"

    @property
    def pretty_name(self) -> str:
        return f"Finite Difference Data Generator ({_FLUX_PRETTY_NAMES[self.flux_name]} flux)"

    @property
    def description(self) -> str:
        return (
            "The finite difference generator uses a finite difference grid of"
            "500 by 500 cells, matching roughly the scale of a"
            "real finite difference problem, Norris/torso3 from UF Matrix Collection,"
            f" using the {_FLUX_PRETTY_NAMES[self.flux_name]} flux function."
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
                title="The university of Florida sparse matrix collection",
                authors=[
                    Author("Timothy A. Davis"),
                    Author("Yifan Hu"),
                ],
                journal="ACM Transactions on Mathematical Software",
                publisher="Association for Computing Machinery (ACM)",
                volume="38",
                number="1",
                pages="1-25",
                year=2011,
                url="https://doi.org/10.1145/2049662.2049663",
                doi="10.1145/2049662.2049663",
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
                name=f"fd2d_test_scale_{self.flux_name}",
                pretty_name="2D Finite Difference Test Problem",
                suites=["test"],
                Nx=100,
                dx=0.1,
                Ny=100,
                dy=0.1,
                Nt=100,
                dt=0.01,
            ),
            FiniteDifference2DDataset(
                name=f"fd2d_realistic_scale_{self.flux_name}",
                pretty_name="2D Finite Difference Realistic Problem",
                suites=["standard"],
                Nx=1000,
                dx=0.1,
                Ny=1000,
                dy=0.1,
                Nt=1000,
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
            "flux_name": self.flux_name,
        }
        return DataInstance(
            inputs=data,
            meta=meta,
        )


class FiniteDifference2DBenchmark(Benchmark):
    @property
    def name(self) -> str:
        return "finite_difference_2d"

    @property
    def pretty_name(self) -> str:
        return "2D Finite Difference"

    @property
    def suites(self) -> list[str]:
        return []

    @property
    def concepts(self) -> str:
        return (
            """
            <ccs2012>
            <concept>
            <concept_id>10002950.10003705.10011686</concept_id>
            <concept_desc>Mathematics of computing~"""
            "Mathematical software performance"
            """</concept_desc>
            <concept_significance>500</concept_significance>
            </concept>
            <concept>
            <concept_id>10010147.10010341.10010349.10010357</concept_id>
            <concept_desc>Computing methodologies~Continuous simulation</concept_desc>
            <concept_significance>500</concept_significance>
            </concept>
            <concept>
            <concept_id>10002950.10003714.10003715.10003750</concept_id>
            <concept_desc>Mathematics of computing~Discretization</concept_desc>
            <concept_significance>500</concept_significance>
            </concept>
            </ccs2012>
        """
        )

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
            FiniteDifference2DGenerator(flux_name="burgers"),
            FiniteDifference2DGenerator(flux_name="buckley_leverett"),
            FiniteDifference2DGenerator(flux_name="linear_advection"),
        ]

    def benchmark(self, xp, data: list, meta: dict):
        u_0, matrix, diff_x, diff_y = data
        timesteps = meta["timesteps"]
        dt = meta["dt"]
        dx = meta["dx"]
        dy = meta["dy"]
        flux_x, flux_y = _resolve_flux(meta["flux_name"])

        Nt = timesteps + 1
        u = xp.zeros((Nt, u_0.shape[0]), dtype=u_0.dtype)
        u[0] = u_0

        alpha = dt / (2 * dx)
        beta = dt / (2 * dy)

        for n in range(Nt - 1):
            u_n = u[n]
            fl_x = flux_x(u_n)
            fl_y = flux_y(u_n)
            u_next = matrix @ u_n - alpha * (diff_x @ fl_x) - beta * (diff_y @ fl_y)
            u[n + 1] = u_next

        return [u]

    def check(self, param):
        super().check(param)
        result = _from_binsparse(self._output[0])
        u0 = _from_binsparse(self._input[0])
        dt = self._meta["dt"]
        dx = self._meta["dx"]
        dy = self._meta["dy"]
        Nx = param.dataset.Nx
        Ny = param.dataset.Ny
        flux_x_fn, flux_y_fn = _resolve_flux(self._meta["flux_name"])

        assert np.allclose(result[0], u0, rtol=1e-12, atol=1e-12)

        time_derivative = np.diff(result, axis=0) / dt
        for timestep in range(time_derivative.shape[0]):
            u_n = result[timestep]
            u_grid = u_n.reshape(Ny, Nx)
            flux_x = flux_x_fn(u_n).reshape(Ny, Nx)
            flux_y = flux_y_fn(u_n).reshape(Ny, Nx)

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

            flux_derivative = flux_difference_x / (2 * dx) + flux_difference_y / (
                2 * dy
            )
            smoothing_derivative = (neighbor_average - u_grid) / dt
            assert np.allclose(
                time_derivative[timestep],
                (smoothing_derivative - flux_derivative).ravel(),
                rtol=1e-12,
                atol=1e-12,
            ), f"{param.dataset.name} has an inconsistent discrete derivative"
