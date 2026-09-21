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


def _lax_freidrichs_matrix_no_flux(Nx):
    matrix = pydata_sparse.DOK((Nx, Nx), dtype=float)
    for i in range(1, Nx):
        matrix[i, i - 1] = 0.5
    for i in range(Nx - 1):
        matrix[i, i + 1] = 0.5

    # periodic BC
    matrix[0, -1] = 0.5
    matrix[-1, 0] = 0.5

    return matrix


def _difference_matrix(Nx):
    matrix = pydata_sparse.DOK((Nx, Nx), dtype=float)
    for i in range(1, Nx):
        matrix[i, i - 1] = -1
    for i in range(Nx - 1):
        matrix[i, i + 1] = 1

    # periodic BC
    matrix[0, -1] = -1
    matrix[-1, 0] = 1
    return matrix


#: Flux functions, keyed by name. Functions can't be serialized in
#: benchmark metadata, so the generator stores this keyword instead and
#: both the generator and the benchmark look the function up by name.
_FLUX_PRETTY_NAMES = {
    "burgers": "Burgers",
    "buckley_leverett": "Buckley-Leverett",
    "linear_advection": "Linear Advection",
}


def _burgers_flux(u):
    return 0.5 * u * u


def _buckley_leverett_flux(u):
    sq = u * u
    return sq / (sq + 0.25 * (1 - u) * (1 - u))


def _linear_advection_flux(u):
    return 1.0 * u


def _resolve_flux(flux_name):
    match flux_name:
        case "burgers":
            return _burgers_flux
        case "buckley_leverett":
            return _buckley_leverett_flux
        case "linear_advection":
            return _linear_advection_flux
        case _:
            raise NotImplementedError(f"Unknown flux_name: {flux_name!r}")


class FiniteDifferenceDataset(Dataset):
    def __init__(self, name, pretty_name, suites, Nx, dx, Nt, dt):
        self._name = name
        self._pretty_name = pretty_name
        self._suites = suites
        self.Nx = Nx
        self.dx = dx
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
        return f"{self.pretty_name}: Nx = {self.Nx}, dx = {self.dx}, dt = {self.dt}."

    @property
    def suites(self) -> list[str]:
        return self._suites

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"


class FiniteDifferenceGenerator(Generator[FiniteDifferenceDataset]):
    def __init__(self, flux_name):
        self.flux_name = flux_name

    @property
    def name(self) -> str:
        return f"finite_difference_inputs_{self.flux_name}"

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
    def datasets(self) -> list[FiniteDifferenceDataset]:
        return [
            FiniteDifferenceDataset(
                name="fd_test_scale",
                pretty_name="Finite Difference Test Problem",
                suites=["test", "trace"],
                Nx=100,
                dx=0.1,
                Nt=100,
                dt=0.01,
            ),
            FiniteDifferenceDataset(
                name="fd_realistic_scale",
                pretty_name="Finite Difference Realistic Problem",
                suites=["standard"],
                Nx=250000,
                dx=0.1,
                Nt=1000,
                dt=0.01,
            ),
        ]

    def generate(self, dataset: FiniteDifferenceDataset):
        # Produce a gentle, sparse initial condition (small amplitudes)
        density = 0.05
        u_0 = np.zeros(dataset.Nx, dtype=float)
        k = max(1, int(dataset.Nx * density))
        rng = np.random.default_rng(0)
        idx = rng.choice(dataset.Nx, size=k, replace=False)
        # small random amplitudes to avoid nonlinear overflow
        u_0[idx] = rng.random(k) * 0.5
        # a modest central pulse (order 1), previously was 10 which caused instability
        u_0[dataset.Nx // 2] = max(u_0[dataset.Nx // 2], 1.0)

        difference = _difference_matrix(dataset.Nx)
        matrix = _lax_freidrichs_matrix_no_flux(dataset.Nx)

        data = [
            _to_binsparse(u_0),
            _to_binsparse(matrix),
            _to_binsparse(difference),
        ]

        meta = {
            "timesteps": dataset.Nt,
            "dt": dataset.dt,
            "dx": dataset.dx,
            "flux_name": self.flux_name,
        }
        return DataInstance(inputs=data, meta=meta)


class FiniteDifferenceBenchmark(Benchmark):
    @property
    def name(self) -> str:
        return "finite_difference"

    @property
    def pretty_name(self) -> str:
        return "1D Finite Difference"

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
            FiniteDifferenceGenerator(flux_name="burgers"),
            FiniteDifferenceGenerator(flux_name="buckley_leverett"),
            FiniteDifferenceGenerator(flux_name="linear_advection"),
        ]

    def benchmark(self, xp, data: list, meta: dict):
        u_0, matrix, dif = data
        timesteps = meta["timesteps"]
        dt = meta["dt"]
        dx = meta["dx"]
        flux = _resolve_flux(meta["flux_name"])
        Nt = timesteps + 1
        alpha = dt / (2 * dx)
        u = xp.zeros((Nt, u_0.shape[0]))
        u[0] = u_0
        for n in range(Nt - 1):
            u_n = u[n]
            f = flux(u_n)
            u_next = matrix @ u_n - alpha * (dif @ f)
            u[n + 1] = u_next
        return [u]

    def check(self, param):
        super().check(param)
        result = _from_binsparse(self._output[0])
        u0 = _from_binsparse(self._input[0])
        dt = self._meta["dt"]
        dx = self._meta["dx"]
        flux_fn = _resolve_flux(self._meta["flux_name"])

        assert np.allclose(result[0], u0, rtol=1e-12, atol=1e-12)

        time_derivative = np.diff(result, axis=0) / dt
        for timestep in range(time_derivative.shape[0]):
            u_n = result[timestep]
            flux = flux_fn(u_n)

            neighbor_average = np.zeros_like(u_n)
            neighbor_average[1:] += 0.5 * u_n[:-1]
            neighbor_average[:-1] += 0.5 * u_n[1:]
            neighbor_average[0] += 0.5 * u_n[-1]
            neighbor_average[-1] += 0.5 * u_n[0]

            flux_difference = np.zeros_like(u_n)
            flux_difference[1:] -= flux[:-1]
            flux_difference[:-1] += flux[1:]
            flux_difference[0] -= flux[-1]
            flux_difference[-1] += flux[0]

            flux_derivative = flux_difference / (2 * dx)
            smoothing_derivative = (neighbor_average - u_n) / dt
            assert np.allclose(
                time_derivative[timestep],
                smoothing_derivative - flux_derivative,
                rtol=1e-12,
                atol=1e-12,
            ), f"{param.dataset.name} has an inconsistent discrete derivative"
