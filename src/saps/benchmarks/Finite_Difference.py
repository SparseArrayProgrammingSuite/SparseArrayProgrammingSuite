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


def _lax_freidrichs_matrix_no_flux(Nx):
    matrix = np.zeros((Nx, Nx))
    for i in range(1, Nx):
        matrix[i, i - 1] = 0.5
    for i in range(Nx - 1):
        matrix[i, i + 1] = 0.5

    # periodic BC
    matrix[0, -1] = 0.5
    matrix[-1, 0] = 0.5

    return matrix


def _difference_matrix(Nx):
    matrix = np.zeros((Nx, Nx))
    for i in range(1, Nx):
        matrix[i, i - 1] = -1
    for i in range(Nx - 1):
        matrix[i, i + 1] = 1

    # periodic BC
    matrix[0, -1] = -1
    matrix[-1, 0] = 1
    return matrix


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
    @property
    def name(self) -> str:
        return "finite_difference_inputs"

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
    def datasets(self) -> list[FiniteDifferenceDataset]:
        return [
            FiniteDifferenceDataset(
                name="default",
                pretty_name="Default",
                suites=[],
                Nx=100,
                dx=0.1,
                Nt=100,
                dt=0.01,
            ),
        ]

    def generate(self, dataset: FiniteDifferenceDataset):
        # Produce a gentle, sparse initial condition (small amplitudes)
        density = 0.05
        u_0 = np.zeros(dataset.Nx, dtype=float)
        k = max(1, int(dataset.Nx * density))
        rng = np.random.default_rng()
        idx = rng.choice(dataset.Nx, size=k, replace=False)
        # small random amplitudes to avoid nonlinear overflow
        u_0[idx] = rng.random(k) * 0.5
        # a modest central pulse (order 1), previously was 10 which caused instability
        u_0[dataset.Nx // 2] = max(u_0[dataset.Nx // 2], 1.0)

        difference = _difference_matrix(dataset.Nx)
        matrix = _lax_freidrichs_matrix_no_flux(dataset.Nx)

        data = (
            xp.to_binsparse(u_0),
            xp.to_binsparse(matrix),
            xp.to_binsparse(difference),
        )

        meta = {
            "timesteps": dataset.Nt,
            "dt": dataset.dt,
            "dx": dataset.dx,
        }
        return data, meta


class _FiniteDifferenceBenchmarkBase(Benchmark):
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
        return [FiniteDifferenceGenerator()]


class BurgersFiniteDifferenceBenchmark(_FiniteDifferenceBenchmarkBase):
    @property
    def name(self) -> str:
        return "burgers_finite_difference"

    @property
    def pretty_name(self) -> str:
        return "1D Finite Difference (Burgers flux)"

    def benchmark(self, data: list, meta: dict):
        u_0, matrix, dif = data
        timesteps = meta["timesteps"]
        dt = meta["dt"]
        dx = meta["dx"]
        Nt = timesteps + 1
        alpha = dt / (2 * dx)
        u = xp.zeros((Nt, u_0.shape[0]))
        u[0] = u_0
        for n in range(Nt - 1):
            u_n = u[n]
            f = 0.5 * u_n * u_n
            u_next = matrix @ u_n - alpha * (dif @ f)
            u[n + 1] = u_next
        return [u]


class BuckleyLeverettFiniteDifferenceBenchmark(_FiniteDifferenceBenchmarkBase):
    @property
    def name(self) -> str:
        return "buckley_leverett_finite_difference"

    @property
    def pretty_name(self) -> str:
        return "1D Finite Difference (Buckley-Leverett flux)"

    def benchmark(self, data: list, meta: dict):
        u_0, matrix, dif = data
        timesteps = meta["timesteps"]
        dt = meta["dt"]
        dx = meta["dx"]
        Nt = timesteps + 1
        alpha = dt / (2 * dx)
        u = xp.zeros((Nt, u_0.shape[0]))
        u[0] = u_0
        for n in range(Nt - 1):
            u_n = u[n]
            sq = u_n * u_n
            f = sq / (sq + (0.25 * (1 - u_n) * (1 - u_n)))
            u_next = matrix @ u_n - alpha * (dif @ f)
            u[n + 1] = u_next
        return [u]


class LinearAdvectionFiniteDifferenceBenchmark(_FiniteDifferenceBenchmarkBase):
    C = 1.0

    @property
    def name(self) -> str:
        return "linear_advection_finite_difference"

    @property
    def pretty_name(self) -> str:
        return "1D Finite Difference (Linear Advection flux)"

    def benchmark(self, data: list, meta: dict):
        u_0, matrix, dif = data
        timesteps = meta["timesteps"]
        dt = meta["dt"]
        dx = meta["dx"]
        Nt = timesteps + 1
        alpha = dt / (2 * dx)
        u = xp.zeros((Nt, u_0.shape[0]))
        u[0] = u_0
        for n in range(Nt - 1):
            u_n = u[n]
            f = self.C * u_n
            u_next = matrix @ u_n - alpha * (dif @ f)
            u[n + 1] = u_next
        return [u]
