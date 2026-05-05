import os
from typing import Any

import numpy as np
from scipy.io import mmread
from scipy.sparse import random

import ssgetpy

import sparseappbench
from sparseappbench.benchmark import (
    Author,
    Benchmark,
    Contributor,
    Dataset,
    Generator,
    Ref,
)
from saps_framework import BinsparseFormat

xp = sparseappbench.xp

def burgers_flux(u):
    return 0.5 * u * u


def buckley_leverett_flux(u):
    sq = u * u
    return sq / (sq + (0.25 * (1 - u) * (1 - u)))


def linear_advection_flux(c):
    def flux(u):
        return c * u

    return flux


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
    def __init__(
        self, name, pretty_name, tags, Nx, dx, Nt, dt, flux
    ):
        self._name = name
        self._pretty_name = pretty_name
        self._tags = tags
        self.Nx = Nx
        self.dx = dx
        self.Nt = Nt
        self.dt = dt
        self.flux = flux

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
    def tags(self) -> list[str]:
        return self._tags

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
            "The purpose of this is to analyze the importance of numerical methods for PDEs, "
            "and applications sparse array theory into these method, through the form of benchmarks."
            "This paticular benchmark analyzes the use of the Lax–Friedrichs method for solving"
            "nonlinear hyberbolic PDEs, with numerical stability and accuracy not seen in FTCS."
            "This benchmark will run a simulation using both Lax–Friedrichs and analyze"
            "core concepts such as numerical stability, conservation law consistency, etc."
        )

    @property
    def tags(self) -> list[str]:
        return ["simulation", "sparse"]

    @property
    def authors(self) -> list[Contributor]:
        return [Contributor("Vilohith Gokarakonda", "vgokarakonda3@gatech.edu")]

    @property
    def references(self) -> list[Ref]:
        return [
            Ref(
                title = "Synthesizing Sound and Precise Abstract Transformers for Nonlinear Hyperbolic PDE Solvers.",
                authors = [
                    Author("Laurel, J."),
                    Author("Laguna, I."),
                    Author("Hückelheim, J."),
                ],
                journal = "Proceedings of the ACM on Programming Languages",
                volume = "9",
                number = "OOPSLA2",
                pages = "1063–1091",
                year = 2025,
                url = "https://doi.org/10.1145/3763088",
            )
        ]

    @property
    def ai_disclosure(self) -> str:
        return (
            "No generative AI was used to construct the benchmark function"
            "itself. Generative AI might have been used to construct tests. This statement"
            "was written by hand."
        )

    @property
    def motivation(self) -> str:
        return (
            "For linear advection, updates are done "
            "using a sparse matrix representation, to updates the spatial coordinates for time t."
        )

    @property
    def datasets(self) -> list[FiniteDifferenceDataset]:
        return [
            FiniteDifferenceDataset(
                name="buckley_leverett_small",
                pretty_name="Buckley-Leverett small",
                tags=["small"],
                Nx=100,
                dx=0.1,
                Nt=100,
                dt=0.01,
                flux=buckley_leverett_flux,
            ),
            FiniteDifferenceDataset(
                name="burgers_small",
                pretty_name="Burgers small",
                tags=["small"],
                Nx=100,
                dx=0.1,
                Nt=100,
                dt=0.01,
                flux=burgers_flux,
            ),
             FiniteDifferenceDataset(
                name="linear_advection_small",
                pretty_name="Linear Advection small",
                tags=["small"],
                Nx=100,
                dx=0.1,
                Nt=100,
                dt=0.01,
                flux=linear_advection_flux(1),
            ),
        ]

    def generate(self, dataset: FiniteDifferenceDataset):
        # Produce a gentle, sparse initial condition (small amplitudes)
        density = 0.05
        u_0 = np.zeros(dataset.Nx, dtype=float)
        k = max(1, int(dataset.Nx * density))
        idx = np.random.choice(dataset.Nx, size=k, replace=False)
        # small random amplitudes to avoid nonlinear overflow
        u_0[idx] = np.random.rand(k) * 0.5
        # a modest central pulse (order 1), previously was 10 which caused instability
        u_0[dataset.Nx // 2] = max(u_0[dataset.Nx // 2], 1.0)

        difference = _difference_matrix(dataset.Nx)
        matrix = _lax_freidrichs_matrix_no_flux(dataset.Nx)

        data = (xp.to_binsparse(u_0), xp.to_binsparse(matrix), xp.to_binsparse(difference))

        meta = {
            "flux": dataset.flux,
            "timesteps": dataset.Nt,
            "dt": dataset.dt,
            "dx": dataset.dx,
        }
        return data, meta

class FiniteDifferenceBenchmark(Benchmark):
    @property
    def tag(self) -> str:
        return "finite_difference"

    @property
    def name(self) -> str:
        return "finite_difference"

    @property
    def pretty_name(self) -> str:
        return "1D Finite Difference Method"

    @property
    def description(self) -> str:
        return (
            "The purpose of this is to analyze the importance of numerical methods for PDEs, "
            "and applications sparse array theory into these method, through the form of benchmarks."
            "This paticular benchmark analyzes the use of the Lax–Friedrichs method for solving"
            "nonlinear hyberbolic PDEs, with numerical stability and accuracy not seen in FTCS."
            "This benchmark will run a simulation using both Lax–Friedrichs and analyze"
            "core concepts such as numerical stability, conservation law consistency, etc."
        )

    @property
    def tags(self) -> list[str]:
        return ["simulation", "sparse"]

    @property
    def authors(self) -> list[Contributor]:
        return [Contributor("Vilohith Gokarakonda", "vgokarakonda3@gatech.edu")]


    @property
    def references(self) -> list[Ref]:
        return [
            Ref(
                title = "Synthesizing Sound and Precise Abstract Transformers for Nonlinear Hyperbolic PDE Solvers.",
                authors = [
                    Author("Laurel, J."),
                    Author("Laguna, I."),
                    Author("Hückelheim, J."),
                ],
                journal = "Proceedings of the ACM on Programming Languages",
                volume = "9",
                number = "OOPSLA2",
                pages = "1063–1091",
                year = 2025,
                url = "https://doi.org/10.1145/3763088",
            )
        ]

    @property
    def ai_disclosure(self) -> str:
        return (
            "No generative AI was used to construct the benchmark function"
            "itself. Generative AI might have been used to construct tests. This statement"
            "was written by hand."
        )

    @property
    def motivation(self) -> str:
        return (
            "Updates are done using a matrix representation, to updates the spatial coordinates for time t."
        )

    @property
    def generators(self):
        return [FiniteDifferenceGenerator()]
        
    def benchmark(self, data: list, meta: dict):
        u0_bench, matrix_bench, difference_bench = data
        flux = meta["flux"]
        timesteps = meta["timesteps"]
        dt = meta["dt"]
        dx = meta["dx"]
        u_0 = u0_bench
        matrix = matrix_bench
        dif = difference_bench
        Nt = timesteps + 1
        alpha = (dt) / (2 * dx)
        u = xp.zeros((Nt, u_0.shape[0]))
        u[0] = u_0
        for n in range(Nt - 1):
            u_n = u[n]
            f = flux(u_n)
            u_next = matrix @ u_n - alpha * (dif @ f)
            u[n + 1] = u_next
        return xp.to_binsparse(u)