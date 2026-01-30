import numpy as np
import scipy as sp

"""
Name: Finite Difference Simulation
Author: Vilohith Gokarakonda
Email: vgokarakonda3@gatech.edu
Motivation (Importance of problem with citation):
The purpose of this is to analyze the importance of numerical methods for PDEs,
and applications sparse array theory into these method, through the form of benchmarks.
This paticular benchmark analyzes the use of the Lax–Friedrichs method for solving
nonlinear hyberbolic PDEs, with numerical stability and accuracy not seen in FTCS.
This benchmark will run a simulation using both Lax–Friedrichs and analyze
core concepts such as numerical stability, comparision to FCTS, etc.
Citation:
Laurel, J., Laguna, I., & Hückelheim, J. (2025).
Synthesizing Sound and Precise Abstract Transformers for
Nonlinear Hyperbolic PDE Solvers.
Proceedings of the ACM on Programming Languages,
9(OOPSLA2), 1063–1091. https://doi.org/10.1145/3763088

Role of sparsity (How sparsity is used in the problem):
The intial conditions are sparse.
Implementation (Where did the reference algorithm come from? With citation.):
Hand-written, direct call to array api function
https://data-apis.org/array-api/latest/API_specification/generated/array_api.matmul.html
Data Generation (How is the data generated? Why is it realistic?):
Sparse-sparse matrix multiplication is sensitive to sparsity patterns and their
interaction. We use random sparsity patterns for now.  Statement on the use of
Generative AI: No generative AI was used to construct the benchmark function
itself. Generative AI might have been used to construct tests. This statement
was written by hand.
"""


def burgers_flux(u):
    return 0.5 * u * u


def buckley_leverett_flux(u):
    sq = u * u
    return sq / (sq + (0.25 * (1 - u) * (1 - u)))


def lax_friedrichs_solver(xp, u_0_bench, dt, dx, flux, timesteps, boundary="dirchlet"):
    u_0 = xp.lazy(u_0_bench)
    flux_0 = flux(u_0)

    u = xp.zeros((timesteps + 1, u_0.shape[0]))  # Intializes the space-time grid
    fluxes = xp.zeros((timesteps + 1, flux_0.shape[0]))

    u = xp.assign(u, 0, u_0)  # assign the intial conditions to the u. like u[0] = u_0
    fluxes = xp.assign(fluxes, 0, flux_0)

    alpha = dt / (2 * dx)
    for n in range(timesteps, dt):
        u_n = u[n]
        # Vector equivalent of doing
        # u[t+1][x] = 0.5(u[t][n+1] - u[t][n-1]) -  alpha (flux(u[t][n+1])
        # - flux(u[t][n-1]))
        u_prev_spatial = xp.roll(u_n, -1)
        u_next_spatial = xp.roll(u_n, 1)
        u_next = 0.5 * (u_next_spatial - u_prev_spatial) - alpha * (
            flux(u_next_spatial) - flux(u_prev_spatial)
        )

        # Accounts for Dirichlet Boundary
        if boundary == "dirichlet":
            u_next = xp.assign(u_next, 0, u_n[0])
            u_next = xp.assign(u_next, -1, u_n[-1])

        u = xp.assign(u, n + 1, u_next)
        fluxes = xp.assign(fluxes, n + 1, flux(u_next))

    return xp.to_benchmark(fluxes)


def lax_freidrichs_data_generator(xp, number_spatial, seed=0.4, density=0.05):
    rng = np.random.default_rng(seed)
    u_0 = sp.sparse(number_spatial, 1, density=density, data_rvs=rng.standard_normal)
    return xp.lazy(u_0)
