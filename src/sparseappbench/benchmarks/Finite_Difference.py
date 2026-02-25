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
core concepts such as numerical stability, conservation law consistency, etc.
Citation:
Laurel, J., Laguna, I., & Hückelheim, J. (2025).
Synthesizing Sound and Precise Abstract Transformers for
Nonlinear Hyperbolic PDE Solvers.
Proceedings of the ACM on Programming Languages,
9(OOPSLA2), 1063–1091. https://doi.org/10.1145/3763088

Role of sparsity (How sparsity is used in the problem):
The intial conditions are sparse. Furthermore, for linear advection, updates are done
using a matrix representation, to updates the spatial coordinates for time t.
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


def linear_advection_flux(c):
    def flux(u):
        return c * u

    return flux


def lax_friedrichs_solver(xp, u0_bench, dt, dx, flux, timesteps):
    u_0 = xp.lazy(u0_bench)

    Nt = timesteps + 1

    # Intializes the space-time grid
    u = xp.zeros((Nt, int(u_0.shape[0])))

    u[0] = u_0

    alpha = dt / (2 * dx)
    for n in range(Nt - 1):
        u_n = u[n]
        # Vector equivalent of doing
        # u[t+1][x] = 0.5(u[t][n+1] - u[t][n-1]) -  alpha (flux(u[t][n+1])
        # - flux(u[t][n-1]))
        # Naturally incorporates periodic BC.
        u_next_spatial = xp.roll(u_n, -1)  # u[i +1]
        u_prev_spatial = xp.roll(u_n, 1)  # u[i -1]
        u_next = 0.5 * (u_next_spatial + u_prev_spatial) - alpha * (
            flux(u_next_spatial) - flux(u_prev_spatial)
        )

        u[n + 1] = u_next
    return xp.to_benchmark(u)


# I made this deterministic
def lax_freidrichs_data_generator(xp, number_spatial, density):
    u_0 = xp.zeros(number_spatial)
    step = int(1 / density)
    indices = xp.arange(0, number_spatial, step)

    u_0[indices] = 1
    return xp.lazy(u_0)


# This can only work for when the flux  = const * u (linear advection)
# Since we would need to symbolically know the flux with the constant out front.
def lax_freidrichs_matrix(xp, number_spatial, dx, dt, const=1):
    Nx = number_spatial
    matrix = xp.zeros((Nx, Nx))
    alpha = (const * dt) / (2 * dx)
    for i in range(1, Nx):
        matrix[i, i - 1] = 0.5 + alpha
    for i in range(Nx - 1):
        matrix[i, i + 1] = 0.5 - alpha

    # periodic BC
    matrix[0, -1] = 0.5 + alpha
    matrix[-1, 0] = 0.5 - alpha

    return xp.lazy(matrix)


def lax_friedrichs_solver_matrix(xp, u0_bench, matrix_bench, timesteps):
    u_0 = xp.lazy(u0_bench)
    matrix = xp.lazy(matrix_bench)
    Nt = timesteps + 1

    u = xp.zeros((Nt, u_0.shape[0]))
    u[0] = u_0
    for n in range(Nt - 1):
        u_n = u[n]
        u_next = matrix @ u_n  # matrix multiply
        u[n + 1] = u_next
    return xp.to_benchmark(u)
