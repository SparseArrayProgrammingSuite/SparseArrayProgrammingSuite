"""
Name: Finite Difference Simulation 2D
Author: Vilohith Gokarakonda
Email: vgokarakonda3@gatech.edu
Motivation (Importance of problem with citation):
The purpose of this is to analyze the importance of numerical methods for PDEs,
and applications sparse array theory into these method, through the form of benchmarks.
This paticular benchmark analyzes the use of the Lax–Friedrichs method for solving
nonlinear hyberbolic PDEs, with numerical stability and accuracy not seen in FTCS.
This benchmark will run a simulation using both Lax–Friedrichs and analyze
core concepts such as numerical stability, conservation law consistency, etc.
This is similar to the other benchmark, but in a 2D case.
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


def lax_freidrichs_data_generator(xp, number_spatial_x, number_spatial_y, density):
    u_0 = xp.zeros(number_spatial_x * number_spatial_y)
    step = int(1 / density)
    indices = xp.arange(0, number_spatial_x * number_spatial_y, step)

    u_0[indices] = 1
    return xp.lazy(u_0)


# This is based on 2D Lax-freidrichs In matrix form.
# Since we would need to symbolically know the flux with the constant out front.
def lax_freidrichs_matrix(
    xp, number_spatial_x, number_spatial_y, dx, dt, dy, const_x=1, const_y=1
):
    N = number_spatial_x * number_spatial_y
    matrix = xp.zeros((N, N))
    alpha = (const_x * dt) / (2 * dx)
    beta = (const_y * dt) / (2 * dy)
    for i in range(1, N):
        if i % number_spatial_x != 0:
            matrix[i, i - 1] = 0.25 + alpha

    for i in range(N - 1):
        if (i + 1) % number_spatial_x != 0:
            matrix[i, i + 1] = 0.25 - alpha

    for j in range(number_spatial_y, N):
        matrix[j, j - number_spatial_y] = 0.25 + beta
    for j in range(N - number_spatial_y):
        matrix[j, j + number_spatial_y] = 0.25 + beta

    # Ignore Periodic BC for now, will add later.

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
