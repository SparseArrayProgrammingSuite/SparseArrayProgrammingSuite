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


def linear_advection_flux_2D(cx, cy):
    def flux_x(u):
        return cx * u

    def flux_y(u):
        return cy * u

    return flux_x, flux_y


# Did it like this to see if there are mistakes in my stencil
def aniso_burgers_flux_2D():
    def burgers_x(u):
        return 0.5 * u * u

    def burgers_y(u):
        return (1 / 3) * u * u

    return burgers_x, burgers_y


def buckley_leverett_flux_2D():
    def buckley_leverett_flux_x(u):
        sq = u * u
        return sq / (sq + (0.25 * (1 - u) * (1 - u)))

    def buckley_leverett_flux_y(u):
        sq = u * u
        return sq / (sq + (0.25 * (1 - u) * (1 - u)))

    return buckley_leverett_flux_x, buckley_leverett_flux_y


def lax_freidrichs_data_generator(xp, number_spatial_x, number_spatial_y, density):
    u_0 = xp.zeros(number_spatial_x * number_spatial_y)
    step = int(1 / density)
    indices = xp.arange(0, number_spatial_x * number_spatial_y, step)

    u_0[indices] = 1
    return xp.lazy(u_0)


# This matrix formula assume Dirichlet BC instead of Periodic BC.
def lax_freidrichs_matrix_no_flux(xp, number_spatial_x, number_spatial_y):
    N = number_spatial_x * number_spatial_y
    matrix = xp.zeros((N, N))
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

    return xp.lazy(matrix)


def difference_matrix_x_direction(xp, number_spatial_x, number_spatial_y):
    N = number_spatial_x * number_spatial_y
    dif_x_matrix = xp.zeros((N, N))
    for i in range(N):
        x = i % number_spatial_x
        if x > 0:
            dif_x_matrix[i, i - 1] = -1
        if x < number_spatial_x - 1:
            dif_x_matrix[i, i + 1] = +1

    return xp.lazy(dif_x_matrix)


def difference_matrix_y_direction(xp, number_spatial_x, number_spatial_y):
    N = number_spatial_x * number_spatial_y
    dif_y_matrix = xp.zeros((N, N))
    for i in range(N):
        y = i // number_spatial_x
        if y > 0:
            dif_y_matrix[i, i - number_spatial_x] = -1
        if y < number_spatial_y - 1:
            dif_y_matrix[i, i + number_spatial_x] = +1

    return xp.lazy(dif_y_matrix)


def lax_friedrichs_solver_matrix_2d(
    xp,
    u0_bench,
    matrix_bench,
    diff_x_bench,
    diff_y_bench,
    timesteps,
    flux_x,
    flux_y,
    dt,
    dx,
    dy,
):
    u_0 = xp.lazy(u0_bench)
    matrix = xp.lazy(matrix_bench)
    diff_x = xp.lazy(diff_x_bench)
    diff_y = xp.lazy(diff_y_bench)

    Nt = timesteps + 1
    u = xp.zeros((Nt, u_0.shape[0]))
    u[0] = u_0

    alpha = dt / (2 * dx)
    beta = dt / (2 * dy)

    for n in range(Nt - 1):
        u_n = u[n]

        fl_x = flux_x(u_n)
        fl_y = flux_y(u_n)

        u_next = matrix @ u_n - alpha * (diff_x @ fl_x) - beta * (diff_y @ fl_y)

        u[n + 1] = u_next

    return xp.to_benchmark(u)
