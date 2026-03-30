"""
Name: Brusselator PDE Solver
Author: Akarsh Duddu
Email: aduddu3@gatech.edu
Motivation (Importance of problem with citation):
Despite its significance, the computational cost of solving 
high-dimensional discretized versions of the cross-diffusive
Brusselator equation can be prohibitive, particularly
in parameter-dependent or long-time simulations.
Küçükseyhan
"Model order reduction for the cross-diffusive Brusselator Equation"
GSC Advanced Research and Reviews, 23 November 2024,
www.ssslab.cn/assets/papers/2025-jiang-boosting.pdf,
https://doi.org/10.30574/gscarr.2024.21.2.0452. Accessed 29 Mar. 2026.
Role of sparsity (How sparsity is used in the problem):
The Brusselator equation is discretized on a 2D grid,
resulting in a large sparse matrix representing the diffusion terms.
The sparsity of this matrix comes from the nature of this diffusion,
as each point only interacts with its immediate neighbors. 
Data Generation (How is the data generated? Why is it realistic?):
The data is generated numerically with the Brusselator PDE,
where the choice of parameters dictates the starting state
and the dynamics of the system.
AI Generation Statement:
No generative AI was used to write the benchmark function itself. Generative
AI was used for debugging. This statement was written by hand.
"""


import numpy as np


def forward_euler(
    xp,
    dydx,
    span,
    y0,
    first_step,
):
    """Forward Euler method of approximating ordinary differential equations (ODEs)."""
    # Builtin range function does not support floating-point step
    curr = span[0]
    inputs = []
    while curr < span[1]:
        inputs.append(curr)
        curr += first_step

    step = first_step
    outputs = [None for _ in inputs]
    outputs[0] = xp.array(y0, dtype=float).flatten()

    for i in range(1, len(inputs)):
        # y_new = y + dy/dx * delta x

        dydt_vector = xp.array(dydx(inputs[i - 1], outputs[i - 1]), dtype=np.complex128).flatten()
        outputs[i] = [outputs[i - 1][j] + dydt_vector[j] * step for j in range(len(y0))]

    return (inputs, outputs)


def limit(a, N):
    return a % N

# y0 = init_brusselator_2d()
def init_brusselator_2d(n):
    u = [0.0] * (n * n * 2)
    for i in range(n):
        for j in range(n):
            fi = i / (n - 1) if n > 1 else 0.0
            fj = j / (n - 1) if n > 1 else 0.0
            u[(i * n + j) * 2]     = float(np.real(22 * (fj * (1 - fj)) ** 1.5))
            u[(i * n + j) * 2 + 1] = float(np.real(27 * (fi * (1 - fi)) ** 1.5))
    return u

def brusselator_dydx(t, u_vec, n, a, b, alpha):
    size = n * n * 2
    brusselator_cb = [0.0] * size
    for i in range(n):
        for j in range(n):
            x = i / (n - 1)
            y = j / (n - 1)
            if (x - 0.3) ** 2 + (y - 0.6) ** 2 <= 0.1**2:
                brusselator_cb[(i * n + j) * 2] = 5

    C = np.zeros((size, size))

    for i in range(n):
        for j in range(n):
            u_idx = (i * n + j) * 2
            v_idx = u_idx + 1

            ip1, im1, jp1, jm1 = (
                limit(i + 1, n),
                limit(i - 1, n),
                limit(j + 1, n),
                limit(j - 1, n),
            )

            for ni, nj in [(ip1, j), (im1, j), (i, jp1), (i, jm1)]:
                C[u_idx][(ni * n + nj) * 2] += alpha
                C[v_idx][(ni * n + nj) * 2 + 1] += alpha

            C[u_idx][u_idx] -= (4 * alpha + (b + 1))
            C[v_idx][v_idx] -= 4 * alpha
            C[v_idx][u_idx] += b

    u_arr = np.array(u_vec, dtype=float)
    
    # Linear part
    lin = C @ u_arr
    lin[0::2] += a

    if t >= 1.1:
        lin += brusselator_cb

    # Non-linear part: u^2 * v
    u_vals = u_arr[0::2]
    v_vals = u_arr[1::2]
    uv2 = u_vals**2 * v_vals
    
    non_lin = np.zeros(size, dtype=float)
    non_lin[0::2] = uv2
    non_lin[1::2] = -uv2
    
    return (lin + non_lin).tolist()

def dg_forward_euler_bruss(n, a, b, alpha, t_max, y0, step):
    """Data Generator for Forward Euler with Brusselator."""

    def dydx(t, u_vec):
        return brusselator_dydx(t, u_vec, n, a, b, alpha)

    return (np, dydx, (0, t_max), y0, step)
