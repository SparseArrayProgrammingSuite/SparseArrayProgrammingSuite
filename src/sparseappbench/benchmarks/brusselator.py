def limit(a, N):
    if a == N + 1:
        return 0
    if a == -1:
        return N
    return a


def brusselator_f(x, y, t):
    a = (x - 0.3) ** 2 + (y - 0.6) ** 2
    if a <= 0.1**2 and t >= 1.1:
        return 5
    return 0

    # a = [u0, v0, u1, v1, ...]


def init_brusselator_2d():
    n = 32
    u = [0] * (n * n * 2)
    for i in range(n):
        for j in range(n):
            u[(i * n + j) * 2] = 22 * (j * (1 - j)) ** 1.5
            u[(i * n + j) * 2 + 1] = 27 * (i * (1 - i)) ** 1.5
    return u


def dg_brusselator():
    n = 32
    brusselator = [0] * n * n * 2
    for i in range(n):
        for j in range(n):
            x = i / (n - 1)
            y = j / (n - 1)
            if (x - 0.3) ** 2 + (y - 0.6) ** 2 <= 0.1**2:
                brusselator[(i * n + j) * 2] = 5

    a = 3.4
    b = 1.0
    alpha = 0.1
    A = [[0] * (n * n * 2)] * (n * n * 2)
    A1 = [[0] * (n * n * 2)] * (n * n * 2)
    A2 = [[0] * (n * n * 2)] * (n * n * 2)
    for i in range(n):
        for j in range(n):
            x = i / (n - 1)
            y = j / (n - 1)
            ip1, im1, jp1, jm1 = (
                limit(i + 1, n),
                limit(i - 1, n),
                limit(j + 1, n),
                limit(j - 1, n),
            )
            # print(f"i={i}, j={j}, ip1={ip1}, im1={im1}, jp1={jp1}, jm1={jm1}")

            # (alpha*nabla^2)
            # du[(i*n + j) * 2] = alpha*(u[(im1* n + j)*2] + u[(ip1* n + j)*2] + 
            # u[(i* n + jp1)*2] + u[(i* n + jm1)*2] - 4*u[(i* n + j)*2])
            A[(i*n + j)*2, (im1 * n + j)*2] += alpha 
            A[(i*n + j)*2, (ip1 * n + j)*2] += alpha
            A[(i*n + j)*2, (i * n + jp1)*2] += alpha 
            A[(i*n + j)*2, (i * n + jm1)*2] += alpha
            A[(i*n + j)*2, (i * n + j)*2] -= 4 * alpha

            # (alpha*nabla^2)
            # du[(i*n + j) * 2+1] = alpha*(u[(im1* n + j)*2+1] + 
            # u[(ip1* n + j)*2+1] + u[(i* n + jp1)*2+1]
            # + u[(i* n + jm1)*2+1] - 4*u[(i* n + j)*2+1])
            A[(i*n + j)*2+1, (im1 * n + j)*2+1] += alpha 
            A[(i*n + j)*2+1, (ip1 * n + j)*2+1] += alpha
            A[(i*n + j)*2+1, (i * n + jp1)*2+1] += alpha 
            A[(i*n + j)*2+1, (i * n + jm1)*2+1] += alpha
            A[(i*n + j)*2+1, (i * n + j)*2+1] -= 4 * alpha + a + 1

            A1[(i * n + j) * 2, (i * n + j) * 2] = 1
            A1[(i * n + j) * 2 + 1, (i * n + j) * 2] = -1
            A2[(i * n + j) * 2, (i * n + j) * 2 + 1] = 1
            A2[(i * n + j) * 2 + 1, (i * n + j) * 2 + 1] = 1

    def du_dt(t, u):
        return A @ u + b + (0 if t < 1.1 else brusselator) + A1 @ u * A2 @ u**2

    def dv_dt(t, u):
        return A @ u + A1 @ u * A2 @ u**2
