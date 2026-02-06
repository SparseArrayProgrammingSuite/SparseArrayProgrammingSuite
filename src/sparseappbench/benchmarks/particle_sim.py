def benchmark_particle_sum(xp, x, y, vx, vy, ax, ay, size, steps):
    # CONSTANTS
    mass = 0.01
    cutoff = 0.01
    min_r = cutoff / 100
    dt = 0.0005

    x = xp.from_benchmark(x)
    y = xp.from_benchmark(y)
    vx = xp.from_benchmark(vx)
    vy = xp.from_benchmark(vy)
    ax = xp.from_benchmark(ax)
    ay = xp.from_benchmark(ay)

    for _ in range(steps):
        x, y, vx, vy, ax, ay = xp.lazy([x, y, vx, vy, ax, ay])

        # compute forces
        dx = x - x.reshape(-1, 1)
        dy = y - y.reshape(-1, 1)
        r2 = dx * dx + dy * dy

        mask = r2 > cutoff * cutoff

        r2 = xp.where(mask, xp.inf, r2)
        r2 = xp.maximum(r2, min_r * min_r)
        r = xp.sqrt(r2)

        coef = (1 - cutoff / r) / r2 / mass
        # coef = xp.where(mask, 0, coef)

        ax = coef * dx
        ay = coef * dy

        ax = xp.sum(ax, axis=1)
        ay = xp.sum(ay, axis=1)

        # move particles
        vx += ax * dt
        vy += ay * dt

        x += vx * dt
        y += vy * dt

        # bounce off walls
        # x
        reflected = (x < 0) | (x > size)
        vx = xp.where(reflected, -vx, vx)

        x1 = xp.abs(x)
        x2 = 2 * size - x
        x = xp.where(x > size, x2, x1)

        # y
        reflected = (y < 0) | (y > size)
        vy = xp.where(reflected, -vy, vy)

        y1 = xp.abs(y)
        y2 = 2 * size - y
        y = xp.where(y > size, y2, y1)

        x, y, vx, vy, ax, ay = xp.compute([x, y, vx, vy, ax, ay])

    x = xp.to_benchmark(x)
    y = xp.to_benchmark(y)
    vx = xp.to_benchmark(vx)
    vy = xp.to_benchmark(vy)
    ax = xp.to_benchmark(ax)
    ay = xp.to_benchmark(ay)

    return (x, y, vx, vy, ax, ay)
