import math

import pytest

import numpy as np

from sparseappbench.benchmarks.particle_sim import benchmark_particle_sum
from sparseappbench.binsparse_format import BinsparseFormat
from sparseappbench.frameworks.numpy_framework import NumpyFramework

# CONSTANTS
nsteps = 1000
savefreq = 10
density = 0.0005
mass = 0.01
cutoff = 0.01
min_r = cutoff / 100
dt = 0.0005


def generate_test_data(num_particles, size, step):
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility!
    x = rng.random(num_particles) * size
    y = rng.random(num_particles) * size
    vx = (rng.random(num_particles) - 0.5) * 0.1
    vy = (rng.random(num_particles) - 0.5) * 0.1
    ax = np.zeros(num_particles)
    ay = np.zeros(num_particles)
    return (
        x,
        y,
        vx,
        vy,
        ax,
        ay,
        size,
        step,
    )  # 10 steps is usually enough for a unit test


@pytest.mark.parametrize(
    "x,y,vx,vy,ax,ay,size,steps",
    [
        # Scenario 1: Manual - Two particles within cutoff
        (
            np.array([0.001, 0.002]),
            np.array([0.001, 0.001]),
            np.array([0.1, 0.0]),
            np.array([0.0, 0.0]),
            np.array([0.0, 0.0]),
            np.array([0.0, 0.0]),
            2,
            1,
        ),
        # Scenario 2: Manual - Wall bounce test
        (
            np.array([0.0001]),
            np.array([0.05]),
            np.array([-0.1]),
            np.array([0.0]),
            np.array([0.0]),
            np.array([0.0]),
            2,
            5,
        ),
        # Scenario 3: Random - 10 particles (Small scale)
        generate_test_data(10, 2, 10),
        # Scenario 4: Random - 50 particles (Density check)
        generate_test_data(50, 2, 20),
    ],
)
def test_particle_sim(x, y, vx, vy, ax, ay, size, steps):
    xp = NumpyFramework()

    ref_particles = [
        Particle(xi, yi, vxi, vyi, axi, ayi)
        for xi, yi, vxi, vyi, axi, ayi in zip(x, y, vx, vy, ax, ay, strict=True)
    ]

    x_bin = BinsparseFormat.from_numpy(x)
    y_bin = BinsparseFormat.from_numpy(y)
    vx_bin = BinsparseFormat.from_numpy(vx)
    vy_bin = BinsparseFormat.from_numpy(vy)
    ax_bin = BinsparseFormat.from_numpy(ax)
    ay_bin = BinsparseFormat.from_numpy(ay)

    (x, y, vx, vy, ax, ay) = benchmark_particle_sum(
        xp, x_bin, y_bin, vx_bin, vy_bin, ax_bin, ay_bin, size, steps
    )

    x = xp.from_benchmark(x)
    y = xp.from_benchmark(y)
    vx = xp.from_benchmark(vx)
    vy = xp.from_benchmark(vy)
    ax = xp.from_benchmark(ax)
    ay = xp.from_benchmark(ay)

    init_simulation(ref_particles, len(ref_particles), size, steps)

    for i, p_ref in enumerate(ref_particles):
        actual = (x[i], y[i], vx[i], vy[i], ax[i], ay[i])
        expected = (p_ref.x, p_ref.y, p_ref.vx, p_ref.vy, p_ref.ax, p_ref.ay)

        msg = f"Mismatch at particle {i}:\n  Expected: {expected}\n  Actual:   {actual}"
        assert actual == expected, msg


class Particle:
    def __init__(self, x=0.0, y=0.0, vx=0.0, vy=0.0, ax=0.0, ay=0.0):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.ax = ax
        self.ay = ay

    # def __repr__(self):
    #     # This tells Python how to represent the object as a string
    #     return (
    #         f"Particle(pos=({self.x}, {self.y}), "
    #         f"vel=({self.vx}, {self.vy}), "
    #         f"acc=({self.ax}, {self.ay}))"
    #     )

    def __eq__(self, other):
        if not isinstance(other, Particle):
            return NotImplemented
        return (
            self.x == other.x
            and self.y == other.y
            and self.vx == other.vx
            and self.vy == other.vy
            and self.ax == other.ax
            and self.ay == other.ay
        )


def apply_force(particle, neighbor):
    dx = neighbor.x - particle.x
    dy = neighbor.y - particle.y
    r2 = dx * dx + dy * dy

    if r2 > cutoff * cutoff:
        return

    r2 = max(r2, min_r * min_r)
    r = math.sqrt(r2)

    coef = (1 - cutoff / r) / r2 / mass
    particle.ax += coef * dx
    particle.ay += coef * dy


def move(p, size):
    p.vx += p.ax * dt
    p.vy += p.ay * dt
    p.x += p.vx * dt
    p.y += p.vy * dt

    # not continuously checking bounds
    if p.x < 0 or p.x > size:
        if p.x < 0:
            p.x = -p.x
        else:
            p.x = 2 * size - p.x

        p.vx = -p.vx

    # not continuously checking bounds
    if p.y < 0 or p.y > size:
        if p.y < 0:
            p.y = -p.y
        else:
            p.y = 2 * size - p.y
        p.vy = -p.vy


def simulate_one_step(parts, num_parts, size):
    for i in range(num_parts):
        parts[i].ax = 0
        parts[i].ay = 0
        for j in range(num_parts):
            apply_force(parts[i], parts[j])

    for i in range(num_parts):
        move(parts[i], size)


def init_simulation(parts, num_parts, size, steps):
    for _ in range(steps):
        simulate_one_step(parts, num_parts, size)
