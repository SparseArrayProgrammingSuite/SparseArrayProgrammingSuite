# BEGIN COPIED TEST FILE: tests/test_particle_sim.py
# import math
#
# import pytest
#
# import numpy as np
#
# import saps.benchmarks.particle_sim as ps
# from frameworks.saps_numpy import NumpyFramework
# from saps_framework import BinsparseFormat
#
# # CONSTANTS
# nsteps = 1000
# savefreq = 10
# density = 0.0005
# mass = 0.01
# cutoff = 0.01
# min_r = cutoff / 100
# dt = 0.0005
#
#
# def generate_test_data(num_particles, size, step):
#     rng = np.random.default_rng(42)  # Fixed seed for reproducibility!
#     x = rng.random(num_particles) * size
#     y = rng.random(num_particles) * size
#     vx = (rng.random(num_particles) - 0.5) * 0.1
#     vy = (rng.random(num_particles) - 0.5) * 0.1
#     return (
#         x,
#         y,
#         vx,
#         vy,
#         size,
#         step,
#     )  # 10 steps is usually enough for a unit test
#
#
# @pytest.mark.parametrize(
#     "x,y,vx,vy,size,steps",
#     [
#         # Scenario 1: Manual - Two particles within cutoff
#         (
#             np.array([0.001, 0.002]),
#             np.array([0.001, 0.001]),
#             np.array([0.1, 0.0]),
#             np.array([0.0, 0.0]),
#             2,
#             1,
#         ),
#         # Scenario 2: Manual - Wall bounce test
#         (
#             np.array([0.0001]),
#             np.array([0.05]),
#             np.array([-0.1]),
#             np.array([0.0]),
#             2,
#             5,
#         ),
#         # Scenario 3: Random - 10 particles (Small scale)
#         generate_test_data(10, 2, 10),
#         # Scenario 4: Random - 50 particles (Density check)
#         generate_test_data(50, 2, 20),
#     ],
# )
# def test_particle_sim(x, y, vx, vy, size, steps):
#     xp = NumpyFramework()
#
#     ref_particles = [
#         Particle(xi, yi, vxi, vyi, 0, 0)
#         for xi, yi, vxi, vyi in zip(x, y, vx, vy, strict=True)
#     ]
#
#     x_bin = BinsparseFormat.from_numpy(x)
#     y_bin = BinsparseFormat.from_numpy(y)
#     vx_bin = BinsparseFormat.from_numpy(vx)
#     vy_bin = BinsparseFormat.from_numpy(vy)
#
#     ps.xp = xp
#     (x, y, vx, vy) = ps.ParticleSimBenchmark().benchmark(
#         (x_bin, y_bin, vx_bin, vy_bin, size, steps), {}
#     )
#
#     init_simulation(ref_particles, len(ref_particles), size, steps)
#
#     for i, p_ref in enumerate(ref_particles):
#         actual = (x[i], y[i], vx[i], vy[i])
#         expected = (p_ref.x, p_ref.y, p_ref.vx, p_ref.vy)
#
#         msg = f"Mismatch at particle {i}:\n  Expected: {expected}\n  Actual:   {actual}"
#         assert actual == expected, msg
#
#
# class Particle:
#     def __init__(self, x=0.0, y=0.0, vx=0.0, vy=0.0, ax=0.0, ay=0.0):
#         self.x = x
#         self.y = y
#         self.vx = vx
#         self.vy = vy
#         self.ax = ax
#         self.ay = ay
#
#     # def __repr__(self):
#     #     # This tells Python how to represent the object as a string
#     #     return (
#     #         f"Particle(pos=({self.x}, {self.y}), "
#     #         f"vel=({self.vx}, {self.vy}), "
#     #         f"acc=({self.ax}, {self.ay}))"
#     #     )
#
#     def __eq__(self, other):
#         if not isinstance(other, Particle):
#             return NotImplemented
#         return (
#             self.x == other.x
#             and self.y == other.y
#             and self.vx == other.vx
#             and self.vy == other.vy
#             and self.ax == other.ax
#             and self.ay == other.ay
#         )
#
#
# def apply_force(particle, neighbor):
#     dx = neighbor.x - particle.x
#     dy = neighbor.y - particle.y
#     r2 = dx * dx + dy * dy
#
#     if r2 > cutoff * cutoff:
#         return
#
#     r2 = max(r2, min_r * min_r)
#     r = math.sqrt(r2)
#
#     coef = (1 - cutoff / r) / r2 / mass
#     particle.ax += coef * dx
#     particle.ay += coef * dy
#
#
# def move(p, size):
#     p.vx += p.ax * dt
#     p.vy += p.ay * dt
#     p.x += p.vx * dt
#     p.y += p.vy * dt
#
#     # not continuously checking bounds
#     if p.x < 0 or p.x > size:
#         if p.x < 0:
#             p.x = -p.x
#         else:
#             p.x = 2 * size - p.x
#
#         p.vx = -p.vx
#
#     # not continuously checking bounds
#     if p.y < 0 or p.y > size:
#         if p.y < 0:
#             p.y = -p.y
#         else:
#             p.y = 2 * size - p.y
#         p.vy = -p.vy
#
#
# def simulate_one_step(parts, num_parts, size):
#     for i in range(num_parts):
#         parts[i].ax = 0
#         parts[i].ay = 0
#         for j in range(num_parts):
#             apply_force(parts[i], parts[j])
#
#     for i in range(num_parts):
#         move(parts[i], size)
#
#
# def init_simulation(parts, num_parts, size, steps):
#     for _ in range(steps):
#         simulate_one_step(parts, num_parts, size)
# END COPIED TEST FILE: tests/test_particle_sim.py

import saps
from saps.benchmark import (
    Author,
    Benchmark,
    Contributor,
    Ref,
)

xp = saps.xp


class ParticleSimBenchmark(Benchmark):
    @property
    def name(self):
        return "particle_sim"

    @property
    def pretty_name(self):
        return "Particle Simulation"

    @property
    def description(self):
        return (
            "Benchmark implementation for Particule_Simulation_Algorithm using sparse"
            " array operations. This benchmark evaluates performance characteristics"
            " and numerical properties."
        )

    @property
    def suites(self):
        return []

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def authors(self):
        return [
            Contributor("Richard Wan", "rwan41@gatech.edu"),
        ]

    @property
    def references(self):
        return [
            Ref(
                title="Particle Simulation Algorithm",
                authors=[Author("CS 267 Staff")],
                url="https://github.com/Berkeley-CS267/hw2-1/blob/master/serial.cpp",
            )
        ]

    @property
    def ai_disclosure(self):
        return (
            "No generative AI was used for the benchmark function itself. Generative"
            " AI might have been used to construct tests. This statement was written by"
            " hand."
        )

    @property
    def motivation(self):
        return (
            "The particle simulation is used to model particle interaction present in"
            " mechanics, biology, astronomy, and other fields on a simplitic level."
        )

    @property
    def generators(self):
        return []

    def benchmark(self, data, meta):
        x, y, vx, vy, size, steps = data
        # CONSTANTS
        mass = 0.01
        cutoff = 0.01
        min_r = cutoff / 100
        dt = 0.0005

        x = xp.from_binsparse(x)
        y = xp.from_binsparse(y)
        vx = xp.from_binsparse(vx)
        vy = xp.from_binsparse(vy)

        for _ in range(steps):
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

        return [x, y, vx, vy]
