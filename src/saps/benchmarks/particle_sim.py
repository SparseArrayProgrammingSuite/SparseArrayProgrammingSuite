import saps
from saps.benchmark import (
    Benchmark,
    Contributor,
    Ref,
    Author,
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
            "Benchmark implementation for Particule_Simulation_Algorithm using sparse array operations. "
            "This benchmark evaluates performance characteristics and numerical properties."
        )

    @property
    def tags(self):
        return ['physics', 'simulation', 'sparse']

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
                authors=[
                    Author("CS 267 Staff")
                ],
                url="https://github.com/Berkeley-CS267/hw2-1/blob/master/serial.cpp"
            )
        ]

    @property
    def ai_disclosure(self):
        return (
            "No generative AI was used for the benchmark function itself. Generative AI might "
            "have been used to construct tests. This statement was written "
            "by hand."
        )

    @property
    def motivation(self):
        return "The particle simulation is used to model particle interaction present in mechanics, biology, astronomy, and other fields on a simplitic level."

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
            x, y, vx, vy = [x, y, vx, vy]

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

            x, y, vx, vy = [x, y, vx, vy]

        x = xp.to_binsparse(x)
        y = xp.to_binsparse(y)
        vx = xp.to_binsparse(vx)
        vy = xp.to_binsparse(vy)

        return (x, y, vx, vy)

