import math

import numpy as np

from binsparse import BinsparseTensor
from binsparse.conversions import from_numpy, to_numpy

from saps.benchmark import (
    Author,
    Benchmark,
    Contributor,
    DataInstance,
    Dataset,
    Generator,
    Ref,
)
from saps.downloaders.ewap import download_ewap_dataset
from saps_framework.binsparse_format import BinsparseFormat

mass = 0.01
cutoff = 0.01
min_r = cutoff / 100
dt = 0.0005


def generate_particle_test_data(num_particles, size, step):
    rng = np.random.default_rng(42)
    x = rng.random(num_particles) * size
    y = rng.random(num_particles) * size
    vx = (rng.random(num_particles) - 0.5) * 0.1
    vy = (rng.random(num_particles) - 0.5) * 0.1
    return x, y, vx, vy, size, step


class ParticleSimDataset(Dataset):
    def __init__(
        self,
        name: str,
        pretty_name: str | None = None,
        description: str | None = None,
        suites: list[str] | None = None,
        values: tuple | None = None,
    ):
        self._name = name
        self._pretty_name = pretty_name or name
        self._description = description or f"Particle simulation input {name}."
        self._suites = suites or []
        self.values = values

    @property
    def name(self) -> str:
        return self._name

    @property
    def pretty_name(self) -> str:
        return self._pretty_name

    @property
    def description(self) -> str:
        return self._description

    @property
    def suites(self) -> list[str]:
        return self._suites

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"


class Particle:
    def __init__(self, x=0.0, y=0.0, vx=0.0, vy=0.0, ax=0.0, ay=0.0):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.ax = ax
        self.ay = ay


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

    if p.x < 0 or p.x > size:
        if p.x < 0:
            p.x = -p.x
        else:
            p.x = 2 * size - p.x

        p.vx = -p.vx

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


def reference_particle_sim(x, y, vx, vy, size, steps):
    ref_particles = [
        Particle(xi, yi, vxi, vyi, 0, 0)
        for xi, yi, vxi, vyi in zip(x, y, vx, vy, strict=True)
    ]
    init_simulation(ref_particles, len(ref_particles), size, steps)
    return [
        np.array([p.x for p in ref_particles]),
        np.array([p.y for p in ref_particles]),
        np.array([p.vx for p in ref_particles]),
        np.array([p.vy for p in ref_particles]),
    ]


class ParticleSimTestGenerator(Generator[ParticleSimDataset]):
    @property
    def name(self):
        return "particle_sim_test_inputs"

    @property
    def pretty_name(self):
        return "Particle Simulation Test Input Generator"

    @property
    def description(self):
        return "Small deterministic particle simulation examples."

    @property
    def suites(self):
        return ["test", "trace"]

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def authors(self):
        return ParticleSimBenchmark().authors

    @property
    def references(self):
        return ParticleSimBenchmark().references

    @property
    def ai_disclosure(self):
        return ParticleSimBenchmark().ai_disclosure

    @property
    def motivation(self):
        return "Provide small particle examples for benchmark correctness checks."

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def datasets(self):
        return [
            ParticleSimDataset(
                "test_particle_sim_two_particles_within_cutoff",
                suites=["test", "trace"],
                values=(
                    np.array([0.001, 0.002]),
                    np.array([0.001, 0.001]),
                    np.array([0.1, 0.0]),
                    np.array([0.0, 0.0]),
                    2,
                    1,
                ),
            ),
            ParticleSimDataset(
                "test_particle_sim_wall_bounce",
                suites=["test", "trace"],
                values=(
                    np.array([0.0001]),
                    np.array([0.05]),
                    np.array([-0.1]),
                    np.array([0.0]),
                    2,
                    5,
                ),
            ),
            ParticleSimDataset(
                "test_particle_sim_random_10",
                suites=["test", "trace"],
                values=generate_particle_test_data(10, 2, 10),
            ),
            ParticleSimDataset(
                "test_particle_sim_random_50",
                suites=["test", "trace"],
                values=generate_particle_test_data(50, 2, 20),
            ),
        ]

    def generate(self, dataset):
        if dataset.values is None:
            raise ValueError("Particle simulation test datasets must define values.")
        x, y, vx, vy, size, steps = dataset.values
        expected = reference_particle_sim(x, y, vx, vy, size, steps)
        return DataInstance(
            inputs=[
                from_numpy(x),
                from_numpy(y),
                from_numpy(vx),
                from_numpy(vy),
            ],
            meta={"size": size, "steps": steps},
            ref_outputs=[from_numpy(value) for value in expected],
        )


class ParticleSimDataset(Dataset):
    def __init__(
        self,
        name: str,
        n_particles: int,
        num_steps: int = 50,
        box_size: float | None = None,
        pretty_name: str | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
    ):
        self._name = name
        self._n_particles = n_particles
        self._num_steps = num_steps
        # Use the CS267 standard density: size = sqrt(n * 0.0005)
        self._box_size = box_size if box_size is not None else math.sqrt(n_particles * 0.0005)
        self._pretty_name = pretty_name or name
        self._description = description or (
            f"Particle Simulation with {n_particles} particles, {num_steps} steps."
        )
        self._tags = tags or ["physics", "simulation", "sparse"]

    @property
    def name(self) -> str:
        return self._name

    @property
    def pretty_name(self) -> str:
        return self._pretty_name

    @property
    def description(self) -> str:
        return self._description

    @property
    def tags(self) -> list[str]:
        return self._tags

    @property
    def n_particles(self) -> int:
        return self._n_particles

    @property
    def num_steps(self) -> int:
        return self._num_steps

    @property
    def box_size(self) -> float:
        return self._box_size


class SyntheticParticleSimDataset(ParticleSimDataset):
    def __init__(
        self,
        name: str,
        n_particles: int,
        pos_distribution: str = "uniform",
        vel_distribution: str = "normal",
        num_steps: int = 50,
        box_size: float | None = None,
        pretty_name: str | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
    ):
        super().__init__(
            name=name,
            n_particles=n_particles,
            num_steps=num_steps,
            box_size=box_size,
            pretty_name=pretty_name,
            description=description,
            tags=tags,
        )
        self._pos_distribution = pos_distribution
        self._vel_distribution = vel_distribution

    @property
    def pos_distribution(self) -> str:
        return self._pos_distribution

    @property
    def vel_distribution(self) -> str:
        return self._vel_distribution



class SyntheticParticleSimGenerator(Generator[SyntheticParticleSimDataset]):
    @property
    def cacheable(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return "synthetic_particle_sim"

    @property
    def pretty_name(self) -> str:
        return "Synthetic Particle Simulation Generator"

    @property
    def description(self) -> str:
        return "Generates synthetic initial conditions for particle simulation benchmarks."

    @property
    def tags(self) -> list[str]:
        return ["physics", "particle-simulation", "sparse"]

    @property
    def authors(self) -> list[Contributor]:
        return []

    @property
    def references(self) -> list[Ref]:
        return []

    @property
    def ai_disclosure(self) -> str:
        return (
            "Generative AI was used to construct the generator and dataset structures."
            " This statement was written by hand."
        )

    @property
    def motivation(self):
        return (
            "The particle simulation is used to model particle interaction present in"
            " mechanics, biology, astronomy, and other fields on a simplitic level."
        )

    @property
    def datasets(self) -> list[SyntheticParticleSimDataset]:
        return [
            SyntheticParticleSimDataset(
                name="uniform_pos_normal_vel_n500",
                n_particles=500,
                num_steps=20,
                pretty_name="Uniform Position, Normal Velocity (N=500)",
            ),
            SyntheticParticleSimDataset(
                name="uniform_pos_normal_vel_n2000",
                n_particles=2000,
                num_steps=20,
                pretty_name="Uniform Position, Normal Velocity (N=2000)",
            ),
            SyntheticParticleSimDataset(
                name="uniform_pos_normal_vel_n5000",
                n_particles=5000,
                num_steps=10,
                pretty_name="Uniform Position, Normal Velocity (N=5000)",
            ),
            SyntheticParticleSimDataset(
                name="gaussian_cluster_zero_vel_n500",
                n_particles=500,
                num_steps=20,
                pos_distribution="gaussian_cluster",
                vel_distribution="zero",
                pretty_name="Gaussian Cluster, Zero Velocity (N=500)",
                description=(
                    "500 particles concentrated in a Gaussian cluster at the box center"
                    " with zero initial velocity. Repulsive forces drive the spreading."
                ),
            ),
            SyntheticParticleSimDataset(
                name="gaussian_cluster_zero_vel_n2000",
                n_particles=2000,
                num_steps=20,
                pos_distribution="gaussian_cluster",
                vel_distribution="zero",
                pretty_name="Gaussian Cluster, Zero Velocity (N=2000)",
                description=(
                    "2000 particles concentrated in a Gaussian cluster at the box center"
                    " with zero initial velocity. Repulsive forces drive the spreading."
                ),
            ),
        ]

    def generate(self, dataset: SyntheticParticleSimDataset):
        rng = np.random.default_rng(42)
        n = dataset.n_particles
        size = dataset.box_size

        if dataset.pos_distribution == "uniform":
            x = rng.uniform(0, size, n).astype(np.float64)
            y = rng.uniform(0, size, n).astype(np.float64)
        elif dataset.pos_distribution == "gaussian_cluster":
            # Tight cluster at box center; sigma ~5% of box size so particles
            # start well within cutoff of each other and spread under repulsion.
            sigma = size * 0.05
            x = np.clip(rng.normal(size / 2, sigma, n), 0, size).astype(np.float64)
            y = np.clip(rng.normal(size / 2, sigma, n), 0, size).astype(np.float64)
        else:
            raise ValueError(f"Unknown pos_distribution: {dataset.pos_distribution!r}")

        if dataset.vel_distribution == "normal":
            vx = rng.normal(0, 0.5, n).astype(np.float64)
            vy = rng.normal(0, 0.5, n).astype(np.float64)
        elif dataset.vel_distribution == "zero":
            vx = np.zeros(n, dtype=np.float64)
            vy = np.zeros(n, dtype=np.float64)
        else:
            raise ValueError(f"Unknown vel_distribution: {dataset.vel_distribution!r}")

        return (
            [
                BinsparseFormat.from_numpy(x),
                BinsparseFormat.from_numpy(y),
                BinsparseFormat.from_numpy(vx),
                BinsparseFormat.from_numpy(vy),
            ],
            {"size": size, "steps": dataset.num_steps},
        )


class ParticleSimGenerator(Generator[ParticleSimDataset]):
    @property
    def cacheable(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return "particle_sim"

    @property
    def pretty_name(self) -> str:
        return "Particle Simulation Real Dataset Generator"

    @property
    def description(self) -> str:
        return "Loads real-world initial conditions for particle simulation benchmarks."

    @property
    def tags(self) -> list[str]:
        return ["physics", "particle-simulation", "sparse"]

    @property
    def authors(self) -> list[Contributor]:
        return []

    @property
    def references(self) -> list[Ref]:
        return [
            Ref(
                title="Learning Social Force Model for Pedestrian Detection",
                authors=[
                    Author("Pellegrini, Stefano"),
                    Author("Ess, Andreas"),
                    Author("Schindler, Konrad"),
                    Author("Van Gool, Luc"),
                ],
                url="http://www.vision.ee.ethz.ch/datasets/downloads/ewap_dataset_light.tgz",
            )
        ]

    @property
    def ai_disclosure(self) -> str:
        return (
            "Generative AI was used to construct the generator and dataset structures."
            " This statement was written by hand."
        )

    @property
    def motivation(self):
        return (
            "The particle simulation is used to model particle interaction present in"
            " mechanics, biology, astronomy, and other fields on a simplitic level."
        )

    @property
    def datasets(self) -> list[ParticleSimDataset]:
        return [
            ParticleSimDataset(
                name="ewap_seq_eth",
                n_particles=75,
                num_steps=50,
                pretty_name="ETH EWAP — seq_eth",
                description=(
                    "Real pedestrian trajectories from the ETH EWAP dataset (scene: seq_eth)."
                    " Each pedestrian's first observed position and velocity serves as a"
                    " particle initial condition, rescaled to the CS267 standard box."
                ),
                tags=["physics", "simulation", "sparse", "real-world", "pedestrian"],
            ),
            ParticleSimDataset(
                name="ewap_seq_hotel",
                n_particles=389,
                num_steps=50,
                pretty_name="ETH EWAP — seq_hotel",
                description=(
                    "Real pedestrian trajectories from the ETH EWAP dataset (scene: seq_hotel)."
                    " Each pedestrian's first observed position and velocity serves as a"
                    " particle initial condition, rescaled to the CS267 standard box."
                ),
                tags=["physics", "simulation", "sparse", "real-world", "pedestrian"],
            ),
        ]

    def generate(self, dataset: ParticleSimDataset):
        # Scene name is encoded after the "ewap_" prefix in the dataset name.
        scene = dataset.name.removeprefix("ewap_")
        return download_ewap_dataset(scene, num_steps=dataset.num_steps)


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
        return """
        <ccs2012>
<concept>
<concept_id>10010147.10010371.10010382.10010383</concept_id>
<concept_desc>Computing methodologies~Image processing</concept_desc>
<concept_significance>500</concept_significance>
</concept>
<concept>
<concept_id>10010147.10010371.10010382.10010236</concept_id>
<concept_desc>Computing methodologies~Computational photography</concept_desc>
<concept_significance>500</concept_significance>
</concept>
<concept>
<concept_id>10010405.10010444.10010087.10010096</concept_id>
<concept_desc>Applied computing~Imaging</concept_desc>
<concept_significance>500</concept_significance>
</concept>
</ccs2012>
"""

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
        return [ParticleSimTestGenerator()]

    def benchmark(self, xp, data, meta):
        x, y, vx, vy = data
        size = meta["size"]
        steps = meta["steps"]

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

    def check(self, param):
        for item in self._output:
            assert isinstance(item, BinsparseTensor), (
                "Output must be in binsparse format"
            )
        if self._ref_outputs is None:
            return

        for i, (actual, expected) in enumerate(
            zip(self._output, self._ref_outputs, strict=True)
        ):
            actual_values = to_numpy(actual)
            expected_values = to_numpy(expected)
            assert np.all(actual_values == expected_values), (
                f"Particle simulation output {i} mismatch for {param.dataset.name}"
            )
