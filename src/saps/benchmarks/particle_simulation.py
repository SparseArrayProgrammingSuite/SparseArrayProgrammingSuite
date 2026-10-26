import itertools
import math
from typing import Any

import numpy as np
from scipy.sparse import csr_array

from binsparse import BinsparseTensor
from binsparse.conversions import from_numpy, to_numpy, to_sparse

from saps.benchmark import (
    Author,
    Benchmark,
    Contributor,
    DataInstance,
    Dataset,
    Generator,
    Ref,
    ReferenceDataset,
    ReferenceGenerator,
    ShellBenchmark,
)
from saps.downloaders.ewap import download_ewap_dataset
from saps.downloaders.nemo import download_nemo_dataset
from saps.util.interval import Interval, add, maximum, minimum


def particle_density_box_size(n_particles: int, density: float) -> float:
    return math.pow(density * n_particles, 1.0 / 3.0)


def generate_particle_test_data(num_particles, size, step, particle_mass):
    rng = np.random.default_rng(42)
    x = rng.random(num_particles) * size
    y = rng.random(num_particles) * size
    z = rng.random(num_particles) * size
    vx = (rng.random(num_particles) - 0.5) * 0.1
    vy = (rng.random(num_particles) - 0.5) * 0.1
    vz = (rng.random(num_particles) - 0.5) * 0.1
    mass = np.asarray(particle_mass, dtype=np.float64)
    return x, y, z, vx, vy, vz, mass, size, step


class ParticleSimulationDataset(Dataset):
    def __init__(
        self,
        name: str,
        n_particles: int | None = None,
        num_steps: int = 50,
        box_size: float | None = None,
        pretty_name: str | None = None,
        description: str | None = None,
        suites: list[str] | None = None,
        values: tuple | None = None,
        tags: list[str] | None = None,
        *,
        parameters: dict[str, Any],
        source_path: str | None = None,
        source_columns: tuple[str, ...] | None = None,
        source_wrap: bool = False,
    ):
        self._name = name
        self._n_particles = (
            n_particles
            if n_particles is not None
            else len(values[0])
            if values is not None
            else 0
        )
        self._num_steps = num_steps
        self._parameters = dict(parameters)
        self._box_size = box_size
        self._pretty_name = pretty_name or name
        self._description = description or (
            f"Particle Simulation with {self._n_particles} particles, "
            f"{num_steps} steps."
        )
        self._suites = suites or []
        self.values = values
        self._tags = tags or ["physics", "simulation", "sparse"]
        self.source_path = source_path
        self.source_columns = source_columns
        self.source_wrap = source_wrap

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
    def box_size(self) -> float | None:
        return self._box_size

    @property
    def parameters(self) -> dict[str, Any]:
        return dict(self._parameters)

    @property
    def metadata(self) -> dict[str, Any]:
        data = super().metadata
        data["n_particles"] = self.n_particles
        data["num_steps"] = self.num_steps
        data["parameters"] = self.parameters
        data["source_path"] = self.source_path
        data["source_columns"] = (
            list(self.source_columns) if self.source_columns is not None else None
        )
        data["source_wrap"] = self.source_wrap
        return data


# Rounding a coordinate into a cell moves it by under 2.2e-10 cells while it
# stays within a million cells of zero, so cells this much wider than the cutoff
# keep every pair within the cutoff in adjacent cells.
_CELL_MARGIN = 1e-9


def _may_interact(position, width, cells):
    """Ordered pairs whose intervals occupy adjacent cells on every axis.

    Each particle occupies the cells of both endpoints of its intervals. Cells
    are `width` wide and wrap every `cells` cells in a periodic box, or never if
    `cells` is None. Other pairs are a whole cell apart on some axis.
    """
    low = np.stack([np.floor(p.lo / width) for p in position]).astype(np.int64)
    high = np.stack([np.floor(p.hi / width) for p in position]).astype(np.int64)
    assert np.all(high - low <= 1), "Interval bounds grew wider than a cell"
    assert np.all(np.abs(low) < 1e6) and np.all(np.abs(high) < 1e6), (
        "Coordinates are too many cells from zero"
    )
    if cells is None:
        # Enough cells that wrapping never folds a cell next to another.
        shape = tuple(high.max(axis=1) - low.min(axis=1) + 3)
    else:
        shape = (cells, cells, cells)

    n = low.shape[1]
    corners = [
        np.where(np.reshape(choice, (3, 1)), high, low)
        for choice in itertools.product((False, True), repeat=3)
    ]
    occupied = np.unique(
        np.concatenate(
            [
                np.ravel_multi_index(c, shape, mode="wrap") * n + np.arange(n)
                for c in corners
            ]
        )
    )
    owners = occupied % n
    keys, cell = np.unique(occupied // n, return_inverse=True)
    members = csr_array((np.ones(len(owners)), (cell, owners)), (len(keys), n))
    coordinates = np.stack(np.unravel_index(keys[cell], shape))
    rows, columns = [], []
    for offset in itertools.product((-1, 0, 1), repeat=3):
        key = np.ravel_multi_index(
            coordinates + np.reshape(offset, (3, 1)), shape, mode="wrap"
        )
        neighbor = np.minimum(np.searchsorted(keys, key), len(keys) - 1)
        found = keys[neighbor] == key
        rows.append(owners[found])
        columns.append(neighbor[found])
    rows, columns = np.concatenate(rows), np.concatenate(columns)
    # Each row of `near` marks the particles in that particle's adjacent cells.
    near = csr_array((np.ones(len(rows)), (rows, columns)), (n, len(keys))) @ members
    i, j = near.nonzero()
    return i[i != j], j[i != j]


def particle_sim_bounds(x, y, z, vx, vy, vz, particle_mass, size, steps, parameters):
    """Interval bounds on every float64 evaluation of the benchmark.

    Repeats the benchmark's arithmetic on the pairs that may interact. Every
    other pair has zero force in every evaluation.
    """
    dt = parameters["dt"]
    gravitational_constant = parameters["gravitational_constant"]
    cutoff = parameters["cutoff"]
    softening = parameters["softening"]

    n = len(x)
    position = [Interval(x), Interval(y), Interval(z)]
    velocity = [Interval(vx), Interval(vy), Interval(vz)]
    particle_mass = Interval(particle_mass)
    periodic = parameters["boundary_model"] == "periodic_box"
    # Cells a margin wider than the cutoff, a whole number per periodic box.
    reach = cutoff * (1 + _CELL_MARGIN)
    cells = max(int(size // reach), 1) if periodic else None
    width = size / cells if periodic else reach

    for _ in range(steps):
        # Other pairs are more than the cutoff apart on some axis in every
        # evaluation, so their wedge, and with it their force, is zero.
        i, j = _may_interact(position, width, cells)
        delta = [p[j] - p[i] for p in position]
        if periodic:
            delta = [d - size * (d / size).round() for d in delta]
        r2 = add(delta[0] * delta[0], delta[1] * delta[1], delta[2] * delta[2])
        r2 = maximum(r2, softening * softening)
        r = r2.sqrt()

        # So do pairs whose distance is at least the cutoff in every evaluation.
        near = r.lo < cutoff
        i, j, r2, r = i[near], j[near], r2[near], r[near]
        delta = [d[near] for d in delta]
        mass = particle_mass if particle_mass.lo.ndim == 0 else particle_mass[j]

        if parameters["force_model"] == "newtonian_gravity":
            coef = gravitational_constant * mass / (r2 * r)
        else:
            coef = gravitational_constant * ((1 - cutoff / r) / r2 / mass)
        wedge = minimum(maximum((cutoff - r) / (0.05 * cutoff), 0.0), 1.0)
        coef = coef * wedge

        # The benchmark sums n terms for each particle, mostly zeros.
        acceleration = [(coef * d).sum_by(i, n, n) for d in delta]
        velocity = [v + a * dt for v, a in zip(velocity, acceleration, strict=True)]
        position = [p + v * dt for p, v in zip(position, velocity, strict=True)]

    return [*position, *velocity]


def _bounds_reference(problem):
    """Every output's lower bound, then every output's upper bound."""
    x, y, z, vx, vy, vz, mass = (to_numpy(a) for a in problem.inputs)
    meta = problem.meta
    bounds = particle_sim_bounds(
        x, y, z, vx, vy, vz, mass, meta["size"], meta["steps"], meta["parameters"]
    )
    return [from_numpy(bound.lo) for bound in bounds] + [
        from_numpy(bound.hi) for bound in bounds
    ]


class ParticleSimulationTestGenerator(Generator[ParticleSimulationDataset]):
    @property
    def name(self):
        return "particle_simulation_test"

    @property
    def pretty_name(self):
        return "Particle Simulation Test"

    @property
    def description(self):
        return (
            "Small deterministic particle simulation examples using the CS267-style "
            "repulsive force parameters: cutoff 0.01, softening 0.0001 from the "
            "CS267 min_r = cutoff / 100 constant, dt 0.0005, G 1.0, and scalar "
            "mass tensor 0.01."
        )

    @property
    def suites(self):
        return ["test"]

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def authors(self):
        return ParticleSimulationBenchmark().authors

    @property
    def references(self):
        return ParticleSimulationBenchmark().references

    @property
    def ai_disclosure(self):
        return (
            "Generative AI was used to construct this generator and its dataset "
            "structures. This statement was written by hand."
        )

    @property
    def motivation(self):
        return (
            "Provide small particle examples for benchmark correctness checks while "
            "exercising the same CS267-derived cutoff, minimum-radius softening, "
            "time step, and unit gravitational constant used by the synthetic and "
            "EWAP repulsive-force datasets."
        )

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def datasets(self):
        return [
            ParticleSimulationDataset(
                "two_particles_within_cutoff",
                pretty_name="Two Particles Within Cutoff",
                suites=["test"],
                parameters={
                    "force_model": "cs267_repulsive",
                    "boundary_model": "periodic_box",
                    "cutoff": 0.01,
                    "softening": 0.0001,
                    "dt": 0.0005,
                    "gravitational_constant": 1.0,
                },
                values=(
                    np.array([0.001, 0.002]),
                    np.array([0.001, 0.001]),
                    np.array([0.001, 0.001]),
                    np.array([0.1, 0.0]),
                    np.array([0.0, 0.0]),
                    np.array([0.0, 0.0]),
                    np.asarray(0.01),
                    2,
                    1,
                ),
            ),
            ParticleSimulationDataset(
                "across_boundary",
                pretty_name="Across Boundary",
                suites=["test"],
                parameters={
                    "force_model": "cs267_repulsive",
                    "boundary_model": "periodic_box",
                    "cutoff": 0.01,
                    "softening": 0.0001,
                    "dt": 0.0005,
                    "gravitational_constant": 1.0,
                },
                # The pair is 0.008 apart through the periodic boundary at x = 0.
                values=(
                    np.array([0.003, 1.995]),
                    np.array([0.05, 0.05]),
                    np.array([0.05, 0.05]),
                    np.array([0.0, 0.0]),
                    np.array([0.0, 0.0]),
                    np.array([0.0, 0.0]),
                    np.asarray(0.01),
                    2,
                    5,
                ),
            ),
            ParticleSimulationDataset(
                "random_n10",
                pretty_name="Random (N=10)",
                suites=["test"],
                parameters={
                    "force_model": "cs267_repulsive",
                    "boundary_model": "periodic_box",
                    "cutoff": 0.01,
                    "softening": 0.0001,
                    "dt": 0.0005,
                    "gravitational_constant": 1.0,
                },
                values=generate_particle_test_data(10, 2, 10, 0.01),
            ),
            ParticleSimulationDataset(
                "random_n50",
                pretty_name="Random (N=50)",
                suites=["test"],
                parameters={
                    "force_model": "cs267_repulsive",
                    "boundary_model": "periodic_box",
                    "cutoff": 0.01,
                    "softening": 0.0001,
                    "dt": 0.0005,
                    "gravitational_constant": 1.0,
                },
                values=generate_particle_test_data(50, 2, 20, 0.01),
            ),
        ]

    def generate(self, dataset):
        if dataset.values is None:
            raise ValueError("Particle simulation test datasets must define values.")
        x, y, z, vx, vy, vz, mass, size, steps = dataset.values
        parameters = dataset.parameters
        bounds = particle_sim_bounds(x, y, z, vx, vy, vz, mass, size, steps, parameters)
        return DataInstance(
            inputs=[
                from_numpy(x),
                from_numpy(y),
                from_numpy(z),
                from_numpy(vx),
                from_numpy(vy),
                from_numpy(vz),
                from_numpy(mass),
            ],
            meta={"size": size, "steps": steps, "parameters": parameters},
            # Every output's lower bound, then every output's upper bound.
            ref_outputs=[from_numpy(bound.lo) for bound in bounds]
            + [from_numpy(bound.hi) for bound in bounds],
        )


class ParticleSimulationCS267Dataset(ParticleSimulationDataset):
    def __init__(
        self,
        name: str,
        n_particles: int,
        seed: int,
        density: float,
        num_steps: int = 50,
        box_size: float | None = None,
        pretty_name: str | None = None,
        description: str | None = None,
        suites: list[str] | None = None,
        tags: list[str] | None = None,
        *,
        parameters: dict[str, Any],
    ):
        super().__init__(
            name=name,
            n_particles=n_particles,
            num_steps=num_steps,
            box_size=box_size,
            pretty_name=pretty_name,
            description=description,
            suites=suites,
            tags=tags,
            parameters=parameters,
        )
        self._seed = seed
        self._density = density

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def density(self) -> float:
        return self._density

    @property
    def metadata(self) -> dict[str, Any]:
        data = super().metadata
        data["seed"] = self.seed
        data["density"] = self.density
        return data


class ParticleSimulationCS267Generator(Generator[ParticleSimulationCS267Dataset]):
    @property
    def cacheable(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return "particle_simulation_cs267"

    @property
    def pretty_name(self) -> str:
        return "Particle Simulation CS267"

    @property
    def description(self) -> str:
        return (
            "Generates synthetic initial conditions for particle simulation "
            "benchmarks. The force parameters follow the Berkeley CS267 homework "
            "constants where available: cutoff 0.01, softening 0.0001 as the "
            "benchmark name for CS267 min_r = cutoff / 100, dt 0.0005, and a "
            "scalar mass tensor 0.01; G 1.0 is used as the benchmark unit scale."
        )

    @property
    def suites(self) -> list[str]:
        return []

    @property
    def tags(self) -> list[str]:
        return ["physics", "particle-simulation", "sparse"]

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def authors(self) -> list[Contributor]:
        return []

    @property
    def references(self) -> list[Ref]:
        return [
            Ref(
                title="CS267 HW2-1: Parallelizing a Particle Simulation",
                authors=[Author("CS 267 Staff")],
                url="https://sites.google.com/lbl.gov/cs267-spr2025/hw-2-1",
            ),
            Ref(
                title="CS267 HW2-1 Starter Code",
                authors=[Author("CS 267 Staff")],
                url="https://github.com/Berkeley-CS267/hw2-1",
            ),
        ]

    @property
    def ai_disclosure(self) -> str:
        return (
            "Generative AI was used to construct this generator and its dataset "
            "structures. This statement was written by hand."
        )

    @property
    def motivation(self):
        return (
            "The CS267 homework initializes particles on a shuffled regular grid "
            "with random velocities so baseline implementations can focus on the "
            "short-range force calculation. Its cutoff, min_r, density, and time "
            "step constants drive this generator; min_r is recorded as softening "
            "so the benchmark can share one radius-regularization parameter across "
            "repulsive and gravitational datasets."
        )

    @property
    def datasets(self) -> list[ParticleSimulationCS267Dataset]:
        return [
            ParticleSimulationCS267Dataset(
                name="hw2_n1000_seed1",
                n_particles=1000,
                num_steps=1000,
                seed=1,
                density=0.0005,
                suites=["standard"],
                parameters={
                    "force_model": "cs267_repulsive",
                    "boundary_model": "periodic_box",
                    "cutoff": 0.01,
                    "softening": 0.0001,
                    "dt": 0.0005,
                    "gravitational_constant": 1.0,
                },
                pretty_name="Homework 2 Initial Conditions (N=1000, Seed 1)",
                description=(
                    "CS267 homework-style particle initialization extended to a "
                    "shuffled near-cubic grid with random velocities in [-1, 1]. "
                    "The cutoff, softening, dt, density, and scalar mass tensor "
                    "come from the CS267-style benchmark setup, with softening "
                    "standing in for the CS267 min_r constant."
                ),
                tags=["physics", "simulation", "sparse", "synthetic", "cs267"],
            ),
        ]

    def generate(self, dataset: ParticleSimulationCS267Dataset) -> DataInstance:
        problem = self.generate_inputs(dataset)
        reference = ParticleSimulationCS267ReferenceGenerator().cached_generate(
            ReferenceDataset(dataset)
        )
        problem.ref_outputs = reference.ref_outputs
        return problem

    def generate_inputs(self, dataset: ParticleSimulationCS267Dataset) -> DataInstance:
        rng = np.random.Generator(np.random.MT19937(dataset.seed))
        n = dataset.n_particles
        parameters = dataset.parameters
        size = particle_density_box_size(n, dataset.density)
        sx = math.ceil(math.pow(n, 1.0 / 3.0))
        sy = math.ceil(math.sqrt(n / sx))
        sz = (n + sx * sy - 1) // (sx * sy)
        shuffle = list(range(n))

        x = np.empty(n, dtype=np.float64)
        y = np.empty(n, dtype=np.float64)
        z = np.empty(n, dtype=np.float64)
        vx = np.empty(n, dtype=np.float64)
        vy = np.empty(n, dtype=np.float64)
        vz = np.empty(n, dtype=np.float64)
        mass = np.asarray(0.01, dtype=np.float64)

        for i in range(n):
            j = int(rng.integers(0, n - i))
            k = shuffle[j]
            shuffle[j] = shuffle[n - i - 1]

            x[i] = size * (1.0 + (k % sx)) / (1 + sx)
            y[i] = size * (1.0 + ((k // sx) % sy)) / (1 + sy)
            z[i] = size * (1.0 + (k // (sx * sy))) / (1 + sz)
            vx[i] = rng.uniform(-1.0, 1.0)
            vy[i] = rng.uniform(-1.0, 1.0)
            vz[i] = rng.uniform(-1.0, 1.0)

        return DataInstance(
            inputs=[
                from_numpy(x),
                from_numpy(y),
                from_numpy(z),
                from_numpy(vx),
                from_numpy(vy),
                from_numpy(vz),
                from_numpy(mass),
            ],
            meta={
                "size": size,
                "steps": dataset.num_steps,
                "n_particles": dataset.n_particles,
                "seed": dataset.seed,
                "density": dataset.density,
                "parameters": parameters,
                "source": "Berkeley CS267 HW2 init_particles extended to 3D",
                "source_dimensions": 3,
                "simulation_dimensions": 3,
            },
        )


class ParticleSimulationCS267ReferenceGenerator(
    ReferenceGenerator[ParticleSimulationCS267Dataset]
):
    @property
    def name(self) -> str:
        return "particle_simulation_cs267_reference"

    @property
    def pretty_name(self) -> str:
        return "Particle Simulation CS267 Reference"

    @property
    def description(self) -> str:
        return (
            "Reference outputs for Particle Simulation CS267, cached apart from its"
            " inputs."
        )

    @property
    def datasets(self) -> list[ReferenceDataset[ParticleSimulationCS267Dataset]]:
        return [
            ReferenceDataset(dataset)
            for dataset in ParticleSimulationCS267Generator().datasets
        ]

    def generate(
        self, dataset: ReferenceDataset[ParticleSimulationCS267Dataset]
    ) -> DataInstance:
        problem = ParticleSimulationCS267Generator().generate_inputs(dataset.dataset)
        return DataInstance(inputs=[], meta={}, ref_outputs=_bounds_reference(problem))


class ParticleSimulationCS267ReferenceShellBenchmark(ShellBenchmark):
    @property
    def generator(self) -> Generator:
        return ParticleSimulationCS267ReferenceGenerator()


class ParticleSimulationEWAPDataset(ParticleSimulationDataset):
    def __init__(
        self,
        name: str,
        scene: str,
        num_steps: int,
        pretty_name: str,
        description: str,
        parameters: dict[str, Any],
        suites: list[str] | None = None,
    ):
        super().__init__(
            name=name,
            num_steps=num_steps,
            pretty_name=pretty_name,
            description=description,
            suites=["standard"] if suites is None else suites,
            tags=["physics", "simulation", "sparse", "pedestrian", "ewap"],
            parameters=parameters,
        )
        self._scene = scene

    @property
    def scene(self) -> str:
        return self._scene

    @property
    def metadata(self) -> dict[str, Any]:
        data = super().metadata
        data["source_scene"] = self.scene
        return data


class ParticleSimulationEWAPGenerator(Generator[ParticleSimulationEWAPDataset]):
    @property
    def cacheable(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return "particle_simulation_ewap"

    @property
    def pretty_name(self) -> str:
        return "Particle Simulation EWAP"

    @property
    def description(self) -> str:
        return (
            "Loads ETH EWAP pedestrian trajectories as particle initial conditions. "
            "EWAP supplies positions and velocities only, so this generator uses the "
            "same CS267-style repulsive parameters as the synthetic dataset: cutoff "
            "0.01, softening 0.0001, dt 0.0005, G 1.0, and scalar mass tensor 0.01."
        )

    @property
    def suites(self) -> list[str]:
        return []

    @property
    def tags(self) -> list[str]:
        return ["physics", "particle-simulation", "sparse", "pedestrian", "ewap"]

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def authors(self) -> list[Contributor]:
        return []

    @property
    def references(self) -> list[Ref]:
        return [
            Ref(
                title="OpenTraj ETH Dataset",
                authors=[Author("OpenTraj Contributors")],
                url=("https://github.com/crowdbotp/OpenTraj/tree/master/datasets/ETH"),
            )
        ]

    @property
    def ai_disclosure(self) -> str:
        return (
            "Generative AI was used to construct this generator and its dataset "
            "structures. This statement was written by hand."
        )

    @property
    def motivation(self):
        return (
            "Pedestrian trajectories provide real-world 2D interaction data that can "
            "exercise the particle simulation benchmark without changing the source "
            "coordinate scale. Because the source data is not a physical N-body "
            "snapshot, the interaction parameters intentionally mirror the "
            "CS267-derived repulsive-force setup rather than being inferred from "
            "the EWAP files."
        )

    @property
    def datasets(self) -> list[ParticleSimulationEWAPDataset]:
        return [
            ParticleSimulationEWAPDataset(
                name="seq_eth",
                scene="seq_eth",
                num_steps=50,
                parameters={
                    "force_model": "cs267_repulsive",
                    "boundary_model": "periodic_box",
                    "cutoff": 0.01,
                    "softening": 0.0001,
                    "dt": 0.0005,
                    "gravitational_constant": 1.0,
                },
                pretty_name="seq_eth",
                description=(
                    "ETH EWAP seq_eth pedestrian trajectories loaded in dataset "
                    "coordinates with z and vz preserved from the source column. "
                    "The force parameters are the CS267-style repulsive benchmark "
                    "constants because EWAP does not provide simulation constants."
                ),
                suites=["standard", "trace"],
            ),
            ParticleSimulationEWAPDataset(
                name="seq_hotel",
                scene="seq_hotel",
                num_steps=50,
                parameters={
                    "force_model": "cs267_repulsive",
                    "boundary_model": "periodic_box",
                    "cutoff": 0.01,
                    "softening": 0.0001,
                    "dt": 0.0005,
                    "gravitational_constant": 1.0,
                },
                pretty_name="seq_hotel",
                description=(
                    "ETH EWAP seq_hotel pedestrian trajectories loaded in dataset "
                    "coordinates with z and vz preserved from the source column. "
                    "The force parameters are the CS267-style repulsive benchmark "
                    "constants because EWAP does not provide simulation constants."
                ),
                suites=["standard", "trace"],
            ),
        ]

    def generate(self, dataset: ParticleSimulationEWAPDataset):
        inputs, meta = download_ewap_dataset(
            dataset.scene,
            num_steps=dataset.num_steps,
        )
        meta["parameters"] = dataset.parameters
        x, y, z, vx, vy, vz, mass = (to_numpy(a) for a in inputs)
        bounds = particle_sim_bounds(
            x, y, z, vx, vy, vz, mass, meta["size"], meta["steps"], dataset.parameters
        )
        return DataInstance(
            inputs=inputs,
            meta=meta,
            # Every output's lower bound, then every output's upper bound.
            ref_outputs=[from_numpy(bound.lo) for bound in bounds]
            + [from_numpy(bound.hi) for bound in bounds],
        )


class ParticleSimulationNEMOGenerator(Generator[ParticleSimulationDataset]):
    @property
    def cacheable(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return "particle_simulation_nemo"

    @property
    def pretty_name(self) -> str:
        return "Particle Simulation NEMO"

    @property
    def description(self) -> str:
        return (
            "Loads real-world initial conditions for particle simulation benchmarks. "
            "NEMO datasets use source mass columns, unscaled archive coordinates, "
            "Newtonian gravity with G 1.0 in N-body units, dt 1/32, softening 0.05, "
            "and short interaction bucket widths chosen to exercise sparse workloads "
            "rather than accurately approximate full gravitational forces."
        )

    @property
    def suites(self) -> list[str]:
        return []

    @property
    def tags(self) -> list[str]:
        return ["physics", "particle-simulation", "sparse"]

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def authors(self) -> list[Contributor]:
        return []

    @property
    def references(self) -> list[Ref]:
        return [
            Ref(
                title="N-Body Data Archive",
                authors=[Author("Peter Teuben")],
                url="https://carma.astro.umd.edu/nemo/archive/",
            ),
            Ref(
                title="On the problem of distribution in globular star clusters",
                authors=[
                    Author("M. C. Plummer"),
                ],
                journal="Monthly Notices of the Royal Astronomical Society",
                volume=71,
                pages="460",
                year=1911,
            ),
            Ref(
                title="The return of the tidal tails in NGC 7252",
                authors=[
                    Author("John Dubinski"),
                    Author("J. Christopher Mihos"),
                    Author("Lars Hernquist"),
                ],
                journal="The Astrophysical Journal",
                volume=462,
                pages="576",
                year=1996,
            ),
        ]

    @property
    def ai_disclosure(self) -> str:
        return (
            "Generative AI was used to construct this generator and its dataset "
            "structures. This statement was written by hand."
        )

    @property
    def motivation(self):
        return (
            "The particle simulation is used to model particle interaction present in"
            " mechanics, biology, astronomy, and other fields on a simplitic level. "
            "For NEMO snapshots, masses come from the archive tables; G 1.0 follows "
            "the usual dimensionless N-body unit convention, softening 0.05 follows "
            "the NEMO eps-style softened-gravity setting. The smaller cutoffs "
            "are workload parameters chosen for sparse interactions on the "
            "preserved source coordinates; they do not meet a physical "
            "force-error target."
        )

    @property
    def datasets(self) -> list[ParticleSimulationDataset]:
        return [
            ParticleSimulationDataset(
                name="plummer_128",
                n_particles=128,
                num_steps=50,
                parameters={
                    "force_model": "newtonian_gravity",
                    "boundary_model": "unbounded",
                    "cutoff": 0.2,
                    "dt": 1.0 / 32.0,
                    "softening": 0.05,
                    "gravitational_constant": 1.0,
                },
                pretty_name="Plummer (N=128)",
                description=(
                    "NEMO Plummer-model equilibrium snapshot generated with mkplummer,"
                    " using mass, position, and velocity columns. The dataset uses "
                    "source masses, G 1.0 N-body units, dt 1/32, softening 0.05, "
                    "and bucket width 0.2 to create a sparse gravity workload. "
                    "This cutoff is not a physically validated approximation."
                ),
                suites=["standard", "trace"],
                tags=["physics", "simulation", "sparse", "astronomy", "n-body"],
                source_path="plummer/tab128.gz",
                source_columns=("mass", "x", "y", "z", "vx", "vy", "vz"),
            ),
            ParticleSimulationDataset(
                name="plummer_1024",
                n_particles=1024,
                num_steps=50,
                parameters={
                    "force_model": "newtonian_gravity",
                    "boundary_model": "unbounded",
                    "cutoff": 0.2,
                    "dt": 1.0 / 32.0,
                    "softening": 0.05,
                    "gravitational_constant": 1.0,
                },
                pretty_name="Plummer (N=1024)",
                description=(
                    "NEMO Plummer-model equilibrium snapshot generated with mkplummer,"
                    " using mass, position, and velocity columns. The dataset uses "
                    "source masses, G 1.0 N-body units, dt 1/32, softening 0.05, "
                    "and bucket width 0.2 to create a sparse gravity workload. "
                    "This cutoff is not a physically validated approximation."
                ),
                suites=["standard", "trace", "train"],
                tags=["physics", "simulation", "sparse", "astronomy", "n-body"],
                source_path="plummer/tab1024.gz",
                source_columns=("mass", "x", "y", "z", "vx", "vy", "vz"),
            ),
            ParticleSimulationDataset(
                name="dubinski_m31",
                n_particles=81920,
                num_steps=10,
                parameters={
                    "force_model": "newtonian_gravity",
                    "boundary_model": "unbounded",
                    "cutoff": 0.1,
                    "dt": 1.0 / 32.0,
                    "softening": 0.05,
                    "gravitational_constant": 1.0,
                },
                pretty_name="Dubinski Milky Way/M31",
                description=(
                    "Dubinski Milky Way/Andromeda collision initial conditions from the"
                    " NEMO archive, stored as mass and six phase-space coordinates. "
                    "The dataset uses source masses, G 1.0 N-body units, dt 1/32, "
                    "softening 0.05, and bucket width 0.1 to create a sparse "
                    "gravity workload. This width is not a physically "
                    "validated approximation."
                ),
                suites=["standard"],
                tags=["physics", "simulation", "sparse", "astronomy", "n-body"],
                source_path="dubinski/dubinski.tab.gz",
                source_columns=("mass", "x", "y", "z", "vx", "vy", "vz"),
            ),
        ]

    def generate(self, dataset: ParticleSimulationDataset) -> DataInstance:
        problem = self.generate_inputs(dataset)
        reference = ParticleSimulationNEMOReferenceGenerator().cached_generate(
            ReferenceDataset(dataset)
        )
        problem.ref_outputs = reference.ref_outputs
        return problem

    def generate_inputs(self, dataset: ParticleSimulationDataset) -> DataInstance:
        if dataset.source_path is None or dataset.source_columns is None:
            raise ValueError(f"Particle dataset {dataset.name} has no NEMO source")
        inputs, meta = download_nemo_dataset(
            dataset.source_path,
            columns=dataset.source_columns,
            wrap=dataset.source_wrap,
            expected_particles=dataset.n_particles,
            num_steps=dataset.num_steps,
            box_size=dataset.box_size,
            include_mass="mass" in dataset.source_columns,
        )
        meta["parameters"] = dataset.parameters
        return DataInstance(inputs=inputs, meta=meta)


class ParticleSimulationNEMOReferenceGenerator(
    ReferenceGenerator[ParticleSimulationDataset]
):
    @property
    def name(self) -> str:
        return "particle_simulation_nemo_reference"

    @property
    def pretty_name(self) -> str:
        return "Particle Simulation NEMO Reference"

    @property
    def description(self) -> str:
        return (
            "Reference outputs for Particle Simulation NEMO, cached apart from its"
            " inputs."
        )

    @property
    def datasets(self) -> list[ReferenceDataset[ParticleSimulationDataset]]:
        return [
            ReferenceDataset(dataset)
            for dataset in ParticleSimulationNEMOGenerator().datasets
        ]

    def generate(
        self, dataset: ReferenceDataset[ParticleSimulationDataset]
    ) -> DataInstance:
        problem = ParticleSimulationNEMOGenerator().generate_inputs(dataset.dataset)
        return DataInstance(inputs=[], meta={}, ref_outputs=_bounds_reference(problem))


class ParticleSimulationNEMOReferenceShellBenchmark(ShellBenchmark):
    @property
    def generator(self) -> Generator:
        return ParticleSimulationNEMOReferenceGenerator()


class ParticleSimulationBenchmark(Benchmark):
    @property
    def name(self):
        return "particle_simulation"

    @property
    def pretty_name(self):
        return "Particle Simulation"

    @property
    def description(self):
        return (
            "Benchmark implementation for Particule_Simulation_Algorithm using sparse"
            " array operations. The cutoff parameter is the interaction radius and"
            " the smallest cell width: forces ramp linearly to zero over the last 5%"
            " of the cutoff radius, and only pairs whose cell indices differ by at"
            " most one on every axis are evaluated. A periodic box wraps its cells"
            " and pairs interact through their nearest periodic image. This"
            " benchmark evaluates performance characteristics and numerical"
            " properties."
        )

    @property
    def suites(self):
        return ["standard-spatial"]

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
            "Generative AI was used to revise the benchmark's bucket interaction"
            " calculation and associated tests."
        )

    @property
    def motivation(self):
        return (
            "The particle simulation is used to model particle interaction present in"
            " mechanics, biology, astronomy, and other fields on a simplitic level."
        )

    @property
    def generators(self):
        return [
            ParticleSimulationTestGenerator(),
            ParticleSimulationCS267Generator(),
            ParticleSimulationEWAPGenerator(),
            ParticleSimulationNEMOGenerator(),
        ]

    def benchmark(self, xp, meta, x, y, z, vx, vy, vz, particle_mass):
        size = meta["size"]
        steps = meta["steps"]
        parameters = meta["parameters"]
        dt = parameters["dt"]
        gravitational_constant = parameters["gravitational_constant"]
        force_model = parameters["force_model"]
        cutoff = parameters["cutoff"]
        softening = parameters["softening"]

        n_particles = x.shape[0]
        particle_index = xp.arange(n_particles)
        periodic = parameters["boundary_model"] == "periodic_box"
        # A periodic box splits into whole buckets at least `cutoff` wide.
        n_buckets = max(int(size // cutoff), 1)
        width = size / n_buckets if periodic else cutoff

        for _ in range(steps):
            # Particles interact when their buckets, `width` wide, are at most
            # one apart on every axis: own @ band @ own.T per axis, where own
            # puts each particle in its bucket and band links adjacent buckets.
            # In a periodic box, buckets wrap around and positions never do.
            bx = xp.astype(xp.floor(x / width), xp.int64)
            by = xp.astype(xp.floor(y / width), xp.int64)
            bz = xp.astype(xp.floor(z / width), xp.int64)
            if periodic:
                bx = bx % n_buckets
                by = by % n_buckets
                bz = bz % n_buckets
                nx = ny = nz = n_buckets
            else:
                bx = bx - xp.min(bx)
                by = by - xp.min(by)
                bz = bz - xp.min(bz)
                nx = int(xp.max(bx)) + 1
                ny = int(xp.max(by)) + 1
                nz = int(xp.max(bz)) + 1

            own_x = xp.zeros((n_particles, nx), dtype=xp.uint8)
            own_y = xp.zeros((n_particles, ny), dtype=xp.uint8)
            own_z = xp.zeros((n_particles, nz), dtype=xp.uint8)
            own_x[particle_index, bx] = 1
            own_y[particle_index, by] = 1
            own_z[particle_index, bz] = 1

            band_x = (
                xp.eye(nx, k=-1, dtype=xp.uint8)
                + xp.eye(nx, dtype=xp.uint8)
                + xp.eye(nx, k=1, dtype=xp.uint8)
            )
            band_y = (
                xp.eye(ny, k=-1, dtype=xp.uint8)
                + xp.eye(ny, dtype=xp.uint8)
                + xp.eye(ny, k=1, dtype=xp.uint8)
            )
            band_z = (
                xp.eye(nz, k=-1, dtype=xp.uint8)
                + xp.eye(nz, dtype=xp.uint8)
                + xp.eye(nz, k=1, dtype=xp.uint8)
            )
            if periodic:
                band_x = band_x + xp.eye(nx, k=1 - nx, dtype=xp.uint8)
                band_x = band_x + xp.eye(nx, k=nx - 1, dtype=xp.uint8)
                band_y = band_y + xp.eye(ny, k=1 - ny, dtype=xp.uint8)
                band_y = band_y + xp.eye(ny, k=ny - 1, dtype=xp.uint8)
                band_z = band_z + xp.eye(nz, k=1 - nz, dtype=xp.uint8)
                band_z = band_z + xp.eye(nz, k=nz - 1, dtype=xp.uint8)

            near_x = xp.matmul(xp.matmul(own_x, band_x), xp.permute_dims(own_x, (1, 0)))
            near_y = xp.matmul(xp.matmul(own_y, band_y), xp.permute_dims(own_y, (1, 0)))
            near_z = xp.matmul(xp.matmul(own_z, band_z), xp.permute_dims(own_z, (1, 0)))
            candidates = (near_x > 0) & (near_y > 0) & (near_z > 0)

            # Masking each side before subtracting only forms differences for
            # candidate pairs, never the full x - x.T.
            dx = candidates * x - candidates * xp.reshape(x, (-1, 1))
            dy = candidates * y - candidates * xp.reshape(y, (-1, 1))
            dz = candidates * z - candidates * xp.reshape(z, (-1, 1))
            if periodic:
                # Each pair uses its nearest periodic image.
                dx = dx - size * xp.round(dx / size)
                dy = dy - size * xp.round(dy / size)
                dz = dz - size * xp.round(dz / size)
            r2 = dx * dx + dy * dy + dz * dz
            # Distance sets force strength only; bucket overlap selects pairs.
            # Nonneighbors have zero force, including the implicit sparse fill.
            r2 = xp.where(candidates, r2, xp.inf)
            r2 = xp.maximum(r2, softening * softening)
            r = xp.sqrt(r2)

            if force_model == "newtonian_gravity":
                coef = gravitational_constant * particle_mass / (r2 * r)
            else:
                coef = gravitational_constant * ((1 - cutoff / r) / r2 / particle_mass)
            # Forces ramp linearly to zero over the last 5% of the cutoff radius,
            # so they stay continuous as a pair moves out of range.
            wedge = xp.minimum(xp.maximum((cutoff - r) / (0.05 * cutoff), 0.0), 1.0)
            coef = coef * wedge

            ax = coef * dx
            ay = coef * dy
            az = coef * dz

            ax = xp.sum(ax, axis=1)
            ay = xp.sum(ay, axis=1)
            az = xp.sum(az, axis=1)

            # move particles
            vx += ax * dt
            vy += ay * dt
            vz += az * dt

            x += vx * dt
            y += vy * dt
            z += vz * dt

        return x, y, z, vx, vy, vz

    def check(self, param):
        for item in self._output:
            assert isinstance(item, BinsparseTensor), (
                "Output must be in binsparse format"
            )
        if self._ref_outputs is None:
            return

        # The reference holds every output's lower bound, then every upper bound.
        count = len(self._output)
        bounds = zip(self._ref_outputs[:count], self._ref_outputs[count:], strict=True)
        for i, (actual, (lower, upper)) in enumerate(
            zip(self._output, bounds, strict=True)
        ):
            try:
                actual_values = to_numpy(actual)
            except TypeError:
                actual_values = to_sparse(actual).todense()
            inside = (to_numpy(lower) <= actual_values) & (
                actual_values <= to_numpy(upper)
            )
            assert np.all(inside), (
                f"Particle simulation output {i} leaves its interval bound at "
                f"{np.count_nonzero(~inside)} particles for {param.dataset.name}"
            )
