import math
from typing import Any

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
from saps.downloaders.nemo import download_nemo_dataset


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


class ParticleSimDataset(Dataset):
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


class Particle:
    def __init__(
        self,
        x=0.0,
        y=0.0,
        z=0.0,
        vx=0.0,
        vy=0.0,
        vz=0.0,
        ax=0.0,
        ay=0.0,
        az=0.0,
        *,
        particle_mass: float,
    ):
        self.x = x
        self.y = y
        self.z = z
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.ax = ax
        self.ay = ay
        self.az = az
        self.mass = particle_mass


def apply_force(particle, neighbor, parameters):
    dx = neighbor.x - particle.x
    dy = neighbor.y - particle.y
    dz = neighbor.z - particle.z
    r2 = dx * dx + dy * dy + dz * dz
    gravitational_constant = parameters["gravitational_constant"]
    cutoff = parameters["cutoff"]

    if r2 > cutoff * cutoff:
        return

    softening = parameters["softening"]
    r2 = max(r2, softening * softening)
    r = math.sqrt(r2)

    if parameters["force_model"] == "newtonian_gravity":
        coef = gravitational_constant * neighbor.mass / (r2 * r)
    else:
        coef = gravitational_constant * ((1 - cutoff / r) / r2 / particle.mass)
    particle.ax += coef * dx
    particle.ay += coef * dy
    particle.az += coef * dz


def move(p, size, parameters):
    dt = parameters["dt"]
    p.vx += p.ax * dt
    p.vy += p.ay * dt
    p.vz += p.az * dt
    p.x += p.vx * dt
    p.y += p.vy * dt
    p.z += p.vz * dt

    if parameters["boundary_model"] != "reflective_box":
        return

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

    if p.z < 0 or p.z > size:
        if p.z < 0:
            p.z = -p.z
        else:
            p.z = 2 * size - p.z
        p.vz = -p.vz


def simulate_one_step(parts, num_parts, size, parameters):
    for i in range(num_parts):
        parts[i].ax = 0
        parts[i].ay = 0
        parts[i].az = 0
        for j in range(num_parts):
            apply_force(parts[i], parts[j], parameters)

    for i in range(num_parts):
        move(parts[i], size, parameters)


def init_simulation(parts, num_parts, size, steps, parameters):
    for _ in range(steps):
        simulate_one_step(parts, num_parts, size, parameters)


def reference_particle_sim(
    x,
    y,
    z,
    vx,
    vy,
    vz,
    size,
    steps,
    parameters,
    particle_mass,
):
    ref_particles = [
        Particle(xi, yi, zi, vxi, vyi, vzi, 0, 0, 0, particle_mass=float(mi))
        for xi, yi, zi, vxi, vyi, vzi, mi in zip(
            x,
            y,
            z,
            vx,
            vy,
            vz,
            _mass_values(particle_mass, len(x)),
            strict=True,
        )
    ]
    init_simulation(ref_particles, len(ref_particles), size, steps, parameters)
    return [
        np.array([p.x for p in ref_particles]),
        np.array([p.y for p in ref_particles]),
        np.array([p.z for p in ref_particles]),
        np.array([p.vx for p in ref_particles]),
        np.array([p.vy for p in ref_particles]),
        np.array([p.vz for p in ref_particles]),
    ]


def _mass_values(particle_mass, n_particles):
    if getattr(particle_mass, "ndim", 0) == 0:
        return np.full(n_particles, float(particle_mass), dtype=np.float64)
    return particle_mass


class ParticleSimTestGenerator(Generator[ParticleSimDataset]):
    @property
    def name(self):
        return "particle_sim_test_inputs"

    @property
    def pretty_name(self):
        return "Particle Simulation Test Input Generator"

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
            ParticleSimDataset(
                "test_particle_sim_two_particles_within_cutoff",
                suites=["test", "trace"],
                parameters={
                    "force_model": "cs267_repulsive",
                    "boundary_model": "reflective_box",
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
            ParticleSimDataset(
                "test_particle_sim_wall_bounce",
                suites=["test", "trace"],
                parameters={
                    "force_model": "cs267_repulsive",
                    "boundary_model": "reflective_box",
                    "cutoff": 0.01,
                    "softening": 0.0001,
                    "dt": 0.0005,
                    "gravitational_constant": 1.0,
                },
                values=(
                    np.array([0.0001]),
                    np.array([0.05]),
                    np.array([0.05]),
                    np.array([-0.1]),
                    np.array([0.0]),
                    np.array([0.0]),
                    np.asarray(0.01),
                    2,
                    5,
                ),
            ),
            ParticleSimDataset(
                "test_particle_sim_random_10",
                suites=["test", "trace"],
                parameters={
                    "force_model": "cs267_repulsive",
                    "boundary_model": "reflective_box",
                    "cutoff": 0.01,
                    "softening": 0.0001,
                    "dt": 0.0005,
                    "gravitational_constant": 1.0,
                },
                values=generate_particle_test_data(10, 2, 10, 0.01),
            ),
            ParticleSimDataset(
                "test_particle_sim_random_50",
                suites=["test", "trace"],
                parameters={
                    "force_model": "cs267_repulsive",
                    "boundary_model": "reflective_box",
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
        expected = reference_particle_sim(
            x,
            y,
            z,
            vx,
            vy,
            vz,
            size,
            steps,
            parameters,
            mass,
        )
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
            ref_outputs=[from_numpy(value) for value in expected],
        )


class SyntheticParticleSimDataset(ParticleSimDataset):
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


class SyntheticBerkeleyCS267ParticleGenerator(Generator[SyntheticParticleSimDataset]):
    @property
    def cacheable(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return "synthetic_berkeley_cs267_particle"

    @property
    def pretty_name(self) -> str:
        return "Synthetic Berkeley CS267 Particle Generator"

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
    def datasets(self) -> list[SyntheticParticleSimDataset]:
        return [
            SyntheticParticleSimDataset(
                name="cs267_hw2_n1000_seed1",
                n_particles=1000,
                num_steps=1000,
                seed=1,
                density=0.0005,
                suites=["standard"],
                parameters={
                    "force_model": "cs267_repulsive",
                    "boundary_model": "reflective_box",
                    "cutoff": 0.01,
                    "softening": 0.0001,
                    "dt": 0.0005,
                    "gravitational_constant": 1.0,
                },
                pretty_name="CS267 HW2 Initial Conditions (N=1000)",
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

    def generate(self, dataset: SyntheticParticleSimDataset):
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


class EWAPParticleSimDataset(ParticleSimDataset):
    def __init__(
        self,
        name: str,
        scene: str,
        num_steps: int,
        pretty_name: str,
        description: str,
        parameters: dict[str, Any],
    ):
        super().__init__(
            name=name,
            num_steps=num_steps,
            pretty_name=pretty_name,
            description=description,
            suites=["standard"],
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


class EWAPParticleSimGenerator(Generator[EWAPParticleSimDataset]):
    @property
    def cacheable(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return "ewap_particle_sim"

    @property
    def pretty_name(self) -> str:
        return "ETH EWAP Particle Simulation Generator"

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
    def datasets(self) -> list[EWAPParticleSimDataset]:
        return [
            EWAPParticleSimDataset(
                name="ewap_seq_eth",
                scene="seq_eth",
                num_steps=50,
                parameters={
                    "force_model": "cs267_repulsive",
                    "boundary_model": "reflective_box",
                    "cutoff": 0.01,
                    "softening": 0.0001,
                    "dt": 0.0005,
                    "gravitational_constant": 1.0,
                },
                pretty_name="ETH EWAP ETH Scene",
                description=(
                    "ETH EWAP seq_eth pedestrian trajectories loaded in dataset "
                    "coordinates with z and vz preserved from the source column. "
                    "The force parameters are the CS267-style repulsive benchmark "
                    "constants because EWAP does not provide simulation constants."
                ),
            ),
            EWAPParticleSimDataset(
                name="ewap_seq_hotel",
                scene="seq_hotel",
                num_steps=50,
                parameters={
                    "force_model": "cs267_repulsive",
                    "boundary_model": "reflective_box",
                    "cutoff": 0.01,
                    "softening": 0.0001,
                    "dt": 0.0005,
                    "gravitational_constant": 1.0,
                },
                pretty_name="ETH EWAP Hotel Scene",
                description=(
                    "ETH EWAP seq_hotel pedestrian trajectories loaded in dataset "
                    "coordinates with z and vz preserved from the source column. "
                    "The force parameters are the CS267-style repulsive benchmark "
                    "constants because EWAP does not provide simulation constants."
                ),
            ),
        ]

    def generate(self, dataset: EWAPParticleSimDataset):
        inputs, meta = download_ewap_dataset(
            dataset.scene,
            num_steps=dataset.num_steps,
        )
        meta["parameters"] = dataset.parameters
        return DataInstance(inputs=inputs, meta=meta)


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
        return (
            "Loads real-world initial conditions for particle simulation benchmarks. "
            "NEMO datasets use source mass columns, unscaled archive coordinates, "
            "Newtonian gravity with G 1.0 in N-body units, dt 1/32, softening 0.05, "
            "and source-scale cutoffs selected for the benchmark datasets."
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
            "the NEMO eps-style softened-gravity setting, and cutoffs are chosen "
            "against the preserved source coordinate scale."
        )

    @property
    def datasets(self) -> list[ParticleSimDataset]:
        return [
            ParticleSimDataset(
                name="nemo_plummer_128",
                n_particles=128,
                num_steps=50,
                parameters={
                    "force_model": "newtonian_gravity",
                    "boundary_model": "unbounded",
                    "cutoff": 1.0,
                    "dt": 1.0 / 32.0,
                    "softening": 0.05,
                    "gravitational_constant": 1.0,
                },
                pretty_name="NEMO Plummer (N=128)",
                description=(
                    "NEMO Plummer-model equilibrium snapshot generated with mkplummer,"
                    " using mass, position, and velocity columns. The dataset uses "
                    "source masses, G 1.0 N-body units, dt 1/32, softening 0.05, "
                    "and cutoff 1.0 for the preserved Plummer coordinate scale."
                ),
                suites=["standard"],
                tags=["physics", "simulation", "sparse", "astronomy", "n-body"],
                source_path="plummer/tab128.gz",
                source_columns=("mass", "x", "y", "z", "vx", "vy", "vz"),
            ),
            ParticleSimDataset(
                name="nemo_plummer_1024",
                n_particles=1024,
                num_steps=50,
                parameters={
                    "force_model": "newtonian_gravity",
                    "boundary_model": "unbounded",
                    "cutoff": 1.0,
                    "dt": 1.0 / 32.0,
                    "softening": 0.05,
                    "gravitational_constant": 1.0,
                },
                pretty_name="NEMO Plummer (N=1024)",
                description=(
                    "NEMO Plummer-model equilibrium snapshot generated with mkplummer,"
                    " using mass, position, and velocity columns. The dataset uses "
                    "source masses, G 1.0 N-body units, dt 1/32, softening 0.05, "
                    "and cutoff 1.0 for the preserved Plummer coordinate scale."
                ),
                suites=["standard"],
                tags=["physics", "simulation", "sparse", "astronomy", "n-body"],
                source_path="plummer/tab1024.gz",
                source_columns=("mass", "x", "y", "z", "vx", "vy", "vz"),
            ),
            ParticleSimDataset(
                name="nemo_dubinski_m31",
                n_particles=81920,
                num_steps=10,
                parameters={
                    "force_model": "newtonian_gravity",
                    "boundary_model": "unbounded",
                    "cutoff": 20.0,
                    "dt": 1.0 / 32.0,
                    "softening": 0.05,
                    "gravitational_constant": 1.0,
                },
                pretty_name="NEMO Dubinski MW/M31",
                description=(
                    "Dubinski Milky Way/Andromeda collision initial conditions from the"
                    " NEMO archive, stored as mass and six phase-space coordinates. "
                    "The dataset uses source masses, G 1.0 N-body units, dt 1/32, "
                    "softening 0.05, and cutoff 20.0 for the larger preserved "
                    "Dubinski coordinate scale."
                ),
                suites=["standard"],
                tags=["physics", "simulation", "sparse", "astronomy", "n-body"],
                source_path="dubinski/dubinski.tab.gz",
                source_columns=("mass", "x", "y", "z", "vx", "vy", "vz"),
            ),
        ]

    def generate(self, dataset: ParticleSimDataset):
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
        return [
            ParticleSimTestGenerator(),
            SyntheticBerkeleyCS267ParticleGenerator(),
            EWAPParticleSimGenerator(),
            ParticleSimGenerator(),
        ]

    def benchmark(self, xp, data, meta):
        x, y, z, vx, vy, vz, particle_mass = data
        size = meta["size"]
        steps = meta["steps"]
        parameters = meta["parameters"]
        dt = parameters["dt"]
        gravitational_constant = parameters["gravitational_constant"]
        force_model = parameters["force_model"]

        for _ in range(steps):
            # compute forces
            dx = x - x.reshape(-1, 1)
            dy = y - y.reshape(-1, 1)
            dz = z - z.reshape(-1, 1)
            r2 = dx * dx + dy * dy + dz * dz
            cutoff = parameters["cutoff"]
            softening = parameters["softening"]
            mask = r2 > cutoff * cutoff
            r2 = xp.where(mask, xp.inf, r2)
            r2 = xp.maximum(r2, softening * softening)
            r = xp.sqrt(r2)

            if force_model == "newtonian_gravity":
                coef = gravitational_constant * particle_mass / (r2 * r)
            else:
                coef = gravitational_constant * ((1 - cutoff / r) / r2 / particle_mass)

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

            if parameters["boundary_model"] != "reflective_box":
                continue

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

            # z
            reflected = (z < 0) | (z > size)
            vz = xp.where(reflected, -vz, vz)

            z1 = xp.abs(z)
            z2 = 2 * size - z
            z = xp.where(z > size, z2, z1)

        return [x, y, z, vx, vy, vz]

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
