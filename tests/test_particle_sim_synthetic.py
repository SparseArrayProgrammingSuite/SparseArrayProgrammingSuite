from __future__ import annotations

import math

import numpy as np

from binsparse.conversions import to_numpy

from saps.benchmarks.particle_sim import (
    ParticleSimBenchmark,
    SyntheticBerkeleyCS267ParticleGenerator,
    particle_density_box_size,
)


def test_synthetic_berkeley_cs267_particle_generator_has_one_dataset():
    generator = SyntheticBerkeleyCS267ParticleGenerator()

    assert [dataset.name for dataset in generator.datasets] == ["cs267_hw2_n1000_seed1"]

    dataset = generator.datasets[0]
    assert dataset.n_particles == 1000
    assert dataset.num_steps == 1000
    assert dataset.seed == 1
    assert dataset.suites == ["standard"]
    assert dataset.parameters == {
        "force_model": "cs267_repulsive",
        "boundary_model": "reflective_box",
        "cutoff": 0.01,
        "softening": 0.0001,
        "dt": 0.0005,
        "gravitational_constant": 1.0,
    }
    assert dataset.density == 0.0005


def test_synthetic_generator_matches_assignment_layout():
    generator = SyntheticBerkeleyCS267ParticleGenerator()
    dataset = generator.datasets[0]
    instance = generator.generate(dataset)
    n = dataset.n_particles
    x = to_numpy(instance.inputs[0])
    y = to_numpy(instance.inputs[1])
    z = to_numpy(instance.inputs[2])
    vx = to_numpy(instance.inputs[3])
    vy = to_numpy(instance.inputs[4])
    vz = to_numpy(instance.inputs[5])
    mass = to_numpy(instance.inputs[6])
    size = instance.meta["size"]

    sx = math.ceil(math.pow(n, 1.0 / 3.0))
    sy = math.ceil(math.sqrt(n / sx))
    sz = (n + sx * sy - 1) // (sx * sy)
    grid_x = {size * (1.0 + (k % sx)) / (1 + sx) for k in range(n)}
    grid_y = {size * (1.0 + ((k // sx) % sy)) / (1 + sy) for k in range(n)}
    grid_z = {size * (1.0 + (k // (sx * sy))) / (1 + sz) for k in range(n)}

    assert math.isclose(
        size,
        particle_density_box_size(n, dataset.density),
    )
    assert len(x) == n
    assert set(x).issubset(grid_x)
    assert set(y).issubset(grid_y)
    assert set(z).issubset(grid_z)
    assert len(set(z)) > 1
    assert np.all(vx >= -1.0)
    assert np.all(vx <= 1.0)
    assert np.all(vy >= -1.0)
    assert np.all(vy <= 1.0)
    assert np.all(vz >= -1.0)
    assert np.all(vz <= 1.0)

    np.testing.assert_allclose(mass, np.asarray(0.01))
    assert mass.shape == ()

    again = generator.generate(generator.datasets[0])
    for actual, expected_input in zip(
        (x, y, z, vx, vy, vz, mass),
        again.inputs,
        strict=True,
    ):
        expected = to_numpy(expected_input)
        np.testing.assert_allclose(actual, expected)


def test_synthetic_generator_outputs_cs267_metadata():
    generator = SyntheticBerkeleyCS267ParticleGenerator()
    dataset = generator.datasets[0]
    instance = generator.generate(dataset)

    assert instance.meta["n_particles"] == dataset.n_particles
    assert instance.meta["steps"] == dataset.num_steps
    assert instance.meta["seed"] == dataset.seed
    assert instance.meta["density"] == dataset.density
    assert instance.meta["parameters"] == dataset.parameters
    assert instance.meta["source"] == "Berkeley CS267 HW2 init_particles extended to 3D"
    assert instance.meta["source_dimensions"] == 3
    assert instance.meta["simulation_dimensions"] == 3

    x = to_numpy(instance.inputs[0])
    assert x.shape == (dataset.n_particles,)
    assert len(instance.inputs) == 7


def test_particle_sim_benchmark_registers_synthetic_generator():
    assert [generator.name for generator in ParticleSimBenchmark().generators] == [
        "particle_sim_test_inputs",
        "synthetic_berkeley_cs267_particle",
        "ewap_particle_sim",
        "particle_sim",
    ]
