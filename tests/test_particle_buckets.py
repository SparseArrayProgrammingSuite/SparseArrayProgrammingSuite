"""Interaction-region and force checks independent of the bucket indicator code."""

from pathlib import Path

import pytest

import numpy as np

from binsparse.conversions import to_numpy

from frameworks.saps_numpy import NumpyFramework
from frameworks.saps_smart import SmartSparseFramework
from saps.benchmarks.particle_sim import (
    ParticleSimBenchmark,
    ParticleSimTestGenerator,
    reference_particle_sim,
)
from saps.framework import load_framework


def tagger_framework():
    path = Path(__file__).resolve().parents[1] / "frameworks/saps_tagger.py"
    return type(load_framework(path))()


@pytest.mark.parametrize(
    "framework", [NumpyFramework, SmartSparseFramework, tagger_framework]
)
@pytest.mark.parametrize(
    "positions, interacts",
    [
        ([[0.1, 0.1, 0.1], [1.9, 1.9, 1.9]], True),  # Adjacent diagonal cells.
        ([[0.99, 0, 0], [2.0, 0, 0]], False),  # Two cells apart.
        ([[-0.1, 0, 0], [0.1, 0, 0]], True),  # Across zero.
        ([[-10.1, 0, 0], [-8.9, 0, 0]], False),  # Never clamp negative ids.
        ([[10.1, 0, 0], [12.1, 0, 0]], False),  # Beyond the original box.
        ([[0.1, 0.1, 0.1], [0.1, 2.1, 0.1]], False),  # Intersect all axes.
        ([[0.1, 0.1, 0.1], [0.1, 0.1, 2.1]], False),
    ],
)
def test_bucket_interaction_region(framework, positions, interacts):
    xp = framework()
    position = np.asarray(positions)
    masses = np.array([0.25, 0.75])
    data = [*position.T, *np.zeros((3, 2)), masses]
    parameters = {
        "force_model": "newtonian_gravity",
        "boundary_model": "unbounded",
        "cutoff": 1.0,
        "softening": 0.05,
        "dt": 0.125,
        "gravitational_constant": 1.0,
    }
    result = ParticleSimBenchmark().benchmark(
        xp,
        {"size": 1.0, "steps": 1, "parameters": parameters},
        *[xp.asarray(a.copy()) for a in data],
    )
    velocity = np.array([np.asarray(a) for a in result[3:]])
    delta = position[1] - position[0]
    force = delta / max(np.dot(delta, delta), 0.05**2) ** 1.5
    expected = np.column_stack([0.75 * force, -0.25 * force]) * 0.125
    if not interacts:
        expected[:] = 0
    np.testing.assert_allclose(velocity, expected, atol=1e-14)
    scalar = reference_particle_sim(
        *[a.copy() for a in data[:6]], 1.0, 1, parameters, masses
    )
    np.testing.assert_allclose(np.array(scalar[3:]), expected, atol=1e-14)


@pytest.mark.parametrize(
    "framework", [NumpyFramework, SmartSparseFramework, tagger_framework]
)
def test_bucket_simulation_matches_scalar_reference(framework):
    # Moving particles cross cells and the original box; unequal masses also
    # exercise nonconstant broadcasting in the sparse backend.
    rng = np.random.default_rng(14)
    data = [
        *rng.uniform(-2, 2, (3, 8)),
        *rng.uniform(-3, 3, (3, 8)),
        rng.uniform(0.1, 1, 8),
    ]
    parameters = {
        "force_model": "newtonian_gravity",
        "boundary_model": "unbounded",
        "cutoff": 1.0,
        "softening": 0.05,
        "dt": 0.1,
        "gravitational_constant": 1.0,
    }
    expected = reference_particle_sim(
        *[a.copy() for a in data[:6]], 1.0, 5, parameters, data[6]
    )
    xp = framework()
    result = ParticleSimBenchmark().benchmark(
        xp,
        {"size": 1.0, "steps": 5, "parameters": parameters},
        *[xp.asarray(a.copy()) for a in data],
    )
    np.testing.assert_allclose([np.asarray(a) for a in result], expected, atol=1e-12)


@pytest.mark.parametrize(
    "framework", [NumpyFramework, SmartSparseFramework, tagger_framework]
)
def test_repulsive_test_datasets_match_updated_reference(framework):
    generator = ParticleSimTestGenerator()
    for dataset in generator.datasets:
        problem = generator.generate(dataset)
        xp = framework()
        result = ParticleSimBenchmark().benchmark(
            xp,
            problem.meta,
            *[xp.from_binsparse(a) for a in problem.inputs],
        )
        np.testing.assert_allclose(
            [np.asarray(a) for a in result],
            [to_numpy(a) for a in problem.ref_outputs],
            atol=1e-12,
        )
