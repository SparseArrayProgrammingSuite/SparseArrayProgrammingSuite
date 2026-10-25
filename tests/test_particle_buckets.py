"""Interaction-region and force checks independent of the bucket indicator code."""

from pathlib import Path

import pytest

import numpy as np

from binsparse.conversions import to_numpy

from frameworks.saps_numpy import NumpyFramework
from frameworks.saps_smart import SmartSparseFramework
from saps.benchmarks.particle_simulation import (
    ParticleSimulationBenchmark,
    ParticleSimulationTestGenerator,
    _may_interact,
    particle_sim_bounds,
)
from saps.framework import load_framework
from saps.util.interval import Interval


def tagger_framework():
    path = Path(__file__).resolve().parents[1] / "frameworks/saps_tagger.py"
    return type(load_framework(path))()


def assert_within(result, bounds):
    for value, bound in zip(result, bounds, strict=True):
        value = np.asarray(value)
        assert np.all((bound.lo <= value) & (value <= bound.hi))


@pytest.mark.parametrize("cells, shifts", [(None, [0.0]), (20, [-2.0, 0.0, 2.0])])
def test_may_interact_keeps_pairs_within_a_cell_on_every_axis(cells, shifts):
    # Twenty cells 0.1 wide make a periodic box 2 wide, so images are 2 apart.
    rng = np.random.default_rng(3)
    lo = rng.uniform(-1, 1, (3, 300))
    hi = lo + rng.uniform(0, 0.05, (3, 300))
    intervals = [Interval(a, b) for a, b in zip(lo, hi, strict=True)]
    i, j = _may_interact(intervals, 0.1, cells)
    gap = np.min(
        [
            np.maximum(
                lo[:, None, :] + shift - hi[:, :, None],
                lo[:, :, None] - shift - hi[:, None, :],
            )
            for shift in shifts
        ],
        axis=0,
    )
    close = np.all(gap < 0.1, axis=0) & ~np.eye(300, dtype=bool)
    assert set(zip(*np.nonzero(close), strict=True)) <= set(zip(i, j, strict=True))


@pytest.mark.parametrize(
    "framework", [NumpyFramework, SmartSparseFramework, tagger_framework]
)
@pytest.mark.parametrize(
    "positions, weight",
    [
        ([[0.9, 0.9, 0.9], [1.1, 1.1, 1.1]], 1.0),  # Adjacent diagonal cells.
        ([[-0.1, 0, 0], [0.1, 0, 0]], 1.0),  # Across zero.
        ([[-9.1, 0, 0], [-8.9, 0, 0]], 1.0),  # Never clamp negative ids.
        ([[11.9, 0, 0], [12.1, 0, 0]], 1.0),  # Beyond the original box.
        ([[0.5, 0, 0], [1.475, 0, 0]], 0.5),  # Halfway down the wedge.
        ([[0.1, 0.1, 0.1], [1.9, 1.9, 1.9]], 0.0),  # Adjacent cells, out of range.
        ([[0.99, 0, 0], [2.0, 0, 0]], 0.0),  # Two cells apart.
    ],
)
def test_bucket_interaction_region(framework, positions, weight):
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
    result = ParticleSimulationBenchmark().benchmark(
        xp,
        {"size": 1.0, "steps": 1, "parameters": parameters},
        *[xp.asarray(a.copy()) for a in data],
    )
    velocity = np.array([np.asarray(a) for a in result[3:]])
    delta = position[1] - position[0]
    force = weight * delta / max(np.dot(delta, delta), 0.05**2) ** 1.5
    expected = np.column_stack([0.75 * force, -0.25 * force]) * 0.125
    np.testing.assert_allclose(velocity, expected, atol=1e-14)
    assert_within(result, particle_sim_bounds(*data, 1.0, 1, parameters))


@pytest.mark.parametrize(
    "framework", [NumpyFramework, SmartSparseFramework, tagger_framework]
)
@pytest.mark.parametrize(
    "positions, weight",
    [
        ([[0.1, 0.5, 0.5], [3.95, 0.5, 0.5]], 1.0),  # Across one face.
        ([[0.1, 0.1, 0.1], [3.9, 3.9, 3.9]], 1.0),  # Across a corner.
        ([[0.2, 0, 0], [7.95, 0, 0]], 1.0),  # Positions never wrap.
        ([[0.5, 0, 0], [2.5, 0, 0]], 0.0),  # Half a box apart.
    ],
)
def test_periodic_pairs_interact_through_nearest_image(framework, positions, weight):
    xp = framework()
    position = np.asarray(positions)
    masses = np.array([0.25, 0.75])
    data = [*position.T, *np.zeros((3, 2)), masses]
    parameters = {
        "force_model": "newtonian_gravity",
        "boundary_model": "periodic_box",
        "cutoff": 1.0,
        "softening": 0.05,
        "dt": 0.125,
        "gravitational_constant": 1.0,
    }
    result = ParticleSimulationBenchmark().benchmark(
        xp,
        {"size": 4.0, "steps": 1, "parameters": parameters},
        *[xp.asarray(a.copy()) for a in data],
    )
    velocity = np.array([np.asarray(a) for a in result[3:]])
    delta = position[1] - position[0]
    delta -= 4.0 * np.round(delta / 4.0)
    force = weight * delta / max(np.dot(delta, delta), 0.05**2) ** 1.5
    expected = np.column_stack([0.75 * force, -0.25 * force]) * 0.125
    np.testing.assert_allclose(velocity, expected, atol=1e-14)
    assert_within(result, particle_sim_bounds(*data, 4.0, 1, parameters))


@pytest.mark.parametrize(
    "framework", [NumpyFramework, SmartSparseFramework, tagger_framework]
)
def test_bucket_simulation_stays_within_bounds(framework):
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
    bounds = particle_sim_bounds(*data, 1.0, 5, parameters)
    for bound in bounds:
        assert np.all(bound.hi - bound.lo <= 1e-10 * np.maximum(1, abs(bound.hi)))
    xp = framework()
    result = ParticleSimulationBenchmark().benchmark(
        xp,
        {"size": 1.0, "steps": 5, "parameters": parameters},
        *[xp.asarray(a.copy()) for a in data],
    )
    assert_within(result, bounds)


@pytest.mark.parametrize(
    "framework", [NumpyFramework, SmartSparseFramework, tagger_framework]
)
def test_repulsive_test_datasets_stay_within_bounds(framework):
    generator = ParticleSimulationTestGenerator()
    for dataset in generator.datasets:
        problem = generator.generate(dataset)
        xp = framework()
        result = ParticleSimulationBenchmark().benchmark(
            xp,
            problem.meta,
            *[xp.from_binsparse(a) for a in problem.inputs],
        )
        lower = [to_numpy(a) for a in problem.ref_outputs[:6]]
        upper = [to_numpy(a) for a in problem.ref_outputs[6:]]
        for value, lo, hi in zip(result, lower, upper, strict=True):
            value = np.asarray(value)
            assert np.all(hi - lo <= 1e-10 * np.maximum(1, abs(hi)))
            assert np.all((lo <= value) & (value <= hi))
