import math

import pytest

import numpy as np
import scipy.integrate

from saps.benchmarks.approx_nn import (
    JLApproxNNDenseGenerator,
    JLApproxNNRandomDataset,
    _collision_probability,
    _tables_needed,
    _tune_lsh,
)


class _Dataset:
    def __init__(self, max_tables, max_projections, target_probability):
        self.max_tables = max_tables
        self.max_projections = max_projections
        self.target_probability = target_probability


@pytest.mark.parametrize("width_ratio", [0.1, 1.0, 4.0, 20.0])
def test_collision_probability_matches_e2lsh_integral(width_ratio):
    expected, _ = scipy.integrate.quad(
        lambda z: (1 - z / width_ratio) * math.sqrt(2 / math.pi) * math.exp(-z * z / 2),
        0,
        width_ratio,
    )
    assert _collision_probability(width_ratio) == pytest.approx(
        expected, rel=1e-8, abs=1e-12
    )


def test_collision_probability_limits_and_monotonicity():
    ratios = [0, 1e-6, 1, 4, 20, math.inf]
    probabilities = [_collision_probability(r) for r in ratios]
    assert probabilities[0] == 0
    assert probabilities[-1] == 1
    # Small-ratio slope matches the density at the origin, sqrt(2/pi)/2.
    assert probabilities[1] / ratios[1] == pytest.approx(
        1 / math.sqrt(2 * math.pi), rel=1e-3
    )
    assert all(a <= b for a, b in zip(probabilities, probabilities[1:], strict=False))


def test_tables_needed_is_the_minimum_that_reaches_target():
    for hit_probability, target in [(0.3, 0.9), (0.05, 0.99), (0.8, 0.999)]:
        tables = _tables_needed(hit_probability, target)
        assert 1 - (1 - hit_probability) ** tables >= target
        assert tables == 1 or 1 - (1 - hit_probability) ** (tables - 1) < target


def test_tables_needed_certain_hit_uses_one_table():
    assert _tables_needed(1.0, 0.999) == 1


def test_tune_lsh_meets_target_with_the_fewest_total_hashes():
    dataset = _Dataset(max_tables=9, max_projections=6, target_probability=0.9)
    n_projections, n_tables, widen, probability = _tune_lsh(dataset)
    assert probability >= dataset.target_probability
    assert 1 <= n_tables <= dataset.max_tables
    assert 1 <= n_projections <= dataset.max_projections
    assert widen == pytest.approx(1.0)
    if n_tables > 1:
        p = _collision_probability(4.0) ** n_projections
        assert 1 - (1 - p) ** (n_tables - 1) < dataset.target_probability


def test_tune_lsh_widens_the_reference_ratio_when_the_table_budget_is_tight():
    dataset = _Dataset(max_tables=1, max_projections=3, target_probability=0.999)
    n_projections, n_tables, widen, probability = _tune_lsh(dataset)
    assert n_tables == 1
    assert probability >= dataset.target_probability
    # A single table can't hit 0.999 at the unscaled reference ratio (4.0).
    assert widen > 1.0


def test_tune_lsh_is_deterministic_and_depends_only_on_the_dataset():
    dataset = _Dataset(max_tables=9, max_projections=6, target_probability=0.9)
    assert _tune_lsh(dataset) == _tune_lsh(dataset)


def test_generate_falls_back_to_a_unit_radius_for_coincident_points():
    generator = JLApproxNNDenseGenerator()
    dataset = JLApproxNNRandomDataset(
        "coincident",
        "Coincident",
        "Coincident",
        [],
        10,
        2,
        3,
        2,
        0.1,
        42,
        max_tables=64,
        max_projections=16,
        candidate_target=100,
        target_probability=0.9,
    )
    instance = generator._instance(dataset, np.zeros((10, 2)), np.zeros((3, 2)))
    assert instance.meta["calibration_radius"] == 1.0
    assert (
        instance.meta["estimated_retrieval_probability"] >= dataset.target_probability
    )


def test_generate_picks_a_tuning_that_meets_target_on_clustered_data():
    rng = np.random.default_rng(10)
    centers = rng.normal(size=(8, 3)) * 20
    data = centers[np.arange(800) % 8] + rng.normal(size=(800, 3))
    query = centers[np.arange(240) % 8] + rng.normal(size=(240, 3))
    dataset = JLApproxNNRandomDataset(
        "clustered",
        "Clustered",
        "Clustered",
        [],
        800,
        3,
        240,
        3,
        0.1,
        42,
        max_tables=9,
        max_projections=6,
        candidate_target=100,
        target_probability=0.9,
    )
    generator = JLApproxNNDenseGenerator()
    instance = generator._instance(dataset, data, query)
    repeated = generator._instance(dataset, data, query)
    scaled = generator._instance(dataset, 13 * data, 13 * query)
    assert instance.meta == repeated.meta
    assert 1 < instance.meta["n_tables"] <= dataset.max_tables
    assert 1 <= instance.meta["n_projections"] <= dataset.max_projections
    assert (
        instance.meta["estimated_retrieval_probability"] >= dataset.target_probability
    )
    assert scaled.meta["n_tables"] == instance.meta["n_tables"]
    assert scaled.meta["n_projections"] == instance.meta["n_projections"]
    assert scaled.meta["bucket_width_scale"] == pytest.approx(
        13 * instance.meta["bucket_width_scale"]
    )
