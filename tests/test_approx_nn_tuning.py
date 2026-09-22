import math

import pytest

import numpy as np

from saps.benchmarks.approx_nn import (
    SimHashApproxNNDenseGenerator,
    SimHashApproxNNRandomDataset,
    _collision_probability,
    _tables_needed,
    _tune_lsh,
)


class _Dataset:
    def __init__(self, max_tables, max_projections, target_probability):
        self.max_tables = max_tables
        self.max_projections = max_projections
        self.target_probability = target_probability


@pytest.mark.parametrize("cosine_similarity", [-1.0, -0.5, 0.0, 0.5, 1.0])
def test_collision_probability_matches_charikar_formula(cosine_similarity):
    expected = 1 - math.acos(cosine_similarity) / math.pi
    assert _collision_probability(cosine_similarity) == pytest.approx(
        expected, rel=1e-8, abs=1e-12
    )


def test_collision_probability_limits_and_monotonicity():
    similarities = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
    probabilities = [_collision_probability(s) for s in similarities]
    assert probabilities[0] == probabilities[1] == 0.0
    assert probabilities[-1] == probabilities[-2] == 1.0
    assert probabilities[3] == pytest.approx(0.5)
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
    n_projections, n_tables, probability = _tune_lsh(dataset, n_features=64)
    assert probability >= dataset.target_probability
    assert 1 <= n_tables <= dataset.max_tables
    assert 1 <= n_projections <= dataset.max_projections
    if n_projections > 1:
        cheaper = _tune_lsh(
            _Dataset(
                max_tables=dataset.max_tables,
                max_projections=n_projections - 1,
                target_probability=dataset.target_probability,
            ),
            n_features=64,
        )
        assert cheaper[2] < dataset.target_probability or (
            cheaper[0] * cheaper[1] >= n_projections * n_tables
        )


def test_tune_lsh_falls_back_to_the_best_effort_when_target_is_unreachable():
    dataset = _Dataset(max_tables=1, max_projections=1, target_probability=0.999999)
    n_projections, n_tables, probability = _tune_lsh(dataset, n_features=10000)
    assert n_projections == 1
    assert n_tables == 1
    # A single table and projection can't reach 0.999999 for a near-orthogonal
    # reference similarity; _tune_lsh should still return its best option
    # rather than looping or raising.
    assert 0 < probability < dataset.target_probability


def test_tune_lsh_is_deterministic_and_depends_only_on_dataset_and_n_features():
    dataset = _Dataset(max_tables=9, max_projections=6, target_probability=0.9)
    assert _tune_lsh(dataset, n_features=64) == _tune_lsh(dataset, n_features=64)


def test_tune_lsh_needs_more_hashes_as_features_grow():
    # More features means a weaker reference similarity (1/sqrt(n_features)),
    # so tuning should never get cheaper as n_features grows.
    dataset = _Dataset(max_tables=64, max_projections=16, target_probability=0.9)
    small = _tune_lsh(dataset, n_features=4)
    large = _tune_lsh(dataset, n_features=4096)
    assert small[0] * small[1] <= large[0] * large[1]


def test_generate_tuning_depends_only_on_shape_not_data_values():
    generator = SimHashApproxNNDenseGenerator()
    dataset = SimHashApproxNNRandomDataset(
        "custom",
        "Custom",
        "Custom",
        [],
        9,
        12,
        3,
        2,
        0.1,
        42,
        max_tables=7,
        max_projections=9,
        candidate_target=4,
        target_probability=0.9,
    )
    zeros = generator._instance(dataset, np.zeros((9, 12)), np.zeros((3, 12)))
    rng = np.random.default_rng(0)
    random_data = generator._instance(
        dataset, rng.standard_normal((9, 12)), rng.standard_normal((3, 12))
    )
    assert zeros.meta["n_projections"] == random_data.meta["n_projections"]
    assert zeros.meta["n_tables"] == random_data.meta["n_tables"]
    assert (
        zeros.meta["estimated_retrieval_probability"]
        == random_data.meta["estimated_retrieval_probability"]
    )
    assert (
        zeros.meta["estimated_retrieval_probability"] >= dataset.target_probability
        or zeros.meta["n_tables"] == dataset.max_tables
    )
