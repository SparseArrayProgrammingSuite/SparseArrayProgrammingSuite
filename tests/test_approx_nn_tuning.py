import math

import pytest

import numpy as np

from saps.benchmarks.approx_nn import (
    SimHashApproxNNDenseGenerator,
    SimHashApproxNNRandomDataset,
    _collision_probability,
    _tune_lsh,
)


class _Dataset:
    def __init__(self, max_tables, max_projections, candidate_target):
        self.max_tables = max_tables
        self.max_projections = max_projections
        self.candidate_target = candidate_target


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


def test_tune_lsh_always_spends_the_full_table_budget():
    for max_tables in (1, 7, 64):
        dataset = _Dataset(max_tables=max_tables, max_projections=9, candidate_target=4)
        _, n_tables, _ = _tune_lsh(dataset, n_features=12, n_samples=9)
        assert n_tables == max_tables


def test_tune_lsh_bounds_expected_accidental_matches_by_candidate_target():
    dataset = _Dataset(max_tables=7, max_projections=16, candidate_target=100)
    n_samples = 63000
    n_projections, n_tables, _ = _tune_lsh(dataset, n_features=784, n_samples=n_samples)
    expected_at_n = n_samples * n_tables * 0.5**n_projections
    expected_at_n_minus_1 = n_samples * n_tables * 0.5 ** (n_projections - 1)
    # n_projections is the fewest signs that bring the expected number of
    # accidental (unrelated-point) matches down to candidate_target.
    assert expected_at_n <= dataset.candidate_target
    assert expected_at_n_minus_1 > dataset.candidate_target


def test_tune_lsh_clamps_to_max_projections_when_target_is_very_tight():
    dataset = _Dataset(max_tables=64, max_projections=16, candidate_target=1)
    n_projections, n_tables, _ = _tune_lsh(dataset, n_features=784, n_samples=10**9)
    assert n_projections == dataset.max_projections
    assert n_tables == dataset.max_tables


def test_tune_lsh_clamps_to_at_least_one_projection():
    dataset = _Dataset(max_tables=4, max_projections=16, candidate_target=10**9)
    n_projections, n_tables, _ = _tune_lsh(dataset, n_features=784, n_samples=10)
    assert n_projections == 1
    assert n_tables == dataset.max_tables


def test_tune_lsh_uses_max_projections_for_a_zero_candidate_target():
    dataset = _Dataset(max_tables=4, max_projections=9, candidate_target=0)
    n_projections, n_tables, _ = _tune_lsh(dataset, n_features=12, n_samples=9)
    assert n_projections == dataset.max_projections
    assert n_tables == dataset.max_tables


def test_tune_lsh_is_deterministic_and_depends_only_on_dataset_shape():
    dataset = _Dataset(max_tables=7, max_projections=9, candidate_target=4)
    assert _tune_lsh(dataset, n_features=12, n_samples=9) == _tune_lsh(
        dataset, n_features=12, n_samples=9
    )


def test_tune_lsh_reports_the_reference_similarity_probability_estimate():
    dataset = _Dataset(max_tables=7, max_projections=9, candidate_target=4)
    n_projections, n_tables, probability = _tune_lsh(
        dataset, n_features=12, n_samples=9
    )
    reference_similarity = 1.0 / math.sqrt(12)
    p = _collision_probability(reference_similarity)
    expected_probability = 1 - (1 - p**n_projections) ** n_tables
    assert probability == pytest.approx(expected_probability)


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
    assert zeros.meta["n_tables"] == random_data.meta["n_tables"] == dataset.max_tables
    assert (
        zeros.meta["estimated_retrieval_probability"]
        == random_data.meta["estimated_retrieval_probability"]
    )
