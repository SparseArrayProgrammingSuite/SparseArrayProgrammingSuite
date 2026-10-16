import itertools
from types import SimpleNamespace

import pytest

import numpy as np

from binsparse.conversions import from_numpy

from saps.benchmarks import weighted_model_counting
from saps.benchmarks.weighted_model_counting import (
    WeightedModelCountingBenchmark,
    WeightedModelCountingMCCompGenerator,
    WeightedModelCountingTestGenerator,
    parse_format,
    referenceWeightedModelCount,
)


def _loop_weighted_model_count(num_vars, clauses, weights):
    total = 0.0
    for bits in itertools.product([False, True], repeat=num_vars):
        if not all(
            any(bits[abs(lit) - 1] == (lit > 0) for lit in clause) for clause in clauses
        ):
            continue
        product = 1.0
        for i in range(1, num_vars + 1):
            product *= weights[i] if bits[i - 1] else weights[-i]
        total += product
    return total


@pytest.mark.parametrize(
    "num_vars,clauses,weights,expected",
    [
        (0, [], {}, 1.0),
        (2, [], {1: 0.3, -1: 0.7, 2: 0.6, -2: 0.4}, 1.0),
        (2, [[]], {1: 0.3, -1: 0.7, 2: 0.6, -2: 0.4}, 0.0),
        (1, [[1], [-1]], {1: 0.7, -1: 0.3}, 0.0),
        (1, [[1]], {1: 0.7, -1: 0.3}, 0.7),
        (2, [[1], [2]], {1: 0.6, -1: 0.4, 2: 0.8, -2: 0.2}, 0.48),
        (2, [[1, 2], [-1, 2]], {1: 0.6, -1: 0.4, 2: 0.8, -2: 0.2}, 0.8),
    ],
)
def test_reference_weighted_model_count_small_formulas(
    num_vars, clauses, weights, expected
):
    result = float(referenceWeightedModelCount(num_vars, clauses, weights))
    assert np.isclose(result, expected, rtol=0, atol=1e-12)


def test_reference_weighted_model_count_matches_loop_on_random_formulas():
    rng = np.random.default_rng(0)
    for _ in range(50):
        num_vars = int(rng.integers(1, 9))
        weights = {}
        for i in range(1, num_vars + 1):
            w = float(rng.uniform(0.05, 0.95))
            weights[i] = w
            weights[-i] = 1.0 - w
        clauses = [
            [
                int(rng.integers(1, num_vars + 1)) * int(rng.choice([-1, 1]))
                for _ in range(int(rng.integers(1, 4)))
            ]
            for _ in range(int(rng.integers(0, 12)))
        ]
        got = float(referenceWeightedModelCount(num_vars, clauses, weights))
        want = _loop_weighted_model_count(num_vars, clauses, weights)
        assert np.isclose(got, want, rtol=0, atol=1e-9)


@pytest.mark.parametrize(
    "dataset", WeightedModelCountingTestGenerator().datasets, ids=lambda d: d.name
)
def test_test_datasets_expected_matches_reference(dataset):
    problem = WeightedModelCountingTestGenerator().generate(dataset)
    num_vars, clauses, weights = parse_format(dataset.cnf_text)
    reference = (
        problem.meta["default_total"]
        if problem.meta["expr"] is None
        else float(referenceWeightedModelCount(num_vars, clauses, weights))
    )
    assert problem.ref_meta["reference_count"] == reference
    assert np.isclose(
        problem.ref_meta["reference_count"],
        dataset.expected,
        rtol=0,
        atol=problem.ref_meta["error_bound"],
    )


def test_mccomp_generator_leaves_projected_instances_unchecked(monkeypatch, tmp_path):
    source_path = tmp_path / "Track4_PWMC" / "random_pwmc_1.cnf"
    source_path.parent.mkdir()
    source_path.write_text(
        """\
        p cnf 2 1
        c t pwmc
        c p show 1 0
        c p weight -1 0.25 0
        c p weight 1 0.75 0
        1 2 0
        c c s exact arb float 7.5E-01
        """
    )
    monkeypatch.setattr(
        weighted_model_counting,
        "download_mccomp_instance",
        lambda name: source_path,
    )
    problem = WeightedModelCountingMCCompGenerator().generate(
        WeightedModelCountingMCCompGenerator().datasets[0]
    )
    # Track4 answers are projected over the "show" variables, so the full-sum
    # benchmark output cannot be checked against them: no reference is attached.
    assert problem.ref_outputs is None
    assert problem.ref_meta is None
    assert problem.meta["source_exact_is_projected"]


def _benchmark(result, expected, reference, *, error_bound=1e-9):
    benchmark = WeightedModelCountingBenchmark()
    benchmark._output = [from_numpy(np.array(result, dtype=np.float64))]
    benchmark._ref_outputs = [from_numpy(np.array(expected, dtype=np.float64))]
    if reference is None:
        benchmark._ref_meta = {"error_bound": error_bound}
    else:
        benchmark._ref_meta = {
            "error_bound": error_bound,
            "reference_count": reference,
        }
    return benchmark


_PARAM = SimpleNamespace(dataset=SimpleNamespace(name="formula"))


def test_check_accepts_result_matching_both_counts():
    _benchmark(0.8, 0.8, 0.8).check(_PARAM)


def test_check_tolerates_floating_point_error_within_bound():
    _benchmark(0.8 + 5e-10, 0.8, 0.8).check(_PARAM)


def test_check_rejects_stored_count_disagreeing_with_reference():
    with pytest.raises(AssertionError, match="NumPy brute-force reference counts 0.4"):
        _benchmark(0.8, 0.8, 0.4).check(_PARAM)


def test_check_rejects_wrong_result():
    with pytest.raises(AssertionError, match="expected 0.8, got 0.6"):
        _benchmark(0.6, 0.8, 0.8).check(_PARAM)


def test_check_requires_reference_count():
    with pytest.raises(AssertionError, match="No NumPy reference count"):
        _benchmark(0.8, 0.8, None).check(_PARAM)


def test_check_skips_when_no_reference_outputs():
    benchmark = WeightedModelCountingBenchmark()
    benchmark._output = [from_numpy(np.array(0.8, dtype=np.float64))]
    benchmark._ref_outputs = None
    benchmark._ref_meta = None
    benchmark.check(_PARAM)
