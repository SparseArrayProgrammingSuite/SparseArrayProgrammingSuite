import itertools
from types import SimpleNamespace

import pytest

import numpy as np

from binsparse.conversions import from_numpy

from saps.benchmark import DataInstance
from saps.benchmarks import model_counting
from saps.benchmarks.model_counting import (
    MCCompDataset,
    ModelCountingBenchmark,
    ModelCountingMCCompGenerator,
    ModelCountingTestGenerator,
    referenceModelCount,
)


def _loop_model_count(num_vars, clauses):
    return sum(
        all(
            any(bits[abs(lit) - 1] == (lit > 0) for lit in clause) for clause in clauses
        )
        for bits in itertools.product([False, True], repeat=num_vars)
    )


@pytest.mark.parametrize(
    "num_vars,clauses,expected",
    [
        (0, [], 1),
        (2, [], 4),
        (3, [[]], 0),
        (3, [[1], []], 0),
        (1, [[1], [-1]], 0),
        (3, [[1], [2], [3]], 1),
        (3, [[1, -3], [2, 3, -1]], 5),
        (2, [[1, -1]], 4),
        (2, [[2, 2]], 2),
    ],
)
def test_reference_model_count_small_formulas(num_vars, clauses, expected):
    assert referenceModelCount(num_vars, clauses) == expected


def test_reference_model_count_matches_loop_on_random_formulas():
    rng = np.random.default_rng(0)
    for _ in range(50):
        num_vars = int(rng.integers(1, 9))
        clauses = [
            [
                int(rng.integers(1, num_vars + 1)) * int(rng.choice([-1, 1]))
                for _ in range(int(rng.integers(1, 4)))
            ]
            for _ in range(int(rng.integers(0, 12)))
        ]
        assert referenceModelCount(num_vars, clauses) == _loop_model_count(
            num_vars, clauses
        )


@pytest.mark.parametrize(
    "dataset", ModelCountingTestGenerator().datasets, ids=lambda d: d.name
)
def test_test_datasets_expected_matches_reference(dataset):
    problem = ModelCountingTestGenerator().generate(dataset)
    assert problem.ref_meta == {"reference_count": dataset.expected}


def test_mccomp_generator_records_reference_count(monkeypatch, tmp_path):
    path = tmp_path / "random_mc_1.cnf"
    path.write_text("p cnf 3 2\nc t mc\n1 -3 0\n2 3 -1 0\nc c s exact arb int 5\n")
    monkeypatch.setattr(
        model_counting,
        "fetch_mccomp_instance",
        lambda _: DataInstance(inputs=[], meta={"local_path": str(path)}),
    )
    problem = ModelCountingMCCompGenerator().generate(
        MCCompDataset("Track1_MC/random_mc_1")
    )
    assert problem.ref_meta == {"reference_count": 5}


def _benchmark(result, expected, reference):
    benchmark = ModelCountingBenchmark()
    benchmark._output = [from_numpy(np.array(result, dtype=np.int64))]
    benchmark._ref_outputs = [from_numpy(np.array(expected, dtype=np.int64))]
    benchmark._ref_meta = None if reference is None else {"reference_count": reference}
    return benchmark


_PARAM = SimpleNamespace(dataset=SimpleNamespace(name="formula"))


def test_check_accepts_result_matching_both_counts():
    _benchmark(5, 5, 5).check(_PARAM)


def test_check_rejects_stored_count_disagreeing_with_reference():
    with pytest.raises(AssertionError, match="NumPy brute-force reference counts 4"):
        _benchmark(5, 5, 4).check(_PARAM)


def test_check_rejects_wrong_result():
    with pytest.raises(AssertionError, match="expected 5, got 6"):
        _benchmark(6, 5, 5).check(_PARAM)


def test_check_requires_reference_count():
    with pytest.raises(AssertionError, match="No NumPy reference count"):
        _benchmark(5, 5, None).check(_PARAM)
