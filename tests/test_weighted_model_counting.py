import pytest

from sparseappbench.benchmarks.weighted_model_counting import (
    benchmark_weighted_model_counting,
)
from sparseappbench.frameworks.numpy_framework import NumpyFramework


@pytest.mark.parametrize(
    "input, expected",
    [
        (
            """\
            c (V1 or V2) and (not V1 or V2)
            p cnf 2 2
            c p weight 1 0.6
            c p weight -1 0.4
            c p weight 2 0.8
            c p weight -2 0.2
            1 2 0
            -1 2 0
            """,
            0.8,
        ),
        (
            """\
            c V1 and not V1 (unsatisfiable)
            p cnf 1 2
            c p weight 1 0.7
            c p weight -1 0.3
            1 0
            -1 0
            """,
            0.0,
        ),
        (
            """\
            c no clauses
            p cnf 2 0
            c p weight 1 0.7
            c p weight -1 0.3
            c p weight 2 0.9
            c p weight -2 0.1
            """,
            1.0,
        ),
        (
            """\
            c V1 or V2 (default weights)
            p cnf 2 1
            1 2 0
            """,
            0.75,
        ),
        (
            """\
            c (V1 or V2) and (not V2 or V3)
            p cnf 3 2
            c p weight 1 0.2
            c p weight -1 0.8
            c p weight 2 0.6
            c p weight -2 0.4
            c p weight 3 0.9
            c p weight -3 0.1
            1 2 0
            -2 3 0
            """,
            0.62,
        ),
    ],
)
def test_weighted_model_counting(input, expected):
    xp = NumpyFramework()

    out = benchmark_weighted_model_counting(xp, input)

    msg = f"Expected {expected} models, but found {out}"
    assert out == expected, msg
