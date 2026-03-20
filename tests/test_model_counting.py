import pytest

from sparseappbench.benchmarks.model_counting import benchmark_model_counting
from sparseappbench.frameworks.numpy_framework import NumpyFramework


@pytest.mark.parametrize(
    "dimacs_txt, expected",
    [
        (
            """\
            p cnf 3 2
            1 -3 0
            2 3 -1 0
            """,
            5,
        ),
        (
            """\
            c contradiction
            p cnf 1 2
            1 0
            -1 0
            """,
            0,
        ),
        (
            """\
            c single_solution
            p cnf 3 3
            1 0
            2 0
            3 0
            """,
            1,
        ),
        (
            """\
            c empty_formula
            p cnf 2 0
            """,
            4,
        ),
    ],
)
def test_model_counting(dimacs_txt, expected):
    xp = NumpyFramework()

    out = benchmark_model_counting(xp, dimacs_txt)

    msg = f"Expected {expected} models, but found {out}"
    assert out == expected, msg
