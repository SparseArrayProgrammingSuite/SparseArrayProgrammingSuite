import pytest

import numpy as np

from sparseappbench.benchmarks.tic_tac import (
    build_win_masks,
    is_terminal,
    minimax_depth2,
    minimax_depth3,
)
from sparseappbench.frameworks.numpy_framework import NumpyFramework


@pytest.mark.parametrize(
    "S,expected_terminal,expected_value",
    [
        (
            np.array(
                [
                    [
                        [[1, 0], [1, 0], [1, 0]],
                        [[0, 1], [0, 1], [0, 0]],
                        [[0, 0], [0, 0], [0, 0]],
                    ]
                ],
                dtype=float,
            ),
            True,
            1.0,
        ),
        (
            np.array(
                [
                    [
                        [[0, 1], [1, 0], [0, 0]],
                        [[0, 1], [1, 0], [0, 0]],
                        [[0, 1], [0, 0], [0, 0]],
                    ]
                ],
                dtype=float,
            ),
            True,
            -1.0,
        ),
        (
            np.array(
                [
                    [
                        [[1, 0], [0, 1], [1, 0]],
                        [[0, 1], [1, 0], [0, 1]],
                        [[0, 1], [1, 0], [0, 1]],
                    ]
                ],
                dtype=float,
            ),
            True,
            0.0,
        ),
    ],
)
def test_is_terminal(S, expected_terminal, expected_value):
    xp = NumpyFramework()
    W = build_win_masks(xp)
    terminal, value = is_terminal(xp, xp.asarray(S), W)
    assert bool(terminal[0]) == expected_terminal
    assert np.isclose(value[0], expected_value, atol=1e-6)


def test_is_terminal_batch():
    xp = NumpyFramework()
    W = build_win_masks(xp)

    S_batch = np.array(
        [
            [
                [[1, 0], [1, 0], [1, 0]],
                [[0, 1], [0, 1], [0, 0]],
                [[0, 0], [0, 0], [0, 0]],
            ],
            [
                [[0, 1], [1, 0], [0, 0]],
                [[0, 1], [1, 0], [0, 0]],
                [[0, 1], [0, 0], [0, 0]],
            ],
            [
                [[1, 0], [0, 1], [1, 0]],
                [[0, 1], [1, 0], [0, 1]],
                [[0, 1], [1, 0], [0, 1]],
            ],
        ],
        dtype=float,
    )

    terminal, values = is_terminal(xp, xp.asarray(S_batch), W)
    assert np.all(terminal)
    assert np.allclose(values, [1.0, -1.0, 0.0], atol=1e-6)


@pytest.mark.parametrize(
    "S,expected",
    [
        # X one move from winning
        (
            np.array(
                [
                    [
                        [[1, 0], [1, 0], [0, 0]],
                        [[0, 1], [0, 1], [0, 0]],
                        [[1, 0], [0, 0], [0, 1]],
                    ]
                ],
                dtype=float,
            ),
            1.0,
        ),
        # O one move from winning
        (
            np.array(
                [
                    [
                        [[1, 0], [0, 1], [0, 0]],
                        [[0, 0], [0, 1], [1, 0]],
                        [[0, 1], [1, 0], [1, 0]],
                    ]
                ],
                dtype=float,
            ),
            -1.0,
        ),
        # one empty cell, draw
        (
            np.array(
                [
                    [
                        [[1, 0], [0, 1], [1, 0]],
                        [[1, 0], [0, 1], [0, 1]],
                        [[0, 1], [1, 0], [0, 0]],
                    ]
                ],
                dtype=float,
            ),
            0.0,
        ),
    ],
)
def test_minimax_depth2(S, expected):
    xp = NumpyFramework()
    W = build_win_masks(xp)
    result = minimax_depth2(xp, xp.asarray(S), W)
    assert np.isclose(result[0], expected, atol=1e-6)


def test_minimax_depth2_batch():
    xp = NumpyFramework()
    W = build_win_masks(xp)

    S_batch = np.array(
        [
            [
                [[1, 0], [1, 0], [0, 0]],
                [[0, 1], [0, 1], [0, 0]],
                [[1, 0], [0, 0], [0, 1]],
            ],
            [
                [[1, 0], [0, 1], [0, 0]],
                [[0, 0], [0, 1], [1, 0]],
                [[0, 1], [1, 0], [1, 0]],
            ],
            [
                [[1, 0], [0, 1], [1, 0]],
                [[1, 0], [0, 1], [0, 1]],
                [[0, 1], [1, 0], [0, 0]],
            ],
        ],
        dtype=float,
    )

    result = minimax_depth2(xp, xp.asarray(S_batch), W)
    expected = np.array([1.0, -1.0, 0.0])
    assert np.allclose(result, expected, atol=1e-6)


@pytest.mark.parametrize(
    "S,expected",
    [
        # X two moves from winning
        # X two moves from winning, it's X's turn (equal pieces)
        (
            np.array(
                [
                    [
                        [[1, 0], [1, 0], [0, 0]],
                        [[0, 1], [0, 0], [0, 0]],
                        [[0, 0], [0, 1], [0, 0]],
                    ]
                ],
                dtype=float,
            ),
            1.0,
        ),
        # O two moves from winning
        (
            np.array(
                [
                    [
                        [[1, 0], [0, 1], [0, 0]],
                        [[0, 0], [0, 1], [1, 0]],
                        [[0, 0], [1, 0], [1, 0]],
                    ]
                ],
                dtype=float,
            ),
            -1.0,
        ),
    ],
)
def test_minimax_depth3(S, expected):
    xp = NumpyFramework()
    W = build_win_masks(xp)
    result = minimax_depth3(xp, xp.asarray(S), W)
    assert np.isclose(result[0], expected, atol=1e-6)
