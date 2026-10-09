import pytest

import numpy as np

from binsparse.conversions import to_numpy

from frameworks.saps_numpy import NumpyFramework
from saps.benchmarks.tictactoe_minimax import (
    TicTacToeMinimaxBenchmark,
    TicTacToeMinimaxBoardsGenerator,
    reference_minimax,
)

_DATASETS = TicTacToeMinimaxBoardsGenerator().datasets
_TRACTABLE = [d for d in _DATASETS if d.depth <= 6]


@pytest.mark.parametrize("dataset", _DATASETS, ids=lambda d: d.name)
def test_cached_expected_matches_recursive_reference(dataset):
    problem = TicTacToeMinimaxBoardsGenerator().generate(dataset)
    cached = to_numpy(problem.ref_outputs[0])
    reference = np.asarray(reference_minimax(dataset.board, dataset.depth))
    np.testing.assert_allclose(cached, reference, atol=1e-9)


@pytest.mark.parametrize("dataset", _TRACTABLE, ids=lambda d: d.name)
def test_benchmark_matches_cached_reference(dataset):
    problem = TicTacToeMinimaxBoardsGenerator().generate(dataset)
    xp = NumpyFramework()
    output = TicTacToeMinimaxBenchmark().benchmark(
        xp, [xp.from_binsparse(x) for x in problem.inputs], problem.meta
    )[0]
    actual = NumpyFramework().from_binsparse(xp.to_binsparse(output))
    expected = to_numpy(problem.ref_outputs[0])
    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_reference_known_terminal_values():
    x_win = [1, 1, 1, -1, -1, 0, 0, 0, 0]
    assert reference_minimax(np.array([_flat_to_board(x_win)]), 0) == 1.0
    o_win = [-1, 1, 1, -1, 1, 0, -1, 0, 0]
    assert reference_minimax(np.array([_flat_to_board(o_win)]), 0) == -1.0
    draw = [1, -1, 1, 1, -1, -1, -1, 1, 1]
    assert reference_minimax(np.array([_flat_to_board(draw)]), 0) == 0.0


def _flat_to_board(flat):
    board = np.zeros((3, 3, 2), dtype=float)
    for idx, cell in enumerate(flat):
        i, j = divmod(idx, 3)
        if cell == 1:
            board[i, j, 0] = 1.0
        elif cell == -1:
            board[i, j, 1] = 1.0
    return board
