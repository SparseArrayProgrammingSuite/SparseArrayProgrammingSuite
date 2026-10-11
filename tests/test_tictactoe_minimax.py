from functools import cache

import pytest

import numpy as np

from binsparse.conversions import to_numpy

from frameworks.saps_numpy import NumpyFramework
from saps.benchmarks.tictactoe_minimax import (
    BOARD_BATCH_NEAR,
    BOARD_DRAW_EARLY,
    BOARD_EMPTY,
    BOARD_O_WINS_MID,
    BOARD_X_WINS_EARLY,
    BOARD_X_WINS_MID,
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
        xp, problem.meta, *(xp.from_binsparse(x) for x in problem.inputs)
    )
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


WINNING_LINES = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
)


@cache
def _reference_value(board: tuple[int, ...], depth: int) -> int:
    """Scalar recursive oracle, independent of the tensorized traversal."""
    for player in (1, -1):
        if any(all(board[cell] == player for cell in line) for line in WINNING_LINES):
            return player
    if depth == 0 or 0 not in board:
        return 0
    player = -1 if board.count(1) > board.count(-1) else 1
    children = [
        _reference_value((*board[:cell], player, *board[cell + 1 :]), depth - 1)
        for cell, value in enumerate(board)
        if value == 0
    ]
    return min(children) if player == -1 else max(children)


@pytest.mark.parametrize("depth", [0, 1, 2, 3, 4])
def test_depth_loop_matches_scalar_minimax(depth):
    labels = np.array(
        [
            [1, 1, 1, -1, -1, 0, 0, 0, 0],  # X already won, with moves left.
            [-1, -1, -1, 1, 1, 0, 1, 0, 0],  # O already won, with moves left.
            [1, -1, 1, 1, -1, -1, -1, 1, 1],  # Full draw.
        ]
    ).reshape(3, 3, 3)
    terminal_boards = np.stack((labels == 1, labels == -1), axis=-1).astype(float)
    boards = np.concatenate(
        [
            BOARD_BATCH_NEAR,
            BOARD_X_WINS_MID,
            BOARD_O_WINS_MID,
            BOARD_X_WINS_EARLY,
            BOARD_DRAW_EARLY,
            BOARD_EMPTY,
            terminal_boards,
        ]
    )
    original = boards.copy()
    expected = [
        _reference_value(
            tuple((board[:, :, 0] - board[:, :, 1]).astype(int).flat), depth
        )
        for board in boards
    ]
    actual = TicTacToeMinimaxBenchmark().benchmark(
        NumpyFramework(), {"depth": depth}, boards
    )
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(boards, original)
