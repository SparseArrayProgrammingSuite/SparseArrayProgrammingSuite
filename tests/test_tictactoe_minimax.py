from functools import cache

import pytest

import numpy as np

from frameworks.saps_numpy import NumpyFramework
from saps.benchmarks.tictactoe_minimax import (
    BOARD_BATCH_NEAR,
    BOARD_DRAW_EARLY,
    BOARD_EMPTY,
    BOARD_O_WINS_MID,
    BOARD_X_WINS_EARLY,
    BOARD_X_WINS_MID,
    TicTacToeMinimaxBenchmark,
)

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
