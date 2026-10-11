from functools import cache

import pytest

import numpy as np

from binsparse.conversions import from_numpy, to_numpy

from frameworks.saps_numpy import NumpyFramework
from frameworks.saps_smart import SmartSparseFramework
from saps.benchmark import Param
from saps.benchmarks.tictactoe_minimax import (
    BOARD_BATCH_NEAR,
    BOARD_DRAW_EARLY,
    BOARD_EMPTY,
    BOARD_O_WINS_MID,
    BOARD_X_WINS_EARLY,
    BOARD_X_WINS_MID,
    TicTacToeMinimaxBenchmark,
    TicTacToeMinimaxBoardsGenerator,
    TicTacToeMinimaxDataset,
    reference_minimax,
)

_DATASETS = TicTacToeMinimaxBoardsGenerator().datasets
_TRACTABLE = [d for d in _DATASETS if d.depth <= 6]


@pytest.mark.parametrize("dataset", _DATASETS, ids=lambda d: d.name)
def test_generated_reference_matches_scalar_minimax(dataset):
    problem = TicTacToeMinimaxBoardsGenerator().generate(dataset)
    actual = to_numpy(problem.ref_outputs[0])
    expected = [
        _reference_value(
            tuple((board[:, :, 0] - board[:, :, 1]).astype(int).flat), dataset.depth
        )
        for board in dataset.board
    ]
    assert to_numpy(problem.inputs[0]).dtype == np.dtype(np.int64)
    assert actual.dtype == np.dtype(np.int64)
    np.testing.assert_array_equal(actual, expected)
    if dataset.expected is not None:
        np.testing.assert_array_equal(actual, dataset.expected)


@pytest.mark.parametrize("dataset", _TRACTABLE, ids=lambda d: d.name)
def test_checker_accepts_benchmark_output(dataset):
    benchmark = TicTacToeMinimaxBenchmark()
    param = Param(TicTacToeMinimaxBoardsGenerator(), dataset)
    benchmark.setup(param, xp=NumpyFramework(), use_cache=False)
    benchmark.run(param)
    benchmark.check(param)
    benchmark.teardown(param)


@pytest.mark.parametrize("board", [BOARD_X_WINS_MID, BOARD_BATCH_NEAR])
def test_checker_uses_recursive_reference(board):
    dataset = TicTacToeMinimaxDataset(
        "incorrect_expected",
        board,
        depth=3,
        expected=np.zeros(board.shape[0], dtype=np.int64),
    )
    benchmark = TicTacToeMinimaxBenchmark()
    param = Param(TicTacToeMinimaxBoardsGenerator(), dataset)
    benchmark.setup(param, xp=NumpyFramework(), use_cache=False)
    benchmark.run(param)
    benchmark.check(param)

    # The dataset's incorrect expected value must not pass the checker.
    benchmark._output = [from_numpy(dataset.expected)]
    with pytest.raises(AssertionError):
        benchmark.check(param)

    benchmark._ref_outputs = None
    with pytest.raises(AssertionError, match="No reference output"):
        benchmark.check(param)
    benchmark.teardown(param)


@pytest.mark.parametrize("value", [1.0, 1.0 + 1e-7])
def test_checker_rejects_floating_point_output(value):
    benchmark = TicTacToeMinimaxBenchmark()
    param = next(
        param for param in benchmark.params if param.dataset.name == "x_wins_mid"
    )
    benchmark.setup(param, xp=NumpyFramework(), use_cache=False)
    benchmark._output = [from_numpy(np.array([value]))]
    with pytest.raises(AssertionError, match="integer dtype"):
        benchmark.check(param)
    del benchmark._output
    benchmark.teardown(param)


def test_reference_known_terminal_values():
    x_win = [1, 1, 1, -1, -1, 0, 0, 0, 0]
    assert reference_minimax(np.array([_flat_to_board(x_win)]), 0) == 1
    o_win = [-1, 1, 1, -1, 1, 0, -1, 0, 0]
    assert reference_minimax(np.array([_flat_to_board(o_win)]), 0) == -1
    draw = [1, -1, 1, 1, -1, -1, -1, 1, 1]
    assert reference_minimax(np.array([_flat_to_board(draw)]), 0) == 0


def _flat_to_board(flat):
    board = np.zeros((3, 3, 2), dtype=np.int64)
    for idx, cell in enumerate(flat):
        i, j = divmod(idx, 3)
        if cell == 1:
            board[i, j, 0] = 1
        elif cell == -1:
            board[i, j, 1] = 1
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


@pytest.mark.parametrize("framework_cls", [NumpyFramework, SmartSparseFramework])
@pytest.mark.parametrize("depth", [0, 1, 2, 3, 4])
def test_depth_loop_matches_scalar_minimax(depth, framework_cls):
    labels = np.array(
        [
            [1, 1, 1, -1, -1, 0, 0, 0, 0],  # X already won, with moves left.
            [-1, -1, -1, 1, 1, 0, 1, 0, 0],  # O already won, with moves left.
            [1, -1, 1, 1, -1, -1, -1, 1, 1],  # Full draw.
        ]
    ).reshape(3, 3, 3)
    terminal_boards = np.stack((labels == 1, labels == -1), axis=-1).astype(np.int64)
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
    xp = framework_cls()
    actual = TicTacToeMinimaxBenchmark().benchmark(
        xp, {"depth": depth}, xp.from_binsparse(from_numpy(boards))
    )
    actual = to_numpy(xp.to_binsparse(actual))
    assert actual.dtype == np.dtype(np.int64)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(boards, original)
