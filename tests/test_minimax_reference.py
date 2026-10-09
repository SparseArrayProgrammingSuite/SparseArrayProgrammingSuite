import pytest

import numpy as np

from sparseappbench.benchmarks.tic_tac import (
    BOARD_BATCH_NEAR,
    BOARD_DRAW_EARLY,
    BOARD_DRAW_NEAR,
    BOARD_O_WINS_MID,
    BOARD_O_WINS_NEAR,
    BOARD_X_WINS_EARLY,
    BOARD_X_WINS_MID,
    BOARD_X_WINS_NEAR,
    TicTacToeBenchmark,
    TicTacToeGenerator,
    build_win_masks,
    minimax_depth2,
    minimax_depth3,
    minimax_depth5,
    reference_minimax,
)
from sparseappbench.frameworks.numpy_framework import NumpyFramework
from sparseappbench.frameworks.sparse_framework import PyDataSparseFramework

_WIN_LINES = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
)


def _winner(board_flat):
    for a, b, c in _WIN_LINES:
        s = board_flat[a] + board_flat[b] + board_flat[c]
        if s == 3:
            return 1
        if s == -3:
            return -1
    return 0


def _board_to_flat(board):
    board = np.asarray(board, dtype=float)
    flat = []
    for i in range(3):
        for j in range(3):
            x = board[i, j, 0]
            o = board[i, j, 1]
            if x > 0 and o > 0:
                raise ValueError("Both players occupy the same cell")
            flat.append(1 if x > 0 else (-1 if o > 0 else 0))
    return flat


def _recursive_minimax(board_flat, depth):
    w = _winner(board_flat)
    if w != 0:
        return w
    if all(cell != 0 for cell in board_flat):
        return 0
    if depth == 0:
        return 0

    count_x = sum(1 for cell in board_flat if cell == 1)
    count_o = sum(1 for cell in board_flat if cell == -1)
    x_to_move = not (count_x > count_o)
    mark = 1 if x_to_move else -1

    best = None
    for idx in range(9):
        if board_flat[idx] != 0:
            continue
        child = list(board_flat)
        child[idx] = mark
        val = _recursive_minimax(child, depth - 1)
        if best is None:
            best = val
        elif x_to_move:
            best = max(best, val)
        else:
            best = min(best, val)
    return best


def _reference_values(batch, depth):
    batch = np.asarray(batch, dtype=float)
    return np.array(
        [
            _recursive_minimax(_board_to_flat(batch[n]), depth)
            for n in range(batch.shape[0])
        ],
        dtype=float,
    )


_TENSOR_MINIMAX = {
    2: minimax_depth2,
    3: minimax_depth3,
    5: minimax_depth5,
}


def _run_tensor(xp, batch, depth):
    W = build_win_masks(xp)
    result = _TENSOR_MINIMAX[depth](xp, xp.asarray(np.asarray(batch, dtype=float)), W)
    if hasattr(result, "todense"):
        result = result.todense()
    return np.asarray(result, dtype=float)


_FIXED_CASES = [
    ("x_wins_near", BOARD_X_WINS_NEAR, 2),
    ("o_wins_near", BOARD_O_WINS_NEAR, 2),
    ("draw_near", BOARD_DRAW_NEAR, 2),
    ("batch_near", BOARD_BATCH_NEAR, 2),
    ("x_wins_mid", BOARD_X_WINS_MID, 3),
    ("o_wins_mid", BOARD_O_WINS_MID, 3),
    ("x_wins_early", BOARD_X_WINS_EARLY, 5),
    ("draw_early", BOARD_DRAW_EARLY, 5),
]


@pytest.fixture(params=[NumpyFramework, PyDataSparseFramework], ids=["numpy", "sparse"])
def xp(request):
    return request.param()


@pytest.mark.parametrize(
    "name,board,depth",
    _FIXED_CASES,
    ids=[c[0] for c in _FIXED_CASES],
)
def test_tensor_matches_recursive_reference(xp, name, board, depth):
    expected = _reference_values(board, depth)
    got = _run_tensor(xp, board, depth)
    assert got.shape == expected.shape, (
        f"{name}: shape mismatch, got {got.shape}, expected {expected.shape}"
    )
    assert np.allclose(got, expected, atol=1e-9), (
        f"{name} (depth {depth}): tensorized minimax {got} "
        f"!= recursive reference {expected}"
    )


def test_reference_known_terminal_values():
    x_win = [1, 1, 1, -1, -1, 0, 0, 0, 0]
    assert _recursive_minimax(x_win, 0) == 1
    o_win = [-1, 1, 1, -1, 1, 0, -1, 0, 0]
    assert _recursive_minimax(o_win, 0) == -1
    draw = [1, -1, 1, 1, -1, -1, -1, 1, 1]
    assert _recursive_minimax(draw, 0) == 0


@pytest.mark.parametrize(
    "board,depth,expected",
    [
        (BOARD_X_WINS_NEAR, 2, 1.0),
        (BOARD_O_WINS_NEAR, 2, -1.0),
        (BOARD_DRAW_NEAR, 2, 0.0),
        (BOARD_X_WINS_MID, 3, 1.0),
        (BOARD_O_WINS_MID, 3, -1.0),
        (BOARD_X_WINS_EARLY, 5, 1.0),
        (BOARD_DRAW_EARLY, 5, 0.0),
    ],
    ids=[
        "x_wins_near",
        "o_wins_near",
        "draw_near",
        "x_wins_mid",
        "o_wins_mid",
        "x_wins_early",
        "draw_early",
    ],
)
def test_recursive_reference_expected_outcomes(board, depth, expected):
    got = _reference_values(board, depth)
    assert np.allclose(got, expected, atol=1e-9), (
        f"reference minimax {got} != labelled outcome {expected}"
    )


def test_generator_caches_expected_value():
    gen = TicTacToeGenerator()
    for dataset in gen.datasets:
        _data, meta = gen.generate(dataset)
        assert "expected" in meta, (
            f"{dataset.name}: generator meta is missing cached 'expected'"
        )
        ref = reference_minimax(dataset.board, dataset.depth)
        assert np.allclose(np.asarray(meta["expected"], dtype=float), ref, atol=1e-9), (
            f"{dataset.name}: cached expected {meta['expected']} != reference {ref}"
        )


def test_benchmark_check_correct_passes():
    bench = TicTacToeBenchmark()
    gen = TicTacToeGenerator()
    for dataset in gen.datasets:
        data, meta = gen.generate(dataset)
        output = bench.benchmark(data, meta)
        bench._output = output
        bench._meta = meta
        bench.check_correct(param=None)
        del bench._output
        del bench._meta
