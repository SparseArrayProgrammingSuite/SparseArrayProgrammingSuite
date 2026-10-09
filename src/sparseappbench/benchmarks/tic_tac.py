import numpy as np

import sparseappbench
from sparseappbench.benchmark import (
    Benchmark,
    BinsparseFormat,
    Contributor,
    Dataset,
    Generator,
    Ref,
)

xp = sparseappbench.xp

"""
Name: Minimax Tic-Tac-Toe

Co-Authors: Aarav Jogekar & Prof. Ahrens

What does this code do: Implement a fully tensorized, non-recursive minimax
search over a tic-tac-toe game. Game states are represented as tensors.
Game state is represented as S[n, i, j, p] of shape (N, 3, 3, 2) where n
indexes boards, i,j are board positions and p is the player channel. Given
any board state within the game, it should return the result of the game.

Statement on the use of Generative AI: No generative AI was used to construct
the benchmark function itself.
Email: ajoglekar32@gatech.edu
"""


def build_win_masks(xp):
    return xp.asarray(
        [
            [[1, 1, 1], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [1, 1, 1], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [1, 1, 1]],
            [[1, 0, 0], [1, 0, 0], [1, 0, 0]],
            [[0, 1, 0], [0, 1, 0], [0, 1, 0]],
            [[0, 0, 1], [0, 0, 1], [0, 0, 1]],
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[0, 0, 1], [0, 1, 0], [1, 0, 0]],
        ]
    )


def check_wins(xp, S, W):
    W_exp = xp.reshape(W, (1, 8, 3, 3, 1))
    S_exp = xp.reshape(S, (S.shape[0], 1, 3, 3, 2))
    scores = xp.sum(W_exp * S_exp, axis=(2, 3))
    return xp.any(scores >= 3, axis=1)


def check_full(xp, S):
    return xp.all(xp.sum(S, axis=3) >= 1, axis=(1, 2))


def is_terminal(xp, S, W):
    N = S.shape[0]
    winner = check_wins(xp, S, W)
    x_wins = winner[:, 0]
    o_wins = winner[:, 1]
    full = check_full(xp, S)
    terminal = x_wins | o_wins | full
    value = xp.where(x_wins, xp.ones(N), xp.where(o_wins, -xp.ones(N), xp.zeros(N)))
    return terminal, value


def whose_turn(xp, S):
    count_X = xp.sum(S[:, :, :, 0], axis=(1, 2))
    count_O = xp.sum(S[:, :, :, 1], axis=(1, 2))
    return xp.where(count_X > count_O, xp.ones(S.shape[0]), xp.zeros(S.shape[0]))


def generate_child(xp, S, W):
    N = S.shape[0]
    empty_flat = xp.reshape(1 - xp.sum(S, axis=3), (N, 9))
    turn = whose_turn(xp, S)
    pos_exp = xp.reshape(xp.eye(9), (1, 9, 3, 3))
    turn_exp = xp.reshape(turn, (N, 1, 1, 1))
    ch0 = xp.reshape(pos_exp * (1 - turn_exp), (N, 9, 3, 3, 1))
    ch1 = xp.reshape(pos_exp * turn_exp, (N, 9, 3, 3, 1))
    delta = xp.concat([ch0, ch1], axis=4)
    valid_exp = xp.reshape(empty_flat, (N, 9, 1, 1, 1))
    children = (xp.reshape(S, (N, 1, 3, 3, 2)) + delta * valid_exp) * valid_exp
    # Above line zeroes out boards
    return xp.reshape(children, (N * 9, 3, 3, 2)), xp.reshape(empty_flat, (N * 9,))


def backup(xp, S, val_children, valid, terminal, leaf_val):
    N = S.shape[0]
    turn = whose_turn(xp, S)
    child_val = xp.where(valid > 0, val_children, xp.zeros(N * 9))
    child_val_grid = xp.reshape(child_val, (N, 9))
    valid_grid = xp.reshape(valid, (N, 9))
    turn_exp = xp.reshape(turn, (N, 1))
    sentinel = xp.where(
        turn_exp > 0,
        float("inf") * xp.ones((N, 9)),
        float("-inf") * xp.ones((N, 9)),
    )
    child_val_masked = xp.where(valid_grid > 0, child_val_grid, sentinel)
    backed_max = xp.max(child_val_masked, axis=1)
    backed_min = xp.min(child_val_masked, axis=1)
    backed = xp.where(turn > 0, backed_min, backed_max)
    return xp.where(terminal, leaf_val, backed)


def minimax_depth2(xp, S_initial, W):
    c1, v1 = generate_child(xp, S_initial, W)
    c2, v2 = generate_child(xp, c1, W)
    t0, val0 = is_terminal(xp, S_initial, W)
    t1, val1 = is_terminal(xp, c1, W)
    t2, val2 = is_terminal(xp, c2, W)
    val2 = xp.where(t2, val2, xp.zeros(c2.shape[0]))
    val1 = backup(xp, c1, val2, v2, t1, val1)
    return backup(xp, S_initial, val1, v1, t0, val0)


def minimax_depth3(xp, S_initial, W):
    c1, v1 = generate_child(xp, S_initial, W)
    c2, v2 = generate_child(xp, c1, W)
    c3, v3 = generate_child(xp, c2, W)
    t0, val0 = is_terminal(xp, S_initial, W)
    t1, val1 = is_terminal(xp, c1, W)
    t2, val2 = is_terminal(xp, c2, W)
    t3, val3 = is_terminal(xp, c3, W)
    val3 = xp.where(t3, val3, xp.zeros(c3.shape[0]))
    val2 = backup(xp, c2, val3, v3, t2, val2)
    val1 = backup(xp, c1, val2, v2, t1, val1)
    return backup(xp, S_initial, val1, v1, t0, val0)


def minimax_depth5(xp, S_initial, W):
    c1, v1 = generate_child(xp, S_initial, W)
    c2, v2 = generate_child(xp, c1, W)
    c3, v3 = generate_child(xp, c2, W)
    c4, v4 = generate_child(xp, c3, W)
    c5, v5 = generate_child(xp, c4, W)
    t0, val0 = is_terminal(xp, S_initial, W)
    t1, val1 = is_terminal(xp, c1, W)
    t2, val2 = is_terminal(xp, c2, W)
    t3, val3 = is_terminal(xp, c3, W)
    t4, val4 = is_terminal(xp, c4, W)
    t5, val5 = is_terminal(xp, c5, W)
    val5 = xp.where(t5, val5, xp.zeros(c5.shape[0]))
    val4 = backup(xp, c4, val5, v5, t4, val4)
    val3 = backup(xp, c3, val4, v4, t3, val3)
    val2 = backup(xp, c2, val3, v3, t2, val2)
    val1 = backup(xp, c1, val2, v2, t1, val1)
    return backup(xp, S_initial, val1, v1, t0, val0)


_REF_WIN_LINES = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
)


def _ref_board_to_flat(board):
    board = np.asarray(board, dtype=float)
    flat = []
    for i in range(3):
        for j in range(3):
            x = board[i, j, 0]
            o = board[i, j, 1]
            flat.append(1 if x > 0 else (-1 if o > 0 else 0))
    return flat


def _ref_winner(flat):
    for a, b, c in _REF_WIN_LINES:
        s = flat[a] + flat[b] + flat[c]
        if s == 3:
            return 1
        if s == -3:
            return -1
    return 0


def _ref_minimax(flat, depth):
    w = _ref_winner(flat)
    if w != 0:
        return w
    if all(cell != 0 for cell in flat):
        return 0
    if depth == 0:
        return 0

    count_x = sum(1 for c in flat if c == 1)
    count_o = sum(1 for c in flat if c == -1)
    x_to_move = not (count_x > count_o)
    mark = 1 if x_to_move else -1

    best = None
    for idx in range(9):
        if flat[idx] != 0:
            continue
        child = list(flat)
        child[idx] = mark
        val = _ref_minimax(child, depth - 1)
        if best is None:
            best = val
        elif x_to_move:
            best = max(best, val)
        else:
            best = min(best, val)
    return best


def reference_minimax(board, depth):
    board = np.asarray(board, dtype=float)
    return np.array(
        [
            _ref_minimax(_ref_board_to_flat(board[n]), depth)
            for n in range(board.shape[0])
        ],
        dtype=float,
    )


# These are the testing boards, used np.
BOARD_X_WINS_NEAR = np.array(
    [[[[1, 0], [1, 0], [0, 0]], [[0, 1], [0, 1], [0, 0]], [[1, 0], [0, 0], [0, 1]]]],
    dtype=float,
)
BOARD_O_WINS_NEAR = np.array(
    [[[[1, 0], [0, 1], [0, 0]], [[0, 0], [0, 1], [1, 0]], [[0, 1], [1, 0], [1, 0]]]],
    dtype=float,
)
BOARD_DRAW_NEAR = np.array(
    [[[[1, 0], [0, 1], [1, 0]], [[1, 0], [0, 1], [0, 1]], [[0, 1], [1, 0], [0, 0]]]],
    dtype=float,
)
BOARD_X_WINS_MID = np.array(
    [[[[1, 0], [1, 0], [0, 0]], [[0, 1], [0, 0], [0, 0]], [[0, 0], [0, 1], [0, 0]]]],
    dtype=float,
)
BOARD_O_WINS_MID = np.array(
    [[[[1, 0], [0, 1], [0, 0]], [[0, 0], [0, 1], [1, 0]], [[0, 0], [1, 0], [1, 0]]]],
    dtype=float,
)
BOARD_X_WINS_EARLY = np.array(
    [[[[0, 0], [0, 1], [0, 0]], [[0, 0], [1, 0], [0, 0]], [[0, 0], [0, 0], [0, 0]]]],
    dtype=float,
)
BOARD_DRAW_EARLY = np.array(
    [[[[0, 1], [0, 0], [1, 0]], [[0, 0], [1, 0], [0, 0]], [[0, 1], [0, 0], [0, 0]]]],
    dtype=float,
)
BOARD_BATCH_NEAR = np.concatenate(
    [BOARD_X_WINS_NEAR, BOARD_O_WINS_NEAR, BOARD_DRAW_NEAR], axis=0
)


class TicTacToeDataset(Dataset):
    def __init__(self, name: str, board: np.ndarray, depth: int):
        self._name = name
        self.board = board
        self.depth = depth

    @property
    def name(self) -> str:
        return self._name

    @property
    def pretty_name(self) -> str:
        return f"TicTacToe {self._name}"

    @property
    def description(self) -> str:
        return f"Batch of {self.board.shape[0]} board(s) at minimax depth {self.depth}."

    @property
    def tags(self) -> list[str]:
        return ["game", "minimax", f"depth{self.depth}"]


class TicTacToeGenerator(Generator[TicTacToeDataset]):
    @property
    def name(self) -> str:
        return "tictactoe_boards"

    @property
    def pretty_name(self) -> str:
        return "Fixed Boards for testing."

    @property
    def description(self) -> str:
        return (
            "These tests covering end-game, mid-game, and early game"
            "through using various minimax at different depths."
        )

    @property
    def tags(self) -> list[str]:
        return ["game", "minimax", "tictactoe"]

    @property
    def authors(self) -> list[Contributor]:
        return [Contributor("Aarav Jogekar", "ajoglekar32@gatech.edu")]

    @property
    def references(self) -> list[Ref]:
        return []

    @property
    def ai_disclosure(self) -> str:
        return (
            "No generative AI was used to construct the benchmark function itself. "
            "This statement was written by hand."
        )

    @property
    def motivation(self) -> str:
        return (
            "Boards have range of sparsity so they go from empty to being very dense"
            "this helps us measure how sparsity can affect performance at different"
            "depths."
        )

    @property
    def datasets(self) -> list[TicTacToeDataset]:
        return [
            TicTacToeDataset("x_wins_near", BOARD_X_WINS_NEAR, depth=2),
            TicTacToeDataset("o_wins_near", BOARD_O_WINS_NEAR, depth=2),
            TicTacToeDataset("draw_near", BOARD_DRAW_NEAR, depth=2),
            TicTacToeDataset("batch_near", BOARD_BATCH_NEAR, depth=2),
            TicTacToeDataset("x_wins_mid", BOARD_X_WINS_MID, depth=3),
            TicTacToeDataset("o_wins_mid", BOARD_O_WINS_MID, depth=3),
            TicTacToeDataset("x_wins_early", BOARD_X_WINS_EARLY, depth=5),
            TicTacToeDataset("draw_early", BOARD_DRAW_EARLY, depth=5),
        ]

    def generate(self, dataset: TicTacToeDataset):
        S_bin = BinsparseFormat.from_numpy(dataset.board)
        expected = reference_minimax(dataset.board, dataset.depth).tolist()
        return ([S_bin], {"depth": dataset.depth, "expected": expected})


class TicTacToeBenchmark(Benchmark):
    @property
    def tag(self) -> str:
        return "tictactoe_minimax"

    @property
    def name(self) -> str:
        return "tictactoe_minimax"

    @property
    def pretty_name(self) -> str:
        return "Tensorized Minimax Tic-Tac-Toe"

    @property
    def description(self) -> str:
        return (
            "Tensorized minimax search over tic-tac-toe. "
            "You are able to test at depths 2, 3, and 5"
        )

    @property
    def tags(self) -> list[str]:
        return ["minimax", "tensor", "sparse"]

    @property
    def authors(self) -> list[Contributor]:
        return [
            Contributor("Aarav Jogekar", "ajoglekar32@gatech.edu"),
            Contributor("Willow Ahrens", "ahrens@gatech.edu"),
        ]

    @property
    def references(self) -> list[Ref]:
        return []

    @property
    def ai_disclosure(self) -> str:
        return (
            "No generative AI was used to construct the benchmark function itself. "
            "This statement was written by hand."
        )

    @property
    def motivation(self) -> str:
        return (
            "This benchmark will do sparse array operations on tic-tac-toe"
            "game trees. Sparsity should increase as you go deeper into the game/tree."
            "Invalid boards states caused by bad moves are zeroed out."
        )

    @property
    def generators(self):
        return [TicTacToeGenerator()]

    def benchmark(self, data: list, meta: dict):
        S_bin = data[0]
        depth = meta.get("depth", 5)
        S = xp.from_binsparse(S_bin)
        W = build_win_masks(xp)

        if depth == 2:
            result = minimax_depth2(xp, S, W)
        elif depth == 3:
            result = minimax_depth3(xp, S, W)
        elif depth == 5:
            result = minimax_depth5(xp, S, W)
        else:
            raise ValueError(
                f"Unsupported minimax depth: {depth} (expected 2, 3, or 5)"
            )

        return [xp.to_binsparse(result)]

    def check_correct(self, param):
        super().check_correct(param)
        expected = self._meta.get("expected")
        if expected is None:
            return
        expected = np.asarray(expected, dtype=float)
        out = xp.from_binsparse(self._output[0])
        if hasattr(out, "todense"):
            out = out.todense()
        out = np.asarray(out, dtype=float)
        assert out.shape == expected.shape, (
            f"tic-tac-toe minimax: output shape {out.shape} != expected "
            f"{expected.shape}"
        )
        assert np.allclose(out, expected, atol=1e-9), (
            f"tic-tac-toe minimax output {out} != cached reference {expected}"
        )
