import numpy as np

from binsparse.conversions import from_numpy, to_numpy

from saps.benchmark import (
    Benchmark,
    Contributor,
    DataInstance,
    Dataset,
    Generator,
    Ref,
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
BOARD_EMPTY = np.zeros((1, 3, 3, 2), dtype=float)
BOARD_BATCH_NEAR = np.concatenate(
    [BOARD_X_WINS_NEAR, BOARD_O_WINS_NEAR, BOARD_DRAW_NEAR], axis=0
)


class TicTacToeMinimaxDataset(Dataset):
    def __init__(
        self,
        name: str,
        board: np.ndarray,
        depth: int,
        expected: np.ndarray | float | None = None,
        suites: list[str] | None = None,
        pretty_name: str | None = None,
    ):
        self._suites = suites or []
        self._name = name
        self._pretty_name = pretty_name or name
        self.board = board
        self.depth = depth
        self.expected = expected

    @property
    def name(self) -> str:
        return self._name

    @property
    def pretty_name(self) -> str:
        return self._pretty_name

    @property
    def description(self) -> str:
        return f"Batch of {self.board.shape[0]} board(s) at minimax depth {self.depth}."

    @property
    def suites(self) -> list[str]:
        return self._suites

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"


class TicTacToeMinimaxBoardsGenerator(Generator[TicTacToeMinimaxDataset]):
    @property
    def cacheable(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return "tictactoe_minimax_boards"

    @property
    def pretty_name(self) -> str:
        return "Tic-Tac-Toe Minimax Boards"

    @property
    def description(self) -> str:
        return (
            "These tests covering end-game, mid-game, and early game"
            "through using various minimax at different depths."
        )

    @property
    def suites(self) -> list[str]:
        return []

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def authors(self) -> list[Contributor]:
        return [Contributor("Aarav Jogekar", "ajoglekar32@gatech.edu")]

    @property
    def references(self) -> list[Ref]:
        return []

    @property
    def ai_disclosure(self) -> str:
        return (
            "No generative AI was used to construct the benchmark function itself."
            " This statement was written by hand."
        )

    @property
    def motivation(self) -> str:
        return (
            "Boards have range of sparsity so they go from empty to being very dense"
            " this helps us measure how sparsity can affect performance at different"
            " depths."
        )

    @property
    def datasets(self) -> list[TicTacToeMinimaxDataset]:
        return [
            TicTacToeMinimaxDataset(
                "x_wins_near",
                BOARD_X_WINS_NEAR,
                depth=2,
                expected=1.0,
                suites=["test"],
                pretty_name="X Wins Near",
            ),
            TicTacToeMinimaxDataset(
                "o_wins_near",
                BOARD_O_WINS_NEAR,
                depth=2,
                expected=-1.0,
                suites=["test"],
                pretty_name="O Wins Near",
            ),
            TicTacToeMinimaxDataset(
                "draw_near",
                BOARD_DRAW_NEAR,
                depth=2,
                expected=0.0,
                suites=["test"],
                pretty_name="Draw Near",
            ),
            TicTacToeMinimaxDataset(
                "batch_near",
                BOARD_BATCH_NEAR,
                depth=2,
                expected=np.array([1.0, -1.0, 0.0]),
                suites=["test"],
                pretty_name="Batch Near",
            ),
            TicTacToeMinimaxDataset(
                "x_wins_mid",
                BOARD_X_WINS_MID,
                depth=3,
                expected=1.0,
                suites=["test"],
                pretty_name="X Wins Mid",
            ),
            TicTacToeMinimaxDataset(
                "o_wins_mid",
                BOARD_O_WINS_MID,
                depth=3,
                expected=-1.0,
                suites=["test"],
                pretty_name="O Wins Mid",
            ),
            TicTacToeMinimaxDataset(
                "x_wins_early",
                BOARD_X_WINS_EARLY,
                depth=5,
                expected=1.0,
                suites=["test"],
                pretty_name="X Wins Early",
            ),
            TicTacToeMinimaxDataset(
                "draw_early",
                BOARD_DRAW_EARLY,
                depth=6,
                expected=0.0,
                suites=["test", "standard", "trace", "train"],
                pretty_name="Draw Early",
            ),
            TicTacToeMinimaxDataset(
                "empty_board",
                BOARD_EMPTY,
                depth=9,
                suites=["stress"],
                pretty_name="Empty Board",
            ),
        ]

    def generate(self, dataset: TicTacToeMinimaxDataset):
        S_bin = from_numpy(dataset.board)
        ref_outputs = None
        if dataset.expected is not None:
            ref_outputs = [from_numpy(np.asarray(dataset.expected))]
        return DataInstance(
            inputs=[S_bin], meta={"depth": dataset.depth}, ref_outputs=ref_outputs
        )


class TicTacToeMinimaxBenchmark(Benchmark):
    @property
    def name(self) -> str:
        return "tictactoe_minimax"

    @property
    def pretty_name(self) -> str:
        return "Tic-Tac-Toe Minimax"

    @property
    def description(self) -> str:
        return (
            "What does this code do: Implement a fully tensorized, non-recursive"
            " minimax search over a tic-tac-toe game. Game states are represented as"
            " tensors. Game state is represented as S[n, i, j, p] of shape (N, 3, 3, 2)"
            " where n indexes boards, i,j are board positions and p is the player"
            " channel. Given any board state within the game, it should return the"
            " result of the game."
        )

    @property
    def suites(self) -> list[str]:
        return ["standard-logic"]

    @property
    def concepts(self) -> str:
        return """
<ccs2012>
<concept>
<concept_id>10010147.10010178.10010205.10010210</concept_id>
<concept_desc>Computing methodologies~Game tree search</concept_desc>
<concept_significance>500</concept_significance>
</concept>
</ccs2012>
"""

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
            "No generative AI was used to construct the benchmark function itself."
            " This statement was written by hand."
        )

    @property
    def motivation(self) -> str:
        return (
            "This benchmark will do sparse array operations on tic-tac-toe"
            "game trees. Sparsity should increase as you go deeper into the game/tree."
            "Invalid boards states caused by bad moves are zeroed out. You can test "
            "empty board all the way to the higher depth starting states"
        )

    @property
    def generators(self):
        return [TicTacToeMinimaxBoardsGenerator()]

    def benchmark(self, xp, meta: dict, S):
        depth = meta.get("depth", 9)
        W = xp.asarray(
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

        if depth == 2:
            N = S.shape[0]
            empty_flat = xp.reshape(1 - xp.sum(S, axis=3), (N, 9))
            count_X = xp.sum(S[:, :, :, 0], axis=(1, 2))
            count_O = xp.sum(S[:, :, :, 1], axis=(1, 2))
            turn = xp.where(
                count_X > count_O, xp.ones(S.shape[0]), xp.zeros(S.shape[0])
            )
            pos_exp = xp.reshape(xp.eye(9), (1, 9, 3, 3))
            turn_exp = xp.reshape(turn, (N, 1, 1, 1))
            ch0 = xp.reshape(pos_exp * (1 - turn_exp), (N, 9, 3, 3, 1))
            ch1 = xp.reshape(pos_exp * turn_exp, (N, 9, 3, 3, 1))
            delta = xp.concat([ch0, ch1], axis=4)
            valid_exp = xp.reshape(empty_flat, (N, 9, 1, 1, 1))
            children = (xp.reshape(S, (N, 1, 3, 3, 2)) + delta * valid_exp) * valid_exp
            # Above line zeroes out boards
            c1, v1 = (
                xp.reshape(children, (N * 9, 3, 3, 2)),
                xp.reshape(empty_flat, (N * 9,)),
            )
            N = c1.shape[0]
            empty_flat = xp.reshape(1 - xp.sum(c1, axis=3), (N, 9))
            count_X = xp.sum(c1[:, :, :, 0], axis=(1, 2))
            count_O = xp.sum(c1[:, :, :, 1], axis=(1, 2))
            turn = xp.where(
                count_X > count_O, xp.ones(c1.shape[0]), xp.zeros(c1.shape[0])
            )
            pos_exp = xp.reshape(xp.eye(9), (1, 9, 3, 3))
            turn_exp = xp.reshape(turn, (N, 1, 1, 1))
            ch0 = xp.reshape(pos_exp * (1 - turn_exp), (N, 9, 3, 3, 1))
            ch1 = xp.reshape(pos_exp * turn_exp, (N, 9, 3, 3, 1))
            delta = xp.concat([ch0, ch1], axis=4)
            valid_exp = xp.reshape(empty_flat, (N, 9, 1, 1, 1))
            children = (xp.reshape(c1, (N, 1, 3, 3, 2)) + delta * valid_exp) * valid_exp
            # Above line zeroes out boards
            c2, v2 = (
                xp.reshape(children, (N * 9, 3, 3, 2)),
                xp.reshape(empty_flat, (N * 9,)),
            )
            N = S.shape[0]
            W_exp = xp.reshape(W, (1, 8, 3, 3, 1))
            S_exp = xp.reshape(S, (S.shape[0], 1, 3, 3, 2))
            scores = xp.sum(W_exp * S_exp, axis=(2, 3))
            winner = xp.any(scores >= 3, axis=1)
            x_wins = winner[:, 0]
            o_wins = winner[:, 1]
            full = xp.all(xp.sum(S, axis=3) >= 1, axis=(1, 2))
            terminal = x_wins | o_wins | full
            value = xp.where(
                x_wins, xp.ones(N), xp.where(o_wins, -xp.ones(N), xp.zeros(N))
            )
            t0, val0 = terminal, value
            N = c1.shape[0]
            W_exp = xp.reshape(W, (1, 8, 3, 3, 1))
            S_exp = xp.reshape(c1, (c1.shape[0], 1, 3, 3, 2))
            scores = xp.sum(W_exp * S_exp, axis=(2, 3))
            winner = xp.any(scores >= 3, axis=1)
            x_wins = winner[:, 0]
            o_wins = winner[:, 1]
            full = xp.all(xp.sum(c1, axis=3) >= 1, axis=(1, 2))
            terminal = x_wins | o_wins | full
            value = xp.where(
                x_wins, xp.ones(N), xp.where(o_wins, -xp.ones(N), xp.zeros(N))
            )
            t1, val1 = terminal, value
            N = c2.shape[0]
            W_exp = xp.reshape(W, (1, 8, 3, 3, 1))
            S_exp = xp.reshape(c2, (c2.shape[0], 1, 3, 3, 2))
            scores = xp.sum(W_exp * S_exp, axis=(2, 3))
            winner = xp.any(scores >= 3, axis=1)
            x_wins = winner[:, 0]
            o_wins = winner[:, 1]
            full = xp.all(xp.sum(c2, axis=3) >= 1, axis=(1, 2))
            terminal = x_wins | o_wins | full
            value = xp.where(
                x_wins, xp.ones(N), xp.where(o_wins, -xp.ones(N), xp.zeros(N))
            )
            t2, val2 = terminal, value
            val2 = xp.where(t2, val2, xp.zeros(c2.shape[0]))
            N = c1.shape[0]
            count_X = xp.sum(c1[:, :, :, 0], axis=(1, 2))
            count_O = xp.sum(c1[:, :, :, 1], axis=(1, 2))
            turn = xp.where(
                count_X > count_O, xp.ones(c1.shape[0]), xp.zeros(c1.shape[0])
            )
            child_val = xp.where(v2 > 0, val2, xp.zeros(N * 9))
            child_val_grid = xp.reshape(child_val, (N, 9))
            valid_grid = xp.reshape(v2, (N, 9))
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
            val1 = xp.where(t1, val1, backed)
            N = S.shape[0]
            count_X = xp.sum(S[:, :, :, 0], axis=(1, 2))
            count_O = xp.sum(S[:, :, :, 1], axis=(1, 2))
            turn = xp.where(
                count_X > count_O, xp.ones(S.shape[0]), xp.zeros(S.shape[0])
            )
            child_val = xp.where(v1 > 0, val1, xp.zeros(N * 9))
            child_val_grid = xp.reshape(child_val, (N, 9))
            valid_grid = xp.reshape(v1, (N, 9))
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
            result = xp.where(t0, val0, backed)
        elif depth == 3:
            N = S.shape[0]
            empty_flat = xp.reshape(1 - xp.sum(S, axis=3), (N, 9))
            count_X = xp.sum(S[:, :, :, 0], axis=(1, 2))
            count_O = xp.sum(S[:, :, :, 1], axis=(1, 2))
            turn = xp.where(
                count_X > count_O, xp.ones(S.shape[0]), xp.zeros(S.shape[0])
            )
            pos_exp = xp.reshape(xp.eye(9), (1, 9, 3, 3))
            turn_exp = xp.reshape(turn, (N, 1, 1, 1))
            ch0 = xp.reshape(pos_exp * (1 - turn_exp), (N, 9, 3, 3, 1))
            ch1 = xp.reshape(pos_exp * turn_exp, (N, 9, 3, 3, 1))
            delta = xp.concat([ch0, ch1], axis=4)
            valid_exp = xp.reshape(empty_flat, (N, 9, 1, 1, 1))
            children = (xp.reshape(S, (N, 1, 3, 3, 2)) + delta * valid_exp) * valid_exp
            # Above line zeroes out boards
            c1, v1 = (
                xp.reshape(children, (N * 9, 3, 3, 2)),
                xp.reshape(empty_flat, (N * 9,)),
            )
            N = c1.shape[0]
            empty_flat = xp.reshape(1 - xp.sum(c1, axis=3), (N, 9))
            count_X = xp.sum(c1[:, :, :, 0], axis=(1, 2))
            count_O = xp.sum(c1[:, :, :, 1], axis=(1, 2))
            turn = xp.where(
                count_X > count_O, xp.ones(c1.shape[0]), xp.zeros(c1.shape[0])
            )
            pos_exp = xp.reshape(xp.eye(9), (1, 9, 3, 3))
            turn_exp = xp.reshape(turn, (N, 1, 1, 1))
            ch0 = xp.reshape(pos_exp * (1 - turn_exp), (N, 9, 3, 3, 1))
            ch1 = xp.reshape(pos_exp * turn_exp, (N, 9, 3, 3, 1))
            delta = xp.concat([ch0, ch1], axis=4)
            valid_exp = xp.reshape(empty_flat, (N, 9, 1, 1, 1))
            children = (xp.reshape(c1, (N, 1, 3, 3, 2)) + delta * valid_exp) * valid_exp
            # Above line zeroes out boards
            c2, v2 = (
                xp.reshape(children, (N * 9, 3, 3, 2)),
                xp.reshape(empty_flat, (N * 9,)),
            )
            N = c2.shape[0]
            empty_flat = xp.reshape(1 - xp.sum(c2, axis=3), (N, 9))
            count_X = xp.sum(c2[:, :, :, 0], axis=(1, 2))
            count_O = xp.sum(c2[:, :, :, 1], axis=(1, 2))
            turn = xp.where(
                count_X > count_O, xp.ones(c2.shape[0]), xp.zeros(c2.shape[0])
            )
            pos_exp = xp.reshape(xp.eye(9), (1, 9, 3, 3))
            turn_exp = xp.reshape(turn, (N, 1, 1, 1))
            ch0 = xp.reshape(pos_exp * (1 - turn_exp), (N, 9, 3, 3, 1))
            ch1 = xp.reshape(pos_exp * turn_exp, (N, 9, 3, 3, 1))
            delta = xp.concat([ch0, ch1], axis=4)
            valid_exp = xp.reshape(empty_flat, (N, 9, 1, 1, 1))
            children = (xp.reshape(c2, (N, 1, 3, 3, 2)) + delta * valid_exp) * valid_exp
            # Above line zeroes out boards
            c3, v3 = (
                xp.reshape(children, (N * 9, 3, 3, 2)),
                xp.reshape(empty_flat, (N * 9,)),
            )
            N = S.shape[0]
            W_exp = xp.reshape(W, (1, 8, 3, 3, 1))
            S_exp = xp.reshape(S, (S.shape[0], 1, 3, 3, 2))
            scores = xp.sum(W_exp * S_exp, axis=(2, 3))
            winner = xp.any(scores >= 3, axis=1)
            x_wins = winner[:, 0]
            o_wins = winner[:, 1]
            full = xp.all(xp.sum(S, axis=3) >= 1, axis=(1, 2))
            terminal = x_wins | o_wins | full
            value = xp.where(
                x_wins, xp.ones(N), xp.where(o_wins, -xp.ones(N), xp.zeros(N))
            )
            t0, val0 = terminal, value
            N = c1.shape[0]
            W_exp = xp.reshape(W, (1, 8, 3, 3, 1))
            S_exp = xp.reshape(c1, (c1.shape[0], 1, 3, 3, 2))
            scores = xp.sum(W_exp * S_exp, axis=(2, 3))
            winner = xp.any(scores >= 3, axis=1)
            x_wins = winner[:, 0]
            o_wins = winner[:, 1]
            full = xp.all(xp.sum(c1, axis=3) >= 1, axis=(1, 2))
            terminal = x_wins | o_wins | full
            value = xp.where(
                x_wins, xp.ones(N), xp.where(o_wins, -xp.ones(N), xp.zeros(N))
            )
            t1, val1 = terminal, value
            N = c2.shape[0]
            W_exp = xp.reshape(W, (1, 8, 3, 3, 1))
            S_exp = xp.reshape(c2, (c2.shape[0], 1, 3, 3, 2))
            scores = xp.sum(W_exp * S_exp, axis=(2, 3))
            winner = xp.any(scores >= 3, axis=1)
            x_wins = winner[:, 0]
            o_wins = winner[:, 1]
            full = xp.all(xp.sum(c2, axis=3) >= 1, axis=(1, 2))
            terminal = x_wins | o_wins | full
            value = xp.where(
                x_wins, xp.ones(N), xp.where(o_wins, -xp.ones(N), xp.zeros(N))
            )
            t2, val2 = terminal, value
            N = c3.shape[0]
            W_exp = xp.reshape(W, (1, 8, 3, 3, 1))
            S_exp = xp.reshape(c3, (c3.shape[0], 1, 3, 3, 2))
            scores = xp.sum(W_exp * S_exp, axis=(2, 3))
            winner = xp.any(scores >= 3, axis=1)
            x_wins = winner[:, 0]
            o_wins = winner[:, 1]
            full = xp.all(xp.sum(c3, axis=3) >= 1, axis=(1, 2))
            terminal = x_wins | o_wins | full
            value = xp.where(
                x_wins, xp.ones(N), xp.where(o_wins, -xp.ones(N), xp.zeros(N))
            )
            t3, val3 = terminal, value
            val3 = xp.where(t3, val3, xp.zeros(c3.shape[0]))
            N = c2.shape[0]
            count_X = xp.sum(c2[:, :, :, 0], axis=(1, 2))
            count_O = xp.sum(c2[:, :, :, 1], axis=(1, 2))
            turn = xp.where(
                count_X > count_O, xp.ones(c2.shape[0]), xp.zeros(c2.shape[0])
            )
            child_val = xp.where(v3 > 0, val3, xp.zeros(N * 9))
            child_val_grid = xp.reshape(child_val, (N, 9))
            valid_grid = xp.reshape(v3, (N, 9))
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
            val2 = xp.where(t2, val2, backed)
            N = c1.shape[0]
            count_X = xp.sum(c1[:, :, :, 0], axis=(1, 2))
            count_O = xp.sum(c1[:, :, :, 1], axis=(1, 2))
            turn = xp.where(
                count_X > count_O, xp.ones(c1.shape[0]), xp.zeros(c1.shape[0])
            )
            child_val = xp.where(v2 > 0, val2, xp.zeros(N * 9))
            child_val_grid = xp.reshape(child_val, (N, 9))
            valid_grid = xp.reshape(v2, (N, 9))
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
            val1 = xp.where(t1, val1, backed)
            N = S.shape[0]
            count_X = xp.sum(S[:, :, :, 0], axis=(1, 2))
            count_O = xp.sum(S[:, :, :, 1], axis=(1, 2))
            turn = xp.where(
                count_X > count_O, xp.ones(S.shape[0]), xp.zeros(S.shape[0])
            )
            child_val = xp.where(v1 > 0, val1, xp.zeros(N * 9))
            child_val_grid = xp.reshape(child_val, (N, 9))
            valid_grid = xp.reshape(v1, (N, 9))
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
            result = xp.where(t0, val0, backed)
        elif depth == 5:
            N = S.shape[0]
            empty_flat = xp.reshape(1 - xp.sum(S, axis=3), (N, 9))
            count_X = xp.sum(S[:, :, :, 0], axis=(1, 2))
            count_O = xp.sum(S[:, :, :, 1], axis=(1, 2))
            turn = xp.where(
                count_X > count_O, xp.ones(S.shape[0]), xp.zeros(S.shape[0])
            )
            pos_exp = xp.reshape(xp.eye(9), (1, 9, 3, 3))
            turn_exp = xp.reshape(turn, (N, 1, 1, 1))
            ch0 = xp.reshape(pos_exp * (1 - turn_exp), (N, 9, 3, 3, 1))
            ch1 = xp.reshape(pos_exp * turn_exp, (N, 9, 3, 3, 1))
            delta = xp.concat([ch0, ch1], axis=4)
            valid_exp = xp.reshape(empty_flat, (N, 9, 1, 1, 1))
            children = (xp.reshape(S, (N, 1, 3, 3, 2)) + delta * valid_exp) * valid_exp
            # Above line zeroes out boards
            c1, v1 = (
                xp.reshape(children, (N * 9, 3, 3, 2)),
                xp.reshape(empty_flat, (N * 9,)),
            )
            N = c1.shape[0]
            empty_flat = xp.reshape(1 - xp.sum(c1, axis=3), (N, 9))
            count_X = xp.sum(c1[:, :, :, 0], axis=(1, 2))
            count_O = xp.sum(c1[:, :, :, 1], axis=(1, 2))
            turn = xp.where(
                count_X > count_O, xp.ones(c1.shape[0]), xp.zeros(c1.shape[0])
            )
            pos_exp = xp.reshape(xp.eye(9), (1, 9, 3, 3))
            turn_exp = xp.reshape(turn, (N, 1, 1, 1))
            ch0 = xp.reshape(pos_exp * (1 - turn_exp), (N, 9, 3, 3, 1))
            ch1 = xp.reshape(pos_exp * turn_exp, (N, 9, 3, 3, 1))
            delta = xp.concat([ch0, ch1], axis=4)
            valid_exp = xp.reshape(empty_flat, (N, 9, 1, 1, 1))
            children = (xp.reshape(c1, (N, 1, 3, 3, 2)) + delta * valid_exp) * valid_exp
            # Above line zeroes out boards
            c2, v2 = (
                xp.reshape(children, (N * 9, 3, 3, 2)),
                xp.reshape(empty_flat, (N * 9,)),
            )
            N = c2.shape[0]
            empty_flat = xp.reshape(1 - xp.sum(c2, axis=3), (N, 9))
            count_X = xp.sum(c2[:, :, :, 0], axis=(1, 2))
            count_O = xp.sum(c2[:, :, :, 1], axis=(1, 2))
            turn = xp.where(
                count_X > count_O, xp.ones(c2.shape[0]), xp.zeros(c2.shape[0])
            )
            pos_exp = xp.reshape(xp.eye(9), (1, 9, 3, 3))
            turn_exp = xp.reshape(turn, (N, 1, 1, 1))
            ch0 = xp.reshape(pos_exp * (1 - turn_exp), (N, 9, 3, 3, 1))
            ch1 = xp.reshape(pos_exp * turn_exp, (N, 9, 3, 3, 1))
            delta = xp.concat([ch0, ch1], axis=4)
            valid_exp = xp.reshape(empty_flat, (N, 9, 1, 1, 1))
            children = (xp.reshape(c2, (N, 1, 3, 3, 2)) + delta * valid_exp) * valid_exp
            # Above line zeroes out boards
            c3, v3 = (
                xp.reshape(children, (N * 9, 3, 3, 2)),
                xp.reshape(empty_flat, (N * 9,)),
            )
            N = c3.shape[0]
            empty_flat = xp.reshape(1 - xp.sum(c3, axis=3), (N, 9))
            count_X = xp.sum(c3[:, :, :, 0], axis=(1, 2))
            count_O = xp.sum(c3[:, :, :, 1], axis=(1, 2))
            turn = xp.where(
                count_X > count_O, xp.ones(c3.shape[0]), xp.zeros(c3.shape[0])
            )
            pos_exp = xp.reshape(xp.eye(9), (1, 9, 3, 3))
            turn_exp = xp.reshape(turn, (N, 1, 1, 1))
            ch0 = xp.reshape(pos_exp * (1 - turn_exp), (N, 9, 3, 3, 1))
            ch1 = xp.reshape(pos_exp * turn_exp, (N, 9, 3, 3, 1))
            delta = xp.concat([ch0, ch1], axis=4)
            valid_exp = xp.reshape(empty_flat, (N, 9, 1, 1, 1))
            children = (xp.reshape(c3, (N, 1, 3, 3, 2)) + delta * valid_exp) * valid_exp
            # Above line zeroes out boards
            c4, v4 = (
                xp.reshape(children, (N * 9, 3, 3, 2)),
                xp.reshape(empty_flat, (N * 9,)),
            )
            N = c4.shape[0]
            empty_flat = xp.reshape(1 - xp.sum(c4, axis=3), (N, 9))
            count_X = xp.sum(c4[:, :, :, 0], axis=(1, 2))
            count_O = xp.sum(c4[:, :, :, 1], axis=(1, 2))
            turn = xp.where(
                count_X > count_O, xp.ones(c4.shape[0]), xp.zeros(c4.shape[0])
            )
            pos_exp = xp.reshape(xp.eye(9), (1, 9, 3, 3))
            turn_exp = xp.reshape(turn, (N, 1, 1, 1))
            ch0 = xp.reshape(pos_exp * (1 - turn_exp), (N, 9, 3, 3, 1))
            ch1 = xp.reshape(pos_exp * turn_exp, (N, 9, 3, 3, 1))
            delta = xp.concat([ch0, ch1], axis=4)
            valid_exp = xp.reshape(empty_flat, (N, 9, 1, 1, 1))
            children = (xp.reshape(c4, (N, 1, 3, 3, 2)) + delta * valid_exp) * valid_exp
            # Above line zeroes out boards
            c5, v5 = (
                xp.reshape(children, (N * 9, 3, 3, 2)),
                xp.reshape(empty_flat, (N * 9,)),
            )
            N = S.shape[0]
            W_exp = xp.reshape(W, (1, 8, 3, 3, 1))
            S_exp = xp.reshape(S, (S.shape[0], 1, 3, 3, 2))
            scores = xp.sum(W_exp * S_exp, axis=(2, 3))
            winner = xp.any(scores >= 3, axis=1)
            x_wins = winner[:, 0]
            o_wins = winner[:, 1]
            full = xp.all(xp.sum(S, axis=3) >= 1, axis=(1, 2))
            terminal = x_wins | o_wins | full
            value = xp.where(
                x_wins, xp.ones(N), xp.where(o_wins, -xp.ones(N), xp.zeros(N))
            )
            t0, val0 = terminal, value
            N = c1.shape[0]
            W_exp = xp.reshape(W, (1, 8, 3, 3, 1))
            S_exp = xp.reshape(c1, (c1.shape[0], 1, 3, 3, 2))
            scores = xp.sum(W_exp * S_exp, axis=(2, 3))
            winner = xp.any(scores >= 3, axis=1)
            x_wins = winner[:, 0]
            o_wins = winner[:, 1]
            full = xp.all(xp.sum(c1, axis=3) >= 1, axis=(1, 2))
            terminal = x_wins | o_wins | full
            value = xp.where(
                x_wins, xp.ones(N), xp.where(o_wins, -xp.ones(N), xp.zeros(N))
            )
            t1, val1 = terminal, value
            N = c2.shape[0]
            W_exp = xp.reshape(W, (1, 8, 3, 3, 1))
            S_exp = xp.reshape(c2, (c2.shape[0], 1, 3, 3, 2))
            scores = xp.sum(W_exp * S_exp, axis=(2, 3))
            winner = xp.any(scores >= 3, axis=1)
            x_wins = winner[:, 0]
            o_wins = winner[:, 1]
            full = xp.all(xp.sum(c2, axis=3) >= 1, axis=(1, 2))
            terminal = x_wins | o_wins | full
            value = xp.where(
                x_wins, xp.ones(N), xp.where(o_wins, -xp.ones(N), xp.zeros(N))
            )
            t2, val2 = terminal, value
            N = c3.shape[0]
            W_exp = xp.reshape(W, (1, 8, 3, 3, 1))
            S_exp = xp.reshape(c3, (c3.shape[0], 1, 3, 3, 2))
            scores = xp.sum(W_exp * S_exp, axis=(2, 3))
            winner = xp.any(scores >= 3, axis=1)
            x_wins = winner[:, 0]
            o_wins = winner[:, 1]
            full = xp.all(xp.sum(c3, axis=3) >= 1, axis=(1, 2))
            terminal = x_wins | o_wins | full
            value = xp.where(
                x_wins, xp.ones(N), xp.where(o_wins, -xp.ones(N), xp.zeros(N))
            )
            t3, val3 = terminal, value
            N = c4.shape[0]
            W_exp = xp.reshape(W, (1, 8, 3, 3, 1))
            S_exp = xp.reshape(c4, (c4.shape[0], 1, 3, 3, 2))
            scores = xp.sum(W_exp * S_exp, axis=(2, 3))
            winner = xp.any(scores >= 3, axis=1)
            x_wins = winner[:, 0]
            o_wins = winner[:, 1]
            full = xp.all(xp.sum(c4, axis=3) >= 1, axis=(1, 2))
            terminal = x_wins | o_wins | full
            value = xp.where(
                x_wins, xp.ones(N), xp.where(o_wins, -xp.ones(N), xp.zeros(N))
            )
            t4, val4 = terminal, value
            N = c5.shape[0]
            W_exp = xp.reshape(W, (1, 8, 3, 3, 1))
            S_exp = xp.reshape(c5, (c5.shape[0], 1, 3, 3, 2))
            scores = xp.sum(W_exp * S_exp, axis=(2, 3))
            winner = xp.any(scores >= 3, axis=1)
            x_wins = winner[:, 0]
            o_wins = winner[:, 1]
            full = xp.all(xp.sum(c5, axis=3) >= 1, axis=(1, 2))
            terminal = x_wins | o_wins | full
            value = xp.where(
                x_wins, xp.ones(N), xp.where(o_wins, -xp.ones(N), xp.zeros(N))
            )
            t5, val5 = terminal, value
            val5 = xp.where(t5, val5, xp.zeros(c5.shape[0]))
            N = c4.shape[0]
            count_X = xp.sum(c4[:, :, :, 0], axis=(1, 2))
            count_O = xp.sum(c4[:, :, :, 1], axis=(1, 2))
            turn = xp.where(
                count_X > count_O, xp.ones(c4.shape[0]), xp.zeros(c4.shape[0])
            )
            child_val = xp.where(v5 > 0, val5, xp.zeros(N * 9))
            child_val_grid = xp.reshape(child_val, (N, 9))
            valid_grid = xp.reshape(v5, (N, 9))
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
            val4 = xp.where(t4, val4, backed)
            N = c3.shape[0]
            count_X = xp.sum(c3[:, :, :, 0], axis=(1, 2))
            count_O = xp.sum(c3[:, :, :, 1], axis=(1, 2))
            turn = xp.where(
                count_X > count_O, xp.ones(c3.shape[0]), xp.zeros(c3.shape[0])
            )
            child_val = xp.where(v4 > 0, val4, xp.zeros(N * 9))
            child_val_grid = xp.reshape(child_val, (N, 9))
            valid_grid = xp.reshape(v4, (N, 9))
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
            val3 = xp.where(t3, val3, backed)
            N = c2.shape[0]
            count_X = xp.sum(c2[:, :, :, 0], axis=(1, 2))
            count_O = xp.sum(c2[:, :, :, 1], axis=(1, 2))
            turn = xp.where(
                count_X > count_O, xp.ones(c2.shape[0]), xp.zeros(c2.shape[0])
            )
            child_val = xp.where(v3 > 0, val3, xp.zeros(N * 9))
            child_val_grid = xp.reshape(child_val, (N, 9))
            valid_grid = xp.reshape(v3, (N, 9))
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
            val2 = xp.where(t2, val2, backed)
            N = c1.shape[0]
            count_X = xp.sum(c1[:, :, :, 0], axis=(1, 2))
            count_O = xp.sum(c1[:, :, :, 1], axis=(1, 2))
            turn = xp.where(
                count_X > count_O, xp.ones(c1.shape[0]), xp.zeros(c1.shape[0])
            )
            child_val = xp.where(v2 > 0, val2, xp.zeros(N * 9))
            child_val_grid = xp.reshape(child_val, (N, 9))
            valid_grid = xp.reshape(v2, (N, 9))
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
            val1 = xp.where(t1, val1, backed)
            N = S.shape[0]
            count_X = xp.sum(S[:, :, :, 0], axis=(1, 2))
            count_O = xp.sum(S[:, :, :, 1], axis=(1, 2))
            turn = xp.where(
                count_X > count_O, xp.ones(S.shape[0]), xp.zeros(S.shape[0])
            )
            child_val = xp.where(v1 > 0, val1, xp.zeros(N * 9))
            child_val_grid = xp.reshape(child_val, (N, 9))
            valid_grid = xp.reshape(v1, (N, 9))
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
            result = xp.where(t0, val0, backed)
        elif depth == 6:
            N = S.shape[0]
            empty_flat = xp.reshape(1 - xp.sum(S, axis=3), (N, 9))
            count_X = xp.sum(S[:, :, :, 0], axis=(1, 2))
            count_O = xp.sum(S[:, :, :, 1], axis=(1, 2))
            turn = xp.where(
                count_X > count_O, xp.ones(S.shape[0]), xp.zeros(S.shape[0])
            )
            pos_exp = xp.reshape(xp.eye(9), (1, 9, 3, 3))
            turn_exp = xp.reshape(turn, (N, 1, 1, 1))
            ch0 = xp.reshape(pos_exp * (1 - turn_exp), (N, 9, 3, 3, 1))
            ch1 = xp.reshape(pos_exp * turn_exp, (N, 9, 3, 3, 1))
            delta = xp.concat([ch0, ch1], axis=4)
            valid_exp = xp.reshape(empty_flat, (N, 9, 1, 1, 1))
            children = (xp.reshape(S, (N, 1, 3, 3, 2)) + delta * valid_exp) * valid_exp
            # Above line zeroes out boards
            c1, v1 = (
                xp.reshape(children, (N * 9, 3, 3, 2)),
                xp.reshape(empty_flat, (N * 9,)),
            )
            N = c1.shape[0]
            empty_flat = xp.reshape(1 - xp.sum(c1, axis=3), (N, 9))
            count_X = xp.sum(c1[:, :, :, 0], axis=(1, 2))
            count_O = xp.sum(c1[:, :, :, 1], axis=(1, 2))
            turn = xp.where(
                count_X > count_O, xp.ones(c1.shape[0]), xp.zeros(c1.shape[0])
            )
            pos_exp = xp.reshape(xp.eye(9), (1, 9, 3, 3))
            turn_exp = xp.reshape(turn, (N, 1, 1, 1))
            ch0 = xp.reshape(pos_exp * (1 - turn_exp), (N, 9, 3, 3, 1))
            ch1 = xp.reshape(pos_exp * turn_exp, (N, 9, 3, 3, 1))
            delta = xp.concat([ch0, ch1], axis=4)
            valid_exp = xp.reshape(empty_flat, (N, 9, 1, 1, 1))
            children = (xp.reshape(c1, (N, 1, 3, 3, 2)) + delta * valid_exp) * valid_exp
            # Above line zeroes out boards
            d5_c1, d5_v1 = (
                xp.reshape(children, (N * 9, 3, 3, 2)),
                xp.reshape(empty_flat, (N * 9,)),
            )
            N = d5_c1.shape[0]
            empty_flat = xp.reshape(1 - xp.sum(d5_c1, axis=3), (N, 9))
            count_X = xp.sum(d5_c1[:, :, :, 0], axis=(1, 2))
            count_O = xp.sum(d5_c1[:, :, :, 1], axis=(1, 2))
            turn = xp.where(
                count_X > count_O, xp.ones(d5_c1.shape[0]), xp.zeros(d5_c1.shape[0])
            )
            pos_exp = xp.reshape(xp.eye(9), (1, 9, 3, 3))
            turn_exp = xp.reshape(turn, (N, 1, 1, 1))
            ch0 = xp.reshape(pos_exp * (1 - turn_exp), (N, 9, 3, 3, 1))
            ch1 = xp.reshape(pos_exp * turn_exp, (N, 9, 3, 3, 1))
            delta = xp.concat([ch0, ch1], axis=4)
            valid_exp = xp.reshape(empty_flat, (N, 9, 1, 1, 1))
            children = (
                xp.reshape(d5_c1, (N, 1, 3, 3, 2)) + delta * valid_exp
            ) * valid_exp
            # Above line zeroes out boards
            d5_c2, d5_v2 = (
                xp.reshape(children, (N * 9, 3, 3, 2)),
                xp.reshape(empty_flat, (N * 9,)),
            )
            N = d5_c2.shape[0]
            empty_flat = xp.reshape(1 - xp.sum(d5_c2, axis=3), (N, 9))
            count_X = xp.sum(d5_c2[:, :, :, 0], axis=(1, 2))
            count_O = xp.sum(d5_c2[:, :, :, 1], axis=(1, 2))
            turn = xp.where(
                count_X > count_O, xp.ones(d5_c2.shape[0]), xp.zeros(d5_c2.shape[0])
            )
            pos_exp = xp.reshape(xp.eye(9), (1, 9, 3, 3))
            turn_exp = xp.reshape(turn, (N, 1, 1, 1))
            ch0 = xp.reshape(pos_exp * (1 - turn_exp), (N, 9, 3, 3, 1))
            ch1 = xp.reshape(pos_exp * turn_exp, (N, 9, 3, 3, 1))
            delta = xp.concat([ch0, ch1], axis=4)
            valid_exp = xp.reshape(empty_flat, (N, 9, 1, 1, 1))
            children = (
                xp.reshape(d5_c2, (N, 1, 3, 3, 2)) + delta * valid_exp
            ) * valid_exp
            # Above line zeroes out boards
            d5_c3, d5_v3 = (
                xp.reshape(children, (N * 9, 3, 3, 2)),
                xp.reshape(empty_flat, (N * 9,)),
            )
            N = d5_c3.shape[0]
            empty_flat = xp.reshape(1 - xp.sum(d5_c3, axis=3), (N, 9))
            count_X = xp.sum(d5_c3[:, :, :, 0], axis=(1, 2))
            count_O = xp.sum(d5_c3[:, :, :, 1], axis=(1, 2))
            turn = xp.where(
                count_X > count_O, xp.ones(d5_c3.shape[0]), xp.zeros(d5_c3.shape[0])
            )
            pos_exp = xp.reshape(xp.eye(9), (1, 9, 3, 3))
            turn_exp = xp.reshape(turn, (N, 1, 1, 1))
            ch0 = xp.reshape(pos_exp * (1 - turn_exp), (N, 9, 3, 3, 1))
            ch1 = xp.reshape(pos_exp * turn_exp, (N, 9, 3, 3, 1))
            delta = xp.concat([ch0, ch1], axis=4)
            valid_exp = xp.reshape(empty_flat, (N, 9, 1, 1, 1))
            children = (
                xp.reshape(d5_c3, (N, 1, 3, 3, 2)) + delta * valid_exp
            ) * valid_exp
            # Above line zeroes out boards
            d5_c4, d5_v4 = (
                xp.reshape(children, (N * 9, 3, 3, 2)),
                xp.reshape(empty_flat, (N * 9,)),
            )
            N = d5_c4.shape[0]
            empty_flat = xp.reshape(1 - xp.sum(d5_c4, axis=3), (N, 9))
            count_X = xp.sum(d5_c4[:, :, :, 0], axis=(1, 2))
            count_O = xp.sum(d5_c4[:, :, :, 1], axis=(1, 2))
            turn = xp.where(
                count_X > count_O, xp.ones(d5_c4.shape[0]), xp.zeros(d5_c4.shape[0])
            )
            pos_exp = xp.reshape(xp.eye(9), (1, 9, 3, 3))
            turn_exp = xp.reshape(turn, (N, 1, 1, 1))
            ch0 = xp.reshape(pos_exp * (1 - turn_exp), (N, 9, 3, 3, 1))
            ch1 = xp.reshape(pos_exp * turn_exp, (N, 9, 3, 3, 1))
            delta = xp.concat([ch0, ch1], axis=4)
            valid_exp = xp.reshape(empty_flat, (N, 9, 1, 1, 1))
            children = (
                xp.reshape(d5_c4, (N, 1, 3, 3, 2)) + delta * valid_exp
            ) * valid_exp
            # Above line zeroes out boards
            d5_c5, d5_v5 = (
                xp.reshape(children, (N * 9, 3, 3, 2)),
                xp.reshape(empty_flat, (N * 9,)),
            )
            N = c1.shape[0]
            W_exp = xp.reshape(W, (1, 8, 3, 3, 1))
            S_exp = xp.reshape(c1, (c1.shape[0], 1, 3, 3, 2))
            scores = xp.sum(W_exp * S_exp, axis=(2, 3))
            winner = xp.any(scores >= 3, axis=1)
            x_wins = winner[:, 0]
            o_wins = winner[:, 1]
            full = xp.all(xp.sum(c1, axis=3) >= 1, axis=(1, 2))
            terminal = x_wins | o_wins | full
            value = xp.where(
                x_wins, xp.ones(N), xp.where(o_wins, -xp.ones(N), xp.zeros(N))
            )
            d5_t0, d5_val0 = terminal, value
            N = d5_c1.shape[0]
            W_exp = xp.reshape(W, (1, 8, 3, 3, 1))
            S_exp = xp.reshape(d5_c1, (d5_c1.shape[0], 1, 3, 3, 2))
            scores = xp.sum(W_exp * S_exp, axis=(2, 3))
            winner = xp.any(scores >= 3, axis=1)
            x_wins = winner[:, 0]
            o_wins = winner[:, 1]
            full = xp.all(xp.sum(d5_c1, axis=3) >= 1, axis=(1, 2))
            terminal = x_wins | o_wins | full
            value = xp.where(
                x_wins, xp.ones(N), xp.where(o_wins, -xp.ones(N), xp.zeros(N))
            )
            d5_t1, d5_val1 = terminal, value
            N = d5_c2.shape[0]
            W_exp = xp.reshape(W, (1, 8, 3, 3, 1))
            S_exp = xp.reshape(d5_c2, (d5_c2.shape[0], 1, 3, 3, 2))
            scores = xp.sum(W_exp * S_exp, axis=(2, 3))
            winner = xp.any(scores >= 3, axis=1)
            x_wins = winner[:, 0]
            o_wins = winner[:, 1]
            full = xp.all(xp.sum(d5_c2, axis=3) >= 1, axis=(1, 2))
            terminal = x_wins | o_wins | full
            value = xp.where(
                x_wins, xp.ones(N), xp.where(o_wins, -xp.ones(N), xp.zeros(N))
            )
            d5_t2, d5_val2 = terminal, value
            N = d5_c3.shape[0]
            W_exp = xp.reshape(W, (1, 8, 3, 3, 1))
            S_exp = xp.reshape(d5_c3, (d5_c3.shape[0], 1, 3, 3, 2))
            scores = xp.sum(W_exp * S_exp, axis=(2, 3))
            winner = xp.any(scores >= 3, axis=1)
            x_wins = winner[:, 0]
            o_wins = winner[:, 1]
            full = xp.all(xp.sum(d5_c3, axis=3) >= 1, axis=(1, 2))
            terminal = x_wins | o_wins | full
            value = xp.where(
                x_wins, xp.ones(N), xp.where(o_wins, -xp.ones(N), xp.zeros(N))
            )
            d5_t3, d5_val3 = terminal, value
            N = d5_c4.shape[0]
            W_exp = xp.reshape(W, (1, 8, 3, 3, 1))
            S_exp = xp.reshape(d5_c4, (d5_c4.shape[0], 1, 3, 3, 2))
            scores = xp.sum(W_exp * S_exp, axis=(2, 3))
            winner = xp.any(scores >= 3, axis=1)
            x_wins = winner[:, 0]
            o_wins = winner[:, 1]
            full = xp.all(xp.sum(d5_c4, axis=3) >= 1, axis=(1, 2))
            terminal = x_wins | o_wins | full
            value = xp.where(
                x_wins, xp.ones(N), xp.where(o_wins, -xp.ones(N), xp.zeros(N))
            )
            d5_t4, d5_val4 = terminal, value
            N = d5_c5.shape[0]
            W_exp = xp.reshape(W, (1, 8, 3, 3, 1))
            S_exp = xp.reshape(d5_c5, (d5_c5.shape[0], 1, 3, 3, 2))
            scores = xp.sum(W_exp * S_exp, axis=(2, 3))
            winner = xp.any(scores >= 3, axis=1)
            x_wins = winner[:, 0]
            o_wins = winner[:, 1]
            full = xp.all(xp.sum(d5_c5, axis=3) >= 1, axis=(1, 2))
            terminal = x_wins | o_wins | full
            value = xp.where(
                x_wins, xp.ones(N), xp.where(o_wins, -xp.ones(N), xp.zeros(N))
            )
            d5_t5, d5_val5 = terminal, value
            d5_val5 = xp.where(d5_t5, d5_val5, xp.zeros(d5_c5.shape[0]))
            N = d5_c4.shape[0]
            count_X = xp.sum(d5_c4[:, :, :, 0], axis=(1, 2))
            count_O = xp.sum(d5_c4[:, :, :, 1], axis=(1, 2))
            turn = xp.where(
                count_X > count_O, xp.ones(d5_c4.shape[0]), xp.zeros(d5_c4.shape[0])
            )
            child_val = xp.where(d5_v5 > 0, d5_val5, xp.zeros(N * 9))
            child_val_grid = xp.reshape(child_val, (N, 9))
            valid_grid = xp.reshape(d5_v5, (N, 9))
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
            d5_val4 = xp.where(d5_t4, d5_val4, backed)
            N = d5_c3.shape[0]
            count_X = xp.sum(d5_c3[:, :, :, 0], axis=(1, 2))
            count_O = xp.sum(d5_c3[:, :, :, 1], axis=(1, 2))
            turn = xp.where(
                count_X > count_O, xp.ones(d5_c3.shape[0]), xp.zeros(d5_c3.shape[0])
            )
            child_val = xp.where(d5_v4 > 0, d5_val4, xp.zeros(N * 9))
            child_val_grid = xp.reshape(child_val, (N, 9))
            valid_grid = xp.reshape(d5_v4, (N, 9))
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
            d5_val3 = xp.where(d5_t3, d5_val3, backed)
            N = d5_c2.shape[0]
            count_X = xp.sum(d5_c2[:, :, :, 0], axis=(1, 2))
            count_O = xp.sum(d5_c2[:, :, :, 1], axis=(1, 2))
            turn = xp.where(
                count_X > count_O, xp.ones(d5_c2.shape[0]), xp.zeros(d5_c2.shape[0])
            )
            child_val = xp.where(d5_v3 > 0, d5_val3, xp.zeros(N * 9))
            child_val_grid = xp.reshape(child_val, (N, 9))
            valid_grid = xp.reshape(d5_v3, (N, 9))
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
            d5_val2 = xp.where(d5_t2, d5_val2, backed)
            N = d5_c1.shape[0]
            count_X = xp.sum(d5_c1[:, :, :, 0], axis=(1, 2))
            count_O = xp.sum(d5_c1[:, :, :, 1], axis=(1, 2))
            turn = xp.where(
                count_X > count_O, xp.ones(d5_c1.shape[0]), xp.zeros(d5_c1.shape[0])
            )
            child_val = xp.where(d5_v2 > 0, d5_val2, xp.zeros(N * 9))
            child_val_grid = xp.reshape(child_val, (N, 9))
            valid_grid = xp.reshape(d5_v2, (N, 9))
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
            d5_val1 = xp.where(d5_t1, d5_val1, backed)
            N = c1.shape[0]
            count_X = xp.sum(c1[:, :, :, 0], axis=(1, 2))
            count_O = xp.sum(c1[:, :, :, 1], axis=(1, 2))
            turn = xp.where(
                count_X > count_O, xp.ones(c1.shape[0]), xp.zeros(c1.shape[0])
            )
            child_val = xp.where(d5_v1 > 0, d5_val1, xp.zeros(N * 9))
            child_val_grid = xp.reshape(child_val, (N, 9))
            valid_grid = xp.reshape(d5_v1, (N, 9))
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
            val1 = xp.where(d5_t0, d5_val0, backed)
            N = S.shape[0]
            W_exp = xp.reshape(W, (1, 8, 3, 3, 1))
            S_exp = xp.reshape(S, (S.shape[0], 1, 3, 3, 2))
            scores = xp.sum(W_exp * S_exp, axis=(2, 3))
            winner = xp.any(scores >= 3, axis=1)
            x_wins = winner[:, 0]
            o_wins = winner[:, 1]
            full = xp.all(xp.sum(S, axis=3) >= 1, axis=(1, 2))
            terminal = x_wins | o_wins | full
            value = xp.where(
                x_wins, xp.ones(N), xp.where(o_wins, -xp.ones(N), xp.zeros(N))
            )
            t0, val0 = terminal, value
            N = S.shape[0]
            count_X = xp.sum(S[:, :, :, 0], axis=(1, 2))
            count_O = xp.sum(S[:, :, :, 1], axis=(1, 2))
            turn = xp.where(
                count_X > count_O, xp.ones(S.shape[0]), xp.zeros(S.shape[0])
            )
            child_val = xp.where(v1 > 0, val1, xp.zeros(N * 9))
            child_val_grid = xp.reshape(child_val, (N, 9))
            valid_grid = xp.reshape(v1, (N, 9))
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
            result = xp.where(t0, val0, backed)
        else:
            N = S.shape[0]
            empty_flat = xp.reshape(1 - xp.sum(S, axis=3), (N, 9))
            count_X = xp.sum(S[:, :, :, 0], axis=(1, 2))
            count_O = xp.sum(S[:, :, :, 1], axis=(1, 2))
            turn = xp.where(
                count_X > count_O, xp.ones(S.shape[0]), xp.zeros(S.shape[0])
            )
            pos_exp = xp.reshape(xp.eye(9), (1, 9, 3, 3))
            turn_exp = xp.reshape(turn, (N, 1, 1, 1))
            ch0 = xp.reshape(pos_exp * (1 - turn_exp), (N, 9, 3, 3, 1))
            ch1 = xp.reshape(pos_exp * turn_exp, (N, 9, 3, 3, 1))
            delta = xp.concat([ch0, ch1], axis=4)
            valid_exp = xp.reshape(empty_flat, (N, 9, 1, 1, 1))
            children = (xp.reshape(S, (N, 1, 3, 3, 2)) + delta * valid_exp) * valid_exp
            # Above line zeroes out boards
            c1, v1 = (
                xp.reshape(children, (N * 9, 3, 3, 2)),
                xp.reshape(empty_flat, (N * 9,)),
            )
            N = c1.shape[0]
            empty_flat = xp.reshape(1 - xp.sum(c1, axis=3), (N, 9))
            count_X = xp.sum(c1[:, :, :, 0], axis=(1, 2))
            count_O = xp.sum(c1[:, :, :, 1], axis=(1, 2))
            turn = xp.where(
                count_X > count_O, xp.ones(c1.shape[0]), xp.zeros(c1.shape[0])
            )
            pos_exp = xp.reshape(xp.eye(9), (1, 9, 3, 3))
            turn_exp = xp.reshape(turn, (N, 1, 1, 1))
            ch0 = xp.reshape(pos_exp * (1 - turn_exp), (N, 9, 3, 3, 1))
            ch1 = xp.reshape(pos_exp * turn_exp, (N, 9, 3, 3, 1))
            delta = xp.concat([ch0, ch1], axis=4)
            valid_exp = xp.reshape(empty_flat, (N, 9, 1, 1, 1))
            children = (xp.reshape(c1, (N, 1, 3, 3, 2)) + delta * valid_exp) * valid_exp
            # Above line zeroes out boards
            c2, v2 = (
                xp.reshape(children, (N * 9, 3, 3, 2)),
                xp.reshape(empty_flat, (N * 9,)),
            )
            N = c2.shape[0]
            empty_flat = xp.reshape(1 - xp.sum(c2, axis=3), (N, 9))
            count_X = xp.sum(c2[:, :, :, 0], axis=(1, 2))
            count_O = xp.sum(c2[:, :, :, 1], axis=(1, 2))
            turn = xp.where(
                count_X > count_O, xp.ones(c2.shape[0]), xp.zeros(c2.shape[0])
            )
            pos_exp = xp.reshape(xp.eye(9), (1, 9, 3, 3))
            turn_exp = xp.reshape(turn, (N, 1, 1, 1))
            ch0 = xp.reshape(pos_exp * (1 - turn_exp), (N, 9, 3, 3, 1))
            ch1 = xp.reshape(pos_exp * turn_exp, (N, 9, 3, 3, 1))
            delta = xp.concat([ch0, ch1], axis=4)
            valid_exp = xp.reshape(empty_flat, (N, 9, 1, 1, 1))
            children = (xp.reshape(c2, (N, 1, 3, 3, 2)) + delta * valid_exp) * valid_exp
            # Above line zeroes out boards
            c3, v3 = (
                xp.reshape(children, (N * 9, 3, 3, 2)),
                xp.reshape(empty_flat, (N * 9,)),
            )
            N = c3.shape[0]
            empty_flat = xp.reshape(1 - xp.sum(c3, axis=3), (N, 9))
            count_X = xp.sum(c3[:, :, :, 0], axis=(1, 2))
            count_O = xp.sum(c3[:, :, :, 1], axis=(1, 2))
            turn = xp.where(
                count_X > count_O, xp.ones(c3.shape[0]), xp.zeros(c3.shape[0])
            )
            pos_exp = xp.reshape(xp.eye(9), (1, 9, 3, 3))
            turn_exp = xp.reshape(turn, (N, 1, 1, 1))
            ch0 = xp.reshape(pos_exp * (1 - turn_exp), (N, 9, 3, 3, 1))
            ch1 = xp.reshape(pos_exp * turn_exp, (N, 9, 3, 3, 1))
            delta = xp.concat([ch0, ch1], axis=4)
            valid_exp = xp.reshape(empty_flat, (N, 9, 1, 1, 1))
            children = (xp.reshape(c3, (N, 1, 3, 3, 2)) + delta * valid_exp) * valid_exp
            # Above line zeroes out boards
            c4, v4 = (
                xp.reshape(children, (N * 9, 3, 3, 2)),
                xp.reshape(empty_flat, (N * 9,)),
            )
            N = c4.shape[0]
            empty_flat = xp.reshape(1 - xp.sum(c4, axis=3), (N, 9))
            count_X = xp.sum(c4[:, :, :, 0], axis=(1, 2))
            count_O = xp.sum(c4[:, :, :, 1], axis=(1, 2))
            turn = xp.where(
                count_X > count_O, xp.ones(c4.shape[0]), xp.zeros(c4.shape[0])
            )
            pos_exp = xp.reshape(xp.eye(9), (1, 9, 3, 3))
            turn_exp = xp.reshape(turn, (N, 1, 1, 1))
            ch0 = xp.reshape(pos_exp * (1 - turn_exp), (N, 9, 3, 3, 1))
            ch1 = xp.reshape(pos_exp * turn_exp, (N, 9, 3, 3, 1))
            delta = xp.concat([ch0, ch1], axis=4)
            valid_exp = xp.reshape(empty_flat, (N, 9, 1, 1, 1))
            children = (xp.reshape(c4, (N, 1, 3, 3, 2)) + delta * valid_exp) * valid_exp
            # Above line zeroes out boards
            c5, v5 = (
                xp.reshape(children, (N * 9, 3, 3, 2)),
                xp.reshape(empty_flat, (N * 9,)),
            )
            N = c5.shape[0]
            empty_flat = xp.reshape(1 - xp.sum(c5, axis=3), (N, 9))
            count_X = xp.sum(c5[:, :, :, 0], axis=(1, 2))
            count_O = xp.sum(c5[:, :, :, 1], axis=(1, 2))
            turn = xp.where(
                count_X > count_O, xp.ones(c5.shape[0]), xp.zeros(c5.shape[0])
            )
            pos_exp = xp.reshape(xp.eye(9), (1, 9, 3, 3))
            turn_exp = xp.reshape(turn, (N, 1, 1, 1))
            ch0 = xp.reshape(pos_exp * (1 - turn_exp), (N, 9, 3, 3, 1))
            ch1 = xp.reshape(pos_exp * turn_exp, (N, 9, 3, 3, 1))
            delta = xp.concat([ch0, ch1], axis=4)
            valid_exp = xp.reshape(empty_flat, (N, 9, 1, 1, 1))
            children = (xp.reshape(c5, (N, 1, 3, 3, 2)) + delta * valid_exp) * valid_exp
            # Above line zeroes out boards
            c6, v6 = (
                xp.reshape(children, (N * 9, 3, 3, 2)),
                xp.reshape(empty_flat, (N * 9,)),
            )
            N = c6.shape[0]
            empty_flat = xp.reshape(1 - xp.sum(c6, axis=3), (N, 9))
            count_X = xp.sum(c6[:, :, :, 0], axis=(1, 2))
            count_O = xp.sum(c6[:, :, :, 1], axis=(1, 2))
            turn = xp.where(
                count_X > count_O, xp.ones(c6.shape[0]), xp.zeros(c6.shape[0])
            )
            pos_exp = xp.reshape(xp.eye(9), (1, 9, 3, 3))
            turn_exp = xp.reshape(turn, (N, 1, 1, 1))
            ch0 = xp.reshape(pos_exp * (1 - turn_exp), (N, 9, 3, 3, 1))
            ch1 = xp.reshape(pos_exp * turn_exp, (N, 9, 3, 3, 1))
            delta = xp.concat([ch0, ch1], axis=4)
            valid_exp = xp.reshape(empty_flat, (N, 9, 1, 1, 1))
            children = (xp.reshape(c6, (N, 1, 3, 3, 2)) + delta * valid_exp) * valid_exp
            # Above line zeroes out boards
            c7, v7 = (
                xp.reshape(children, (N * 9, 3, 3, 2)),
                xp.reshape(empty_flat, (N * 9,)),
            )
            N = c7.shape[0]
            empty_flat = xp.reshape(1 - xp.sum(c7, axis=3), (N, 9))
            count_X = xp.sum(c7[:, :, :, 0], axis=(1, 2))
            count_O = xp.sum(c7[:, :, :, 1], axis=(1, 2))
            turn = xp.where(
                count_X > count_O, xp.ones(c7.shape[0]), xp.zeros(c7.shape[0])
            )
            pos_exp = xp.reshape(xp.eye(9), (1, 9, 3, 3))
            turn_exp = xp.reshape(turn, (N, 1, 1, 1))
            ch0 = xp.reshape(pos_exp * (1 - turn_exp), (N, 9, 3, 3, 1))
            ch1 = xp.reshape(pos_exp * turn_exp, (N, 9, 3, 3, 1))
            delta = xp.concat([ch0, ch1], axis=4)
            valid_exp = xp.reshape(empty_flat, (N, 9, 1, 1, 1))
            children = (xp.reshape(c7, (N, 1, 3, 3, 2)) + delta * valid_exp) * valid_exp
            # Above line zeroes out boards
            c8, v8 = (
                xp.reshape(children, (N * 9, 3, 3, 2)),
                xp.reshape(empty_flat, (N * 9,)),
            )
            N = c8.shape[0]
            empty_flat = xp.reshape(1 - xp.sum(c8, axis=3), (N, 9))
            count_X = xp.sum(c8[:, :, :, 0], axis=(1, 2))
            count_O = xp.sum(c8[:, :, :, 1], axis=(1, 2))
            turn = xp.where(
                count_X > count_O, xp.ones(c8.shape[0]), xp.zeros(c8.shape[0])
            )
            pos_exp = xp.reshape(xp.eye(9), (1, 9, 3, 3))
            turn_exp = xp.reshape(turn, (N, 1, 1, 1))
            ch0 = xp.reshape(pos_exp * (1 - turn_exp), (N, 9, 3, 3, 1))
            ch1 = xp.reshape(pos_exp * turn_exp, (N, 9, 3, 3, 1))
            delta = xp.concat([ch0, ch1], axis=4)
            valid_exp = xp.reshape(empty_flat, (N, 9, 1, 1, 1))
            children = (xp.reshape(c8, (N, 1, 3, 3, 2)) + delta * valid_exp) * valid_exp
            # Above line zeroes out boards
            c9, v9 = (
                xp.reshape(children, (N * 9, 3, 3, 2)),
                xp.reshape(empty_flat, (N * 9,)),
            )
            N = S.shape[0]
            W_exp = xp.reshape(W, (1, 8, 3, 3, 1))
            S_exp = xp.reshape(S, (S.shape[0], 1, 3, 3, 2))
            scores = xp.sum(W_exp * S_exp, axis=(2, 3))
            winner = xp.any(scores >= 3, axis=1)
            x_wins = winner[:, 0]
            o_wins = winner[:, 1]
            full = xp.all(xp.sum(S, axis=3) >= 1, axis=(1, 2))
            terminal = x_wins | o_wins | full
            value = xp.where(
                x_wins, xp.ones(N), xp.where(o_wins, -xp.ones(N), xp.zeros(N))
            )
            t0, val0 = terminal, value
            N = c1.shape[0]
            W_exp = xp.reshape(W, (1, 8, 3, 3, 1))
            S_exp = xp.reshape(c1, (c1.shape[0], 1, 3, 3, 2))
            scores = xp.sum(W_exp * S_exp, axis=(2, 3))
            winner = xp.any(scores >= 3, axis=1)
            x_wins = winner[:, 0]
            o_wins = winner[:, 1]
            full = xp.all(xp.sum(c1, axis=3) >= 1, axis=(1, 2))
            terminal = x_wins | o_wins | full
            value = xp.where(
                x_wins, xp.ones(N), xp.where(o_wins, -xp.ones(N), xp.zeros(N))
            )
            t1, val1 = terminal, value
            N = c2.shape[0]
            W_exp = xp.reshape(W, (1, 8, 3, 3, 1))
            S_exp = xp.reshape(c2, (c2.shape[0], 1, 3, 3, 2))
            scores = xp.sum(W_exp * S_exp, axis=(2, 3))
            winner = xp.any(scores >= 3, axis=1)
            x_wins = winner[:, 0]
            o_wins = winner[:, 1]
            full = xp.all(xp.sum(c2, axis=3) >= 1, axis=(1, 2))
            terminal = x_wins | o_wins | full
            value = xp.where(
                x_wins, xp.ones(N), xp.where(o_wins, -xp.ones(N), xp.zeros(N))
            )
            t2, val2 = terminal, value
            N = c3.shape[0]
            W_exp = xp.reshape(W, (1, 8, 3, 3, 1))
            S_exp = xp.reshape(c3, (c3.shape[0], 1, 3, 3, 2))
            scores = xp.sum(W_exp * S_exp, axis=(2, 3))
            winner = xp.any(scores >= 3, axis=1)
            x_wins = winner[:, 0]
            o_wins = winner[:, 1]
            full = xp.all(xp.sum(c3, axis=3) >= 1, axis=(1, 2))
            terminal = x_wins | o_wins | full
            value = xp.where(
                x_wins, xp.ones(N), xp.where(o_wins, -xp.ones(N), xp.zeros(N))
            )
            t3, val3 = terminal, value
            N = c4.shape[0]
            W_exp = xp.reshape(W, (1, 8, 3, 3, 1))
            S_exp = xp.reshape(c4, (c4.shape[0], 1, 3, 3, 2))
            scores = xp.sum(W_exp * S_exp, axis=(2, 3))
            winner = xp.any(scores >= 3, axis=1)
            x_wins = winner[:, 0]
            o_wins = winner[:, 1]
            full = xp.all(xp.sum(c4, axis=3) >= 1, axis=(1, 2))
            terminal = x_wins | o_wins | full
            value = xp.where(
                x_wins, xp.ones(N), xp.where(o_wins, -xp.ones(N), xp.zeros(N))
            )
            t4, val4 = terminal, value
            N = c5.shape[0]
            W_exp = xp.reshape(W, (1, 8, 3, 3, 1))
            S_exp = xp.reshape(c5, (c5.shape[0], 1, 3, 3, 2))
            scores = xp.sum(W_exp * S_exp, axis=(2, 3))
            winner = xp.any(scores >= 3, axis=1)
            x_wins = winner[:, 0]
            o_wins = winner[:, 1]
            full = xp.all(xp.sum(c5, axis=3) >= 1, axis=(1, 2))
            terminal = x_wins | o_wins | full
            value = xp.where(
                x_wins, xp.ones(N), xp.where(o_wins, -xp.ones(N), xp.zeros(N))
            )
            t5, val5 = terminal, value
            N = c6.shape[0]
            W_exp = xp.reshape(W, (1, 8, 3, 3, 1))
            S_exp = xp.reshape(c6, (c6.shape[0], 1, 3, 3, 2))
            scores = xp.sum(W_exp * S_exp, axis=(2, 3))
            winner = xp.any(scores >= 3, axis=1)
            x_wins = winner[:, 0]
            o_wins = winner[:, 1]
            full = xp.all(xp.sum(c6, axis=3) >= 1, axis=(1, 2))
            terminal = x_wins | o_wins | full
            value = xp.where(
                x_wins, xp.ones(N), xp.where(o_wins, -xp.ones(N), xp.zeros(N))
            )
            t6, val6 = terminal, value
            N = c7.shape[0]
            W_exp = xp.reshape(W, (1, 8, 3, 3, 1))
            S_exp = xp.reshape(c7, (c7.shape[0], 1, 3, 3, 2))
            scores = xp.sum(W_exp * S_exp, axis=(2, 3))
            winner = xp.any(scores >= 3, axis=1)
            x_wins = winner[:, 0]
            o_wins = winner[:, 1]
            full = xp.all(xp.sum(c7, axis=3) >= 1, axis=(1, 2))
            terminal = x_wins | o_wins | full
            value = xp.where(
                x_wins, xp.ones(N), xp.where(o_wins, -xp.ones(N), xp.zeros(N))
            )
            t7, val7 = terminal, value
            N = c8.shape[0]
            W_exp = xp.reshape(W, (1, 8, 3, 3, 1))
            S_exp = xp.reshape(c8, (c8.shape[0], 1, 3, 3, 2))
            scores = xp.sum(W_exp * S_exp, axis=(2, 3))
            winner = xp.any(scores >= 3, axis=1)
            x_wins = winner[:, 0]
            o_wins = winner[:, 1]
            full = xp.all(xp.sum(c8, axis=3) >= 1, axis=(1, 2))
            terminal = x_wins | o_wins | full
            value = xp.where(
                x_wins, xp.ones(N), xp.where(o_wins, -xp.ones(N), xp.zeros(N))
            )
            t8, val8 = terminal, value
            N = c9.shape[0]
            W_exp = xp.reshape(W, (1, 8, 3, 3, 1))
            S_exp = xp.reshape(c9, (c9.shape[0], 1, 3, 3, 2))
            scores = xp.sum(W_exp * S_exp, axis=(2, 3))
            winner = xp.any(scores >= 3, axis=1)
            x_wins = winner[:, 0]
            o_wins = winner[:, 1]
            full = xp.all(xp.sum(c9, axis=3) >= 1, axis=(1, 2))
            terminal = x_wins | o_wins | full
            value = xp.where(
                x_wins, xp.ones(N), xp.where(o_wins, -xp.ones(N), xp.zeros(N))
            )
            t9, val9 = terminal, value
            val9 = xp.where(t9, val9, xp.zeros(c9.shape[0]))
            N = c8.shape[0]
            count_X = xp.sum(c8[:, :, :, 0], axis=(1, 2))
            count_O = xp.sum(c8[:, :, :, 1], axis=(1, 2))
            turn = xp.where(
                count_X > count_O, xp.ones(c8.shape[0]), xp.zeros(c8.shape[0])
            )
            child_val = xp.where(v9 > 0, val9, xp.zeros(N * 9))
            child_val_grid = xp.reshape(child_val, (N, 9))
            valid_grid = xp.reshape(v9, (N, 9))
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
            val8 = xp.where(t8, val8, backed)
            N = c7.shape[0]
            count_X = xp.sum(c7[:, :, :, 0], axis=(1, 2))
            count_O = xp.sum(c7[:, :, :, 1], axis=(1, 2))
            turn = xp.where(
                count_X > count_O, xp.ones(c7.shape[0]), xp.zeros(c7.shape[0])
            )
            child_val = xp.where(v8 > 0, val8, xp.zeros(N * 9))
            child_val_grid = xp.reshape(child_val, (N, 9))
            valid_grid = xp.reshape(v8, (N, 9))
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
            val7 = xp.where(t7, val7, backed)
            N = c6.shape[0]
            count_X = xp.sum(c6[:, :, :, 0], axis=(1, 2))
            count_O = xp.sum(c6[:, :, :, 1], axis=(1, 2))
            turn = xp.where(
                count_X > count_O, xp.ones(c6.shape[0]), xp.zeros(c6.shape[0])
            )
            child_val = xp.where(v7 > 0, val7, xp.zeros(N * 9))
            child_val_grid = xp.reshape(child_val, (N, 9))
            valid_grid = xp.reshape(v7, (N, 9))
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
            val6 = xp.where(t6, val6, backed)
            N = c5.shape[0]
            count_X = xp.sum(c5[:, :, :, 0], axis=(1, 2))
            count_O = xp.sum(c5[:, :, :, 1], axis=(1, 2))
            turn = xp.where(
                count_X > count_O, xp.ones(c5.shape[0]), xp.zeros(c5.shape[0])
            )
            child_val = xp.where(v6 > 0, val6, xp.zeros(N * 9))
            child_val_grid = xp.reshape(child_val, (N, 9))
            valid_grid = xp.reshape(v6, (N, 9))
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
            val5 = xp.where(t5, val5, backed)
            N = c4.shape[0]
            count_X = xp.sum(c4[:, :, :, 0], axis=(1, 2))
            count_O = xp.sum(c4[:, :, :, 1], axis=(1, 2))
            turn = xp.where(
                count_X > count_O, xp.ones(c4.shape[0]), xp.zeros(c4.shape[0])
            )
            child_val = xp.where(v5 > 0, val5, xp.zeros(N * 9))
            child_val_grid = xp.reshape(child_val, (N, 9))
            valid_grid = xp.reshape(v5, (N, 9))
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
            val4 = xp.where(t4, val4, backed)
            N = c3.shape[0]
            count_X = xp.sum(c3[:, :, :, 0], axis=(1, 2))
            count_O = xp.sum(c3[:, :, :, 1], axis=(1, 2))
            turn = xp.where(
                count_X > count_O, xp.ones(c3.shape[0]), xp.zeros(c3.shape[0])
            )
            child_val = xp.where(v4 > 0, val4, xp.zeros(N * 9))
            child_val_grid = xp.reshape(child_val, (N, 9))
            valid_grid = xp.reshape(v4, (N, 9))
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
            val3 = xp.where(t3, val3, backed)
            N = c2.shape[0]
            count_X = xp.sum(c2[:, :, :, 0], axis=(1, 2))
            count_O = xp.sum(c2[:, :, :, 1], axis=(1, 2))
            turn = xp.where(
                count_X > count_O, xp.ones(c2.shape[0]), xp.zeros(c2.shape[0])
            )
            child_val = xp.where(v3 > 0, val3, xp.zeros(N * 9))
            child_val_grid = xp.reshape(child_val, (N, 9))
            valid_grid = xp.reshape(v3, (N, 9))
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
            val2 = xp.where(t2, val2, backed)
            N = c1.shape[0]
            count_X = xp.sum(c1[:, :, :, 0], axis=(1, 2))
            count_O = xp.sum(c1[:, :, :, 1], axis=(1, 2))
            turn = xp.where(
                count_X > count_O, xp.ones(c1.shape[0]), xp.zeros(c1.shape[0])
            )
            child_val = xp.where(v2 > 0, val2, xp.zeros(N * 9))
            child_val_grid = xp.reshape(child_val, (N, 9))
            valid_grid = xp.reshape(v2, (N, 9))
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
            val1 = xp.where(t1, val1, backed)
            N = S.shape[0]
            count_X = xp.sum(S[:, :, :, 0], axis=(1, 2))
            count_O = xp.sum(S[:, :, :, 1], axis=(1, 2))
            turn = xp.where(
                count_X > count_O, xp.ones(S.shape[0]), xp.zeros(S.shape[0])
            )
            child_val = xp.where(v1 > 0, val1, xp.zeros(N * 9))
            child_val_grid = xp.reshape(child_val, (N, 9))
            valid_grid = xp.reshape(v1, (N, 9))
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
            result = xp.where(t0, val0, backed)

        return result

    def check(self, param):
        super().check(param)
        if self._ref_outputs is None:
            return
        actual = to_numpy(self._output[0])
        expected = to_numpy(self._ref_outputs[0])
        assert np.allclose(actual, expected, atol=1e-6)
