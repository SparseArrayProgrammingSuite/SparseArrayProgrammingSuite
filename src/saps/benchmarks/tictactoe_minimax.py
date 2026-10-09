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
            "The original benchmark function was written by hand. Generative AI "
            "assisted with refactoring the depth traversal into loops and adding "
            "regression tests."
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

        # Expand each level, retaining parent boards and legal-move masks.
        boards = [S]
        valid_moves = []
        level = 0
        while level < depth:
            board = boards[level]
            N = board.shape[0]
            empty_flat = xp.reshape(1 - xp.sum(board, axis=3), (N, 9))
            count_X = xp.sum(board[:, :, :, 0], axis=(1, 2))
            count_O = xp.sum(board[:, :, :, 1], axis=(1, 2))
            turn = xp.where(count_X > count_O, xp.ones(N), xp.zeros(N))
            pos_exp = xp.reshape(xp.eye(9), (1, 9, 3, 3))
            turn_exp = xp.reshape(turn, (N, 1, 1, 1))
            ch0 = xp.reshape(pos_exp * (1 - turn_exp), (N, 9, 3, 3, 1))
            ch1 = xp.reshape(pos_exp * turn_exp, (N, 9, 3, 3, 1))
            delta = xp.concat([ch0, ch1], axis=4)
            valid_exp = xp.reshape(empty_flat, (N, 9, 1, 1, 1))
            children = (
                xp.reshape(board, (N, 1, 3, 3, 2)) + delta * valid_exp
            ) * valid_exp
            # Invalid moves produce zeroed-out boards.
            boards.append(xp.reshape(children, (N * 9, 3, 3, 2)))
            valid_moves.append(xp.reshape(empty_flat, (N * 9,)))
            level += 1

        # Evaluate terminal positions at every level of the tree.
        terminals = []
        values = []
        level = 0
        while level <= depth:
            board = boards[level]
            N = board.shape[0]
            W_exp = xp.reshape(W, (1, 8, 3, 3, 1))
            S_exp = xp.reshape(board, (N, 1, 3, 3, 2))
            scores = xp.sum(W_exp * S_exp, axis=(2, 3))
            winner = xp.any(scores >= 3, axis=1)
            x_wins = winner[:, 0]
            o_wins = winner[:, 1]
            full = xp.all(xp.sum(board, axis=3) >= 1, axis=(1, 2))
            terminal = x_wins | o_wins | full
            value = xp.where(
                x_wins, xp.ones(N), xp.where(o_wins, -xp.ones(N), xp.zeros(N))
            )
            terminals.append(terminal)
            values.append(value)
            level += 1

        # Nonterminal leaves have value zero at the search horizon.
        result = xp.where(terminals[-1], values[-1], xp.zeros(boards[-1].shape[0]))
        level = depth - 1
        while level >= 0:
            board = boards[level]
            valid = valid_moves[level]
            N = board.shape[0]
            count_X = xp.sum(board[:, :, :, 0], axis=(1, 2))
            count_O = xp.sum(board[:, :, :, 1], axis=(1, 2))
            turn = xp.where(count_X > count_O, xp.ones(N), xp.zeros(N))
            child_val = xp.where(valid > 0, result, xp.zeros(N * 9))
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
            result = xp.where(terminals[level], values[level], backed)
            level -= 1

        return result

    def check(self, param):
        super().check(param)
        if self._ref_outputs is None:
            return
        actual = to_numpy(self._output[0])
        expected = to_numpy(self._ref_outputs[0])
        assert np.allclose(actual, expected, atol=1e-6)
