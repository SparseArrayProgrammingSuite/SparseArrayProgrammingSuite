"""
Name: Minmax Tic-Tac Toe

Co-Authors: Aarav Jogekar & Prof. Ahrens

What does this code do: Implement a fully  tensorized, non recursive minimax
search over a tic-tac-toe game. Game states are represented as tensors.
Game state is represented as S[n, i, j, p] of shape (N, 3, 3, 2) where n
indexes boards, i,j are board positions and p is the player channel.


Statement on the use of Generative AI: No generative AI was used to construct
the benchmark function itself.
Email: ajoglekar32@gatech.edu
"""


def build_win_masks(xp):
    return xp.array(
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
    # (N,2) outputs that indicates who has one what board
    # [:, 0] if X got it, [:, 1] O won won
    W_exp = xp.reshape(W, (1, 8, 3, 3, 1))
    S_exp = xp.reshape(S, (S.shape[0], 1, 3, 3, 2))
    scores = xp.sum(W_exp * S_exp, axis=(2, 3))
    return xp.any(scores >= 3.0, axis=1)


def check_full(xp, S):
    # Just says if a board is full good for status check
    return xp.all(xp.sum(S, axis=3) >= 1, axis=(1, 2))


def is_terminal(xp, S, W):
    # Returns what boards are done and what the numeric outcome is for those
    N = S.shape[0]
    winner = check_wins(xp, S, W)
    x_wins = winner[:, 0]
    o_wins = winner[:, 1]
    full = check_full(xp, S)
    terminal = x_wins | o_wins | full
    value = xp.where(x_wins, xp.ones(N), xp.where(o_wins, -xp.ones(N), xp.zeros(N)))
    return terminal, value


def whose_turn(xp, S):
    # Counts board pieces and figures out turn
    count_X = xp.sum(S[:, :, :, 0], axis=(1, 2))
    count_O = xp.sum(S[:, :, :, 1], axis=(1, 2))
    return xp.where(count_X > count_O, xp.ones(S.shape[0]), xp.zeros(S.shape[0]))


def generate_child(xp, S, W):
    # Generates all legal next board states for every board in the batch.
    # Builds the move tensor m and computes S' = S + m

    # After this 1 means empty.
    N = S.shape[0]
    empty_flat = xp.reshape(1.0 - xp.sum(S, axis=3), (N, 9))
    turn = whose_turn(xp, S)

    # one hot encodings of possible cells
    pos_exp = xp.reshape(xp.eye(9), (1, 9, 3, 3))
    turn_exp = xp.reshape(turn, (N, 1, 1, 1))

    ch0 = xp.reshape(pos_exp * (1.0 - turn_exp), (N, 9, 3, 3, 1))
    ch1 = xp.reshape(pos_exp * turn_exp, (N, 9, 3, 3, 1))
    delta = xp.concat([ch0, ch1], axis=4)

    valid_exp = xp.reshape(empty_flat, (N, 9, 1, 1, 1))
    children = xp.reshape(S, (N, 1, 3, 3, 2)) + delta * valid_exp

    return xp.reshape(children, (N * 9, 3, 3, 2)), xp.reshape(empty_flat, (N * 9,))


def minimax(xp, S_initial, W, max_depth=3):
    pass


def tictactoe(xp, board_binsaprse):
    # takes input reshapes it and calls minimax and then returns the result
    S = xp.from_binsparse(board_binsaprse)
    W = build_win_masks(xp)
    values = minimax(xp, S, W, max_depth=9)
    return xp.to_binsparse(values)
