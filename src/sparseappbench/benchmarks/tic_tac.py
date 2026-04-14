"""
Name: Minmax Tic-Tac Toe

Co-Authors: Aarav Jogekar & Prof. Ahrens

What does this code do: Implement a fully  tensorized, non recursive minimax
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
    # (N,2) outputs that indicates who has one what board
    # [:, 0] if X got it, [:, 1] O won won
    W_exp = xp.reshape(W, (1, 8, 3, 3, 1))
    S_exp = xp.reshape(S, (S.shape[0], 1, 3, 3, 2))
    scores = xp.sum(W_exp * S_exp, axis=(2, 3))
    return xp.any(scores >= 3, axis=1)


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
    empty_flat = xp.reshape(1 - xp.sum(S, axis=3), (N, 9))
    turn = whose_turn(xp, S)

    # one hot encodings of possible cells
    pos_exp = xp.reshape(xp.eye(9), (1, 9, 3, 3))
    turn_exp = xp.reshape(turn, (N, 1, 1, 1))

    ch0 = xp.reshape(pos_exp * (1 - turn_exp), (N, 9, 3, 3, 1))
    ch1 = xp.reshape(pos_exp * turn_exp, (N, 9, 3, 3, 1))
    delta = xp.concat([ch0, ch1], axis=4)

    valid_exp = xp.reshape(empty_flat, (N, 9, 1, 1, 1))
    children = xp.reshape(S, (N, 1, 3, 3, 2)) + delta * valid_exp

    return xp.reshape(children, (N * 9, 3, 3, 2)), xp.reshape(empty_flat, (N * 9,))


def minimax(xp, S_initial, W):

    c1, v1 = generate_child(xp, S_initial, W)
    c2, v2 = generate_child(xp, c1, W)
    c3, v3 = generate_child(xp, c2, W)
    c4, v4 = generate_child(xp, c3, W)
    c5, v5 = generate_child(xp, c4, W)
    c6, v6 = generate_child(xp, c5, W)
    c7, v7 = generate_child(xp, c6, W)
    c8, v8 = generate_child(xp, c7, W)
    c9, v9 = generate_child(xp, c8, W)

    t0, val0 = is_terminal(xp, S_initial, W)
    t1, val1 = is_terminal(xp, c1, W)
    t2, val2 = is_terminal(xp, c2, W)
    t3, val3 = is_terminal(xp, c3, W)
    t4, val4 = is_terminal(xp, c4, W)
    t5, val5 = is_terminal(xp, c5, W)
    t6, val6 = is_terminal(xp, c6, W)
    t7, val7 = is_terminal(xp, c7, W)
    t8, val8 = is_terminal(xp, c8, W)
    t9, val9 = is_terminal(xp, c9, W)

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

    val9 = xp.where(t9, val9, xp.zeros(c9.shape[0]))
    val8 = backup(xp, c8, val9, v9, t8, val8)
    val7 = backup(xp, c7, val8, v8, t7, val7)
    val6 = backup(xp, c6, val7, v7, t6, val6)
    val5 = backup(xp, c5, val6, v6, t5, val5)
    val4 = backup(xp, c4, val5, v5, t4, val4)
    val3 = backup(xp, c3, val4, v4, t3, val3)
    val2 = backup(xp, c2, val3, v3, t2, val2)
    val1 = backup(xp, c1, val2, v2, t1, val1)
    return backup(xp, S_initial, val1, v1, t0, val0)


def minimax_depth2(xp, S_initial, W):
    c1, v1 = generate_child(xp, S_initial, W)
    c2, v2 = generate_child(xp, c1, W)

    t0, val0 = is_terminal(xp, S_initial, W)
    t1, val1 = is_terminal(xp, c1, W)
    t2, val2 = is_terminal(xp, c2, W)

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

    val5 = xp.where(t5, val5, xp.zeros(c5.shape[0]))
    val4 = backup(xp, c4, val5, v5, t4, val4)
    val3 = backup(xp, c3, val4, v4, t3, val3)
    val2 = backup(xp, c2, val3, v3, t2, val2)
    val1 = backup(xp, c1, val2, v2, t1, val1)
    return backup(xp, S_initial, val1, v1, t0, val0)


def tictactoe(xp, board_binsparse):
    # takes input reshapes it and calls minimax and then returns the result
    S = xp.from_binsparse(board_binsparse)
    W = build_win_masks(xp)
    values = minimax(xp, S, W)
    return xp.to_binsparse(values)
