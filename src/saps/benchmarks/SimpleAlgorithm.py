"""
Revised simplex method for LP.

Solves:
    minimize    c^T x
    subject to  A x = b
                x >= 0

Since we have to use minimal for-loops, only two for loops have been used:
  1. the outer pivot loop, which is inherently sequential (each pivot
     depends on the basis produced by the previous one), and
  2. a short, bounded (<= m iterations) cleanup loop at the end of Phase 1
     that evicts any leftover artificial variables sitting in the basis
     at zero level.
"""

from __future__ import annotations
import numpy as np


class LPResult:
    def __init__(self, status, x, objective, basis, iterations, message=""):
        self.status = status          
        self.x = x
        self.objective = objective
        self.basis = basis
        self.iterations = iterations
        self.message = message

    def __repr__(self):
        return (f"LPResult(status={self.status!r}, objective={self.objective}, "
                f"iterations={self.iterations})")


def _pivot(A, b, c, basis, Binv, rule="bland", tol=1e-9):
   
    m, n = A.shape

    xB = Binv @ b                                   # current basic solution

    nonbasic_mask = np.ones(n, dtype=bool)
    nonbasic_mask[basis] = False
    nonbasic = np.nonzero(nonbasic_mask)[0]

    y = Binv.T @ c[basis]                           # simplex multipliers
    s_N = c[nonbasic] - A[:, nonbasic].T @ y        # reduced costs, vectorized

    improving = s_N < -tol
    if not improving.any():
        return basis, Binv, xB, "optimal"

    if rule == "bland":
        entering = nonbasic[improving].min()        
    else:
        entering = nonbasic[np.argmin(s_N)]         

    d = Binv @ A[:, entering]                       

    positive = d > tol
    if not positive.any():
        return basis, Binv, xB, "unbounded"

    # ratio test
    ratios = np.where(positive, xB / np.where(positive, d, 1.0), np.inf)
    min_ratio = ratios.min()

    if rule == "bland":
        tied = np.isclose(ratios, min_ratio, atol=tol)
        leaving_row = np.nonzero(tied)[0][np.argmin(basis[tied])]
    else:
        leaving_row = np.argmin(ratios)

    pivot_val = d[leaving_row]
    eta = -d / pivot_val
    eta[leaving_row] = 1.0 / pivot_val
    E = np.eye(m)
    E[:, leaving_row] = eta
    new_Binv = E @ Binv

    new_basis = basis.copy()
    new_basis[leaving_row] = entering

    return new_basis, new_Binv, xB, "continue"


def _run_pivots(A, b, c, basis, Binv, rule, max_iter):
    status = "continue"
    it = 0
    while status == "continue" and it < max_iter:
        basis, Binv, xB, status = _pivot(A, b, c, basis, Binv, rule)
        it += 1
    return basis, Binv, xB, status, it


def _phase1(A, b, rule, max_iter, tol=1e-9):
    """
    Find an initial feasible basis by minimizing the sum of artificial
    variables on the augmented matrix [A | I] x_aug = b, b >= 0.
    """
    m, n = A.shape

    A_aug = np.hstack([A, np.eye(m)])
    c_aug = np.concatenate([np.zeros(n), np.ones(m)])
    basis = np.arange(n, n + m)         # artificials start basic
    Binv = np.eye(m)

    basis, Binv, xB, status, _ = _run_pivots(A_aug, b, c_aug, basis, Binv, rule, max_iter)

    phase1_obj = c_aug[basis] @ xB
    if phase1_obj > 1e-7:
        return None, None, "infeasible"

    # Evict any artificial variables still sitting in the basis (at val 0)
    for row in range(m):
        if basis[row] >= n:
            tableau_row = Binv[row] @ A_aug[:, :n]      # this row, original vars only
            candidates = np.nonzero(np.abs(tableau_row) > tol)[0]
            if candidates.size == 0:
                continue  # redundant constraint row; leave artificial at 0
            entering = candidates[0]
            d_col = Binv @ A_aug[:, entering]
            pivot_val = d_col[row]
            eta = -d_col / pivot_val
            eta[row] = 1.0 / pivot_val
            E = np.eye(m)
            E[:, row] = eta
            Binv = E @ Binv
            basis[row] = entering

    return basis, Binv, "ok"


def revised_simplex(A, b, c, rule="bland", max_iter=10_000):
    """
    Solve  minimize c^T x  s.t.  A x = b, x >= 0  via the revised simplex
    method, running Phase 1 automatically to find a starting basis.

    A : (m, n) array
    b : (m,) array   
    c : (n,) array
    """
    A = np.asarray(A, dtype=float).copy()
    b = np.asarray(b, dtype=float).copy()
    c = np.asarray(c, dtype=float)
    m, n = A.shape
  
    flip = b < 0
    A[flip] *= -1
    b[flip] *= -1

    basis, Binv, p1status = _phase1(A, b, rule, max_iter)
    if p1status == "infeasible":
        return LPResult("infeasible", None, None, None, 0,
                         message="No feasible point satisfies A x = b, x >= 0.")

    basis, Binv, xB, status, iters = _run_pivots(A, b, c, basis, Binv, rule, max_iter)

    if status == "continue":
        return LPResult("iteration_limit", None, None, basis, iters,
                         message="Max iterations reached without convergence.")

    x = np.zeros(n)
    x[basis] = xB
    obj = c @ x if status == "optimal" else None

    return LPResult(status, x if status == "optimal" else None, obj, basis, iters)


def solve_lp(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, rule="bland", max_iter=10_000):
    c = np.asarray(c, dtype=float)
    n = c.shape[0]

    blocks_A = []
    blocks_b = []

    if A_ub is not None:
        A_ub = np.asarray(A_ub, dtype=float)
        b_ub = np.asarray(b_ub, dtype=float)
        n_slack = A_ub.shape[0]
        A_ub_std = np.hstack([A_ub, np.eye(n_slack)])
        blocks_A.append(A_ub_std)
        blocks_b.append(b_ub)
        pad_cols = n_slack
    else:
        pad_cols = 0

    if A_eq is not None:
        A_eq = np.asarray(A_eq, dtype=float)
        b_eq = np.asarray(b_eq, dtype=float)
        A_eq_std = np.hstack([A_eq, np.zeros((A_eq.shape[0], pad_cols))])
        blocks_A.append(A_eq_std)
        blocks_b.append(b_eq)

    if not blocks_A:
        raise ValueError("Provide at least one of (A_ub, b_ub) or (A_eq, b_eq).")

    width = max(block.shape[1] for block in blocks_A)
    blocks_A = [np.hstack([blk, np.zeros((blk.shape[0], width - blk.shape[1]))])
                for blk in blocks_A]

    A_std = np.vstack(blocks_A)
    b_std = np.concatenate(blocks_b)
    c_std = np.concatenate([c, np.zeros(width - n)])

    result = revised_simplex(A_std, b_std, c_std, rule=rule, max_iter=max_iter)
    if result.x is not None:
        result.x = result.x[:n]   # drop slack columns from the solution
    return result

# I  have added an example below to show that the algorithm works!
if __name__ == "__main__":
    #   maximize 3x + 5y
    #   subject to x <= 4, 2y <= 12, 3x + 2y <= 18, x, y >= 0
    # Sol: x=2, y=6, max value = 36
    c = [-3, -5]  #  maximize 3x + 5y is the same as minimize -3x - 5y 
    A_ub = [[1, 0], [0, 2], [3, 2]]
    b_ub = [4, 12, 18]

    result = solve_lp(c, A_ub=A_ub, b_ub=b_ub, rule="bland")
    print(result)
    print("x =", result.x)
    print("max 3x+5y =", -result.objective)
