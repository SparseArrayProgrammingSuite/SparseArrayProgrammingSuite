"""
Revised simplex method for linear programming.

Solves standard-form linear programs:

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

from binsparse import BinsparseTensor
from binsparse.conversions import from_numpy, to_numpy, to_scipy

from saps.benchmark import (
    Author,
    Benchmark,
    Contributor,
    DataInstance,
    Dataset,
    Generator,
    Ref,
)
from saps.benchmarks.suitesparse import (
    _LPNETLIB_PROBLEMS,
    SuiteSparseDataset,
    fetch_suitesparse_matrix,
)

_STATUS_OPTIMAL = 0
_STATUS_INFEASIBLE = 1
_STATUS_UNBOUNDED = 2
_STATUS_ITERATION_LIMIT = 3
# Added this to show what each of them mean.
_STATUS_NAMES = {
    _STATUS_OPTIMAL: "optimal",
    _STATUS_INFEASIBLE: "infeasible",
    _STATUS_UNBOUNDED: "unbounded",
    _STATUS_ITERATION_LIMIT: "iteration limit",
}


def _unit_vector(xp, size, index):
    positions = xp.arange(size)
    return xp.where(positions == index, 1.0, 0.0)


def _set_at(xp, arr, index, value):
    positions = xp.arange(arr.shape[0])
    return xp.where(positions == index, xp.asarray(value), arr)


def _first_true_index(xp, mask):
    size = mask.shape[0]
    positions = xp.arange(size)
    return int(xp.min(xp.where(mask, positions, size)))


def _onehot_rows(xp, basis, width):
    idx = xp.arange(width)
    return xp.astype(idx[None, :] == basis[:, None], xp.float64)


# Keeping the basis inverse as a product of elementary matrices, one per pivot, is
# the product form of Dantzig and Orchard-Hays (1954): each iteration multiplies the
# previous inverse by a single elementary matrix rather than inverting again.
# INSPIRATION: same rank-one Binv update as Simplex.py lines 128-132, except theirs
# reads Binv[:, l] where a unit vector belongs, so it only holds on the first pivot.
def _eta_matrix(xp, m, d, leaving_row):
    e_col = _unit_vector(xp, m, leaving_row)
    pivot_val = d[leaving_row]
    eta = xp.where(xp.arange(m) == leaving_row, 1.0 / pivot_val, -d / pivot_val)
    diff = eta - e_col
    return xp.eye(m) + xp.reshape(diff, (m, 1)) @ xp.reshape(e_col, (1, m))


def _pivot(xp, A, b, c, basis, Binv, tol=1e-9):
    m = A.shape[0]
    width = A.shape[1]

    xB = Binv @ b  # Inspiration: one matrix–vector
    # product, Binv · b, giving the values of the basic variables. Line 47

    onehot = _onehot_rows(xp, basis, width)
    basic_mask = xp.any(onehot > 0, axis=0)
    # INSPIRATION: dual vector and reduced costs as in Simplex.py 59-69. Theirs walks
    # the columns one scalar at a time; this prices all of them in one expression.
    c_basis = onehot @ c
    y = Binv.T @ c_basis
    s = xp.where(basic_mask, xp.inf, c - A.T @ y)

    # INSPIRATION: stop when no reduced cost is negative, as in Simplex.py 70-74.
    # Theirs compares against an exact 0 and enters on the first negative column;
    # this enters on the most negative one, which is Dantzig's rule
    improving = s < -tol
    if not bool(xp.any(improving)):
        return basis, Binv, xB, "optimal"

    entering = int(xp.argmin(xp.where(improving, s, xp.inf)))

    # INSPIRATION: direction Binv @ A[:, j] and the unbounded test from Simplex.py
    # 84-93. Theirs tests u[i] >= 0, which misses a column that is all zeros.
    d = Binv @ A[:, entering]
    positive = d > tol
    if not bool(xp.any(positive)):
        return basis, Binv, xB, "unbounded"

    # INSPIRATION: minimum-ratio test from Simplex.py 97-105, with xp.inf standing in
    # for their 1e9+7 sentinel. Note, I added the tie-break.
    ratios = xp.where(positive, xB / xp.where(positive, d, 1.0), xp.inf)
    min_ratio = xp.min(ratios)

    # Among the rows that tie for the smallest ratio, leave on the one with the
    # largest |d|. _eta_matrix divides by that entry, so a near-zero one
    # multiplies whatever rounding error Binv already carries by 1/d, and since
    # Binv is only ever updated the error never washes back out. Bartels and Golub
    # (1969) make the same point about carrying an inverse instead of refactorizing,
    # and take the other way out of it, which is to keep an LU factorization.
    tied = xp.abs(ratios - min_ratio) <= tol
    leaving_row = int(xp.argmax(xp.where(tied, xp.abs(d), -1.0)))

    new_Binv = _eta_matrix(xp, m, d, leaving_row) @ Binv
    # INSPIRATION: swapping one basis index, which Simplex.py 135 does as C[l] = j.
    # _set_at returns a new array instead, so the caller's basis is never mutated.
    new_basis = _set_at(xp, basis, leaving_row, entering)

    return new_basis, new_Binv, xB, "continue"


def _run_pivots(xp, A, b, c, basis, Binv, max_iter):
    status = "continue"
    it = 0
    xB = Binv @ b
    # INSPIRATION: the pivot loop of Simplex.py 52. Bounded by max_iter here, and the
    # body returns a status rather than breaking, so both phases can reuse it.
    while status == "continue" and it < max_iter:
        basis, Binv, xB, status = _pivot(xp, A, b, c, basis, Binv)
        it += 1
    return basis, Binv, xB, status, it


def _phase1(xp, A, b, max_iter, tol=1e-9):
    m, n = A.shape

    # INSPIRATION: identity block, zeros-then-ones cost vector and starting basis as in
    # Simplex.py 222-223, 182-184 and 214, where those columns are real error terms.
    A_aug = xp.concat([A, xp.eye(m)], axis=1)
    c_aug = xp.concat([xp.zeros((n,)), xp.ones((m,))])
    basis = xp.arange(n, n + m)
    Binv = xp.eye(m)

    basis, Binv, xB, status, _ = _run_pivots(xp, A_aug, b, c_aug, basis, Binv, max_iter)
    if status == "continue":
        # Phase 1 ran out of iterations, so nothing has been learned about
        # feasibility yet. Falling through to the test below would report a
        # budget that was too small as a property of the problem.
        return None, None, "iteration_limit"

    onehot_basis = _onehot_rows(xp, basis, n + m)
    phase1_obj = xp.sum((onehot_basis @ c_aug) * xB)
    if float(phase1_obj) > 1e-7:
        return None, None, "infeasible"

    # Bounded cleanup loop (<= m iterations): evict any artificial variable
    # still sitting in the basis at a zero level, replacing it with an
    # original variable that has a nonzero coefficient in that row.
    for row in range(m):
        if int(basis[row]) >= n:
            tableau_row = Binv[row] @ A_aug[:, :n]
            mask = xp.abs(tableau_row) > tol
            if not bool(xp.any(mask)):
                continue
            entering = _first_true_index(xp, mask)

            d_col = Binv @ A_aug[:, entering]
            Binv = _eta_matrix(xp, m, d_col, row) @ Binv
            basis = _set_at(xp, basis, row, entering)

    return basis, Binv, "ok"


def _solve_standard_form(xp, A, b, c, max_iter=10_000):
    m, n = A.shape

    # INSPIRATION: forcing b >= 0 by flipping rows, like Simplex.py 171-173. Theirs
    # flips by class label; this flips wherever b is negative, for any problem.
    flip = b < 0
    sign = xp.where(flip, -1.0, 1.0)
    A = A * xp.reshape(sign, (m, 1))
    b = b * sign

    basis, Binv, p1status = _phase1(xp, A, b, max_iter)
    if p1status == "infeasible":
        return xp.zeros((n,)), _STATUS_INFEASIBLE
    if p1status == "iteration_limit":
        return xp.zeros((n,)), _STATUS_ITERATION_LIMIT

    basis, Binv, xB, status, iters = _run_pivots(xp, A, b, c, basis, Binv, max_iter)

    if status == "continue":
        return xp.zeros((n,)), _STATUS_ITERATION_LIMIT
    if status == "unbounded":
        return xp.zeros((n,)), _STATUS_UNBOUNDED

    onehot = _onehot_rows(xp, basis, n)
    x = xp.reshape(xp.reshape(xB, (1, m)) @ onehot, (n,))
    return x, _STATUS_OPTIMAL


# Data preparation helpers
def _standard_form_from_inequalities(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None):
    """Convert a (possibly mixed) inequality/equality LP into standard form
    A x = b, x >= 0 by adding slack variables. This is a data-preparation
    step run with plain NumPy while building datasets.
    """
    c = np.asarray(c, dtype=float)
    n = c.shape[0]

    blocks_A = []
    blocks_b = []

    if A_ub is not None:
        A_ub = np.asarray(A_ub, dtype=float)
        b_ub = np.asarray(b_ub, dtype=float)
        n_slack = A_ub.shape[0]
        blocks_A.append(np.hstack([A_ub, np.eye(n_slack)]))
        blocks_b.append(b_ub)
        pad_cols = n_slack
    else:
        pad_cols = 0

    if A_eq is not None:
        A_eq = np.asarray(A_eq, dtype=float)
        b_eq = np.asarray(b_eq, dtype=float)
        blocks_A.append(np.hstack([A_eq, np.zeros((A_eq.shape[0], pad_cols))]))
        blocks_b.append(b_eq)

    if not blocks_A:
        raise ValueError("Provide at least one of (A_ub, b_ub) or (A_eq, b_eq).")

    width = max(block.shape[1] for block in blocks_A)
    blocks_A = [
        np.hstack([blk, np.zeros((blk.shape[0], width - blk.shape[1]))])
        for blk in blocks_A
    ]

    A_std = np.vstack(blocks_A)
    b_std = np.concatenate(blocks_b)
    c_std = np.concatenate([c, np.zeros(width - n)])
    return A_std, b_std, c_std


def _standard_form_from_bounds(A, b, c, lo, hi):
    """Convert a bounded LP into standard form A x = b, x >= 0.

    LPnetlib states each problem as min c^T x subject to A x = b and
    lo <= x <= hi. A variable with a finite lower bound is shifted down to
    zero, a variable bounded only from above is reflected, and a free variable
    is split into a positive and a negative part. Every finite upper bound that
    survives the shift then costs one extra equality row and one slack
    variable, so the row count here can be much larger than the one the
    SuiteSparse index reports for A.

    Returns (A_std, b_std, c_std, offset), where offset is the constant the
    substitutions contribute to the objective. Like
    _standard_form_from_inequalities this is a data-preparation step run with
    plain NumPy while building datasets.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).copy()
    c = np.asarray(c, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)

    m, n = A.shape
    columns = []
    costs = []
    uppers = []
    offset = 0.0

    for j in range(n):
        column = A[:, j]
        if np.isfinite(lo[j]):
            # y = x - lo, so the column is unchanged and the bound moves to b.
            b -= column * lo[j]
            offset += c[j] * lo[j]
            columns.append(column)
            costs.append(c[j])
            uppers.append(hi[j] - lo[j])
        elif np.isfinite(hi[j]):
            # y = hi - x flips the sign of the column and of the cost.
            b -= column * hi[j]
            offset += c[j] * hi[j]
            columns.append(-column)
            costs.append(-c[j])
            uppers.append(np.inf)
        else:
            # Free variable: x = y_plus - y_minus, neither part bounded above.
            columns.extend((column, -column))
            costs.extend((c[j], -c[j]))
            uppers.extend((np.inf, np.inf))

    A_cols = np.column_stack(columns)
    costs = np.asarray(costs)
    uppers = np.asarray(uppers)

    bounded = np.flatnonzero(np.isfinite(uppers))
    n_bounded = bounded.shape[0]
    width = A_cols.shape[1]

    A_std = np.zeros((m + n_bounded, width + n_bounded))
    A_std[:m, :width] = A_cols
    bound_rows = np.arange(n_bounded)
    A_std[m + bound_rows, bounded] = 1.0
    A_std[m + bound_rows, width + bound_rows] = 1.0

    b_std = np.concatenate([b, uppers[bounded]])
    c_std = np.concatenate([costs, np.zeros(n_bounded)])
    return A_std, b_std, c_std, float(offset)


def _reference_solution(A, b, c, lo, hi):
    """Solve an LP with SciPy so the benchmark has an answer to check against.

    The reference is taken from the problem as LPnetlib states it, before
    _standard_form_from_bounds has touched it. Solving the converted form
    instead would hide a mistake in the conversion.

    Only the objective value is kept. A linear program can have many optimal
    solutions, so two correct solvers might give us diff answers.
    So, we compare the optimal value.

    Returns (status, objective) using this module's status codes. The objective
    is None whenever there is no finite value to compare, and both are None if
    SciPy could not settle the problem, so that a dataset we are unable to
    cross-check still runs.
    """
    from scipy.optimize import linprog

    bounds = list(zip(lo, hi, strict=True))
    try:
        result = linprog(c, A_eq=A, b_eq=b, bounds=bounds, method="highs")
        # chose "highs" to let it choose the most efficient
        # methods as some problems stress the system.
    except (ValueError, TypeError):
        return None, None

    if result.status == 0:
        return _STATUS_OPTIMAL, float(result.fun)
    if result.status == 2:
        return _STATUS_INFEASIBLE, None
    if result.status == 3:
        return _STATUS_UNBOUNDED, None
    return None, None


# Generator classes.


class LinearProgrammingDataset(Dataset):
    def __init__(
        self,
        name: str,
        pretty_name: str | None = None,
        description: str | None = None,
        suites: list[str] | None = None,
        A: np.ndarray | None = None,
        b: np.ndarray | None = None,
        c: np.ndarray | None = None,
        max_iter: int = 10_000,
        expected_x: np.ndarray | None = None,
        expected_status: int = _STATUS_OPTIMAL,
    ):
        self._name = name
        self._pretty_name = pretty_name or name
        self._description = description or f"Linear programming input {name}."
        self._suites = suites or []
        self.A = A
        self.b = b
        self.c = c
        self.max_iter = max_iter
        self.expected_x = expected_x
        self.expected_status = expected_status

    @property
    def name(self) -> str:
        return self._name

    @property
    def pretty_name(self) -> str:
        return self._pretty_name

    @property
    def description(self) -> str:
        return self._description

    @property
    def suites(self) -> list[str]:
        return self._suites

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"


class LinearProgrammingTestGenerator(Generator[LinearProgrammingDataset]):
    @property
    def name(self) -> str:
        return "lp_test_inputs"

    @property
    def pretty_name(self) -> str:
        return "Linear Programming Test Input Generator"

    @property
    def description(self) -> str:
        return "Small deterministic LP examples with reference outputs."

    @property
    def suites(self) -> list[str]:
        return ["test"]

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def authors(self) -> list[Contributor]:
        return []

    @property
    def references(self) -> list[Ref]:
        return []

    @property
    def ai_disclosure(self) -> str:
        return (
            "Generative AI was used to construct the generator and dataset "
            "structures. This statement was written by hand."
        )

    @property
    def motivation(self) -> str:
        return (
            "Provide small LP examples covering distinct outcome cases "
            "for benchmark correctness checks."
        )

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def datasets(self) -> list[LinearProgrammingDataset]:
        # 1. Bounded LP with a unique optimum.
        #    maximize 3x + 5y  <=>  minimize -3x - 5y
        #    s.t. x <= 4, 2y <= 12, 3x + 2y <= 18, x, y >= 0
        #    optimum: x=2, y=6, answer=-36
        A1, b1, c1 = _standard_form_from_inequalities(
            c=[-3, -5],
            A_ub=[[1, 0], [0, 2], [3, 2]],
            b_ub=[4, 12, 18],
        )

        # 2. Infeasible: x + y = -5 with x, y >= 0 can never hold since the
        #    left side is always nonnegative.
        A2, b2, c2 = _standard_form_from_inequalities(
            c=[1, 1],
            A_eq=[[1, 1]],
            b_eq=[-5],
        )

        # 3. Unbounded: minimize -x s.t. y <= 5, x, y >= 0. Nothing bounds x
        #    from above, so the objective is unbounded below.
        A3, b3, c3 = _standard_form_from_inequalities(
            c=[-1, 0],
            A_ub=[[0, 1]],
            b_ub=[5],
        )

        # 4. Degenerate tie: two identical constraints on x force a tie in
        #    the ratio test, using the leaving row tie-break.
        #    maximize x + y <=> minimize -x - y
        #    s.t. x <= 4, x <= 4, y <= 4, x, y >= 0
        #    optimum: x=4, y=4, answer=-8
        A4, b4, c4 = _standard_form_from_inequalities(
            c=[-1, -1],
            A_ub=[[1, 0], [1, 0], [0, 1]],
            b_ub=[4, 4, 4],
        )

        A5, b5, c5 = _standard_form_from_inequalities(
            c=[1],
            A_eq=[[1]],
            b_eq=[0],
        )

        return [
            LinearProgrammingDataset(
                "test_lp_bounded_unique_optimum",
                suites=["test"],
                A=A1,
                b=b1,
                c=c1,
                expected_x=np.array([2.0, 6.0, 2.0, 0.0, 0.0]),
                expected_status=_STATUS_OPTIMAL,
            ),
            LinearProgrammingDataset(
                "test_lp_infeasible",
                suites=["test"],
                A=A2,
                b=b2,
                c=c2,
                expected_x=np.zeros(A2.shape[1]),
                expected_status=_STATUS_INFEASIBLE,
            ),
            LinearProgrammingDataset(
                "test_lp_unbounded",
                suites=["test"],
                A=A3,
                b=b3,
                c=c3,
                expected_x=np.zeros(A3.shape[1]),
                expected_status=_STATUS_UNBOUNDED,
            ),
            LinearProgrammingDataset(
                "test_lp_degenerate_tie",
                suites=["test"],
                A=A4,
                b=b4,
                c=c4,
                expected_x=np.array([4.0, 4.0, 0.0, 0.0, 0.0]),
                expected_status=_STATUS_OPTIMAL,
            ),
            LinearProgrammingDataset(
                "test_lp_singleton",
                suites=["test"],
                A=A5,
                b=b5,
                c=c5,
                expected_x=np.array([0.0]),
                expected_status=_STATUS_OPTIMAL,
            ),
        ]

    def generate(self, dataset: LinearProgrammingDataset) -> DataInstance:
        if (
            dataset.A is None
            or dataset.b is None
            or dataset.c is None
            or dataset.expected_x is None
        ):
            raise ValueError("LP test datasets must define A, b, c, and expected_x.")
        return DataInstance(
            inputs=[
                from_numpy(dataset.A),
                from_numpy(dataset.b),
                from_numpy(dataset.c),
            ],
            meta={"max_iter": dataset.max_iter},
            ref_outputs=[
                from_numpy(dataset.expected_x),
                from_numpy(np.array([dataset.expected_status])),
            ],
        )


# The members of _LPNETLIB_PROBLEMS that this solver settles within its iteration
# budget, measured by running them; the rest are listed but left untagged.
_LPNETLIB_TRACTABLE = [
    "lp_adlittle",
    "lp_afiro",
    "lp_beaconfd",
    "lp_blend",
    "lp_brandy",
    "lp_e226",
    "lp_israel",
    "lp_kb2",
    "lp_lotfi",
    "lp_recipe",
    "lp_sc105",
    "lp_sc205",
    "lp_sc50a",
    "lp_sc50b",
    "lp_scagr7",
    "lp_share1b",
    "lp_share2b",
    "lp_stocfor1",
    "lpi_bgprtr",
    "lpi_box1",
    "lpi_ex72a",
    "lpi_ex73a",
    "lpi_forest6",
    "lpi_galenet",
    "lpi_itest2",
    "lpi_itest6",
    "lpi_klein1",
    "lpi_woodinfe",
]


class LPNetlibDataset(SuiteSparseDataset):
    def __init__(
        self,
        source_name: str,
        suites: list[str] | None = None,
        max_iter: int = 20_000,
        expected_status: int = _STATUS_OPTIMAL,
    ):
        super().__init__(
            source_name,
            source_name=f"LPnetlib/{source_name}",
            pretty_name=f"LPnetlib {source_name}",
            description=(
                f"Netlib linear program {source_name} from the SuiteSparse LPnetlib"
                " group, converted to standard form."
            ),
            suites=suites,
        )
        self.max_iter = max_iter
        self.expected_status = expected_status


class LPNetlibGenerator(Generator[LPNetlibDataset]):
    @property
    def name(self) -> str:
        return "lpnetlib_inputs"

    @property
    def pretty_name(self) -> str:
        return "LPnetlib Linear Program Generator"

    @property
    def description(self) -> str:
        return (
            "Linear programs from the LPnetlib group of the SuiteSparse Matrix"
            " Collection, converted from the bounded form the collection stores"
            " into the standard form the revised simplex method solves. The"
            " standard-form matrix is materialized densely, since the solver"
            " maintains a dense basis inverse."
        )

    @property
    def suites(self) -> list[str]:
        return []

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def authors(self) -> list[Contributor]:
        return [Contributor("Kabir Sahni", "ksahni30@gatech.edu")]

    @property
    def references(self) -> list[Ref]:
        return [
            Ref(
                title="The university of Florida sparse matrix collection",
                authors=[
                    Author("Timothy A. Davis"),
                    Author("Yifan Hu"),
                ],
                journal="ACM Transactions on Mathematical Software",
                publisher="Association for Computing Machinery (ACM)",
                volume="38",
                number="1",
                pages="1-25",
                year=2011,
                url="https://doi.org/10.1145/2049662.2049663",
                doi="10.1145/2049662.2049663",
            ),
            Ref(
                title=(
                    "Electronic mail distribution of linear programming test problems"
                ),
                authors=[Author("David M. Gay")],
                journal="Mathematical Programming Society COAL Newsletter",
                volume="13",
                pages="10-12",
                year=1985,
            ),
        ]

    @property
    def ai_disclosure(self) -> str:
        return (
            "No generative AI was used to write the benchmark function itself."
            " Generative AI was used to construct this generator, its dataset list,"
            " and the standard-form conversion helper."
        )

    @property
    def motivation(self) -> str:
        return (
            "Netlib is the reference test set that linear programming solvers report"
            " against, so it measures the simplex benchmark on the problems the field"
            " actually uses rather than on generated ones. The problems are naturally"
            " sparse and span three orders of magnitude in size, and the group"
            " includes a set of deliberately infeasible problems that exercise the"
            " Phase 1 termination path on real data."
        )

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def datasets(self) -> list[LPNetlibDataset]:
        return [
            LPNetlibDataset(
                name,
                suites=["standard", "trace"] if name in _LPNETLIB_TRACTABLE else [],
                expected_status=(
                    _STATUS_INFEASIBLE if name.startswith("lpi_") else _STATUS_OPTIMAL
                ),
            )
            for name in _LPNETLIB_PROBLEMS
        ]

    def generate(self, dataset: LPNetlibDataset) -> DataInstance:
        raw = fetch_suitesparse_matrix(dataset.source_name)
        A_bin, b_bin, c_bin, lo_bin, hi_bin = raw.inputs
        try:
            A_dense = to_scipy(A_bin).toarray()
        except TypeError:
            A_dense = to_numpy(A_bin)
        b, c, lo, hi = (to_numpy(vector) for vector in (b_bin, c_bin, lo_bin, hi_bin))
        meta = raw.meta
        A_std, b_std, c_std, offset = _standard_form_from_bounds(A_dense, b, c, lo, hi)
        reference_status, reference_objective = _reference_solution(
            A_dense, b, c, lo, hi
        )

        ref_outputs = None
        if dataset.expected_status != _STATUS_OPTIMAL:
            # The lpi_ problems are the infeasible members of the collection, so
            # the status is known ahead of the run even though no solution is.
            ref_outputs = [
                from_numpy(np.zeros(A_std.shape[1])),
                from_numpy(np.array([dataset.expected_status])),
            ]

        return DataInstance(
            inputs=[
                from_numpy(A_std),
                from_numpy(b_std),
                from_numpy(c_std),
            ],
            meta={
                **meta,
                "max_iter": dataset.max_iter,
                "standard_form_shape": A_std.shape,
                "objective_offset": offset,
            },
            ref_outputs=ref_outputs,
            ref_meta={
                "check_solution": True,
                "feasibility_tol": 1e-6,
                "objective_tol": 1e-6,
                "expect_feasible": dataset.expected_status == _STATUS_OPTIMAL,
                "reference_status": reference_status,
                "reference_objective": reference_objective,
            },
        )


class LinearProgrammingBenchmark(Benchmark):
    @property
    def name(self):
        return "lp_simplex"

    @property
    def pretty_name(self):
        return "Revised Simplex Method for Linear Programming"

    @property
    def description(self):
        return (
            "The revised simplex method solves standard-form linear programs"
            " min c^T x s.t. A x = b, x >= 0 by moving between basic feasible"
            " solutions, maintaining and updating a basis inverse rather than"
            " a full tableau. It is a foundational algorithm in operations"
            " research and numerical optimization."
        )

    @property
    def suites(self) -> list[str]:
        return []

    @property
    def concepts(self) -> str:
        return (
            "<ccs2012>"
            "<concept>"
            "<concept_id>10002950.10003705.10011686</concept_id>"
            "<concept_desc>Mathematics of computing~Mathematical software "
            "performance</concept_desc>"
            "<concept_significance>500</concept_significance>"
            "</concept>"
            "<concept>"
            "<concept_id>10002950.10003714.10003715</concept_id>"
            "<concept_desc>Mathematics of computing~Numerical analysis"
            "</concept_desc>"
            "<concept_significance>500</concept_significance>"
            "</concept>"
            "<concept>"
            "<concept_id>10002950.10003714.10003716.10011138.10010041"
            "</concept_id>"
            "<concept_desc>Mathematics of computing~Linear programming"
            "</concept_desc>"
            "<concept_significance>500</concept_significance>"
            "</concept>"
            "</ccs2012>"
        )

    @property
    def authors(self) -> list[Contributor]:
        return [
            Contributor("Kabir Sahni", "ksahni30@gatech.edu"),
        ]

    @property
    def references(self):
        return [
            Ref(
                title="Revised simplex method",
                authors=[Author("Wikipedia contributors")],
                journal="Wikipedia, The Free Encyclopedia",
                city="San Francisco",
                year=2024,
            ),
            Ref(
                title="Simplex_Method",
                authors=[Author("sayhitosandy")],
                journal="GitHub repository",
                city="",
                year=2020,
            ),
            Ref(
                title="The product form for the inverse in the simplex method",
                authors=[
                    Author("George B. Dantzig"),
                    Author("Wm. Orchard-Hays"),
                ],
                journal="Mathematical Tables and Other Aids to Computation",
                volume="8",
                number="46",
                pages="64-67",
                year=1954,
                url="https://doi.org/10.1090/S0025-5718-1954-0061469-8",
                doi="10.1090/S0025-5718-1954-0061469-8",
            ),
            Ref(
                title=(
                    "The simplex method of linear programming using LU decomposition"
                ),
                authors=[
                    Author("Richard H. Bartels"),
                    Author("Gene H. Golub"),
                ],
                journal="Communications of the ACM",
                publisher="Association for Computing Machinery (ACM)",
                volume="12",
                number="5",
                pages="266-268",
                year=1969,
                url="https://doi.org/10.1145/362946.362974",
                doi="10.1145/362946.362974",
            ),
        ]

    @property
    def ai_disclosure(self):
        return (
            "Generative AI might have been used to construct tests and to "
            "ensure the file follows CONTRIBUTING.md. This statement was "
            "written by hand."
        )

    @property
    def motivation(self):
        return (
            "Linear programming is a core building block for iterative"
            " numerical solvers, resource allocation, and scheduling problems"
            " throughout scientific computing. The revised simplex method is"
            " a good stress test for array-based frameworks because, aside"
            " from the outer pivot loop (each pivot depends on the basis"
            " produced by the previous one) and a short bounded cleanup step"
            " after Phase 1, the pricing step, ratio test, and basis-inverse"
            " update can all be expressed as collective array operations"
            " rather than scalar loops."
        )

    @property
    def generators(self):
        return [
            LinearProgrammingTestGenerator(),
            LPNetlibGenerator(),
        ]

    def benchmark(self, xp, data: list, meta: dict):
        A, b, c = data[0], data[1], data[2]
        max_iter = meta.get("max_iter", 10_000)

        x, status_code = _solve_standard_form(xp, A, b, c, max_iter=max_iter)
        status = xp.reshape(xp.asarray(status_code), (1,))
        return [x, status]

    def _check_solution(self, param):
        """To check that the sol is optimal, we compare the solution vector
        itself is never compared against SciPy's. We don't compare the column vector
        that gives us this solution because an LP's optimal aren't unique and
        two correct solvers often return
        different ones, so we would get false negatives.
        """
        status = int(to_numpy(self._output[1])[0])
        if self._ref_meta.get("expect_feasible"):
            assert status != _STATUS_INFEASIBLE, (
                f"LP {param.dataset.name} is one of the feasible Netlib problems"
                " but the solver reported it infeasible"
            )

        reference_status = self._ref_meta.get("reference_status")
        if reference_status is not None and status != _STATUS_ITERATION_LIMIT:
            assert status == reference_status, (
                f"LP {param.dataset.name} came out as"
                f" {_STATUS_NAMES[status]} but SciPy reports"
                f" {_STATUS_NAMES[reference_status]}"
            )

        if status != _STATUS_OPTIMAL:
            return

        A_bin, b_bin, c_bin = self._input
        A = to_numpy(A_bin)
        b = to_numpy(b_bin)
        c = to_numpy(c_bin)
        x = to_numpy(self._output[0])

        feasibility_tol = self._ref_meta.get("feasibility_tol", 1e-6)
        residual = float(np.max(np.abs(A @ x - b)))
        assert residual <= feasibility_tol, (
            f"LP solution for {param.dataset.name} is reported optimal but violates"
            f" A x = b by {residual}"
        )
        smallest = float(np.min(x))
        assert smallest >= -feasibility_tol, (
            f"LP solution for {param.dataset.name} is reported optimal but has a"
            f" negative entry {smallest}"
        )

        reference_objective = self._ref_meta.get("reference_objective")
        if reference_objective is None:
            return

        # The solver works on the converted problem, so the constant that the
        # bound substitutions moved out of the objective has to go back in
        # before the two values describe the same thing.
        objective = float(c @ x) + self._meta["objective_offset"]
        difference = abs(objective - reference_objective)
        objective_tol = self._ref_meta.get("objective_tol", 1e-6)
        allowed = objective_tol * max(1.0, abs(reference_objective))
        assert difference <= allowed, (
            f"LP solution for {param.dataset.name} is feasible but its objective"
            f" {objective} is not the optimal value {reference_objective}"
        )

    def check(self, param):
        for item in self._output:
            assert isinstance(item, BinsparseTensor), (
                "Output must be in binsparse format"
            )
        if self._ref_meta and self._ref_meta.get("check_solution"):
            self._check_solution(param)
        if self._ref_outputs is None:
            return

        x_actual = to_numpy(self._output[0])
        status_actual = int(to_numpy(self._output[1])[0])

        x_expected = to_numpy(self._ref_outputs[0])
        status_expected = int(to_numpy(self._ref_outputs[1])[0])

        assert status_actual == status_expected, (
            f"LP status mismatch for {param.dataset.name}:"
            f" expected {status_expected}, got {status_actual}"
        )
        if status_expected == _STATUS_OPTIMAL:
            assert x_actual.shape == x_expected.shape, (
                f"LP solution shape mismatch for {param.dataset.name}"
            )
            assert np.allclose(x_actual, x_expected, atol=1e-6), (
                f"LP solution mismatch for {param.dataset.name}:"
                f" expected {x_expected}, got {x_actual}"
            )
