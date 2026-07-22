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

from saps.benchmark import (
    Author,
    Benchmark,
    Contributor,
    DataInstance,
    Dataset,
    Generator,
    Ref,
)
from saps_framework.binsparse_format import BinsparseFormat

_STATUS_OPTIMAL = 0
_STATUS_INFEASIBLE = 1
_STATUS_UNBOUNDED = 2
_STATUS_ITERATION_LIMIT = 3


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


def _eta_matrix(xp, m, d, leaving_row):
    e_col = _unit_vector(xp, m, leaving_row)
    pivot_val = d[leaving_row]
    eta = xp.where(xp.arange(m) == leaving_row, 1.0 / pivot_val, -d / pivot_val)
    diff = eta - e_col
    return xp.eye(m) + xp.reshape(diff, (m, 1)) @ xp.reshape(e_col, (1, m))


def _pivot(xp, A, b, c, basis, Binv, rule, tol=1e-9):
    m = A.shape[0]
    width = A.shape[1]

    xB = Binv @ b

    onehot = _onehot_rows(xp, basis, width)
    basic_mask = xp.any(onehot > 0, axis=0)
    c_basis = onehot @ c
    y = Binv.T @ c_basis
    s = xp.where(basic_mask, xp.inf, c - A.T @ y)

    improving = s < -tol
    if not bool(xp.any(improving)):
        return basis, Binv, xB, "optimal"

    if rule == "bland":
        entering = _first_true_index(xp, improving)
    else:
        entering = int(xp.argmin(xp.where(improving, s, xp.inf)))

    d = Binv @ A[:, entering]
    positive = d > tol
    if not bool(xp.any(positive)):
        return basis, Binv, xB, "unbounded"

    ratios = xp.where(positive, xB / xp.where(positive, d, 1.0), xp.inf)
    min_ratio = xp.min(ratios)

    if rule == "bland":
        tied = xp.abs(ratios - min_ratio) <= tol
        candidate_basis = xp.where(tied, basis, xp.asarray(width))
        min_basis_val = xp.min(candidate_basis)
        leaving_row = _first_true_index(xp, candidate_basis == min_basis_val)
    else:
        leaving_row = int(xp.argmin(ratios))

    new_Binv = _eta_matrix(xp, m, d, leaving_row) @ Binv
    new_basis = _set_at(xp, basis, leaving_row, entering)

    return new_basis, new_Binv, xB, "continue"


def _run_pivots(xp, A, b, c, basis, Binv, rule, max_iter):
    status = "continue"
    it = 0
    xB = Binv @ b
    while status == "continue" and it < max_iter:
        basis, Binv, xB, status = _pivot(xp, A, b, c, basis, Binv, rule)
        it += 1
    return basis, Binv, xB, status, it


def _phase1(xp, A, b, rule, max_iter, tol=1e-9):
    m, n = A.shape

    A_aug = xp.concat([A, xp.eye(m)], axis=1)
    c_aug = xp.concat([xp.zeros((n,)), xp.ones((m,))])
    basis = xp.arange(n, n + m)
    Binv = xp.eye(m)

    basis, Binv, xB, _status, _ = _run_pivots(
        xp, A_aug, b, c_aug, basis, Binv, rule, max_iter
    )

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


def _solve_standard_form(xp, A, b, c, rule="bland", max_iter=10_000):
    m, n = A.shape

    flip = b < 0
    sign = xp.where(flip, -1.0, 1.0)
    A = A * xp.reshape(sign, (m, 1))
    b = b * sign

    basis, Binv, p1status = _phase1(xp, A, b, rule, max_iter)
    if p1status == "infeasible":
        return xp.zeros((n,)), _STATUS_INFEASIBLE

    basis, Binv, xB, status, iters = _run_pivots(
        xp, A, b, c, basis, Binv, rule, max_iter
    )

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


def _random_transportation_problem(rng, n_sources, n_sinks):
    demand = rng.integers(5, 25, size=n_sinks).astype(float)
    total = demand.sum()
    cuts = np.sort(rng.uniform(0, total, size=n_sources - 1))
    supply = np.diff(np.concatenate([[0.0], cuts, [total]]))

    cost = rng.integers(1, 20, size=(n_sources, n_sinks)).astype(float)

    n_vars = n_sources * n_sinks
    A = np.zeros((n_sources + n_sinks, n_vars))
    for i in range(n_sources):
        A[i, i * n_sinks : (i + 1) * n_sinks] = 1.0
    for j in range(n_sinks):
        A[n_sources + j, j::n_sinks] = 1.0
    b = np.concatenate([supply, demand])
    c = cost.reshape(-1)
    return A, b, c


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
        rule: str = "bland",
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
        self.rule = rule
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
        return ["test", "trace"]

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
        #    the ratio test, exercising Bland's rule tie-breaking.
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
                suites=["test", "trace"],
                A=A1,
                b=b1,
                c=c1,
                expected_x=np.array([2.0, 6.0, 2.0, 0.0, 0.0]),
                expected_status=_STATUS_OPTIMAL,
            ),
            LinearProgrammingDataset(
                "test_lp_infeasible",
                suites=["test", "trace"],
                A=A2,
                b=b2,
                c=c2,
                expected_x=np.zeros(A2.shape[1]),
                expected_status=_STATUS_INFEASIBLE,
            ),
            LinearProgrammingDataset(
                "test_lp_unbounded",
                suites=["test", "trace"],
                A=A3,
                b=b3,
                c=c3,
                expected_x=np.zeros(A3.shape[1]),
                expected_status=_STATUS_UNBOUNDED,
            ),
            LinearProgrammingDataset(
                "test_lp_degenerate_tie",
                suites=["test", "trace"],
                A=A4,
                b=b4,
                c=c4,
                expected_x=np.array([4.0, 4.0, 0.0, 0.0, 0.0]),
                expected_status=_STATUS_OPTIMAL,
            ),
            LinearProgrammingDataset(
                "test_lp_singleton",
                suites=["test", "trace"],
                A=A5,
                b=b5,
                c=c5,
                expected_x=np.array([0.0]),
                expected_status=_STATUS_OPTIMAL,
            ),
        ]

    def generate(self, dataset: LinearProgrammingDataset) -> DataInstance:
        if dataset.A is None or dataset.b is None or dataset.c is None:
            raise ValueError("LP test datasets must define A, b, and c.")
        return DataInstance(
            inputs=[
                BinsparseFormat.from_numpy(dataset.A),
                BinsparseFormat.from_numpy(dataset.b),
                BinsparseFormat.from_numpy(dataset.c),
            ],
            meta={"rule": dataset.rule, "max_iter": dataset.max_iter},
            ref_outputs=[
                BinsparseFormat.from_numpy(dataset.expected_x),
                BinsparseFormat.from_numpy(np.array([dataset.expected_status])),
            ],
        )


class LinearProgrammingGenerator(Generator[LinearProgrammingDataset]):
    @property
    def name(self) -> str:
        return "lp_transportation_inputs"

    @property
    def pretty_name(self) -> str:
        return "Transportation Problem Input Generator"

    @property
    def description(self) -> str:
        return (
            "Randomly generated, balanced transportation problems (classic "
            "supply/demand LPs), already in standard equality-constraint form."
        )

    @property
    def suites(self) -> list[str]:
        return ["standard"]

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
            "The transportation problem is a standard operations-research "
            "LP application (matching supply to demand at minimum shipping "
            "cost) and gives a realistic, naturally sparse-friendly stress "
            "test at a range of sizes."
        )

    @property
    def datasets(self) -> list[LinearProgrammingDataset]:
        sizes = [(3, 3), (5, 5), (8, 6)]
        datasets = []
        for n_sources, n_sinks in sizes:
            datasets.append(
                LinearProgrammingDataset(
                    name=f"transportation_{n_sources}x{n_sinks}",
                    pretty_name=f"Transportation problem ({n_sources}x{n_sinks})",
                    description=(
                        f"Randomly generated balanced transportation problem "
                        f"with {n_sources} sources and {n_sinks} sinks (fixed "
                        f"seed {n_sources * 1000 + n_sinks})."
                    ),
                    suites=["dynamic"],
                    rule="bland",
                    max_iter=10_000,
                )
            )
        return datasets

    def generate(self, dataset: LinearProgrammingDataset) -> DataInstance:
        if not dataset.name.startswith("transportation_"):
            raise ValueError(f"Unsupported LP dataset: {dataset.name}")
        n_sources, n_sinks = (int(v) for v in dataset.name.split("_")[1].split("x"))
        seed = n_sources * 1000 + n_sinks
        rng = np.random.default_rng(seed=seed)
        A, b, c = _random_transportation_problem(rng, n_sources, n_sinks)
        return DataInstance(
            inputs=[
                BinsparseFormat.from_numpy(A),
                BinsparseFormat.from_numpy(b),
                BinsparseFormat.from_numpy(c),
            ],
            meta={"rule": dataset.rule, "max_iter": dataset.max_iter, "seed": seed},
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
        return "<ccs2012></ccs2012>"

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
        return [LinearProgrammingGenerator(), LinearProgrammingTestGenerator()]

    def benchmark(self, xp, data: list, meta: dict):
        A, b, c = data[0], data[1], data[2]
        rule = meta.get("rule", "bland")
        max_iter = meta.get("max_iter", 10_000)

        x, status_code = _solve_standard_form(xp, A, b, c, rule=rule, max_iter=max_iter)
        status = xp.reshape(xp.asarray(status_code), (1,))
        return [x, status]

    def check(self, param):
        for item in self._output:
            assert isinstance(item, BinsparseFormat), (
                "Output must be in binsparse format"
            )
        if self._ref_outputs is None:
            return

        x_actual = self._output[0].data["values"].reshape(self._output[0].data["shape"])
        status_actual = int(self._output[1].data["values"][0])

        x_expected = (
            self._ref_outputs[0]
            .data["values"]
            .reshape(self._ref_outputs[0].data["shape"])
         )
        status_expected = int(self._ref_outputs[1].data["values"][0])

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
