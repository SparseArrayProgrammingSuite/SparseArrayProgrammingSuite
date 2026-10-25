from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from textwrap import indent
from typing import Any, TypeVar

import numpy as np
import scipy.sparse as scipy_sparse

from binsparse import BinsparseTensor
from binsparse.conversions import from_numpy, from_scipy, to_numpy, to_scipy

from saps.benchmark import (
    Benchmark,
    Contributor,
    DataInstance,
    Dataset,
    Generator,
    Ref,
)
from saps.codegen import define_function
from saps.downloaders.slicot import (
    SLICOT_BENCHMARK_PAGE_URL,
    load_slicot_problem,
    slicot_problem_metadata,
    slicot_source_url,
)


def _matrix(array: BinsparseTensor):
    """A dense array as NumPy, or a sparse matrix as SciPy CSR."""
    try:
        return to_numpy(array)
    except TypeError:
        return to_scipy(array).tocsr()


def _step_input(t):
    """A simple 5V step input starting at t=0."""
    return 5.0 if t >= 0 else 0.0


def _rc_derivatives(t, state, meta):
    """RC circuit derivatives."""
    R, C = meta["R"], meta["C"]
    tau = R * C
    Vs = _step_input(t)
    return [(Vs - state[0]) / tau]


def _rlc_derivatives(t, state, meta):
    """RLC circuit derivatives."""
    R, L, C = meta["R"], meta["L"], meta["C"]
    Vc = state[0]
    dVc = state[1]
    Vs = _step_input(t)
    d2Vc = (Vs - Vc - R * C * dVc) / (L * C)
    return (dVc, d2Vc)


def _lotka_volterra_derivatives(t, state, meta):
    """Lotka-Volterra derivatives."""
    a, b, c, d = meta["a"], meta["b"], meta["c"], meta["d"]
    x, y = state
    dxdt = a * x - b * x * y
    dydt = d * x * y - c * y
    return (dxdt, dydt)


def _limit(a, N):
    """Periodic boundary condition wrapper."""
    return a % N


def _init_brusselator_2d(n):
    """Initialize 2D Brusselator state."""
    u = [0.0] * (n * n * 2)
    for i in range(n):
        for j in range(n):
            fi = i / (n - 1) if n > 1 else 0.0
            fj = j / (n - 1) if n > 1 else 0.0
            u[(i * n + j) * 2] = float(np.real(22 * (fj * (1 - fj)) ** 1.5))
            u[(i * n + j) * 2 + 1] = float(np.real(27 * (fi * (1 - fi)) ** 1.5))
    return u


def _construct_brusselator_matrix(n, alpha, b):
    """Construct the diffusion/reaction matrix for Brusselator."""
    size = n * n * 2
    C = np.zeros((size, size))

    for i in range(n):
        for j in range(n):
            u_idx = (i * n + j) * 2
            v_idx = u_idx + 1

            ip1, im1, jp1, jm1 = (
                _limit(i + 1, n),
                _limit(i - 1, n),
                _limit(j + 1, n),
                _limit(j - 1, n),
            )

            for ni, nj in [(ip1, j), (im1, j), (i, jp1), (i, jm1)]:
                C[u_idx][(ni * n + nj) * 2] += alpha
                C[v_idx][(ni * n + nj) * 2 + 1] += alpha

            C[u_idx][u_idx] -= 4 * alpha + (b + 1)
            C[v_idx][v_idx] -= 4 * alpha
            C[v_idx][u_idx] += b

    return C


def _brusselator_forcing(n):
    """Forcing term applied inside a disk of the 2D grid once t >= 1.1."""
    brusselator_cb = [0.0] * (n * n * 2)
    for i in range(n):
        for j in range(n):
            x = i / (n - 1)
            y = j / (n - 1)
            if (x - 0.3) ** 2 + (y - 0.6) ** 2 <= 0.1**2:
                brusselator_cb[(i * n + j) * 2] = 5
    return brusselator_cb


def _brusselator_derivatives(t, u_vec, meta, C, brusselator_cb):
    """Brusselator derivatives with diffusion on 2D grid."""
    a = meta["a"]
    u_arr = np.array(u_vec, dtype=float)

    lin = C @ u_arr
    lin[0::2] += a

    if t >= 1.1:
        lin += np.array(brusselator_cb)

    u_vals = u_arr[0::2]
    v_vals = u_arr[1::2]
    uv2 = u_vals**2 * v_vals

    non_lin = np.zeros(len(u_vec), dtype=float)
    non_lin[0::2] = uv2
    non_lin[1::2] = -uv2

    return (lin + non_lin).tolist()


def _linear_system_derivatives(t, state, meta, A, B):
    """Linear state-space derivatives for dx/dt = A x + B u."""
    input_value = meta["input_value"]
    state_array = np.asarray(state)
    input_dtype = np.result_type(B.dtype, type(input_value), float)
    input_array = np.full(B.shape[1], input_value, dtype=input_dtype)
    return (A @ state_array + B @ input_array).tolist()


def _resolve_derivatives(problem_name):
    """Select a derivative for the SciPy reference check."""
    match problem_name:
        case "ode_rc":
            return _rc_derivatives
        case "ode_rlc":
            return _rlc_derivatives
        case "ode_lotka_volterra":
            return _lotka_volterra_derivatives
        case "ode_brusselator":
            return _brusselator_derivatives
        case "ode_slicot":
            return _linear_system_derivatives
        case _:
            raise NotImplementedError(f"Unknown ODE problem: {problem_name!r}")


# Running error analysis for the checks. Each method's step is recomputed with
# a bound on how far its rounding can take it from the exact step: an operation
# rounds at most once per term, and an error e in f's argument moves f by at
# most spread(t, x, e). Higham's gamma_n uses eps, as saps.util.error_bounds does.
_EPS = np.finfo(np.float64).eps
# Each rounding can also lose this much to underflow (Higham, section 2.1).
_UNDERFLOW = np.finfo(np.float64).smallest_subnormal


def _gamma(n, *arrays):
    # A complex operation is up to four real ones, as in saps.util.error_bounds.
    if any(np.iscomplexobj(array) for array in arrays):
        n = 4 * n
    return n * _EPS / (1 - n * _EPS)


def _row_counts(matrix):
    """The stored nonzeros in each row: the terms each row of a product sums."""
    if hasattr(matrix, "indptr"):
        return np.diff(matrix.indptr)
    return np.count_nonzero(matrix, axis=1)


class _StepModel:
    """A problem's derivative f(t, x) with the pieces its error bound needs.

    magnitude(t, x) is f with every term made nonnegative, spread(t, x, e)
    bounds |f(t, x + d) - f(t, x)| over |d| <= e, and rounds counts the
    roundings in each component of f.
    """

    def __init__(self, f, magnitude, spread, rounds):
        self.f, self.magnitude, self.spread, self.rounds = f, magnitude, spread, rounds

    def evaluate(self, t, x, error):
        """f(t, x), and a bound on its distance from f at the exact argument."""
        value = np.asarray(self.f(t, x))
        error = np.broadcast_to(np.asarray(error, dtype=np.float64), value.shape)
        bound = _gamma(self.rounds + 1, value, x) * self.magnitude(t, x)
        bound = bound + 4 * (self.rounds + 1) * _UNDERFLOW
        return value, bound + self.spread(t, x, error)


def _axpy(y, scale, k, k_error):
    """y + k * scale, as the methods compute it, with its error bound."""
    value = y + k * scale
    rounding = _gamma(2, y, k) * (np.abs(y) + abs(scale) * np.abs(k)) + 8 * _UNDERFLOW
    return value, abs(scale) * k_error + rounding


def _product_spread(x, y, ex, ey):
    """Bounds |(x + dx)(y + dy) - x y| over |dx| <= ex, |dy| <= ey."""
    return (np.abs(x) + ex) * (np.abs(y) + ey) - np.abs(x) * np.abs(y)


def _step_model(meta, data) -> _StepModel:
    match meta["problem_name"]:
        case "ode_rc":
            tau = meta["R"] * meta["C"]
            return _StepModel(
                lambda t, x: _rc_derivatives(t, list(x), meta),
                lambda t, x: (abs(_step_input(t)) + np.abs(x)) / abs(tau),
                lambda t, x, e: e / abs(tau),
                2,
            )
        case "ode_rlc":
            rc, lc = meta["R"] * meta["C"], meta["L"] * meta["C"]
            return _StepModel(
                lambda t, x: _rlc_derivatives(t, list(x), meta),
                lambda t, x: np.array(
                    [
                        abs(x[1]),
                        (abs(_step_input(t)) + abs(x[0]) + abs(rc * x[1])) / abs(lc),
                    ]
                ),
                lambda t, x, e: np.array([e[1], (e[0] + abs(rc) * e[1]) / abs(lc)]),
                np.array([0, 4]),
            )
        case "ode_lotka_volterra":
            a, b, c, d = meta["a"], meta["b"], meta["c"], meta["d"]

            def spread(t, x, e):
                xy = _product_spread(x[0], x[1], e[0], e[1])
                return np.array(
                    [abs(a) * e[0] + abs(b) * xy, abs(d) * xy + abs(c) * e[1]]
                )

            return _StepModel(
                lambda t, x: _lotka_volterra_derivatives(t, list(x), meta),
                lambda t, x: np.array(
                    [
                        abs(a * x[0]) + abs(b * x[0] * x[1]),
                        abs(d * x[0] * x[1]) + abs(c * x[1]),
                    ]
                ),
                spread,
                4,
            )
        case "ode_brusselator":
            dense_c, forcing = data
            c = scipy_sparse.csr_array(dense_c)
            abs_c, a = abs(c), meta["a"]

            def constant(t):
                values = np.zeros(c.shape[0])
                values[0::2] = abs(a)
                return values + (np.abs(forcing) if t >= 1.1 else 0)

            def reaction(x, e):
                # u^2 v enters both components of each grid point.
                u, v, eu, ev = x[0::2], x[1::2], e[0::2], e[1::2]
                change = (np.abs(u) + eu) ** 2 * (np.abs(v) + ev) - u**2 * np.abs(v)
                return np.repeat(change, 2)

            def f(t, x):
                # _brusselator_derivatives, with C sparse.
                lin = c @ x
                lin[0::2] += a
                if t >= 1.1:
                    lin += forcing
                uv2 = x[0::2] ** 2 * x[1::2]
                lin[0::2] += uv2
                lin[1::2] -= uv2
                return lin

            return _StepModel(
                f,
                lambda t, x: abs_c @ np.abs(x) + constant(t) + reaction(x, 0 * x),
                lambda t, x, e: abs_c @ e + reaction(x, e),
                _row_counts(c) + 5,
            )
        case "ode_slicot":
            A, B = data
            u = np.full(B.shape[1], meta["input_value"], dtype=np.float64)
            abs_a, forcing = abs(A), abs(B) @ np.abs(u)
            return _StepModel(
                lambda t, x: _linear_system_derivatives(t, list(x), meta, A, B),
                lambda t, x: abs_a @ np.abs(x) + forcing,
                lambda t, x, e: abs_a @ e,
                _row_counts(A) + _row_counts(B) + 1,
            )
        case name:
            raise NotImplementedError(f"No step model for ODE problem {name!r}")


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


class ODERCDataset(Dataset):
    def __init__(
        self, name, pretty_name, description, suites, R, C, t_max, V_C_initial, step
    ):
        self._name = name
        self._pretty_name = pretty_name
        self._description = description
        self._suites = suites
        self.R = R
        self.C = C
        self.t_max = t_max
        self.V_C_initial = V_C_initial
        self.step = step

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


class ODERLCDataset(Dataset):
    def __init__(
        self, name, pretty_name, description, suites, R, L, C, t_max, y0, step
    ):
        self._name = name
        self._pretty_name = pretty_name
        self._description = description
        self._suites = suites
        self.R = R
        self.L = L
        self.C = C
        self.t_max = t_max
        self.y0 = y0
        self.step = step

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


class ODELotkaVolterraDataset(Dataset):
    def __init__(
        self, name, pretty_name, description, suites, a, b, c, d, t_max, y0, step
    ):
        self._name = name
        self._pretty_name = pretty_name
        self._description = description
        self._suites = suites
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.t_max = t_max
        self.y0 = y0
        self.step = step

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


class ODEBrusselatorDataset(Dataset):
    def __init__(
        self, name, pretty_name, description, suites, n, a, b, alpha, t_max, step
    ):
        self._name = name
        self._pretty_name = pretty_name
        self._description = description
        self._suites = suites
        self.n = n
        self.a = a
        self.b = b
        self.alpha = alpha
        self.t_max = t_max
        self.step = step

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


class ODESLICOTDataset(Dataset):
    def __init__(
        self,
        name: str,
        *,
        suites: list[str] | None = None,
        t_max: float = 0.1,
        step: float = 0.01,
        input_value: float = 1.0,
    ):
        self._name = name
        self.source_name = f"{name}.mat"
        self.problem = slicot_problem_metadata(self.source_name)
        self._suites = suites or []
        self.t_max = t_max
        self.step = step
        self.input_value = input_value

    @property
    def name(self) -> str:
        return self._name

    @property
    def pretty_name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        if self.problem.description:
            return f"SLICOT model-reduction ODE: {self.problem.description}."
        return "SLICOT model-reduction ODE."

    @property
    def suites(self) -> list[str]:
        return self._suites

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def metadata(self) -> dict[str, object]:
        data = super().metadata
        data.update(
            {
                "source_name": self.problem.mat_filename,
                "source_url": slicot_source_url(self.problem.mat_filename),
                "source_order": self.problem.order,
                "source_inputs": self.problem.inputs,
                "source_outputs": self.problem.outputs,
                "assumed_E": "identity",
                "step": self.step,
            }
        )
        return data


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------


_AKARSH = [Contributor("Akarsh Duddu", "aduddu3@gatech.edu")]
_AI_DISCLOSURE = (
    "The original benchmark algorithms were written without generative AI."
    " Generative AI was used for debugging, generator-time inlining, and tests."
)


TDataset = TypeVar("TDataset", bound=Dataset)


class _ODEGenerator(Generator[TDataset], ABC):
    array_parameters: tuple[str, ...] = ()
    # Each snippet reads t, state, and meta and assigns dydt_vector. Its locals
    # must not overwrite the solver's inputs, outputs, step, or RK4 stages.
    derivative_source: str

    def generate_benchmark_function(
        self,
        dataset: TDataset,
        problem: DataInstance,
        benchmark: Callable[..., Any],
    ) -> Callable[..., Any]:
        solver = getattr(benchmark, "__self__", None)
        if not isinstance(solver, _ODEBenchmarkBase):
            raise TypeError("ODE generation requires a bound ODE benchmark method")
        source = solver.benchmark_source(self.array_parameters, self.derivative_source)
        return define_function(
            source,
            f"<saps-generated {type(solver).__qualname__}.{self.name}.{dataset.name}>",
            {"np": np},
        )


class ODERCGenerator(_ODEGenerator[ODERCDataset]):
    derivative_source = """\
R, C = meta["R"], meta["C"]
tau = R * C
Vs = 5.0 if t >= 0 else 0.0
dydt_vector = [(Vs - state[0]) / tau]
"""

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return "ode_rc"

    @property
    def pretty_name(self) -> str:
        return "Ordinary Differential Equation (ODE) Resistor-Capacitor (RC) Circuit"

    @property
    def description(self) -> str:
        return "RC circuit ODE."

    @property
    def suites(self) -> list[str]:
        return []

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def authors(self) -> list[Contributor]:
        return _AKARSH

    @property
    def references(self) -> list[Ref]:
        return []

    @property
    def ai_disclosure(self) -> str:
        return _AI_DISCLOSURE

    @property
    def motivation(self) -> str:
        return ""

    @property
    def datasets(self) -> list[ODERCDataset]:
        return [
            ODERCDataset(
                name="small",
                pretty_name="Small",
                description="Small RC circuit",
                suites=["test"],
                R=1000.0,
                C=0.001,
                t_max=0.05,
                V_C_initial=0.0,
                step=0.0001,
            ),
        ]

    def generate(self, dataset: ODERCDataset):
        meta = {
            "problem_name": self.name,
            "span": (0, dataset.t_max),
            "y0": [dataset.V_C_initial],
            "step": dataset.step,
            "R": dataset.R,
            "C": dataset.C,
        }
        return DataInstance(inputs=[], meta=meta)


class ODERLCGenerator(_ODEGenerator[ODERLCDataset]):
    derivative_source = """\
R, L, C = meta["R"], meta["L"], meta["C"]
Vc = state[0]
dVc = state[1]
Vs = 5.0 if t >= 0 else 0.0
d2Vc = (Vs - Vc - R * C * dVc) / (L * C)
dydt_vector = (dVc, d2Vc)
"""

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return "ode_rlc"

    @property
    def pretty_name(self) -> str:
        return (
            "Ordinary Differential Equation (ODE) Resistor-Inductor-Capacitor (RLC)"
            " Circuit"
        )

    @property
    def description(self) -> str:
        return "RLC circuit ODE."

    @property
    def suites(self) -> list[str]:
        return []

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def authors(self) -> list[Contributor]:
        return _AKARSH

    @property
    def references(self) -> list[Ref]:
        return []

    @property
    def ai_disclosure(self) -> str:
        return _AI_DISCLOSURE

    @property
    def motivation(self) -> str:
        return ""

    @property
    def datasets(self) -> list[ODERLCDataset]:
        return [
            ODERLCDataset(
                name="small",
                pretty_name="Small",
                description="Small RLC circuit",
                suites=["test"],
                R=100.0,
                L=0.001,
                C=0.0000001,
                t_max=0.0001,
                y0=[0.0, 0.0],
                step=0.0000001,
            ),
        ]

    def generate(self, dataset: ODERLCDataset):
        meta = {
            "problem_name": self.name,
            "span": (0, dataset.t_max),
            "y0": list(dataset.y0),
            "step": dataset.step,
            "R": dataset.R,
            "L": dataset.L,
            "C": dataset.C,
        }
        return DataInstance(inputs=[], meta=meta)


class ODELotkaVolterraGenerator(_ODEGenerator[ODELotkaVolterraDataset]):
    derivative_source = """\
a, b, c, d = meta["a"], meta["b"], meta["c"], meta["d"]
x, y = state
dxdt = a * x - b * x * y
dydt = d * x * y - c * y
dydt_vector = (dxdt, dydt)
"""

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return "ode_lotka_volterra"

    @property
    def pretty_name(self) -> str:
        return "Ordinary Differential Equation (ODE) Lotka-Volterra"

    @property
    def description(self) -> str:
        return "Lotka-Volterra ODE."

    @property
    def suites(self) -> list[str]:
        return []

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def authors(self) -> list[Contributor]:
        return _AKARSH

    @property
    def references(self) -> list[Ref]:
        return []

    @property
    def ai_disclosure(self) -> str:
        return _AI_DISCLOSURE

    @property
    def motivation(self) -> str:
        return ""

    @property
    def datasets(self) -> list[ODELotkaVolterraDataset]:
        return [
            ODELotkaVolterraDataset(
                name="small",
                pretty_name="Small",
                description="Small Lotka-Volterra system",
                suites=["test"],
                a=0.1,
                b=0.02,
                c=0.3,
                d=0.01,
                t_max=2.0,
                y0=[40.0, 9.0],
                step=0.001,
            ),
        ]

    def generate(self, dataset: ODELotkaVolterraDataset):
        meta = {
            "problem_name": self.name,
            "span": (0, dataset.t_max),
            "y0": list(dataset.y0),
            "step": dataset.step,
            "a": dataset.a,
            "b": dataset.b,
            "c": dataset.c,
            "d": dataset.d,
        }
        return DataInstance(inputs=[], meta=meta)


class ODEBrusselatorGenerator(_ODEGenerator[ODEBrusselatorDataset]):
    array_parameters = ("C", "brusselator_cb")
    derivative_source = """\
a = meta["a"]
u_arr = xp.asarray(state, dtype=xp.float64)
lin = C @ u_arr
lin[0::2] += a
if t >= 1.1:
    lin += brusselator_cb
u_vals = u_arr[0::2]
v_vals = u_arr[1::2]
uv2 = u_vals**2 * v_vals
non_lin = xp.zeros(len(state), dtype=xp.float64)
non_lin[0::2] = uv2
non_lin[1::2] = -uv2
dydt_vector = (lin + non_lin).tolist()
"""

    def __init__(self, train: bool = False):
        self.train = train

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return "ode_brusselator"

    @property
    def pretty_name(self) -> str:
        return "Ordinary Differential Equation (ODE) Brusselator"

    @property
    def description(self) -> str:
        return "2D Brusselator ODE with diffusion."

    @property
    def suites(self) -> list[str]:
        return []

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def authors(self) -> list[Contributor]:
        return _AKARSH

    @property
    def references(self) -> list[Ref]:
        return []

    @property
    def ai_disclosure(self) -> str:
        return _AI_DISCLOSURE

    @property
    def motivation(self) -> str:
        return ""

    @property
    def datasets(self) -> list[ODEBrusselatorDataset]:
        return [
            ODEBrusselatorDataset(
                name="2x2",
                pretty_name="2x2 Grid",
                description="Tiny 2D Brusselator correctness test",
                suites=["test"],
                n=2,
                a=3.4,
                b=1.0,
                alpha=0.01,
                t_max=0.1,
                step=0.01,
            ),
            ODEBrusselatorDataset(
                name="100x100",
                pretty_name="100x100 Grid",
                description="2D Brusselator with 100x100 grid",
                suites=["standard", "trace", "train"]
                if self.train
                else ["standard", "trace"],
                n=100,
                a=3.4,
                b=1.0,
                alpha=0.01,
                t_max=1.0,
                step=0.01,
            ),
        ]

    def generate(self, dataset: ODEBrusselatorDataset):
        # Built here rather than in the dataset: C is dense (2n^2)^2, about
        # 3.2 GB for n=100, and datasets are listed far more often than generated.
        n = dataset.n
        C = _construct_brusselator_matrix(n, dataset.alpha, dataset.b)
        meta = {
            "problem_name": self.name,
            "span": (0, dataset.t_max),
            "y0": list(_init_brusselator_2d(n)),
            "step": dataset.step,
            "n": n,
            "a": dataset.a,
            "alpha": dataset.alpha,
        }
        return DataInstance(
            inputs=[
                from_numpy(C),
                from_numpy(np.asarray(_brusselator_forcing(n))),
            ],
            meta=meta,
        )


class ODESLICOTGenerator(_ODEGenerator[ODESLICOTDataset]):
    array_parameters = ("A", "B")
    derivative_source = """\
input_value = meta["input_value"]
state_array = xp.asarray(state)
input_dtype = np.result_type(B.dtype, type(input_value), float)
input_array = xp.asarray([input_value] * B.shape[1], dtype=input_dtype)
dydt_vector = (A @ state_array + B @ input_array).tolist()
"""

    def __init__(
        self,
        trace_datasets: tuple[str, ...] = (),
        train_dataset: str | None = None,
    ):
        self.trace_datasets = trace_datasets
        self.train_dataset = train_dataset

    @property
    def name(self) -> str:
        return "ode_slicot"

    @property
    def pretty_name(self) -> str:
        return "Ordinary Differential Equation (ODE) SLICOT"

    @property
    def description(self) -> str:
        return (
            "Loads SLICOT model-reduction problems without explicit E matrices, "
            "preserving stored sparse matrices, treating E as the identity, and "
            "defaulting missing B to a normalized single-input vector."
        )

    @property
    def suites(self) -> list[str]:
        return []

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def authors(self) -> list[Contributor]:
        return []

    @property
    def references(self) -> list[Ref]:
        return [
            Ref(
                title=(
                    "Benchmark examples for model reduction of linear time "
                    "invariant dynamical systems"
                ),
                authors=[],
                url=SLICOT_BENCHMARK_PAGE_URL,
            )
        ]

    @property
    def ai_disclosure(self) -> str:
        return "Generative AI was used to implement this generator."

    @property
    def motivation(self) -> str:
        return (
            "SLICOT model-reduction examples provide realistic linear dynamical "
            "systems for ODE integration benchmarks."
        )

    @property
    def datasets(self) -> list[ODESLICOTDataset]:
        # Base timesteps, scaled by each method's step_multiplier at setup.
        # Validated over t_max=0.1 at the 0.05 absolute-error tolerance.
        datasets = [
            ODESLICOTDataset("eady", suites=["standard", "trace"]),
            ODESLICOTDataset("CDplayer", suites=["standard"], step=4e-5),
            ODESLICOTDataset(
                "fom",
                suites=["standard", "trace", "train"]
                if self.train_dataset == "fom"
                else ["standard", "trace"],
                step=0.001,
            ),
            ODESLICOTDataset("random", suites=["standard"], step=5e-5),
            ODESLICOTDataset("pde", suites=["standard", "trace"], step=0.001),
            ODESLICOTDataset(
                "heat-cont",
                suites=["standard", "trace", "train"]
                if self.train_dataset == "heat-cont"
                else ["standard", "trace"],
                step=0.001,
            ),
            ODESLICOTDataset("Orr-Som", suites=["standard", "trace"]),
            ODESLICOTDataset("iss", suites=["standard", "trace"]),
            ODESLICOTDataset("build", suites=["standard", "trace"]),
            ODESLICOTDataset("beam", suites=["standard", "trace"], step=0.001),
        ]
        for dataset in datasets:
            if dataset.name in self.trace_datasets and "trace" not in dataset.suites:
                dataset.suites.append("trace")
        return datasets

    def generate(self, dataset: ODESLICOTDataset):
        from scipy import sparse as scipy_sparse

        variables, source_meta = load_slicot_problem(dataset.source_name)
        if "E" in variables:
            raise ValueError(
                f"SLICOT {dataset.source_name} has an explicit E matrix; "
                "ODESLICOTGenerator only supports identity-E systems"
            )
        if "A" not in variables:
            raise ValueError(f"SLICOT {dataset.source_name} must define A")

        A_value = variables["A"]
        if scipy_sparse.issparse(A_value):
            A = A_value.tocoo(copy=False)
        else:
            A = np.asarray(A_value)
            if A.ndim != 2:
                raise ValueError(
                    f"SLICOT {dataset.source_name} variable A must be two-dimensional"
                )
        if A.shape[0] != A.shape[1]:
            raise ValueError(f"SLICOT {dataset.source_name} A must be square")

        if "B" in variables:
            B_value = variables["B"]
            if scipy_sparse.issparse(B_value):
                B = B_value.tocoo(copy=False)
            else:
                B = np.asarray(B_value)
                if B.ndim != 2:
                    raise ValueError(
                        f"SLICOT {dataset.source_name} variable B must be "
                        "two-dimensional"
                    )
            assumed_B = None
        else:
            B = np.ones((A.shape[0], 1), dtype=np.result_type(A.dtype, float))
            B /= np.sqrt(A.shape[0])
            assumed_B = "normalized_uniform_vector"

        if B.shape[0] != A.shape[0]:
            raise ValueError(
                f"SLICOT {dataset.source_name} B rows must match A dimension"
            )

        meta = {
            "problem_name": self.name,
            "span": (0, dataset.t_max),
            "y0": [0.0] * A.shape[0],
            "step": dataset.step,
            "input_value": dataset.input_value,
            "source_name": dataset.problem.mat_filename,
            "source_title": dataset.problem.title,
            "source_url": source_meta["source_url"],
            "source_page_url": source_meta["source_page_url"],
            "source_order": dataset.problem.order,
            "source_inputs": dataset.problem.inputs,
            "source_outputs": dataset.problem.outputs,
            "num_states": A.shape[0],
            "input_dimension": B.shape[1],
            "assumed_E": "identity",
            "assumed_B": assumed_B,
            "A_storage": "sparse" if scipy_sparse.issparse(A) else "dense",
            "B_storage": "sparse" if scipy_sparse.issparse(B) else "dense",
        }
        return DataInstance(
            inputs=[
                from_scipy(A) if scipy_sparse.issparse(A) else from_numpy(A),
                from_scipy(B) if scipy_sparse.issparse(B) else from_numpy(B),
            ],
            meta=meta,
        )


# ---------------------------------------------------------------------------
# Integration-method benchmarks
# ---------------------------------------------------------------------------


class _ODEBenchmarkBase(Benchmark, ABC):
    step_multiplier = 1.0
    slicot_trace_datasets: tuple[str, ...] = ()
    slicot_train_dataset: str | None = None
    brusselator_train = False

    def benchmark(self, xp, meta, *data_args):
        raise NotImplementedError("ODE benchmark functions are generated during setup")

    @abstractmethod
    def benchmark_source(self, parameters: tuple[str, ...], dydt: str) -> str:
        """Paste a derivative snippet into this solver's source template."""
        ...

    @property
    def suites(self):
        return ["standard-timestepping"]

    @property
    def generators(self):
        return [
            ODERCGenerator(),
            ODERLCGenerator(),
            ODELotkaVolterraGenerator(),
            ODEBrusselatorGenerator(train=self.brusselator_train),
            ODESLICOTGenerator(
                trace_datasets=self.slicot_trace_datasets,
                train_dataset=self.slicot_train_dataset,
            ),
        ]

    @property
    def metadata(self):
        return {
            **super().metadata,
            "step_multiplier": self.step_multiplier,
        }

    def setup(self, param, **kwargs):
        super().setup(param, **kwargs)
        # Older cached inputs do not identify their problem. Use the selected
        # generator and preserve the cached matrices without modifying its metadata.
        self._meta = {"problem_name": param.generator.name, **self._meta}
        if isinstance(param.dataset, ODESLICOTDataset):
            # Apply current method settings even when the cached timestep is stale.
            self._meta["step"] = param.dataset.step * self.step_multiplier

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def authors(self):
        return _AKARSH

    @property
    def references(self):
        return []

    @property
    def ai_disclosure(self):
        return _AI_DISCLOSURE

    @property
    def motivation(self):
        return ""

    @abstractmethod
    def _step(self, model: _StepModel, t: float, step: float, y):
        """This method's step from y at time t, and a bound on its rounding."""

    def check(self, param):
        super().check(param)
        time = to_numpy(self._output[0])
        y_out = to_numpy(self._output[1])
        assert np.all(np.isfinite(y_out)), (
            f"Non-finite ODE output at step={self._meta['step']}"
        )
        span, step, y0 = self._meta["span"], self._meta["step"], self._meta["y0"]
        grid = []
        curr = span[0]
        while curr < span[1]:
            grid.append(curr)
            curr += step
        np.testing.assert_array_equal(time, grid)
        assert y_out.shape == (len(grid), len(y0))
        np.testing.assert_array_equal(y_out[0], y0)
        model = _step_model(self._meta, [_matrix(item) for item in self._input])
        # Recompute each step from the framework's previous state. Both this
        # step and the framework's are within the bound of the exact one.
        for i in range(1, len(grid)):
            value, bound = self._step(model, grid[i - 1], step, y_out[i - 1])
            error = np.abs(y_out[i] - value)
            worst = int(np.argmax(error - 2 * bound))
            assert error[worst] <= 2 * bound[worst], (
                f"{self.name} step {i} of {param.dataset.name} is off by "
                f"{error[worst]:.3g} in component {worst}, more than rounding "
                f"allows ({2 * bound[worst]:.3g})"
            )


class ForwardEulerBenchmark(_ODEBenchmarkBase):
    step_multiplier = 0.01

    def _step(self, model, t, step, y):
        k, k_error = model.evaluate(t, y, 0)
        return _axpy(y, step, k, k_error)

    slicot_trace_datasets = ("beam", "fom", "heat-cont")
    slicot_train_dataset = "fom"

    @property
    def name(self):
        return "forward_euler"

    @property
    def pretty_name(self):
        return "Forward Euler"

    @property
    def description(self):
        return "Integrates ODE initial-value problems with the forward Euler method."

    def benchmark_source(self, parameters: tuple[str, ...], dydt: str) -> str:
        signature = ", ".join(("xp", "meta", *parameters))
        return f"""\
def benchmark({signature}):
    span = meta["span"]
    y0 = meta["y0"]
    step = meta["step"]
    curr = span[0]
    inputs = []
    while curr < span[1]:
        inputs.append(curr)
        curr += step

    outputs = [None for _ in inputs]
    outputs[0] = y0
    for i in range(1, len(inputs)):
        t = inputs[i - 1]
        state = outputs[i - 1]
{indent(dydt, "        ").rstrip()}
        outputs[i] = [
            outputs[i - 1][j] + dydt_vector[j] * step for j in range(len(y0))
        ]
    return (np.asarray(inputs), np.asarray(outputs))
"""


class BackwardEulerBenchmark(_ODEBenchmarkBase):
    step_multiplier = 0.02

    def _step(self, model, t, step, y):
        # Ten fixed-point iterations of y_next = y + step * f(t + step, y_next).
        guess, error = y, np.zeros_like(y)
        for _ in range(10):
            k, k_error = model.evaluate(t + step, guess, error)
            guess, error = _axpy(y, step, k, k_error)
        return guess, error

    slicot_train_dataset = "heat-cont"

    @property
    def name(self):
        return "backward_euler"

    @property
    def pretty_name(self):
        return "Backward Euler"

    @property
    def description(self):
        return (
            "Integrates ODE initial-value problems with backward Euler, "
            "using ten fixed-point iterations per step."
        )

    def benchmark_source(self, parameters: tuple[str, ...], dydt: str) -> str:
        signature = ", ".join(("xp", "meta", *parameters))
        return f"""\
def benchmark({signature}):
    span = meta["span"]
    y0 = meta["y0"]
    step = meta["step"]
    curr = span[0]
    inputs = []
    while curr < span[1]:
        inputs.append(curr)
        curr += step

    outputs = [None for _ in inputs]
    outputs[0] = y0
    for i in range(1, len(inputs)):
        y_guess = outputs[i - 1]
        for _ in range(10):
            t = inputs[i]
            state = y_guess
{indent(dydt, "            ").rstrip()}
            y_guess = [
                outputs[i - 1][j] + dydt_vector[j] * step for j in range(len(y0))
            ]
        outputs[i] = y_guess
    return (np.asarray(inputs), np.asarray(outputs))
"""


class RK4Benchmark(_ODEBenchmarkBase):
    step_multiplier = 1.0

    def _step(self, model, t, step, y):
        k1, e1 = model.evaluate(t, y, 0)
        k2, e2 = model.evaluate(t + step / 2, *_axpy(y, step / 2, k1, e1))
        k3, e3 = model.evaluate(t + step / 2, *_axpy(y, step / 2, k2, e2))
        k4, e4 = model.evaluate(t + step, *_axpy(y, step, k3, e3))
        # ((k1 + 2 k2) + 2 k3) + k4 rounds three times; doubling is exact.
        total = k1 + 2 * k2 + 2 * k3 + k4
        magnitude = np.abs(k1) + 2 * np.abs(k2) + 2 * np.abs(k3) + np.abs(k4)
        total_error = e1 + 2 * e2 + 2 * e3 + e4 + _gamma(3, total) * magnitude
        return _axpy(y, step / 6, total, total_error)

    brusselator_train = True
    slicot_trace_datasets = (
        "CDplayer",
        "random",
        "beam",
        "fom",
        "heat-cont",
    )

    @property
    def name(self):
        return "rk4"

    @property
    def pretty_name(self):
        return "Fourth-Order Runge-Kutta (RK4)"

    @property
    def description(self):
        return (
            "Integrates ODE initial-value problems with the classical "
            "fourth-order Runge-Kutta method."
        )

    def benchmark_source(self, parameters: tuple[str, ...], dydt: str) -> str:
        signature = ", ".join(("xp", "meta", *parameters))
        return f"""\
def benchmark({signature}):
    span = meta["span"]
    y0 = meta["y0"]
    step = meta["step"]
    curr = span[0]
    inputs = []
    while curr < span[1]:
        inputs.append(curr)
        curr += step

    outputs = [None for _ in inputs]
    outputs[0] = y0
    for i in range(1, len(inputs)):
        y_prev = outputs[i - 1]
        t = inputs[i - 1]
        state = y_prev
{indent(dydt, "        ").rstrip()}
        k1 = dydt_vector
        k2_state = [y_prev[j] + (step / 2) * k1[j] for j in range(len(y0))]
        t = inputs[i - 1] + step / 2
        state = k2_state
{indent(dydt, "        ").rstrip()}
        k2 = dydt_vector
        k3_state = [y_prev[j] + (step / 2) * k2[j] for j in range(len(y0))]
        t = inputs[i - 1] + step / 2
        state = k3_state
{indent(dydt, "        ").rstrip()}
        k3 = dydt_vector
        k4_state = [y_prev[j] + step * k3[j] for j in range(len(y0))]
        t = inputs[i - 1] + step
        state = k4_state
{indent(dydt, "        ").rstrip()}
        k4 = dydt_vector
        outputs[i] = [
            y_prev[j] + (step / 6) * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j])
            for j in range(len(y0))
        ]
    return (np.asarray(inputs), np.asarray(outputs))
"""
