from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from saps.benchmark import (
    Benchmark,
    Contributor,
    DataInstance,
    Dataset,
    Generator,
    Ref,
)



def _step_input(t):
    """A simple 5V step input starting at t=0."""
    return 5.0 if t >= 0 else 0.0


def _rc_derivatives(t, state, R, C, source_voltage):
    """RC circuit derivatives."""
    tau = R * C
    Vs = source_voltage(t)
    return [(Vs - state[0]) / tau]


def _rlc_derivatives(t, state, R, L, C, source_voltage):
    """RLC circuit derivatives."""
    Vc = state[0]
    dVc = state[1]
    Vs = source_voltage(t)
    d2Vc = (Vs - Vc - R * C * dVc) / (L * C)
    return (dVc, d2Vc)


def _lotka_volterra_derivatives(t, state, a, b, c, d):
    """Lotka-Volterra derivatives."""
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


def _brusselator_derivatives(t, u_vec, n, a, alpha, C, brusselator_cb):
    """Brusselator derivatives with diffusion on 2D grid."""
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


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


class RCDataset(Dataset):
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


class RLCDataset(Dataset):
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


class LotkaVolterraDataset(Dataset):
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


class BrusselatorDataset(Dataset):
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
        self.y0 = _init_brusselator_2d(n)
        self.C = _construct_brusselator_matrix(n, alpha, b)

        size = n * n * 2
        self.brusselator_cb = [0.0] * size
        for i in range(n):
            for j in range(n):
                x = i / (n - 1)
                y = j / (n - 1)
                if (x - 0.3) ** 2 + (y - 0.6) ** 2 <= 0.1**2:
                    self.brusselator_cb[(i * n + j) * 2] = 5

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


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------


_AKARSH = [Contributor("Akarsh Duddu", "aduddu3@gatech.edu")]
_AI_DISCLOSURE = (
    "No generative AI was used to write the benchmark function itself."
    " Generative AI was used for debugging. This statement was written by hand"
)


class RCGenerator(Generator[RCDataset]):
    @property
    def cacheable(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return "rc"

    @property
    def pretty_name(self) -> str:
        return "RC Circuit"

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
    def datasets(self) -> list[RCDataset]:
        return [
            RCDataset(
                name="rc_small",
                pretty_name="RC Small",
                description="Small RC circuit",
                suites=["test"],
                R=1000.0,
                C=0.001,
                t_max=0.05,
                V_C_initial=0.0,
                step=0.0001,
            ),
        ]

    def generate(self, dataset: RCDataset):
        meta = {
            "span": (0, dataset.t_max),
            "y0": [dataset.V_C_initial],
            "step": dataset.step,
            "R": dataset.R,
            "C": dataset.C,
        }
        return DataInstance(inputs=[], meta=meta)


class RLCGenerator(Generator[RLCDataset]):
    @property
    def cacheable(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return "rlc"

    @property
    def pretty_name(self) -> str:
        return "RLC Circuit"

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
    def datasets(self) -> list[RLCDataset]:
        return [
            RLCDataset(
                name="rlc_small",
                pretty_name="RLC Small",
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

    def generate(self, dataset: RLCDataset):
        meta = {
            "span": (0, dataset.t_max),
            "y0": list(dataset.y0),
            "step": dataset.step,
            "R": dataset.R,
            "L": dataset.L,
            "C": dataset.C,
        }
        return DataInstance(inputs=[], meta=meta)


class LotkaVolterraGenerator(Generator[LotkaVolterraDataset]):
    @property
    def cacheable(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return "lotka_volterra"

    @property
    def pretty_name(self) -> str:
        return "Lotka-Volterra"

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
    def datasets(self) -> list[LotkaVolterraDataset]:
        return [
            LotkaVolterraDataset(
                name="lotka_volterra_small",
                pretty_name="Lotka-Volterra Small",
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

    def generate(self, dataset: LotkaVolterraDataset):
        meta = {
            "span": (0, dataset.t_max),
            "y0": list(dataset.y0),
            "step": dataset.step,
            "a": dataset.a,
            "b": dataset.b,
            "c": dataset.c,
            "d": dataset.d,
        }
        return DataInstance(inputs=[], meta=meta)


class BrusselatorGenerator(Generator[BrusselatorDataset]):
    @property
    def cacheable(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return "brusselator"

    @property
    def pretty_name(self) -> str:
        return "Brusselator"

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
    def datasets(self) -> list[BrusselatorDataset]:
        return [
            BrusselatorDataset(
                name="brusselator_tiny",
                pretty_name="Brusselator Tiny",
                description="Tiny 2D Brusselator correctness test",
                suites=["test"],
                n=2,
                a=3.4,
                b=1.0,
                alpha=0.01,
                t_max=0.1,
                step=0.01,
            ),
            BrusselatorDataset(
                name="brusselator_4",
                pretty_name="Brusselator 4x4",
                description="2D Brusselator with 4x4 grid",
                suites=["standard"],
                n=4,
                a=3.4,
                b=1.0,
                alpha=0.01,
                t_max=1.0,
                step=0.01,
            ),
        ]

    def generate(self, dataset: BrusselatorDataset):
        meta = {
            "span": (0, dataset.t_max),
            "y0": list(dataset.y0),
            "step": dataset.step,
            "n": dataset.n,
            "a": dataset.a,
            "alpha": dataset.alpha,
            "C": dataset.C,
            "brusselator_cb": dataset.brusselator_cb,
        }
        return DataInstance(inputs=[], meta=meta)


# ---------------------------------------------------------------------------
# Integration-scheme abstract bases
# ---------------------------------------------------------------------------


class _OdeBenchmarkBase(Benchmark, ABC):
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

    def _error_tolerance(self):
        return 0.05

    def _comparison_output(self, y, ref):
        return y, ref

    @abstractmethod
    def _dydt(self, t, y, meta):
        raise NotImplementedError

    def check(self, param):
        super().check(param)
        from scipy.integrate import solve_ivp

        time = self._output[0].data["values"].reshape(self._output[0].data["shape"])
        y_out = self._output[1].data["values"].reshape(self._output[1].data["shape"])
        rhs = lambda t, y: self._dydt(t, list(y), self._meta)  # noqa: E731
        ref = solve_ivp(
            rhs,
            self._meta["span"],
            self._meta["y0"],
            t_eval=time,
        ).y.T
        actual_y, ref_y = self._comparison_output(np.asarray(y_out), ref)
        error = np.max(np.abs(actual_y - ref_y))
        assert error < self._error_tolerance()


class _ForwardEulerBase(_OdeBenchmarkBase):
    @property
    def suites(self):
        return []

    def benchmark(self, xp, data, meta):
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
            dydt_vector = self._dydt(inputs[i - 1], outputs[i - 1], meta)
            outputs[i] = [
                outputs[i - 1][j] + dydt_vector[j] * step for j in range(len(y0))
            ]
        return (np.asarray(inputs), np.asarray(outputs))


class _BackwardEulerBase(_OdeBenchmarkBase):
    @property
    def suites(self):
        return []

    def benchmark(self, xp, data, meta):
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
                dydt_vector = self._dydt(inputs[i], y_guess, meta)
                y_guess = [
                    outputs[i - 1][j] + dydt_vector[j] * step for j in range(len(y0))
                ]
            outputs[i] = y_guess
        return (np.asarray(inputs), np.asarray(outputs))


class _RK4Base(_OdeBenchmarkBase):
    @property
    def suites(self):
        return []

    def benchmark(self, xp, data, meta):
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
            k1 = self._dydt(inputs[i - 1], y_prev, meta)
            k2_state = [y_prev[j] + (step / 2) * k1[j] for j in range(len(y0))]
            k2 = self._dydt(inputs[i - 1] + step / 2, k2_state, meta)
            k3_state = [y_prev[j] + (step / 2) * k2[j] for j in range(len(y0))]
            k3 = self._dydt(inputs[i - 1] + step / 2, k3_state, meta)
            k4_state = [y_prev[j] + step * k3[j] for j in range(len(y0))]
            k4 = self._dydt(inputs[i - 1] + step, k4_state, meta)
            outputs[i] = [
                y_prev[j] + (step / 6) * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j])
                for j in range(len(y0))
            ]
        return (np.asarray(inputs), np.asarray(outputs))


# ---------------------------------------------------------------------------
# RC concrete benchmarks
# ---------------------------------------------------------------------------


class _RCMixin:
    @property
    def description(self):
        return "RC circuit ODE."

    @property
    def generators(self):
        return [RCGenerator()]

    def _dydt(self, t, y, meta):
        return _rc_derivatives(t, y, meta["R"], meta["C"], _step_input)


class RCForwardEuler(_RCMixin, _ForwardEulerBase):
    @property
    def name(self):
        return "rc_forward_euler"

    @property
    def pretty_name(self):
        return "RC Circuit — Forward Euler"


class RCBackwardEuler(_RCMixin, _BackwardEulerBase):
    @property
    def name(self):
        return "rc_backward_euler"

    @property
    def pretty_name(self):
        return "RC Circuit — Backward Euler"


class RCRK4(_RCMixin, _RK4Base):
    @property
    def name(self):
        return "rc_rk4"

    @property
    def pretty_name(self):
        return "RC Circuit — RK4"


# ---------------------------------------------------------------------------
# RLC concrete benchmarks
# ---------------------------------------------------------------------------


class _RLCMixin:
    @property
    def description(self):
        return "RLC circuit ODE."

    @property
    def generators(self):
        return [RLCGenerator()]

    def _dydt(self, t, y, meta):
        return _rlc_derivatives(t, y, meta["R"], meta["L"], meta["C"], _step_input)

    def _comparison_output(self, y, ref):
        return y[:, 0], ref[:, 0]


class RLCForwardEuler(_RLCMixin, _ForwardEulerBase):
    @property
    def name(self):
        return "rlc_forward_euler"

    @property
    def pretty_name(self):
        return "RLC Circuit — Forward Euler"


class RLCBackwardEuler(_RLCMixin, _BackwardEulerBase):
    @property
    def name(self):
        return "rlc_backward_euler"

    @property
    def pretty_name(self):
        return "RLC Circuit — Backward Euler"


class RLCRK4(_RLCMixin, _RK4Base):
    @property
    def name(self):
        return "rlc_rk4"

    @property
    def pretty_name(self):
        return "RLC Circuit — RK4"


# ---------------------------------------------------------------------------
# Lotka-Volterra concrete benchmarks
# ---------------------------------------------------------------------------


class _LotkaVolterraMixin:
    @property
    def description(self):
        return "Lotka-Volterra ODE."

    @property
    def generators(self):
        return [LotkaVolterraGenerator()]

    def _dydt(self, t, y, meta):
        return _lotka_volterra_derivatives(
            t, y, meta["a"], meta["b"], meta["c"], meta["d"]
        )

    def _error_tolerance(self):
        return 10.0


class LotkaVolterraForwardEuler(_LotkaVolterraMixin, _ForwardEulerBase):
    @property
    def name(self):
        return "lotka_volterra_forward_euler"

    @property
    def pretty_name(self):
        return "Lotka-Volterra — Forward Euler"


class LotkaVolterraBackwardEuler(_LotkaVolterraMixin, _BackwardEulerBase):
    @property
    def name(self):
        return "lotka_volterra_backward_euler"

    @property
    def pretty_name(self):
        return "Lotka-Volterra — Backward Euler"


class LotkaVolterraRK4(_LotkaVolterraMixin, _RK4Base):
    @property
    def name(self):
        return "lotka_volterra_rk4"

    @property
    def pretty_name(self):
        return "Lotka-Volterra — RK4"


# ---------------------------------------------------------------------------
# Brusselator concrete benchmarks
# ---------------------------------------------------------------------------


class _BrusselatorMixin:
    @property
    def description(self):
        return "2D Brusselator ODE with diffusion."

    @property
    def generators(self):
        return [BrusselatorGenerator()]

    def _dydt(self, t, y, meta):
        return _brusselator_derivatives(
            t,
            y,
            meta["n"],
            meta["a"],
            meta["alpha"],
            meta["C"],
            meta["brusselator_cb"],
        )

    def _error_tolerance(self):
        return 0.5

    def _comparison_output(self, y, ref):
        return y.real, ref


class BrusselatorForwardEuler(_BrusselatorMixin, _ForwardEulerBase):
    @property
    def name(self):
        return "brusselator_forward_euler"

    @property
    def pretty_name(self):
        return "Brusselator — Forward Euler"


class BrusselatorBackwardEuler(_BrusselatorMixin, _BackwardEulerBase):
    @property
    def name(self):
        return "brusselator_backward_euler"

    @property
    def pretty_name(self):
        return "Brusselator — Backward Euler"


class BrusselatorRK4(_BrusselatorMixin, _RK4Base):
    @property
    def name(self):
        return "brusselator_rk4"

    @property
    def pretty_name(self):
        return "Brusselator — RK4"
