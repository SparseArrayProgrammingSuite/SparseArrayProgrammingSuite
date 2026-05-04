from __future__ import annotations

import numpy as np

import sparseappbench
from sparseappbench.benchmark import (
    Author,
    Benchmark,
    Contributor,
    Dataset,
    Generator,
    Ref,
)

xp = sparseappbench.xp

def _step_input(t):
    """A simple 5V step input starting at t=0."""
    return 5.0 if t >= 0 else 0.0


def _rc_derivatives(t, state, R, C, source_voltage):
    """RC circuit derivatives."""
    tau = R * C
    Vs = source_voltage(t)  # Get the current source voltage
    return [(Vs - state[0]) / tau]  # dV/dt


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
            u[(i * n + j) * 2]     = float(np.real(22 * (fj * (1 - fj)) ** 1.5))
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

            C[u_idx][u_idx] -= (4 * alpha + (b + 1))
            C[v_idx][v_idx] -= 4 * alpha
            C[v_idx][u_idx] += b

    return C


def _brusselator_derivatives(t, u_vec, n, a, alpha, C, brusselator_cb):
    """Brusselator derivatives with diffusion on 2D grid."""
    u_arr = np.array(u_vec, dtype=float)
    
    # Linear part
    lin = C @ u_arr
    lin[0::2] += a

    if t >= 1.1:
        lin += np.array(brusselator_cb)

    # Non-linear part: u^2 * v
    u_vals = u_arr[0::2]
    v_vals = u_arr[1::2]
    uv2 = u_vals**2 * v_vals
    
    non_lin = np.zeros(len(u_vec), dtype=float)
    non_lin[0::2] = uv2
    non_lin[1::2] = -uv2
    
    return (lin + non_lin).tolist()


class RCDataset(Dataset):
    def __init__(self, name, pretty_name, description, tags, R, C, t_max, V_C_initial, step, source_voltage):
        self._name = name
        self._pretty_name = pretty_name
        self._description = description
        self._tags = tags
        self.R = R
        self.C = C
        self.t_max = t_max
        self.V_C_initial = V_C_initial
        self.step = step
        self.source_voltage = source_voltage

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
    def tags(self) -> list[str]:
        return self._tags


class RLCDataset(Dataset):
    def __init__(self, name, pretty_name, description, tags, R, L, C, t_max, y0, step, source_voltage):
        self._name = name
        self._pretty_name = pretty_name
        self._description = description
        self._tags = tags
        self.R = R
        self.L = L
        self.C = C
        self.t_max = t_max
        self.y0 = y0
        self.step = step
        self.source_voltage = source_voltage

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
    def tags(self) -> list[str]:
        return self._tags


class LotkaVolterraDataset(Dataset):
    def __init__(self, name, pretty_name, description, tags, a, b, c, d, t_max, y0, step):
        self._name = name
        self._pretty_name = pretty_name
        self._description = description
        self._tags = tags
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
    def tags(self) -> list[str]:
        return self._tags


class BrusselatorDataset(Dataset):
    def __init__(self, name, pretty_name, description, tags, n, a, b, alpha, t_max, step):
        self._name = name
        self._pretty_name = pretty_name
        self._description = description
        self._tags = tags
        self.n = n
        self.a = a
        self.b = b
        self.alpha = alpha
        self.t_max = t_max
        self.step = step
        self.y0 = _init_brusselator_2d(n)
        
        # Pre-construct matrix and forcing term
        self.C = _construct_brusselator_matrix(n, alpha, b)
        
        # Construct forcing term
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
    def tags(self) -> list[str]:
        return self._tags


class RCGenerator(Generator[RCDataset]):
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
    def tags(self) -> list[str]:
        return ["ode", "rc-circuit"]

    @property
    def authors(self) -> list[Contributor]:
        return [Contributor("Akarsh Duddu", "aduddu3@gatech.edu")]

    @property
    def references(self) -> list[Ref]:
        return []

    @property
    def ai_disclosure(self) -> str:
        return (
            "No generative AI was used to write the benchmark function itself. Generative"
            "AI was used for debugging. This statement was written by hand"
        )


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
                tags=["small"],
                R=1000.0,
                C=0.001,
                t_max=5.0,
                V_C_initial=0.0,
                step=0.000001,
                source_voltage=_step_input,
            ),
        ]

    def generate(self, dataset: RCDataset):
        def dVdt(t, state):
            return _rc_derivatives(t, state, dataset.R, dataset.C, dataset.source_voltage)

        return (
            dVdt,
            (0, dataset.t_max),
            [dataset.V_C_initial],
            dataset.step,
        ), {}


class RLCGenerator(Generator[RLCDataset]):
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
    def tags(self) -> list[str]:
        return ["ode", "rlc-circuit"]

    @property
    def authors(self) -> list[Contributor]:
        return [Contributor("Akarsh Duddu", "aduddu3@gatech.edu")]

    @property
    def references(self) -> list[Ref]:
        return []

    @property
    def ai_disclosure(self) -> str:
        return (
            "No generative AI was used to write the benchmark function itself. Generative"
            "AI was used for debugging. This statement was written by hand"
        )

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
                tags=["small"],
                R=100.0,
                L=0.001,
                C=0.0000001,
                t_max=0.01,
                y0=[0.0, 0.0],
                step=0.0000001,
                source_voltage=_step_input,
            ),
        ]

    def generate(self, dataset: RLCDataset):
        def dVdt(t, state):
            return _rlc_derivatives(t, state, dataset.R, dataset.L, dataset.C, dataset.source_voltage)

        return (
            dVdt,
            (0, dataset.t_max),
            dataset.y0,
            dataset.step,
        ), {}


class LotkaVolterraGenerator(Generator[LotkaVolterraDataset]):
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
    def tags(self) -> list[str]:
        return ["ode", "lotka-volterra"]

    @property
    def authors(self) -> list[Contributor]:
        return [Contributor("Akarsh Duddu", "aduddu3@gatech.edu")]

    @property
    def references(self) -> list[Ref]:
        return []

    @property
    def ai_disclosure(self) -> str:
        return (
            "No generative AI was used to write the benchmark function itself. Generative"
            "AI was used for debugging. This statement was written by hand"
        )

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
                tags=["small"],
                a=0.1,
                b=0.02,
                c=0.3,
                d=0.01,
                t_max=100.0,
                y0=[40.0, 9.0],
                step=0.00001,
            ),
        ]

    def generate(self, dataset: LotkaVolterraDataset):
        def dydt(t, state):
            return _lotka_volterra_derivatives(t, state, dataset.a, dataset.b, dataset.c, dataset.d)

        return (
            dydt,
            (0, dataset.t_max),
            dataset.y0,
            dataset.step,
        ), {}


class BrusselatorGenerator(Generator[BrusselatorDataset]):
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
    def tags(self) -> list[str]:
        return ["ode", "brusselator"]

    @property
    def authors(self) -> list[Contributor]:
        return [Contributor("Akarsh Duddu", "aduddu3@gatech.edu")]

    @property
    def references(self) -> list[Ref]:
        return []

    @property
    def ai_disclosure(self) -> str:
        return (
            "No generative AI was used to write the benchmark function itself. Generative"
            "AI was used for debugging. This statement was written by hand"
        )

    @property
    def motivation(self) -> str:
        return ""

    @property
    def datasets(self) -> list[BrusselatorDataset]:
        return [
            BrusselatorDataset(
                name="brusselator_4",
                pretty_name="Brusselator 4x4",
                description="2D Brusselator with 4x4 grid",
                tags=["small"],
                n=4,
                a=3.4,
                b=1.0,
                alpha=0.01,
                t_max=1.0,
                step=0.01,
            ),
        ]

    def generate(self, dataset: BrusselatorDataset):
        def dudt(t, state):
            return _brusselator_derivatives(t, state, dataset.n, dataset.a, dataset.alpha, dataset.C, dataset.brusselator_cb)

        return (
            dudt,
            (0, dataset.t_max),
            dataset.y0,
            dataset.step,
        ), {}


class ForwardEuler(Benchmark):
    @property
    def name(self):
        return "forward_euler"

    @property
    def pretty_name(self):
        return "Forward Euler ODE Solver"

    @property
    def description(self):
        return "Forward Euler method for solving various ODE systems."

    @property
    def tags(self):
        return ["ode", "forward-euler", "integration"]

    @property
    def authors(self):
        return [Contributor("Akarsh Duddu", "aduddu3@gatech.edu")]

    @property
    def references(self):
        return []

    @property
    def ai_disclosure(self):
        return (
            "No generative AI was used to write the benchmark function itself. Generative"
            "AI was used for debugging. This statement was written by hand"
        )


    @property
    def motivation(self):
        return ""

    @property
    def generators(self):
        return [
            RCGenerator(),
            RLCGenerator(),
            LotkaVolterraGenerator(),
            BrusselatorGenerator(),
        ]

    def benchmark(self, data, meta):
        dydx, span, y0, first_step = data
        
        # Forward Euler integration
        curr = span[0]
        inputs = []
        while curr < span[1]:
            inputs.append(curr)
            curr += first_step

        step = first_step
        outputs = [None for _ in inputs]
        outputs[0] = y0

        for i in range(1, len(inputs)):
            dydt_vector = dydx(inputs[i - 1], outputs[i - 1])
            outputs[i] = [outputs[i - 1][j] + dydt_vector[j] * step for j in range(len(y0))]

        return (inputs, outputs)







class BackwardEuler(Benchmark):
    @property
    def name(self):
        return "backward_euler"

    @property
    def pretty_name(self):
        return "Backward Euler ODE Solver"

    @property
    def description(self):
        return "Backward Euler method for solving various ODE systems."

    @property
    def tags(self):
        return ["ode", "backward-euler", "integration"]

    @property
    def authors(self):
        return [Contributor("Akarsh Duddu", "aduddu3@gatech.edu")]

    @property
    def references(self):
        return []

    @property
    def ai_disclosure(self):
        return (
            "No generative AI was used to write the benchmark function itself. Generative"
            "AI was used for debugging. This statement was written by hand"
        )

    @property
    def motivation(self):
        return ""

    @property
    def generators(self):
        return [
            RCGenerator(),
            RLCGenerator(),
            LotkaVolterraGenerator(),
            BrusselatorGenerator(),
        ]

    def benchmark(self, data, meta):
        dydx, span, y0, first_step = data
        
        # Backward Euler integration with fixed point iteration
        curr = span[0]
        inputs = []
        while curr < span[1]:
            inputs.append(curr)
            curr += first_step

        step = first_step
        outputs = [None for _ in inputs]
        outputs[0] = y0

        # y_n+1 = y_n + dy/dx(x_n+1, y_n+1) * delta x
        # Fixed point iteration
        for i in range(1, len(inputs)):
            y_guess = outputs[i - 1]  # initial guess
            for _ in range(10):
                dydt_vector = dydx(inputs[i], y_guess)
                y_guess = [outputs[i - 1][j] + dydt_vector[j] * step for j in range(len(y0))]
            outputs[i] = y_guess

        return (inputs, outputs)





class RK4(Benchmark):
    @property
    def name(self):
        return "rk4"

    @property
    def pretty_name(self):
        return "Runge-Kutta 4th Order ODE Solver"

    @property
    def description(self):
        return "Runge-Kutta 4th order method for solving various ODE systems."

    @property
    def tags(self):
        return ["ode", "rk4", "integration"]

    @property
    def authors(self):
        return [Contributor("Akarsh Duddu", "aduddu3@gatech.edu")]

    @property
    def references(self):
        return []

    @property
    def ai_disclosure(self):
        return (
            "No generative AI was used to write the benchmark function itself. Generative"
            "AI was used for debugging. This statement was written by hand"
        )

    @property
    def motivation(self):
        return ""

    @property
    def generators(self):
        return [
            RCGenerator(),
            RLCGenerator(),
            LotkaVolterraGenerator(),
            BrusselatorGenerator(),
        ]

    def benchmark(self, data, meta):
        dydx, span, y0, first_step = data
        
        # Runge-Kutta 4th order integration
        curr = span[0]
        inputs = []
        while curr < span[1]:
            inputs.append(curr)
            curr += first_step

        step = first_step
        outputs = [None for _ in inputs]
        outputs[0] = y0

        for i in range(1, len(inputs)):
            y_prev = outputs[i - 1]
            k1 = dydx(inputs[i - 1], y_prev)
            k2_state = [y_prev[j] + (step / 2) * k1[j] for j in range(len(y0))]
            k2 = dydx(inputs[i - 1] + step / 2, k2_state)
            k3_state = [y_prev[j] + (step / 2) * k2[j] for j in range(len(y0))]
            k3 = dydx(inputs[i - 1] + step / 2, k3_state)
            k4_state = [y_prev[j] + step * k3[j] for j in range(len(y0))]
            k4 = dydx(inputs[i - 1] + step, k4_state)
            outputs[i] = [
                y_prev[j] + (step / 6) * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j])
                for j in range(len(y0))
            ]

        return (inputs, outputs)