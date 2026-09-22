from __future__ import annotations

from abc import ABC

import numpy as np

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
from saps.downloaders.slicot import (
    SLICOT_BENCHMARK_PAGE_URL,
    load_slicot_problem,
    slicot_problem_metadata,
    slicot_source_url,
)


def _dense_binsparse_array(array: BinsparseTensor):
    try:
        return to_numpy(array)
    except TypeError:
        return to_scipy(array).toarray()


def _step_input(t):
    """A simple 5V step input starting at t=0."""
    return 5.0 if t >= 0 else 0.0


def _rc_derivatives(t, state, data, meta):
    """RC circuit derivatives."""
    R, C = meta["R"], meta["C"]
    tau = R * C
    Vs = _step_input(t)
    return [(Vs - state[0]) / tau]


def _rlc_derivatives(t, state, data, meta):
    """RLC circuit derivatives."""
    R, L, C = meta["R"], meta["L"], meta["C"]
    Vc = state[0]
    dVc = state[1]
    Vs = _step_input(t)
    d2Vc = (Vs - Vc - R * C * dVc) / (L * C)
    return (dVc, d2Vc)


def _lotka_volterra_derivatives(t, state, data, meta):
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


def _brusselator_derivatives(t, u_vec, data, meta):
    """Brusselator derivatives with diffusion on 2D grid."""
    C, brusselator_cb = data
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


def _linear_system_derivatives(t, state, data, meta):
    """Linear state-space derivatives for dx/dt = A x + B u."""
    A, B = data
    input_value = meta["input_value"]
    state_array = np.asarray(state)
    input_dtype = np.result_type(B.dtype, type(input_value), float)
    input_array = np.full(B.shape[1], input_value, dtype=input_dtype)
    return (A @ state_array + B @ input_array).tolist()


def _resolve_derivatives(problem_name):
    match problem_name:
        case "rc":
            return _rc_derivatives
        case "rlc":
            return _rlc_derivatives
        case "lotka_volterra":
            return _lotka_volterra_derivatives
        case "brusselator":
            return _brusselator_derivatives
        case "slicot_ode":
            return _linear_system_derivatives
        case _:
            raise NotImplementedError(f"Unknown ODE problem: {problem_name!r}")


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


class SLICOTDataset(Dataset):
    def __init__(
        self,
        source_name: str,
        *,
        suites: list[str] | None = None,
        t_max: float = 0.1,
        step: float = 0.01,
        input_value: float = 1.0,
    ):
        self.source_name = source_name
        self.problem = slicot_problem_metadata(source_name)
        self._suites = suites or []
        self.t_max = t_max
        self.step = step
        self.input_value = input_value

    @property
    def name(self) -> str:
        return f"slicot_{self.problem.name}".replace("-", "_").lower()

    @property
    def pretty_name(self) -> str:
        return f"SLICOT {self.problem.title}"

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
            "problem_name": self.name,
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
            "problem_name": self.name,
            "span": (0, dataset.t_max),
            "y0": list(dataset.y0),
            "step": dataset.step,
            "R": dataset.R,
            "L": dataset.L,
            "C": dataset.C,
        }
        return DataInstance(inputs=[], meta=meta, ref_meta={"check_components": [0]})


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
            "problem_name": self.name,
            "span": (0, dataset.t_max),
            "y0": list(dataset.y0),
            "step": dataset.step,
            "a": dataset.a,
            "b": dataset.b,
            "c": dataset.c,
            "d": dataset.d,
        }
        return DataInstance(inputs=[], meta=meta, ref_meta={"error_tolerance": 10.0})


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
                description="2D Brusselator with 100x100 grid",
                suites=["standard", "trace"],
                n=100,
                a=3.4,
                b=1.0,
                alpha=0.01,
                t_max=1.0,
                step=0.01,
            ),
        ]

    def generate(self, dataset: BrusselatorDataset):
        meta = {
            "problem_name": self.name,
            "span": (0, dataset.t_max),
            "y0": list(dataset.y0),
            "step": dataset.step,
            "n": dataset.n,
            "a": dataset.a,
            "alpha": dataset.alpha,
        }
        return DataInstance(
            inputs=[
                from_numpy(dataset.C),
                from_numpy(np.asarray(dataset.brusselator_cb)),
            ],
            meta=meta,
            ref_meta={"error_tolerance": 0.5, "real_output": True},
        )


class SLICOTGenerator(Generator[SLICOTDataset]):
    @property
    def name(self) -> str:
        return "slicot_ode"

    @property
    def pretty_name(self) -> str:
        return "SLICOT Model-Reduction ODE"

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
    def datasets(self) -> list[SLICOTDataset]:
        # Base timesteps, scaled by each method's step_multiplier at setup.
        # Validated over t_max=0.1 at the 0.05 absolute-error tolerance.
        return [
            SLICOTDataset("eady.mat", suites=["standard", "trace"]),
            SLICOTDataset("CDplayer.mat", suites=["standard"], step=4e-5),
            SLICOTDataset("fom.mat", suites=["standard", "trace"], step=0.001),
            SLICOTDataset("random.mat", suites=["standard"], step=5e-5),
            SLICOTDataset("pde.mat", suites=["standard", "trace"], step=0.001),
            SLICOTDataset("heat-cont.mat", suites=["standard", "trace"], step=0.001),
            SLICOTDataset("Orr-Som.mat", suites=["standard", "trace"]),
            SLICOTDataset("iss.mat", suites=["standard", "trace"]),
            SLICOTDataset("build.mat", suites=["standard", "trace"]),
            SLICOTDataset("beam.mat", suites=["standard"], step=0.001),
        ]

    def generate(self, dataset: SLICOTDataset):
        from scipy import sparse as scipy_sparse

        variables, source_meta = load_slicot_problem(dataset.source_name)
        if "E" in variables:
            raise ValueError(
                f"SLICOT {dataset.source_name} has an explicit E matrix; "
                "SLICOTGenerator only supports identity-E systems"
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


class _OdeBenchmarkBase(Benchmark, ABC):
    step_multiplier = 1.0

    @property
    def suites(self):
        return []

    @property
    def generators(self):
        return [
            RCGenerator(),
            RLCGenerator(),
            LotkaVolterraGenerator(),
            BrusselatorGenerator(),
            SLICOTGenerator(),
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
        if isinstance(param.dataset, SLICOTDataset):
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

    def check(self, param):
        super().check(param)
        from scipy.integrate import solve_ivp

        time = to_numpy(self._output[0])
        y_out = to_numpy(self._output[1])
        assert np.all(np.isfinite(y_out)), (
            f"Non-finite ODE output at step={self._meta['step']}"
        )
        data = [_dense_binsparse_array(item) for item in self._input]
        dydt = _resolve_derivatives(self._meta["problem_name"])
        rhs = lambda t, y: dydt(t, list(y), data, self._meta)  # noqa: E731
        y0 = np.asarray(
            self._meta["y0"],
            dtype=np.result_type(y_out.dtype, *(item.dtype for item in data), float),
        )
        solution = solve_ivp(
            rhs,
            self._meta["span"],
            y0,
            t_eval=time,
            rtol=1e-8,
            atol=1e-10,
        )
        assert solution.success, f"ODE reference integration failed: {solution.message}"
        ref_meta = self._ref_meta or {}
        actual_y, ref_y = np.asarray(y_out), solution.y.T
        if ref_meta.get("real_output"):
            actual_y = actual_y.real
        if "check_components" in ref_meta:
            components = ref_meta["check_components"]
            actual_y, ref_y = actual_y[:, components], ref_y[:, components]
        tolerance = ref_meta.get("error_tolerance", 0.05)
        error = np.max(np.abs(actual_y - ref_y))
        assert error < tolerance, (
            f"ODE maximum absolute error {error:.6g} exceeds "
            f"tolerance {tolerance} at step={self._meta['step']}"
        )


class ForwardEuler(_OdeBenchmarkBase):
    step_multiplier = 0.01

    @property
    def name(self):
        return "forward_euler"

    @property
    def pretty_name(self):
        return "Forward Euler"

    @property
    def description(self):
        return "Integrates ODE initial-value problems with the forward Euler method."

    def benchmark(self, xp, data, meta):
        dydt = _resolve_derivatives(meta["problem_name"])
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
            dydt_vector = dydt(inputs[i - 1], outputs[i - 1], data, meta)
            outputs[i] = [
                outputs[i - 1][j] + dydt_vector[j] * step for j in range(len(y0))
            ]
        return (np.asarray(inputs), np.asarray(outputs))


class BackwardEuler(_OdeBenchmarkBase):
    step_multiplier = 0.02

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

    def benchmark(self, xp, data, meta):
        dydt = _resolve_derivatives(meta["problem_name"])
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
                dydt_vector = dydt(inputs[i], y_guess, data, meta)
                y_guess = [
                    outputs[i - 1][j] + dydt_vector[j] * step for j in range(len(y0))
                ]
            outputs[i] = y_guess
        return (np.asarray(inputs), np.asarray(outputs))


class RungeKutta(_OdeBenchmarkBase):
    step_multiplier = 1.0

    @property
    def name(self):
        return "runge_kutta"

    @property
    def pretty_name(self):
        return "Runge-Kutta (RK4)"

    @property
    def description(self):
        return (
            "Integrates ODE initial-value problems with the classical "
            "fourth-order Runge-Kutta method."
        )

    def benchmark(self, xp, data, meta):
        dydt = _resolve_derivatives(meta["problem_name"])
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
            k1 = dydt(inputs[i - 1], y_prev, data, meta)
            k2_state = [y_prev[j] + (step / 2) * k1[j] for j in range(len(y0))]
            k2 = dydt(inputs[i - 1] + step / 2, k2_state, data, meta)
            k3_state = [y_prev[j] + (step / 2) * k2[j] for j in range(len(y0))]
            k3 = dydt(inputs[i - 1] + step / 2, k3_state, data, meta)
            k4_state = [y_prev[j] + step * k3[j] for j in range(len(y0))]
            k4 = dydt(inputs[i - 1] + step, k4_state, data, meta)
            outputs[i] = [
                y_prev[j] + (step / 6) * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j])
                for j in range(len(y0))
            ]
        return (np.asarray(inputs), np.asarray(outputs))
