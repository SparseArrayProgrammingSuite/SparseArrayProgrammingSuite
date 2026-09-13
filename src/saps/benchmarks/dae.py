"""Fixed-step BDF2 benchmarks for linear descriptor systems E y' = A y + B u."""

from numbers import Integral
from typing import Any

import numpy as np
import scipy.sparse as scipy_sparse
from scipy.sparse.linalg import splu

import sparse as pydata_sparse
from binsparse import BinsparseTensor
from binsparse.conversions import from_numpy, from_scipy, to_numpy, to_sparse

from saps.benchmark import (
    Author,
    Benchmark,
    Contributor,
    DataInstance,
    Dataset,
    Generator,
    Ref,
)
from saps.downloaders import slicot


# Core adapted from John Burkardt's MIT-licensed BDF2 implementation.
# Keep its layout so the numerical changes remain easy to review.
# fmt: off
def bdf2 ( xp, f, tspan, y0, n, E, startup_factors, bdf2_factors ):

#*****************************************************************************80
#
## bdf2() solves the descriptor system E y' = f(t, y).
#
#  Discussion:
#
#    The first step uses backward Euler.
#    Precomputed LU factors are supplied for the two implicit step matrices.
#
#  Licensing:
#
#    This code is distributed under the MIT license.
#
#  Modified:
#
#    29 May 2022
#
#  Author:
#
#    John Burkardt
#
#  Input:
#
#    function handle f: evaluates the right hand side of the ODE.
#
#    real tspan(2): the starting and ending times.
#
#    real y0(m): the initial conditions.
#
#    integer n: the number of steps.
#
#    matrix E: the descriptor mass matrix.
#
#    startup_factors, bdf2_factors: (L, U, row_order, column_order).
#
#  Output:
#
#    t: time points; y: Python list of n+1 solution vectors.
#
  if isinstance(n, bool) or not isinstance(n, Integral) or n < 1:
    raise ValueError("n must be a positive integer.")
  if (len(tspan) != 2 or not xp.all(xp.isfinite(xp.asarray(tspan)))
      or tspan[1] <= tspan[0]):
    raise ValueError("tspan must contain two finite, increasing times.")
  n = int(n)
  y0 = xp.reshape(xp.asarray(y0, dtype=xp.float64), (-1,))

  t = xp.linspace ( tspan[0], tspan[1], n + 1 )
  y = []

  dt = ( tspan[1] - tspan[0] ) / float ( n )

  for i in range ( n + 1 ):

    if ( i == 0 ):

      y.append ( xp.asarray ( y0, copy=True ) )

    elif ( i == 1 ):

      to = t[i-1]
      yo = y[i-1]
      th = t[i]
      yh = xp.asarray ( yo, copy=True )

      rhs = E @ (yh - yo) - (th - to) * f ( th, yh )
      L, U, row_order, column_order = startup_factors
      correction = xp.linalg.solve ( L, xp.take(rhs, row_order, axis=0) )
      correction = xp.linalg.solve ( U, correction )
      yh = yh - xp.take(correction, column_order, axis=0)

      y.append ( yh )

    else:

      y1 = y[i-2]
      y2 = y[i-1]
      t3 = t[i]
      y3 = xp.asarray ( y[i-1], copy=True )

      rhs = E @ (3.0 * y3 - 4.0 * y2 + y1) - 2.0 * dt * f ( t3, y3 )
      L, U, row_order, column_order = bdf2_factors
      correction = xp.linalg.solve ( L, xp.take(rhs, row_order, axis=0) )
      correction = xp.linalg.solve ( U, correction )
      y3 = y3 - xp.take(correction, column_order, axis=0)

      y.append ( y3 )

    if i > 0 and not xp.all(xp.isfinite(y[i])):
      raise RuntimeError("Descriptor step produced a non-finite solution.")

  return t, y

# fmt: on


class DescriptorDAEDataset(Dataset):
    def __init__(
        self,
        name: str,
        *,
        pretty_name: str | None = None,
        description: str | None = None,
        suites: list[str] | None = None,
        E: Any | None = None,
        A: Any | None = None,
        B: Any | None = None,
        y0: list[float] | np.ndarray | None = None,
        u: list[float] | np.ndarray | None = None,
        source_name: str | None = None,
        t_max: float = 1.0,
        step: float = 0.01,
        ref_meta: dict[str, Any] | None = None,
    ):
        self._name = name
        self._pretty_name = pretty_name
        self._description = description
        self._suites = suites or []
        self.E = E
        self.A = A
        self.B = B
        self.y0 = None if y0 is None else np.asarray(y0, dtype=np.float64)
        self.u = None if u is None else np.asarray(u, dtype=np.float64)
        self.source_name = source_name
        self.t_max = t_max
        self.step = step
        self.ref_meta = ref_meta or {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def pretty_name(self) -> str:
        return self._pretty_name or self._name

    @property
    def description(self) -> str:
        if self._description is not None:
            return self._description
        if self.source_name is not None:
            return f"SLICOT descriptor-system DAE dataset {self.source_name}."
        return "Descriptor-system DAE dataset."

    @property
    def suites(self) -> list[str]:
        return self._suites

    @property
    def concepts(self) -> str:
        return """
<ccs2012>
<concept>
<concept_id>10002950.10003705.10003707</concept_id>
<concept_desc>Mathematics of computing~Solvers</concept_desc>
<concept_significance>500</concept_significance>
</concept>
<concept>
<concept_id>10002950.10003714.10003715</concept_id>
<concept_desc>Mathematics of computing~Numerical analysis</concept_desc>
<concept_significance>500</concept_significance>
</concept>
<concept>
<concept_id>10002950.10003714.10003716</concept_id>
<concept_desc>Mathematics of computing~Differential equations</concept_desc>
<concept_significance>300</concept_significance>
</concept>
</ccs2012>
"""

    @property
    def metadata(self) -> dict[str, Any]:
        metadata = super().metadata
        metadata["step"] = self.step
        metadata["t_max"] = self.t_max
        if self.source_name is not None:
            metadata["source_name"] = self.source_name
        return metadata


class _DescriptorDAEGenerator(Generator[DescriptorDAEDataset]):
    def generate(self, dataset: DescriptorDAEDataset) -> DataInstance:
        source_meta = None
        if dataset.source_name is not None:
            variables, source_meta = slicot.load_slicot_problem(dataset.source_name)
            variables = {key.lower(): value for key, value in variables.items()}
            for name in ("a", "b", "e"):
                if name not in variables:
                    raise ValueError(f"SLICOT MAT file is missing {name.upper()!r}.")
            A, B, E = variables["a"], variables["b"], variables["e"]
        else:
            if dataset.E is None or dataset.A is None or dataset.B is None:
                raise ValueError("DAE test datasets must define E, A and B.")
            A, B, E = dataset.A, dataset.B, dataset.E
        matrices = []
        for value in (A, E, B):
            if isinstance(value, pydata_sparse.SparseArray):
                value = value.to_scipy_sparse()
            matrices.append(
                value.astype(np.float64).tocsr()
                if scipy_sparse.issparse(value)
                else np.asarray(value, dtype=np.float64)
            )
        A, E, B = matrices
        for name, matrix in (("A", A), ("E", E)):
            if (
                matrix.ndim != 2
                or matrix.shape[0] != matrix.shape[1]
                or not matrix.shape[0]
            ):
                raise ValueError(f"{name} must be a nonempty square matrix.")
        if B.ndim == 1:
            B = B.reshape(-1, 1)
        if B.ndim != 2 or B.shape[0] != A.shape[0]:
            raise ValueError(f"B must have {A.shape[0]} rows.")
        if E.shape != A.shape:
            raise ValueError(f"E shape {E.shape} must match A shape {A.shape}.")

        input_count = B.shape[1]
        y0 = (
            np.zeros(A.shape[0], dtype=np.float64) if dataset.y0 is None else dataset.y0
        )
        u = np.ones(input_count, dtype=np.float64) if dataset.u is None else dataset.u
        if y0.shape != (A.shape[0],):
            raise ValueError(f"y0 must have shape {(A.shape[0],)}, got {y0.shape}.")
        if u.shape != (input_count,):
            raise ValueError(f"u must have shape {(input_count,)}, got {u.shape}.")

        meta = {
            "span": (0.0, dataset.t_max),
            "step": dataset.step,
            "y0": y0.tolist(),
            "input": u.tolist(),
            "order": A.shape[0],
            "input_count": input_count,
        }
        if dataset.source_name is not None:
            meta["source_name"] = dataset.source_name
        if source_meta is not None:
            meta["source"] = {
                key: value
                for key, value in source_meta.items()
                if isinstance(value, (str, int, float, bool, type(None)))
            }

        step = dataset.step
        if not np.isfinite(step) or step <= 0:
            raise ValueError("step must be finite and positive.")
        if not np.isfinite(dataset.t_max) or dataset.t_max <= 0:
            raise ValueError("t_max must be finite and positive.")
        # Store the exact grid used for the factors so the solver reuses it.
        n = max(1, int(np.ceil(dataset.t_max / step)))
        dt = dataset.t_max / n
        meta["n"] = n
        meta["dt"] = dt
        startup_lu = splu(scipy_sparse.csc_matrix(E - dt * A))
        bdf2_lu = splu(scipy_sparse.csc_matrix(3 * E - 2 * dt * A))
        # Pr @ matrix @ Pc = L @ U. Store gather indices for Pr and Pc.
        inputs = [
            E,
            A,
            B,
            -A,
            E,
            startup_lu.L,
            startup_lu.U,
            np.argsort(startup_lu.perm_r),
            startup_lu.perm_c.copy(),
            bdf2_lu.L,
            bdf2_lu.U,
            np.argsort(bdf2_lu.perm_r),
            bdf2_lu.perm_c.copy(),
        ]
        return DataInstance(
            inputs=[
                from_scipy(value.tocoo())
                if scipy_sparse.issparse(value)
                else from_numpy(value)
                for value in inputs
            ],
            meta=meta,
            ref_meta=dataset.ref_meta,
        )

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def concepts(self) -> str:
        return """
<ccs2012>
<concept>
<concept_id>10002950.10003705.10003707</concept_id>
<concept_desc>Mathematics of computing~Solvers</concept_desc>
<concept_significance>500</concept_significance>
</concept>
<concept>
<concept_id>10002950.10003714.10003715</concept_id>
<concept_desc>Mathematics of computing~Numerical analysis</concept_desc>
<concept_significance>500</concept_significance>
</concept>
<concept>
<concept_id>10002950.10003714.10003716</concept_id>
<concept_desc>Mathematics of computing~Differential equations</concept_desc>
<concept_significance>300</concept_significance>
</concept>
</ccs2012>
"""

    @property
    def authors(self) -> list[Contributor]:
        return [Contributor("Willow Ahrens", "willow.marie.ahrens@gmail.com")]

    @property
    def ai_disclosure(self) -> str:
        return (
            "Generative AI was used to help organize this benchmark module and checks."
        )


class DescriptorDAETestGenerator(_DescriptorDAEGenerator):
    @property
    def name(self) -> str:
        return "dae_test_inputs"

    @property
    def pretty_name(self) -> str:
        return "DAE Test Data Generator"

    @property
    def description(self) -> str:
        return "Inlined descriptor systems for DAE solver correctness tests."

    @property
    def suites(self) -> list[str]:
        return ["test", "trace"]

    @property
    def references(self) -> list[Ref]:
        return []

    @property
    def motivation(self) -> str:
        return "Uses a small singular-mass descriptor system to verify DAE steps."

    @property
    def datasets(self) -> list[DescriptorDAEDataset]:
        return [
            DescriptorDAEDataset(
                "test_slicot_descriptor_2",
                pretty_name="Tiny Descriptor DAE",
                description="Two-variable index-1 descriptor system.",
                suites=["test", "trace"],
                E=np.array([[1.0, 0.0], [0.0, 0.0]]),
                A=np.array([[-2.0, 1.0], [1.0, -1.0]]),
                B=np.array([[1.0], [0.0]]),
                y0=[0.0, 0.0],
                u=[1.0],
                t_max=0.4,
                step=0.1,
                ref_meta={
                    "check_discrete_residual": True,
                    "check_jacobians": True,
                    "residual_tol": 1e-7,
                },
            )
        ]


class SlicotDAEGenerator(_DescriptorDAEGenerator):
    @property
    def name(self) -> str:
        return "slicot_dae_inputs"

    @property
    def pretty_name(self) -> str:
        return "SLICOT DAE Data Generator"

    @property
    def description(self) -> str:
        return "Loads SLICOT descriptor-system benchmarks for implicit DAE solvers."

    @property
    def suites(self) -> list[str]:
        return []

    @property
    def references(self) -> list[Ref]:
        return [
            Ref(
                title="SLICOT benchmark examples for model reduction",
                authors=[],
                url=slicot.SLICOT_BENCHMARK_PAGE_URL,
            )
        ]

    @property
    def motivation(self) -> str:
        return (
            "Uses SLICOT models with an explicit E matrix as descriptor systems. "
            "Implicit DAE steps solve linear systems built from residual Jacobians."
        )

    @property
    def datasets(self) -> list[DescriptorDAEDataset]:
        return [
            DescriptorDAEDataset(
                "tline",
                source_name="tline.mat",
                pretty_name="SLICOT Transmission line model",
                description="SLICOT example of a transmission line model.",
                suites=["standard"],
                t_max=0.02,
                step=0.01,
            ),
            DescriptorDAEDataset(
                "peec",
                source_name="peec.mat",
                pretty_name="SLICOT PEEC model",
                description="SLICOT partial element equivalent circuit model.",
                suites=["standard"],
                t_max=0.02,
                step=0.01,
            ),
            DescriptorDAEDataset(
                "heat-disc",
                source_name="heat-disc.mat",
                pretty_name="SLICOT Heat equation (discrete case)",
                description="SLICOT discretization of the previous equation.",
                suites=["standard"],
                t_max=0.02,
                step=0.01,
            ),
            DescriptorDAEDataset(
                "MNA_1",
                source_name="MNA_1.mat",
                pretty_name="SLICOT MNA example - 1",
                description="SLICOT Modified Nodal Analysis model.",
                suites=["standard"],
                t_max=0.02,
                step=0.01,
            ),
            DescriptorDAEDataset(
                "MNA_2",
                source_name="MNA_2.mat",
                pretty_name="SLICOT MNA example - 2",
                description="SLICOT Modified Nodal Analysis model.",
                suites=["standard"],
                t_max=0.02,
                step=0.01,
            ),
            DescriptorDAEDataset(
                "MNA_3",
                source_name="MNA_3.mat",
                pretty_name="SLICOT MNA example - 3",
                description="SLICOT Modified Nodal Analysis model.",
                suites=["standard"],
                t_max=0.02,
                step=0.01,
            ),
            DescriptorDAEDataset(
                "MNA_4",
                source_name="MNA_4.mat",
                pretty_name="SLICOT MNA example - 4",
                description="SLICOT Modified Nodal Analysis model.",
                suites=["standard"],
                t_max=0.02,
                step=0.01,
            ),
            DescriptorDAEDataset(
                "MNA_5",
                source_name="MNA_5.mat",
                pretty_name="SLICOT MNA example - 5",
                description="SLICOT Modified Nodal Analysis model.",
                suites=["standard"],
                t_max=0.02,
                step=0.01,
            ),
        ]


class _DescriptorDAEBenchmark(Benchmark):
    @property
    def suites(self) -> list[str]:
        return []

    @property
    def concepts(self) -> str:
        return """
<ccs2012>
<concept>
<concept_id>10002950.10003705.10003707</concept_id>
<concept_desc>Mathematics of computing~Solvers</concept_desc>
<concept_significance>500</concept_significance>
</concept>
<concept>
<concept_id>10002950.10003714.10003715</concept_id>
<concept_desc>Mathematics of computing~Numerical analysis</concept_desc>
<concept_significance>500</concept_significance>
</concept>
<concept>
<concept_id>10002950.10003714.10003716</concept_id>
<concept_desc>Mathematics of computing~Differential equations</concept_desc>
<concept_significance>300</concept_significance>
</concept>
</ccs2012>
"""

    @property
    def authors(self) -> list[Contributor]:
        return [Contributor("Willow Ahrens", "willow.marie.ahrens@gmail.com")]

    @property
    def references(self) -> list[Ref]:
        return [
            *SlicotDAEGenerator().references,
            Ref(
                title="BDF2: Backward Differentiation Formula of Order 2",
                authors=[Author("John Burkardt")],
                url="https://people.sc.fsu.edu/~jburkardt/py_src/bdf2/bdf2.py",
            ),
        ]

    @property
    def ai_disclosure(self) -> str:
        return (
            "Generative AI was used to adapt the found benchmark code to the "
            "benchmark format, with minimal changes to the benchmark code itself."
        )

    @property
    def motivation(self) -> str:
        return "Solves descriptor-form DAEs using explicit residual Jacobians."

    @property
    def generators(self) -> list[Generator[DescriptorDAEDataset]]:
        return [DescriptorDAETestGenerator(), SlicotDAEGenerator()]

    def check(self, param):
        for item in self._output:
            assert isinstance(item, BinsparseTensor), (
                "Output must be in binsparse format"
            )
        if not self._ref_meta:
            return

        matrices = []
        for item in self._input[:5]:
            try:
                matrices.append(to_numpy(item))
            except TypeError:
                matrices.append(np.asarray(to_sparse(item).todense()))
        E, A, B, jac_y, jac_yp = matrices
        if self._ref_meta.get("check_jacobians"):
            np.testing.assert_allclose(jac_y, -A)
            np.testing.assert_allclose(jac_yp, E)
        if not self._ref_meta.get("check_discrete_residual"):
            return

        outputs = []
        for item in self._output:
            try:
                outputs.append(to_numpy(item))
            except TypeError:
                outputs.append(np.asarray(to_sparse(item).todense()))
        time, y, yp = outputs
        forcing = B @ np.asarray(self._meta["input"], dtype=np.float64)
        tol = self._ref_meta.get("residual_tol", 1e-8)
        assert y.shape == (len(time), A.shape[0])
        assert yp.shape == y.shape
        np.testing.assert_allclose(y[0], self._meta["y0"])
        dt = time[1] - time[0]
        np.testing.assert_allclose(np.diff(time), dt)
        np.testing.assert_allclose(yp[1], (y[1] - y[0]) / dt, atol=tol)
        np.testing.assert_allclose(
            yp[2:], (3 * y[2:] - 4 * y[1:-1] + y[:-2]) / (2 * dt), atol=tol
        )
        for step_index in range(1, len(time)):
            residual = jac_yp @ yp[step_index] + jac_y @ y[step_index] - forcing
            assert np.linalg.norm(residual) < tol, (
                f"{self.name} residual too high at step {step_index}: {residual}"
            )


class SlicotDAEBDF(_DescriptorDAEBenchmark):
    @property
    def name(self) -> str:
        return "slicot_dae_bdf"

    @property
    def pretty_name(self) -> str:
        return "SLICOT DAE BDF"

    @property
    def description(self) -> str:
        return "Fixed-step BDF2 for SLICOT DAEs with LU factors from the generator."

    def benchmark(self, xp, data, meta):
        E, A, B, jac_y, jac_yp = data[:5]
        L_start, U_start, rows_start, cols_start, L_bdf, U_bdf, rows_bdf, cols_bdf = (
            data[5:]
        )
        start, stop = meta["span"]
        n, dt = meta["n"], meta["dt"]
        forcing = B @ xp.asarray(meta["input"], dtype=xp.float64)
        time, y = bdf2(
            xp,
            lambda t, state: A @ state + forcing,
            (start, stop),
            meta["y0"],
            n,
            E=jac_yp,
            startup_factors=(L_start, U_start, rows_start, cols_start),
            bdf2_factors=(L_bdf, U_bdf, rows_bdf, cols_bdf),
        )
        # No initial derivative is supplied; row zero is a placeholder.
        yp = [xp.zeros_like(y[0]), (y[1] - y[0]) / dt]
        yp.extend(
            (3 * y[i] - 4 * y[i - 1] + y[i - 2]) / (2 * dt) for i in range(2, len(y))
        )
        return [time, xp.stack(y, axis=0), xp.stack(yp, axis=0)]
