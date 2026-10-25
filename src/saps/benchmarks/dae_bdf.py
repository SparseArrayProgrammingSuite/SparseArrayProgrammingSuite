"""Fixed-step BDF2 benchmarks for linear descriptor systems E y' = A y + B u."""

from numbers import Integral
from typing import Any

import numpy as np
import scipy.sparse as scipy_sparse
from scipy.sparse.linalg import splu

import sparse as pydata_sparse
from binsparse import BinsparseTensor
from binsparse.conversions import from_numpy, from_scipy, to_numpy, to_scipy, to_sparse

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


class DAEBDFDataset(Dataset):
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


class _DescriptorDAEGenerator(Generator[DAEBDFDataset]):
    def generate(self, dataset: DAEBDFDataset) -> DataInstance:
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


class DAEBDFTestGenerator(_DescriptorDAEGenerator):
    @property
    def name(self) -> str:
        return "dae_bdf_test"

    @property
    def pretty_name(self) -> str:
        return (
            "Differential-Algebraic Equation (DAE) Backward Differentiation Formula"
            " (BDF) Test"
        )

    @property
    def description(self) -> str:
        return "Inlined descriptor systems for DAE solver correctness tests."

    @property
    def suites(self) -> list[str]:
        return ["test"]

    @property
    def references(self) -> list[Ref]:
        return []

    @property
    def motivation(self) -> str:
        return "Uses a small singular-mass descriptor system to verify DAE steps."

    @property
    def datasets(self) -> list[DAEBDFDataset]:
        return [
            DAEBDFDataset(
                "tiny_descriptor",
                pretty_name="Tiny Descriptor",
                description="Two-variable index-1 descriptor system.",
                suites=["test"],
                E=np.array([[1.0, 0.0], [0.0, 0.0]]),
                A=np.array([[-2.0, 1.0], [1.0, -1.0]]),
                B=np.array([[1.0], [0.0]]),
                y0=[0.0, 0.0],
                u=[1.0],
                t_max=0.4,
                step=0.1,
            )
        ]


class DAEBDFSLICOTGenerator(_DescriptorDAEGenerator):
    @property
    def name(self) -> str:
        return "dae_bdf_slicot"

    @property
    def pretty_name(self) -> str:
        return (
            "Differential-Algebraic Equation (DAE) Backward Differentiation Formula"
            " (BDF) SLICOT"
        )

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
    def datasets(self) -> list[DAEBDFDataset]:
        return [
            DAEBDFDataset(
                "tline",
                source_name="tline.mat",
                description="SLICOT example of a transmission line model.",
                suites=["standard", "trace"],
                t_max=0.02,
                step=0.01,
            ),
            DAEBDFDataset(
                "peec",
                source_name="peec.mat",
                description="SLICOT partial element equivalent circuit model.",
                suites=["standard", "trace"],
                t_max=0.02,
                step=0.01,
            ),
            DAEBDFDataset(
                "heat-disc",
                source_name="heat-disc.mat",
                description="SLICOT discretization of the previous equation.",
                suites=["standard", "trace"],
                t_max=0.02,
                step=0.01,
            ),
            DAEBDFDataset(
                "MNA_1",
                source_name="MNA_1.mat",
                description="SLICOT Modified Nodal Analysis model.",
                suites=["standard", "trace"],
                t_max=0.02,
                step=0.01,
            ),
            DAEBDFDataset(
                "MNA_2",
                source_name="MNA_2.mat",
                description="SLICOT Modified Nodal Analysis model.",
                suites=["standard", "trace"],
                t_max=0.02,
                step=0.01,
            ),
            DAEBDFDataset(
                "MNA_3",
                source_name="MNA_3.mat",
                description="SLICOT Modified Nodal Analysis model.",
                suites=["standard", "trace", "train"],
                t_max=0.02,
                step=0.01,
            ),
            DAEBDFDataset(
                "MNA_4",
                source_name="MNA_4.mat",
                description="SLICOT Modified Nodal Analysis model.",
                suites=["standard", "trace"],
                t_max=0.02,
                step=0.01,
            ),
            DAEBDFDataset(
                "MNA_5",
                source_name="MNA_5.mat",
                description="SLICOT Modified Nodal Analysis model.",
                suites=["standard", "trace"],
                t_max=0.02,
                step=0.01,
            ),
        ]


_EPS = np.finfo(np.float64).eps
# Each rounding can also lose this much to underflow (Higham, section 2.1).
_UNDERFLOW = np.finfo(np.float64).smallest_subnormal


def _gamma(n):
    """Higham's gamma_n, with eps as in saps.util.error_bounds."""
    return n * _EPS / (1 - n * _EPS)


def _matrix(array: BinsparseTensor):
    """A dense array as NumPy, or a sparse matrix as SciPy CSR."""
    try:
        return to_numpy(array)
    except TypeError:
        return to_scipy(array).tocsr()


def _step_magnitude(abs_E, abs_A, forcing_abs, coefficients, scale, states):
    """|E| sum |c_j| |y_j| + |scale| (|A| |y_0| + |B| |u|): the terms of a step's
    residual E sum c_j y_j - scale (A y_0 + B u), made nonnegative."""
    weighted = sum(
        abs(c) * np.abs(state) for c, state in zip(coefficients, states, strict=True)
    )
    return abs_E @ weighted + abs(scale) * (abs_A @ np.abs(states[0]) + forcing_abs)


def _dense(array: BinsparseTensor):
    try:
        return to_numpy(array)
    except TypeError:
        return np.asarray(to_sparse(array).todense())


def _row_counts(matrix):
    if scipy_sparse.issparse(matrix):
        return np.diff(scipy_sparse.csr_array(matrix).indptr)
    return np.count_nonzero(matrix, axis=-1)


class _DescriptorDAEBenchmark(Benchmark):
    @property
    def suites(self) -> list[str]:
        return ["standard-timestepping"]

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
            *DAEBDFSLICOTGenerator().references,
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
    def generators(self) -> list[Generator[DAEBDFDataset]]:
        return [DAEBDFTestGenerator(), DAEBDFSLICOTGenerator()]

    def check(self, param):
        for item in self._output:
            assert isinstance(
                item, BinsparseTensor
            ), "Output must be in binsparse format"
        inputs = [_matrix(item) for item in self._input]
        E, A, B, jac_y, jac_yp = inputs[:5]
        startup, bdf2 = inputs[5:9], inputs[9:13]
        assert abs(jac_y + A).max() == 0, "jac_y must be -A"
        assert abs(jac_yp - E).max() == 0, "jac_yp must be E"

        time, y, yp = (_dense(item) for item in self._output)
        start, stop = self._meta["span"]
        n, dt = self._meta["n"], self._meta["dt"]
        h = (stop - start) / float(n)
        assert y.shape == (n + 1, A.shape[0]) and yp.shape == y.shape
        np.testing.assert_allclose(
            time, np.linspace(start, stop, n + 1), rtol=_gamma(2), atol=0
        )
        np.testing.assert_array_equal(y[0], self._meta["y0"])

        abs_E, abs_A = abs(E), abs(A)
        u = np.asarray(self._meta["input"], dtype=np.float64)
        forcing = B @ u
        forcing_abs = abs(B) @ np.abs(u)
        rounds = max(
            int(np.max(_row_counts(matrix), initial=0))
            for matrix in (E, A, B, *startup[:2], *bdf2[:2])
        )
        gamma = _gamma(rounds + 6)

        # yp is a backward difference of y.
        np.testing.assert_array_equal(yp[0], 0)
        differences = [(y[1] - y[0]) / dt]
        magnitudes = [(np.abs(y[1]) + np.abs(y[0])) / dt]
        for i in range(2, n + 1):
            differences.append((3 * y[i] - 4 * y[i - 1] + y[i - 2]) / (2 * dt))
            magnitudes.append(
                (3 * np.abs(y[i]) + 4 * np.abs(y[i - 1]) + np.abs(y[i - 2])) / (2 * dt)
            )
        error = np.abs(yp[1:] - np.array(differences))
        assert np.all(
            error <= 2 * _gamma(4) * np.array(magnitudes)
        ), f"yp isn't the backward difference of y for {param.dataset.name}"

        # Each step solves its implicit equation with one correction through
        # the given LU factors of its matrix. Its residual can be no larger
        # than the rounding in that right-hand side and in this residual, the
        # backward error of the triangular solves and the factorization
        # (gamma |L| |U| on the correction), the factorization's own step size
        # if it differs from the step's, and rounding y_i.
        for i in range(1, n + 1):
            if i == 1:
                coefficients, scale, factor_scale = (
                    (1.0, -1.0, 0.0),
                    time[1] - time[0],
                    dt,
                )
                L, U, rows, cols = startup
            else:
                coefficients, scale, factor_scale = (3.0, -4.0, 1.0), 2.0 * h, 2.0 * dt
                L, U, rows, cols = bdf2
            states = [y[i], y[i - 1], y[i - 2] if i > 1 else y[0]]
            combination = sum(
                c * state for c, state in zip(coefficients, states, strict=True)
            )
            residual = E @ combination - scale * (A @ y[i] + forcing)
            terms = (abs_E, abs_A, forcing_abs, coefficients, scale)
            # The step's right-hand side evaluates the residual at its guess,
            # y[i - 1].
            magnitude = _step_magnitude(*terms, states) + _step_magnitude(
                *terms, [y[i - 1], *states[1:]]
            )

            correction = y[i - 1] - y[i]
            z = np.empty_like(correction)
            z[cols] = np.abs(correction)
            solve = np.empty_like(correction)
            solve[rows] = abs(L) @ (abs(U) @ z)
            bound = (
                gamma * (magnitude + 2 * solve)
                + 4 * (rounds + 6) * _UNDERFLOW
                + abs(scale - factor_scale) * (abs_A @ np.abs(correction))
                + _gamma(1)
                * (
                    coefficients[0] * abs_E @ np.abs(y[i])
                    + abs(scale) * abs_A @ np.abs(y[i])
                )
            )
            excess = np.abs(residual) - bound
            worst = int(np.argmax(excess))
            assert excess[worst] <= 0, (
                f"{self.name} step {i} of {param.dataset.name} has residual "
                f"{abs(residual[worst]):.3g} in row {worst}, more than rounding "
                f"allows ({bound[worst]:.3g})"
            )


class DAEBDFBenchmark(_DescriptorDAEBenchmark):
    @property
    def name(self) -> str:
        return "dae_bdf"

    @property
    def pretty_name(self) -> str:
        return (
            "Differential-Algebraic Equation (DAE) Backward Differentiation Formula"
            " (BDF)"
        )

    @property
    def description(self) -> str:
        return "Fixed-step BDF2 for SLICOT DAEs with LU factors from the generator."

    def benchmark(
        self,
        xp,
        meta,
        E,
        A,
        B,
        jac_y,
        jac_yp,
        L_start,
        U_start,
        rows_start,
        cols_start,
        L_bdf,
        U_bdf,
        rows_bdf,
        cols_bdf,
    ):
        start, stop = meta["span"]
        n, dt = meta["n"], meta["dt"]
        forcing = B @ xp.asarray(meta["input"], dtype=xp.float64)
        tspan = (start, stop)
        y0 = meta["y0"]
        startup_factors = (L_start, U_start, rows_start, cols_start)
        bdf2_factors = (L_bdf, U_bdf, rows_bdf, cols_bdf)
        # Core adapted from John Burkardt's MIT-licensed BDF2 implementation.
        # Keep its layout so the numerical changes remain easy to review.
        # fmt: off

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

        t = xp.linspace ( tspan[0], tspan[1], n + 1, dtype=xp.float64 )
        y = []

        h = ( tspan[1] - tspan[0] ) / float ( n )

        for i in range ( n + 1 ):

          if ( i == 0 ):

            y.append ( xp.asarray ( y0, copy=True ) )

          elif ( i == 1 ):

            to = t[i-1]
            yo = y[i-1]
            th = t[i]
            yh = xp.asarray ( yo, copy=True )

            rhs = jac_yp @ (yh - yo) - (th - to) * ( A @ yh + forcing )
            L, U, row_order, column_order = startup_factors
            correction = xp.linalg.solve ( L, xp.take(rhs, row_order, axis=0) )
            correction = xp.linalg.solve ( U, correction )
            yh = yh - xp.take(correction, column_order, axis=0)

            y.append ( yh )

          else:

            y1 = y[i-2]
            y2 = y[i-1]
            t3 = t[i]  # noqa: F841
            y3 = xp.asarray ( y[i-1], copy=True )

            rhs = jac_yp @ (3.0 * y3 - 4.0 * y2 + y1) - 2.0 * h * ( A @ y3 + forcing )
            L, U, row_order, column_order = bdf2_factors
            correction = xp.linalg.solve ( L, xp.take(rhs, row_order, axis=0) )
            correction = xp.linalg.solve ( U, correction )
            y3 = y3 - xp.take(correction, column_order, axis=0)

            y.append ( y3 )

          if i > 0 and not xp.all(xp.isfinite(y[i])):
            raise RuntimeError("Descriptor step produced a non-finite solution.")

        time, y = t, y

        # fmt: on
        # No initial derivative is supplied; row zero is a placeholder.
        yp = [xp.zeros_like(y[0]), (y[1] - y[0]) / dt]
        yp.extend(
            (3 * y[i] - 4 * y[i - 1] + y[i - 2]) / (2 * dt) for i in range(2, len(y))
        )
        return time, xp.stack(y, axis=0), xp.stack(yp, axis=0)
