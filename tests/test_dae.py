import pytest

import numpy as np
import scipy.sparse as sp

from binsparse.conversions import from_numpy, to_scipy

from frameworks.saps_numpy import NumpyFramework
from frameworks.saps_scipy import SciPyFramework
from frameworks.saps_smart import SmartSparseFramework
from frameworks.saps_sparse import PyDataSparseFramework
from saps.benchmark import Param
from saps.benchmarks import dae_bdf
from saps.benchmarks.dae_bdf import (
    DAEBDFBenchmark,
    DAEBDFDataset,
    DAEBDFSLICOTGenerator,
    DAEBDFSuiteSparseDataset,
    DAEBDFSuiteSparseGenerator,
    DAEBDFTestGenerator,
    _dense,
)
from saps.downloaders.suitesparse import SUITESPARSE_PRIMARY


@pytest.mark.parametrize("framework", [NumpyFramework, PyDataSparseFramework])
def test_descriptor_benchmark(framework):
    benchmark = DAEBDFBenchmark()
    param = next(
        p for p in benchmark.params if isinstance(p.generator, DAEBDFTestGenerator)
    )
    benchmark.setup(param, use_cache=False, xp=framework())
    benchmark.run(param)
    benchmark.check(param)


@pytest.mark.parametrize("sparse", [False, True])
def test_singular_mass_with_inconsistent_initial_state(sparse):
    E = np.diag([2.0, 0.0])
    A = np.diag([-2.0, -1.0])
    B = np.array([[2.0], [3.0]])
    generator = DAEBDFTestGenerator()
    problem = generator.generate(
        DAEBDFDataset(
            "singular",
            E=sp.csr_matrix(E) if sparse else E,
            A=sp.csr_matrix(A) if sparse else A,
            B=sp.csr_matrix(B) if sparse else B,
            y0=[0.0, 0.0],
            u=[1.0],
            t_max=1.0,
            step=0.03,
        )
    )
    xp = PyDataSparseFramework() if sparse else NumpyFramework()
    data = [xp.from_binsparse(value) for value in problem.inputs]
    time, y, yp = [
        np.asarray(value.todense() if hasattr(value, "todense") else value)
        for value in DAEBDFBenchmark().benchmark(xp, problem.meta, *data)
    ]
    assert time[-1] == 1.0
    np.testing.assert_allclose(y[1:, 1], 3.0)
    np.testing.assert_allclose(
        (E @ yp[1:].T - A @ y[1:].T).T, np.tile(B[:, 0], (len(time) - 1, 1)), atol=1e-12
    )
    np.testing.assert_allclose(y[-1, 0], 1 - np.exp(-1), atol=3e-4)


def test_descriptor_second_order_convergence():
    errors = []
    for n in [20, 40, 80]:
        problem = DAEBDFTestGenerator().generate(
            DAEBDFDataset(
                "decay",
                E=np.eye(1),
                A=-np.eye(1),
                B=np.zeros((1, 1)),
                y0=[1.0],
                t_max=1.0,
                step=1 / n,
            )
        )
        data = [NumpyFramework().from_binsparse(value) for value in problem.inputs]
        _, y, _ = DAEBDFBenchmark().benchmark(NumpyFramework(), problem.meta, *data)
        errors.append(abs(y[-1, 0] - np.exp(-1)))
    assert errors[0] / errors[1] > 3.8
    assert errors[1] / errors[2] > 3.8


@pytest.mark.parametrize("n", [0, -1, 1.5, True])
def test_invalid_step_count(n):
    problem = DAEBDFTestGenerator().generate(
        DAEBDFDataset(
            "decay",
            E=np.eye(1),
            A=-np.eye(1),
            B=np.zeros((1, 1)),
            y0=[1.0],
            t_max=1.0,
            step=0.5,
        )
    )
    data = [NumpyFramework().from_binsparse(value) for value in problem.inputs]
    with pytest.raises(ValueError, match="positive integer"):
        DAEBDFBenchmark().benchmark(NumpyFramework(), {**problem.meta, "n": n}, *data)


def test_slicot_datasets_are_delisted():
    benchmark_generators = DAEBDFBenchmark().generators
    assert any(isinstance(g, DAEBDFSLICOTGenerator) for g in benchmark_generators)
    datasets = DAEBDFSLICOTGenerator().datasets
    assert len(datasets) == 8
    assert all(dataset.suites == [] for dataset in datasets)


def test_suitesparse_datasets_exercise_bdf2():
    datasets = DAEBDFSuiteSparseGenerator().datasets
    assert datasets
    assert all(d.t_max > d.step for d in datasets)
    assert len({d.name for d in datasets}) == len(datasets)


def _serve_suitesparse(monkeypatch, files):
    """Serve a model reduction problem's matrices in place of SuiteSparse's."""
    calls = []

    def fake_load(source_name, names):
        calls.append((source_name, names))
        meta = {"dataset_name": source_name, "matrix_group": "Fake", "n": 2}
        return {name: sp.csr_matrix(files[name]) for name in names}, meta

    monkeypatch.setattr(dae_bdf, "load_suitesparse_model_reduction", fake_load)
    return calls


def test_suitesparse_first_order_problem_keeps_its_sparse_matrices(monkeypatch):
    E = np.diag([2.0, 0.0])
    A = np.array([[-2.0, 1.0], [0.0, -1.0]])
    B = np.array([[2.0], [3.0]])
    calls = _serve_suitesparse(monkeypatch, {"E": E, SUITESPARSE_PRIMARY: A, "B": B})
    dataset = DAEBDFSuiteSparseDataset(
        "Fake/first", E="E", A=SUITESPARSE_PRIMARY, B="B", t_max=0.05, step=0.01
    )

    problem = DAEBDFSuiteSparseGenerator().generate(dataset)

    assert calls == [("Fake/first", ("E", SUITESPARSE_PRIMARY, "B"))]
    E_out, A_out, B_out = (to_scipy(value) for value in problem.inputs[:3])
    np.testing.assert_array_equal(E_out.toarray(), E)
    np.testing.assert_array_equal(A_out.toarray(), A)
    np.testing.assert_array_equal(B_out.toarray(), B)
    assert problem.meta["source"]["dataset_name"] == "Fake/first"
    assert dataset.name == "first"
    assert dataset.metadata["system"] == {
        "E": "E",
        "A": SUITESPARSE_PRIMARY,
        "B": "B",
    }


@pytest.mark.parametrize("damped", [False, True])
def test_suitesparse_second_order_problem_becomes_first_order(monkeypatch, damped):
    K = np.array([[4.0, -1.0], [-1.0, 3.0]])
    M = np.diag([2.0, 1.0])
    D = np.array([[0.5, 0.0], [0.0, 0.25]])
    B = np.array([[1.0], [0.0]])
    _serve_suitesparse(monkeypatch, {SUITESPARSE_PRIMARY: K, "M": M, "E": D, "B": B})
    dataset = DAEBDFSuiteSparseDataset(
        "Fake/second",
        M="M",
        D="E" if damped else None,
        K=SUITESPARSE_PRIMARY,
        B="B",
        t_max=0.05,
        step=0.01,
    )

    problem = DAEBDFSuiteSparseGenerator().generate(dataset)

    E_out, A_out, B_out = (to_scipy(value).toarray() for value in problem.inputs[:3])
    zero, identity = np.zeros((2, 2)), np.eye(2)
    np.testing.assert_array_equal(E_out, np.block([[identity, zero], [zero, M]]))
    np.testing.assert_array_equal(
        A_out, np.block([[zero, identity], [-K, -D if damped else zero]])
    )
    np.testing.assert_array_equal(B_out, np.vstack([np.zeros((2, 1)), B]))
    assert problem.meta["order"] == 4
    assert "second-order" in dataset.description


@pytest.mark.parametrize("framework", [NumpyFramework, SciPyFramework])
def test_suitesparse_complex_problem_runs_and_checks(monkeypatch, framework):
    K = np.array([[4.0, -1.0], [-1.0, 3.0]]) * (1 + 0.02j)
    _serve_suitesparse(
        monkeypatch,
        {SUITESPARSE_PRIMARY: K, "M": np.eye(2), "B": np.array([[1.0], [0.0]])},
    )
    generator = DAEBDFSuiteSparseGenerator()
    dataset = DAEBDFSuiteSparseDataset(
        "Fake/complex", M="M", K=SUITESPARSE_PRIMARY, B="B", t_max=0.05, step=0.01
    )
    benchmark = DAEBDFBenchmark()
    param = Param(generator, dataset)

    benchmark.setup(param, use_cache=False, xp=framework())
    benchmark.run(param)
    benchmark.check(param)

    y = _dense(benchmark._output[1])
    assert np.iscomplexobj(y)
    assert np.abs(y[-1].imag).max() > 0


def test_suitesparse_dataset_names_one_system_form():
    with pytest.raises(ValueError, match="either E and A, or M and K"):
        DAEBDFSuiteSparseDataset(
            "Fake/both", E="E", A="A", M="M", K="K", B="B", t_max=0.05, step=0.01
        )
    with pytest.raises(ValueError, match="either E and A, or M and K"):
        DAEBDFSuiteSparseDataset("Fake/neither", B="B", t_max=0.05, step=0.01)


def test_lu_permutations():
    # Nontrivial row and column permutations expose gather/scatter mistakes.
    matrix = sp.csc_matrix(
        [
            [1.0, 2.0, 0.0, 4.0],
            [1.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 2.0, 1.0],
            [2.0, 2.0, 1.0, 0.0],
        ]
    )
    rhs = np.array([1.0, 2.0, 3.0, 4.0])
    problem = DAEBDFTestGenerator().generate(
        DAEBDFDataset(
            "permuted",
            E=matrix,
            A=sp.csc_matrix(matrix.shape),
            B=rhs,
            t_max=1.0,
            step=1.0,
        )
    )
    data = [NumpyFramework().from_binsparse(value) for value in problem.inputs]
    assert not np.array_equal(data[7], np.arange(4))
    assert not np.array_equal(data[8], np.arange(4))
    for offset, coefficient in ((5, 1), (9, 3)):
        L, U, rows, cols = data[offset : offset + 4]
        np.testing.assert_array_equal(np.sort(rows), np.arange(4))
        np.testing.assert_array_equal(np.sort(cols), np.arange(4))
        permuted = (coefficient * matrix).toarray()[rows][:, np.argsort(cols)]
        np.testing.assert_allclose(permuted, L @ U)
    _, y, _ = DAEBDFBenchmark().benchmark(NumpyFramework(), problem.meta, *data)
    np.testing.assert_allclose(matrix @ y[1], rhs, atol=1e-12)


def test_factorization_only_in_generator(monkeypatch):
    import saps.benchmarks.dae_bdf as dae

    benchmark = DAEBDFBenchmark()
    param = next(
        p for p in benchmark.params if isinstance(p.generator, DAEBDFTestGenerator)
    )
    calls = []
    original = dae.splu

    def count(matrix):
        calls.append(matrix)
        return original(matrix)

    monkeypatch.setattr(dae, "splu", count)
    benchmark.setup(param, use_cache=False, xp=PyDataSparseFramework())
    assert len(calls) == 2
    monkeypatch.setattr(dae, "splu", lambda *args, **kwargs: pytest.fail("timed LU"))
    benchmark.run(param)
    benchmark.check(param)


def test_timed_solver_does_not_use_host_array_libraries():
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(DAEBDFBenchmark.benchmark)))
    assert not any(
        isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store)
        for node in ast.walk(tree)
    )
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert not names.intersection(
        {
            "np",
            "scipy_sparse",
            "pydata_sparse",
            "splu",
            "spsolve_triangular",
            "from_numpy",
            "from_scipy",
            "to_numpy",
            "to_sparse",
        }
    )


@pytest.mark.parametrize(
    "framework_cls",
    [NumpyFramework, SciPyFramework, PyDataSparseFramework, SmartSparseFramework],
)
def test_dae_check_verifies_every_step(framework_cls):
    benchmark = DAEBDFBenchmark()
    param = next(p for p in benchmark.params if p.dataset.name == "tiny_descriptor")
    benchmark.setup(param, xp=framework_cls(), use_cache=False)
    benchmark.run(param)
    benchmark.check(param)

    time, y, yp = benchmark._output
    states = _dense(y).copy()
    states[2, 0] *= 1 + 1e-9
    benchmark._output = [time, from_numpy(states), yp]
    # The changed state breaks yp's backward difference as well as the step.
    with pytest.raises(AssertionError, match="backward difference|more than rounding"):
        benchmark.check(param)
