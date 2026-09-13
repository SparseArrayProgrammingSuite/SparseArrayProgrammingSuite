import pytest

import numpy as np
import scipy.sparse as sp

from frameworks.saps_numpy import NumpyFramework
from frameworks.saps_sparse import PyDataSparseFramework
from saps.benchmarks.dae import (
    DescriptorDAEDataset,
    DescriptorDAETestGenerator,
    SlicotDAEBDF,
    SlicotDAEGenerator,
    bdf2,
)


@pytest.mark.parametrize("framework", [NumpyFramework, PyDataSparseFramework])
def test_descriptor_benchmark(framework):
    benchmark = SlicotDAEBDF()
    param = next(
        p
        for p in benchmark.params
        if isinstance(p.generator, DescriptorDAETestGenerator)
    )
    benchmark.setup(param, use_cache=False, xp=framework())
    benchmark.run(param)
    benchmark.check(param)


@pytest.mark.parametrize("sparse", [False, True])
def test_singular_mass_with_inconsistent_initial_state(sparse):
    E = np.diag([2.0, 0.0])
    A = np.diag([-2.0, -1.0])
    B = np.array([[2.0], [3.0]])
    generator = DescriptorDAETestGenerator()
    problem = generator.generate(
        DescriptorDAEDataset(
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
        for value in SlicotDAEBDF().benchmark(xp, data, problem.meta)
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
        problem = DescriptorDAETestGenerator().generate(
            DescriptorDAEDataset(
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
        _, y, _ = SlicotDAEBDF().benchmark(NumpyFramework(), data, problem.meta)
        errors.append(abs(y[-1, 0] - np.exp(-1)))
    assert errors[0] / errors[1] > 3.8
    assert errors[1] / errors[2] > 3.8


@pytest.mark.parametrize("n", [0, -1, 1.5, True])
def test_invalid_step_count(n):
    with pytest.raises(ValueError, match="positive integer"):
        bdf2(
            NumpyFramework(),
            lambda t, y: -y,
            (0.0, 1.0),
            [1.0],
            n,
            E=np.eye(1),
            startup_factors=(),
            bdf2_factors=(),
        )


def test_slicot_datasets_exercise_bdf2():
    assert all(d.t_max > d.step for d in SlicotDAEGenerator().datasets)


def test_slicot_dae_generator_uses_explicit_descriptor_problems():
    datasets = SlicotDAEGenerator().datasets

    assert [dataset.source_name for dataset in datasets] == [
        "tline.mat",
        "peec.mat",
        "heat-disc.mat",
        "MNA_1.mat",
        "MNA_2.mat",
        "MNA_3.mat",
        "MNA_4.mat",
        "MNA_5.mat",
    ]
    assert all(dataset.suites == ["standard"] for dataset in datasets)


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
    problem = DescriptorDAETestGenerator().generate(
        DescriptorDAEDataset(
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
    _, y, _ = SlicotDAEBDF().benchmark(NumpyFramework(), data, problem.meta)
    np.testing.assert_allclose(matrix @ y[1], rhs, atol=1e-12)


def test_factorization_only_in_generator(monkeypatch):
    import saps.benchmarks.dae as dae

    benchmark = SlicotDAEBDF()
    param = next(
        p
        for p in benchmark.params
        if isinstance(p.generator, DescriptorDAETestGenerator)
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

    for function in (bdf2, SlicotDAEBDF.benchmark):
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
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
