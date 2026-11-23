import pytest

import numpy as np
import scipy.sparse as sp

from binsparse.conversions import to_scipy

from frameworks.saps_scipy import SciPyFramework
from saps.benchmark import Param
from saps.benchmarks import ode
from saps.benchmarks.ode import (
    BackwardEulerBenchmark,
    ForwardEulerBenchmark,
    ODESuiteSparseDataset,
    ODESuiteSparseGenerator,
    RK4Benchmark,
)
from saps.downloaders.suitesparse import SUITESPARSE_PRIMARY


def _serve_suitesparse(monkeypatch, files):
    """Serve a model reduction problem's matrices in place of SuiteSparse's."""
    calls = []

    def fake_load(source_name, names):
        calls.append((source_name, names))
        return {name: sp.csr_matrix(files[name]) for name in names}, {}

    monkeypatch.setattr(ode, "load_suitesparse_model_reduction", fake_load)
    return calls


def _dataset(**kwargs):
    return ODESuiteSparseDataset(
        "Fake/system", A=SUITESPARSE_PRIMARY, B="B", t_max=0.1, step=0.01, **kwargs
    )


def test_suitesparse_ode_lists_only_problems_that_omit_e():
    datasets = ODESuiteSparseGenerator().datasets
    assert [dataset.source_name for dataset in datasets] == ["Rommes/ww_36_pmec_36"]
    assert all(dataset.metadata["assumed_E"] == "identity" for dataset in datasets)


def test_suitesparse_ode_generator_keeps_a_and_b_sparse(monkeypatch):
    A = np.array([[-2.0, 1.0], [0.0, -3.0]])
    B = np.array([[0.0], [1.0]])
    calls = _serve_suitesparse(monkeypatch, {SUITESPARSE_PRIMARY: A, "B": B})

    instance = ODESuiteSparseGenerator().generate(_dataset())

    assert calls == [("Fake/system", (SUITESPARSE_PRIMARY, "B"))]
    A_out, B_out = (to_scipy(value) for value in instance.inputs)
    assert sp.issparse(A_out) and sp.issparse(B_out)
    np.testing.assert_array_equal(A_out.toarray(), A)
    np.testing.assert_array_equal(B_out.toarray(), B)
    assert instance.meta["problem_name"] == "ode_suitesparse"
    assert instance.meta["assumed_E"] == "identity"
    assert instance.meta["span"] == (0, 0.1)
    assert instance.meta["y0"] == [0.0, 0.0]


def test_suitesparse_ode_generator_rejects_mismatched_b(monkeypatch):
    _serve_suitesparse(
        monkeypatch, {SUITESPARSE_PRIMARY: np.eye(2), "B": np.ones((3, 1))}
    )
    with pytest.raises(ValueError, match="square A and a B"):
        ODESuiteSparseGenerator().generate(_dataset())


@pytest.mark.parametrize(
    ("benchmark_cls", "expected_step"),
    [
        (ForwardEulerBenchmark, 0.0001),
        (BackwardEulerBenchmark, 0.0002),
        (RK4Benchmark, 0.01),
    ],
)
def test_suitesparse_ode_runs_each_method_at_its_scaled_step(
    monkeypatch, benchmark_cls, expected_step
):
    _serve_suitesparse(
        monkeypatch,
        {SUITESPARSE_PRIMARY: np.array([[-1.0, 0.5], [0.0, -2.0]]), "B": np.eye(2)},
    )
    generator = ODESuiteSparseGenerator()
    param = Param(generator, _dataset())
    benchmark = benchmark_cls()

    benchmark.setup(param, use_cache=False, xp=SciPyFramework())
    benchmark.run(param)
    benchmark.check(param)

    assert benchmark._meta["step"] == expected_step
    assert benchmark._meta["problem_name"] == "ode_suitesparse"
