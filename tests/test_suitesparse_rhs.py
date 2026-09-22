import json
from copy import copy
from unittest.mock import Mock

import pytest

import numpy as np
from scipy import sparse

from binsparse.conversions import from_numpy, from_scipy, to_numpy, to_scipy

from saps.benchmark import DataInstance
from saps.benchmarks import GMRES, cg, jacobi, lsqr, preconditioned_cg, suitesparse
from saps.storage import LocalStorageBackend


@pytest.fixture
def raw_system():
    return DataInstance(
        inputs=[
            from_scipy(sparse.diags([2.0, 3.0, 4.0], format="coo")),
            from_numpy(np.array([[2.0, 8.0], [6.0, 15.0], [12.0, 24.0]])),
        ],
        meta={"has_b_file": True},
    )


def test_rhs_selections_share_one_cached_matrix(monkeypatch, tmp_path, raw_system):
    backend = LocalStorageBackend(
        tmp_path / "remote", tmp_path / "manifest.json", tmp_path / "cache"
    )
    monkeypatch.setattr(
        suitesparse.SuiteSparseMatrixGenerator, "backend", property(lambda _: backend)
    )
    load = Mock(
        return_value=(
            to_scipy(raw_system.inputs[0]),
            to_numpy(raw_system.inputs[1]),
            raw_system.meta,
        )
    )
    monkeypatch.setattr(suitesparse, "load_suitesparse_matrix", load)
    generator = suitesparse.SuiteSparseMatrixGenerator()
    dataset = next(d for d in generator.datasets if d.name == "HB/orani678")
    assert backend.upload_dataset(generator, dataset)
    load.assert_called_once_with("HB/orani678")
    manifest = json.loads(backend.manifest_path.read_text())
    assert list(manifest) == ["suitesparse_matrix.HB/orani678"]
    prefix = backend.prefix(
        generator, dataset, manifest[next(iter(manifest))]["digest"]
    )
    (backend.cache_dir / prefix).unlink()
    download = Mock(wraps=backend.download_file)
    monkeypatch.setattr(backend, "download_file", download)
    load.side_effect = AssertionError("Raw inputs should be cached")

    for index in (1, 0, 1):
        matrix, b, real = suitesparse.fetch_suitesparse_linear_system(
            dataset.source_name, rhs_index=index
        )
        np.testing.assert_array_equal(to_scipy(matrix).toarray(), np.diag([2, 3, 4]))
        np.testing.assert_array_equal(b, to_numpy(raw_system.inputs[1])[:, index])
        assert real
    assert download.call_count == 1
    assert load.call_count == 1
    assert len(list(backend.cache_dir.rglob("*.bsp.h5"))) == 1
    assert len(list((tmp_path / "remote").rglob("*.bsp.h5"))) == 1


@pytest.mark.parametrize("rhs_index", [None, 0])
def test_single_rhs_is_selected(monkeypatch, raw_system, rhs_index):
    raw_system.inputs[1] = from_numpy(np.array([2.0, 6.0, 12.0]))
    monkeypatch.setattr(suitesparse, "fetch_suitesparse_matrix", lambda _: raw_system)
    _, b, real = suitesparse.fetch_suitesparse_linear_system(
        "test/matrix", rhs_index=rhs_index
    )
    np.testing.assert_array_equal(b, [2, 6, 12])
    assert real


@pytest.mark.parametrize("has_rhs", [False, True])
def test_unspecified_multi_or_missing_rhs_keeps_synthetic_fallback(
    monkeypatch, raw_system, has_rhs
):
    if not has_rhs:
        raw_system.inputs = raw_system.inputs[:1]
    monkeypatch.setattr(suitesparse, "fetch_suitesparse_matrix", lambda _: raw_system)
    _, b, real = suitesparse.fetch_suitesparse_linear_system("test/matrix")
    expected = suitesparse.random_rhs_for_matrix(to_scipy(raw_system.inputs[0]).tocoo())
    np.testing.assert_array_equal(b, expected)
    assert not real


@pytest.mark.parametrize(("rhs_count", "rhs_index"), [(2, -1), (2, 2), (1, 1), (0, 0)])
def test_invalid_rhs_selection_raises(monkeypatch, raw_system, rhs_count, rhs_index):
    if rhs_count == 0:
        raw_system.inputs = raw_system.inputs[:1]
    elif rhs_count == 1:
        raw_system.inputs[1] = from_numpy(to_numpy(raw_system.inputs[1])[:, 0])
    monkeypatch.setattr(suitesparse, "fetch_suitesparse_matrix", lambda _: raw_system)
    with pytest.raises(ValueError, match="rhs_index|no compatible RHS"):
        suitesparse.fetch_suitesparse_linear_system("test/matrix", rhs_index=rhs_index)


@pytest.mark.parametrize(
    "generator",
    [
        cg.CGGenerator(),
        jacobi.JacobiGenerator(),
        GMRES.GMRESGenerator(),
        lsqr.LSQRGenerator(),
        preconditioned_cg.BlockJacobiCGGenerator(),
        preconditioned_cg.JacobiCGGenerator(),
    ],
    ids=lambda g: g.name,
)
def test_solver_generators_select_rhs_without_caching(
    monkeypatch, raw_system, generator
):
    fetch = Mock(return_value=raw_system)
    monkeypatch.setattr(suitesparse, "fetch_suitesparse_matrix", fetch)
    assert not generator.cacheable
    dataset = copy(next(d for d in generator.datasets if getattr(d, "A", None) is None))
    for index in (0, 1):
        dataset.rhs_index = index
        problem = generator.cached_generate(dataset)
        assert problem.inputs[0] is raw_system.inputs[0]
        np.testing.assert_array_equal(
            to_numpy(problem.inputs[1]), to_numpy(raw_system.inputs[1])[:, index]
        )
    assert fetch.call_count == 2
    assert all(
        call.args == (dataset.source_name,) and not call.kwargs
        for call in fetch.call_args_list
    )
