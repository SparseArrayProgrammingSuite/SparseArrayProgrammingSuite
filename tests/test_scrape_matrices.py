from __future__ import annotations

import math

import pytest

import numpy as np
import scipy as sp

import saps.utils.scrape_matrices as scrape_matrices


def test_jacobi_convergence_forwards_tol_to_eigsh(monkeypatch):
    calls = []

    def fake_eigsh(*args, **kwargs):
        calls.append(kwargs)
        return np.array([0.5])

    monkeypatch.setattr(sp.sparse.linalg, "eigsh", fake_eigsh)

    A = sp.sparse.eye(3, format="csr")
    result = scrape_matrices.check_jacobi_normalized_convergence(A, tol=1e-7)

    assert result == 0.5
    assert calls[0]["tol"] == 1e-7


def test_cg_convergence_forwards_tol_to_each_eigsh_call(monkeypatch):
    eigenvalues = iter([np.array([4.0]), np.array([1.0])])
    calls = []

    def fake_eigsh(*args, **kwargs):
        calls.append(kwargs)
        return next(eigenvalues)

    monkeypatch.setattr(sp.sparse.linalg, "eigsh", fake_eigsh)

    A = sp.sparse.eye(3, format="csr")
    result = scrape_matrices.check_cg_normalized_convergence(A, tol=1e-8)

    assert result == pytest.approx(1 / 3)
    assert [call["tol"] for call in calls] == [1e-8 / (2 + 1e-8), 1e-8 / (2 + 1e-8)]


def test_preconditioned_cg_checks_forward_tol(monkeypatch):
    calls = []

    def fake_check_cg(A, M=None, tol=1e-3):
        calls.append((A, M, tol))
        return 0.25

    monkeypatch.setattr(
        scrape_matrices, "check_cg_normalized_convergence", fake_check_cg
    )

    A = sp.sparse.eye(3, format="csr")

    assert scrape_matrices.check_jacobi_cg_normalized_convergence(A, tol=1e-6) == 0.25
    assert (
        scrape_matrices.check_block_jacobi_cg_normalized_convergence(A, tol=1e-5)
        == 0.25
    )
    assert [call[2] for call in calls] == [1e-6, 1e-5]


def test_lsqr_convergence_uses_sqrt_corrected_tol_for_svds(monkeypatch):
    singular_values = {
        "LM": np.array([6.0]),
        "SM": np.array([2.0]),
    }
    calls = []

    def fake_svds(*args, **kwargs):
        calls.append(kwargs)
        return singular_values[kwargs["which"]]

    monkeypatch.setattr(sp.sparse.linalg, "svds", fake_svds)

    A = sp.sparse.eye(3, format="csr")
    result = scrape_matrices.check_lsqr_normalized_convergence(A, tol=1e-6)

    assert result == pytest.approx(0.5)
    assert [call["tol"] for call in calls] == [
        math.sqrt(1e-6 / (2 + 1e-6)),
        math.sqrt(1e-6 / (2 + 1e-6)),
    ]
