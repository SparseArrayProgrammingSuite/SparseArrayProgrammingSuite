import importlib
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
scorch = pytest.importorskip("scorch")


@pytest.fixture
def saps_scorch(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "frameworks"))
    return importlib.import_module("saps_scorch")


@pytest.fixture
def scorch_calls(monkeypatch):
    calls = []
    for name in ("einsum", "matmul"):
        original = getattr(scorch, name)

        def spy(*args, op_name=name, op=original, **kwargs):
            calls.append(op_name)
            return op(*args, **kwargs)

        monkeypatch.setattr(scorch, name, spy)
    return calls


def random_sparse(shape, dtype, seed):
    generator = torch.Generator().manual_seed(seed)
    dense = torch.rand(*shape, generator=generator, dtype=dtype)
    dense[torch.rand(*shape, generator=generator) > 0.3] = 0
    return dense.to_sparse()


def to_dense(array):
    return array.to_dense() if array.layout != torch.strided else array


@pytest.mark.parametrize(
    ("prgm", "operands"),
    [
        ("C[i,j] += A[i,k] * B[k,j]", ("A", "B")),
        ("C[i,j] += A[i,k] * D[k,j]", ("A", "D")),
        ("y[i] += A[i,j] * x[j]", ("A", "x")),
        ("C[i,j] = A[i,j] * A[i,j]", ("A",)),
        ("s[i] += A[i,j]", ("A",)),
    ],
)
def test_float32_sum_of_products_runs_in_scorch(
    saps_scorch, scorch_calls, prgm, operands
):
    arrays = {
        "A": random_sparse((12, 9), torch.float32, 0),
        "B": random_sparse((9, 7), torch.float32, 1),
        "D": torch.rand(9, 7, generator=torch.Generator().manual_seed(2)),
        "x": torch.rand(9, generator=torch.Generator().manual_seed(3)),
    }
    kwargs = {name: arrays[name] for name in operands}
    xp = saps_scorch.ScorchFramework()

    actual = xp.einsum(prgm, **kwargs)
    expected = xp.einsum(prgm, **{name: to_dense(a) for name, a in kwargs.items()})

    assert scorch_calls == ["einsum"]
    assert actual.layout in (torch.strided, torch.sparse_coo)
    torch.testing.assert_close(to_dense(actual), expected)


@pytest.mark.parametrize(
    ("prgm", "shapes"),
    [
        ("C[j,i] += A[i,k] * B[k,j]", {"A": (12, 9), "B": (9, 7)}),
        ("C[i,j] += A[i,k] * B[j,k]", {"A": (12, 9), "B": (7, 9)}),
        ("C[i,j] = A[i,j] * B[i,j]", {"A": (12, 9), "B": (1, 9)}),
        ("C[i,j,k] = A[i,j,k] * B[i,j,k]", {"A": (4, 5, 6), "B": (4, 5, 6)}),
        ("C[i,j] += A[i,j,k]", {"A": (4, 5, 6)}),
    ],
)
def test_patterns_scorch_gets_wrong_fall_back(
    saps_scorch, scorch_calls, monkeypatch, prgm, shapes
):
    from saps_framework.einsum import Einsum

    monkeypatch.setattr(Einsum, "run", lambda self, xp, kwargs: "fallback")
    kwargs = {
        name: random_sparse(shape, torch.float32, seed)
        for seed, (name, shape) in enumerate(shapes.items())
    }
    xp = saps_scorch.ScorchFramework()

    assert xp.einsum(prgm, **kwargs) == "fallback"
    assert scorch_calls == []


def test_float64_einsum_falls_back_to_pytorch_once(saps_scorch, scorch_calls):
    A = random_sparse((12, 9), torch.float64, 0)
    x = torch.rand(9, dtype=torch.float64)
    xp = saps_scorch.ScorchFramework()

    for _ in range(2):
        actual = xp.einsum("y[i] += A[i,j] * x[j]", A=A, x=x)
        torch.testing.assert_close(to_dense(actual), A.to_dense() @ x)

    assert scorch_calls == ["einsum"]


def test_non_sum_reduction_does_not_use_scorch(saps_scorch, scorch_calls):
    A = random_sparse((12, 9), torch.float32, 0).to_dense()
    xp = saps_scorch.ScorchFramework()

    actual = xp.einsum("m[i] max= A[i,j]", A=A)

    assert scorch_calls == []
    torch.testing.assert_close(actual, A.amax(dim=1))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("rhs_shape", [(9,), (9, 7)])
def test_sparse_matmul_runs_in_scorch(saps_scorch, scorch_calls, dtype, rhs_shape):
    A = random_sparse((12, 9), dtype, 0)
    b = torch.rand(*rhs_shape, dtype=dtype)
    xp = saps_scorch.ScorchFramework()

    actual = xp.matmul(A, b)

    assert scorch_calls == ["matmul"]
    torch.testing.assert_close(to_dense(actual), A.to_dense() @ b)
