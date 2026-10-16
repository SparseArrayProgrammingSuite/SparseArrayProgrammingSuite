import pytest

import numpy as np

import sparse as sp
import torch
from binsparse.conversions import from_numpy, from_sparse

from frameworks.saps_numpy import NumpyFramework
from frameworks.saps_pytorch import PytorchFramework
from frameworks.saps_smart import SmartSparseFramework
from frameworks.saps_sparse import PyDataSparseFramework


@pytest.fixture(
    params=[
        NumpyFramework,
        PyDataSparseFramework,
        SmartSparseFramework,
        PytorchFramework,
    ]
)
def xp(request):
    return request.param()


def array(xp, value):
    value = np.asarray(value)
    tensor = (
        from_numpy(value)
        if isinstance(xp, (NumpyFramework, PytorchFramework))
        else from_sparse(sp.COO(value))
    )
    return xp.from_binsparse(tensor)


def dense(xp, value):
    return NumpyFramework().from_binsparse(xp.to_binsparse(value))


@pytest.mark.parametrize("n", [3, 4, 5])
def test_native_mttkrp(xp, n, monkeypatch):
    if isinstance(xp, SmartSparseFramework):
        # Exercise the native fallback independently of Smart's blocked path.
        monkeypatch.setattr(xp.kernels, "_blocked_einsum", lambda *args: NotImplemented)
    rng = np.random.default_rng(42)
    shape = (3,) * n
    tensor = rng.random(shape)
    tensor[tensor < 0.7] = 0
    factors = [rng.random((3, 2)) for _ in range(n - 1)]
    labels = ["row"] + [f"axis{i}" for i in range(n - 1)]
    expression = "Y[row, rank] += X[" + ",".join(labels) + "]"
    kwargs = {"X": array(xp, tensor)}
    for i, factor in enumerate(factors):
        expression += f" * F{i}[axis{i}, rank]"
        kwargs[f"F{i}"] = array(xp, factor)
    indices = "ijklm"[:n]
    expected = np.einsum(
        indices + "," + ",".join(c + "r" for c in indices[1:]) + "->ir",
        tensor,
        *factors,
    )
    backend = (
        torch
        if isinstance(xp, PytorchFramework)
        else (np if isinstance(xp, NumpyFramework) else sp)
    )
    original = backend.einsum
    calls = []

    def spy(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(backend, "einsum", spy)
    result = xp.einsum(expression, **kwargs)
    np.testing.assert_allclose(dense(xp, result), expected)
    assert len(calls) == 1


@pytest.mark.parametrize("reduction", ["+", "|", "or"])
@pytest.mark.parametrize("product", ["*", "&"])
@pytest.mark.parametrize("scalar", [False, True])
def test_boolean_contraction_semantics(xp, reduction, product, scalar):
    a = np.array([[True, True, False], [False, True, True]])
    b = np.array([[True, False], [True, True], [False, True]])
    output = "" if scalar else "row, col"
    got = dense(
        xp,
        xp.einsum(
            f"C[{output}] {reduction}= A[row, k] {product} B[k, col]",
            A=array(xp, a),
            B=array(xp, b),
        ),
    )
    products = a[:, :, None] & b[None, :, :]
    axes = None if scalar else 1
    expected = products.sum(axis=axes) if reduction == "+" else products.any(axis=axes)
    np.testing.assert_array_equal(got, expected)
    assert got.dtype == expected.dtype


@pytest.mark.parametrize(
    "expression,expected",
    [
        ("C[i] min= A[i,j] + B[j]", [1.0, 2.0]),
        ("C[i] += A[i,j] - B[j]", [-2.0, 4.0]),
    ],
)
def test_extended_expression_fallback(xp, expression, expected, monkeypatch):
    a = array(xp, [[1.0, 0.0], [2.0, 5.0]])
    b = array(xp, [0.0, 3.0])

    def fail(*args, **kwargs):
        pytest.fail("unsupported expressions must use the interpreter")

    backend = (
        torch
        if isinstance(xp, PytorchFramework)
        else (np if isinstance(xp, NumpyFramework) else sp)
    )
    monkeypatch.setattr(backend, "einsum", fail)
    np.testing.assert_allclose(dense(xp, xp.einsum(expression, A=a, B=b)), expected)


def test_narrow_integer_product_preserves_overflow(xp):
    a = np.array([[100, 100]], dtype=np.int8)
    b = np.array([2, 2], dtype=np.int8)
    got = dense(xp, xp.einsum("C[i] += A[i,j] * B[j]", A=array(xp, a), B=array(xp, b)))
    np.testing.assert_array_equal(got, (a * b).sum(axis=1))


def test_broadcast_label_fallback(xp):
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = np.array([2.0])
    got = dense(xp, xp.einsum("C[i] += A[i,j] * B[j]", A=array(xp, a), B=array(xp, b)))
    np.testing.assert_allclose(got, (a * b).sum(axis=1))


@pytest.mark.parametrize("framework", [PyDataSparseFramework, SmartSparseFramework])
def test_nonzero_fill_uses_interpreter(framework):
    xp = framework()
    a = sp.COO([[0], [1]], [0.0], shape=(2, 3), fill_value=2.0)
    b = np.array([1.0, 2.0, 3.0])
    got = dense(
        xp,
        xp.einsum(
            "C[i] += A[i,j] * B[j]", A=xp.from_binsparse(from_sparse(a)), B=array(xp, b)
        ),
    )
    np.testing.assert_allclose(got, (a.todense() * b).sum(axis=1))


def test_integer_bitwise_is_not_arithmetic_multiplication(xp):
    a = np.array([[3, 5], [6, 7]])
    b = np.array([2, 4])
    result = xp.einsum("C[i] += A[i,j] & B[j]", A=array(xp, a), B=array(xp, b))
    np.testing.assert_array_equal(dense(xp, result), (a & b).sum(axis=1))


def test_numpy_integer_bitwise_or_reduction():
    xp = NumpyFramework()
    a = np.array([[3, 5], [6, 7]])
    b = np.array([2, 4])
    result = xp.einsum("C[i] |= A[i,j] & B[j]", A=a, B=b)
    np.testing.assert_array_equal(result, np.bitwise_or.reduce(a & b, axis=1))


def test_torch_native_einsum_preserves_gradients_and_promotes_dtype():
    xp = PytorchFramework()
    a = torch.randn(3, 4, dtype=torch.float32, requires_grad=True)
    b = torch.randn(4, 2, dtype=torch.float64, requires_grad=True)
    result = xp.einsum("C[row,col] += A[row,k] * B[k,col]", A=a, B=b)
    expected = a.double() @ b
    torch.testing.assert_close(result, expected)
    gradients = torch.autograd.grad(result.sum(), (a, b))
    expected_gradients = torch.autograd.grad(expected.sum(), (a, b))
    for actual, reference in zip(gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, reference)


def test_torch_sparse_einsum_keeps_interpreter(monkeypatch):
    from saps_framework.einsum import Einsum

    xp = PytorchFramework()
    array = torch.eye(3).to_sparse()
    sentinel = object()
    calls = []

    def fallback(parsed, framework, kwargs):
        calls.append(kwargs["A"])
        return sentinel

    def fail(*args, **kwargs):
        pytest.fail("Sparse operands must not enter the dense Torch fast path")

    monkeypatch.setattr(Einsum, "run", fallback)
    monkeypatch.setattr(torch, "einsum", fail)
    monkeypatch.setattr(torch.Tensor, "to_dense", fail)
    assert xp.einsum("C[i] += A[i,j]", A=array) is sentinel
    assert calls == [array]


def test_torch_native_einsum_is_compiled():
    xp = PytorchFramework()
    graphs = []

    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward

    a, b = torch.randn(3, 4), torch.randn(4, 2)
    with torch._dynamo.config.patch(suppress_errors=False):
        compiled = torch.compile(xp.einsum, backend=backend)
        result = compiled("C[i,j] += A[i,k] * B[k,j]", A=a, B=b)
    torch.testing.assert_close(result, a @ b)
    assert torch.einsum in {
        node.target for graph in graphs for node in graph.graph.nodes
    }
