"""PyTorch cases of tests that cover several frameworks, plus PyTorch-only checks.

Only pixi's pytorch environment installs PyTorch (`pixi run test-pytorch`). The
imported tests rerun here against this module's `xp` fixture.
"""

import pytest

import numpy as np

torch = pytest.importorskip("torch")

from frameworks.saps_pytorch import PytorchFramework  # noqa: E402
from tests.test_approx_nn_lsh import (  # noqa: E402, F401
    lsh_runner,
    test_lsh_exhaustive_candidates_match_cosine_knn,
    test_lsh_queries_stop_independently_and_exclude_non_candidates,
    test_lsh_shortens_the_required_prefix_until_opposite_signed_projections_collide,
    test_lsh_unions_tables_and_returns_distinct_neighbors,
)
from tests.test_graph_counting_checks import (  # noqa: E402, F401
    benchmark,
    test_counting_contractions_preserve_int64,
)
from tests.test_native_einsum import (  # noqa: E402, F401
    test_boolean_contraction_semantics,
    test_broadcast_label_fallback,
    test_extended_expression_fallback,
    test_integer_bitwise_is_not_arithmetic_multiplication,
    test_narrow_integer_product_preserves_overflow,
    test_native_mttkrp,
)
from tests.test_replace import (  # noqa: E402, F401
    dense,
    test_replace_promotes_integer_values,
    test_replace_scalar_and_empty_arrays,
    test_replace_values_without_mutating_input,
)


@pytest.fixture
def xp():
    return PytorchFramework()


@pytest.fixture
def einsum_backend():
    return torch


@pytest.fixture
def run_lsh(xp):
    return lsh_runner(xp, sparse_inputs=False)


def test_torch_native_einsum_preserves_gradients_and_promotes_dtype(xp):
    a = torch.randn(3, 4, dtype=torch.float32, requires_grad=True)
    b = torch.randn(4, 2, dtype=torch.float64, requires_grad=True)
    result = xp.einsum("C[row,col] += A[row,k] * B[k,col]", A=a, B=b)
    expected = a.double() @ b
    torch.testing.assert_close(result, expected)
    gradients = torch.autograd.grad(result.sum(), (a, b))
    expected_gradients = torch.autograd.grad(expected.sum(), (a, b))
    for actual, reference in zip(gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, reference)


def test_torch_sparse_einsum_keeps_interpreter(xp, monkeypatch):
    from saps_framework.einsum import Einsum

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


def test_torch_native_einsum_is_compiled(xp):
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


@pytest.mark.parametrize("format", ["coo", "csr", "csc"])
def test_replace_zero_fill_keeps_layout(xp, format):
    values = np.array([[0.0, 2.0, np.nan], [2.0, 0.0, 4.0]])
    array = torch.as_tensor(values.copy()).to_sparse(
        layout=getattr(torch, f"sparse_{format}")
    )
    changed = xp.replace(array, 2, 0)
    assert changed.layout == array.layout
    np.testing.assert_array_equal(dense(changed), [[0, 0, np.nan], [0, 0, 4]])
    np.testing.assert_array_equal(
        dense(xp.replace(array, 0, 3)), [[3, 2, np.nan], [2, 3, 4]]
    )
    np.testing.assert_array_equal(
        dense(xp.replace(array, np.nan, 0)), [[0, 2, 0], [2, 0, 4]]
    )
    np.testing.assert_array_equal(dense(array), values)


def test_replace_combines_duplicate_sparse_coordinates_first(xp):
    array = torch.sparse_coo_tensor([[0, 0], [1, 1]], [2.0, 3.0], (2, 3))
    np.testing.assert_array_equal(dense(xp.replace(array, 5, 0)), np.zeros((2, 3)))
    assert dense(array)[0, 1] == 5
