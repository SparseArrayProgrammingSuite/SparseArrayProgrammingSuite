import pytest

torch = pytest.importorskip("torch")

from frameworks.saps_pytorch import PytorchFramework  # noqa: E402


@pytest.fixture
def xp():
    return PytorchFramework()


def random_mask(shape, seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(shape, generator=generator) < 0.3


def test_where_accepts_sparse_condition(xp):
    # MSBFS passes its sources as a sparse frontier.
    condition = random_mask((3, 4), seed=0)
    level = torch.zeros((3, 4), dtype=torch.int64)

    result = xp.where(condition.to_sparse(), 7, level)

    torch.testing.assert_close(result, torch.where(condition, 7, level))


@pytest.mark.parametrize("x1", ["dense", "scalar"])
def test_where_with_zero_else_stays_sparse(xp, x1):
    # A stored False must not select x1, whatever its value.
    condition = torch.sparse_coo_tensor(
        torch.tensor([[0, 0, 1], [0, 1, 2]]), torch.tensor([True, False, True]), (2, 3)
    )
    values = (
        torch.tensor([[1.5, float("inf"), 2.0], [3.0, float("nan"), float("inf")]])
        if x1 == "dense"
        else 7
    )

    result = xp.where(condition, values, 0)

    assert result.layout == torch.sparse_coo
    expected = torch.where(condition.to_dense(), torch.as_tensor(values), 0)
    torch.testing.assert_close(result.to_dense(), expected, equal_nan=True)


def test_logical_not_accepts_sparse(xp):
    mask = random_mask((3, 4), seed=1)

    torch.testing.assert_close(xp.logical_not(mask.to_sparse()), ~mask)


@pytest.mark.parametrize("name", ["logical_and", "logical_or", "logical_xor"])
@pytest.mark.parametrize("sparse_side", [0, 1])
def test_binary_logical_ops_accept_sparse_with_dense(xp, name, sparse_side):
    # BFS: frontier from a sparse einsum combined with a dense visited mask.
    masks = [random_mask((3, 4), seed=2), random_mask((3, 4), seed=3)]
    operands = list(masks)
    operands[sparse_side] = operands[sparse_side].to_sparse()

    result = getattr(xp, name)(*operands)

    assert result.layout == torch.strided
    torch.testing.assert_close(result, getattr(torch, name)(*masks))
