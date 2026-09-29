import pytest

import numpy as np

import sparse as sp

from frameworks.saps_smart import SmartSparseFramework


@pytest.fixture
def xp():
    return SmartSparseFramework()


def test_operators_on_raw_and_wrapped_arrays_route_through_framework(xp):
    dense = np.arange(6.0).reshape(2, 3)
    eye = xp.eye(2, 3)

    for result in (dense + eye, eye + dense, dense @ xp.eye(3), -eye, eye[0]):
        assert result.mod is xp
    np.testing.assert_array_equal(dense - eye, dense - np.eye(2, 3))


def test_broadcast_mixed_add_densifies(xp):
    # The tic_tac pattern: a dense operand that must broadcast against a sparse
    # one, where pydata/sparse would raise instead of densifying.
    dense = np.arange(6.0).reshape(2, 1, 3)
    sparse = xp.reshape(xp.eye(3), (1, 3, 3))

    result = dense + sparse

    assert isinstance(result.array, np.ndarray)
    assert result.fill_value == 0
    np.testing.assert_array_equal(result, dense + np.eye(3)[None])


def test_mixed_op_with_constant_result_stays_sparse(xp):
    dense = np.arange(9.0).reshape(3, 3)

    result = xp.eye(3) * dense

    assert isinstance(result.array, sp.SparseArray)
    np.testing.assert_array_equal(result, np.eye(3) * dense)


def test_dense_result_remembers_fill_from_operands(xp):
    shifted = xp.eye(3) - 1.0
    assert isinstance(shifted.array, sp.SparseArray)
    assert shifted.fill_value == -1

    result = shifted + np.arange(3.0)

    assert isinstance(result.array, np.ndarray)
    assert result.fill_value == -1
    np.testing.assert_array_equal(result, np.eye(3) - 1.0 + np.arange(3.0))


def test_matmul_densifies_nonzero_fill(xp):
    # pydata/sparse contractions refuse sparse operands with a nonzero fill.
    shifted = xp.eye(2) - 1.0
    assert shifted.fill_value == -1

    result = shifted @ np.ones((2, 3))

    np.testing.assert_array_equal(result, (np.eye(2) - 1.0) @ np.ones((2, 3)))


@pytest.mark.parametrize("name", ["concat", "stack"])
def test_concat_mixed_members(xp, name):
    dense = np.arange(4.0).reshape(2, 2)

    result = getattr(xp, name)([dense, xp.eye(2)], axis=1)

    assert isinstance(result.array, sp.SparseArray)
    expected = getattr(np, "concatenate" if name == "concat" else name)(
        [dense, np.eye(2)], axis=1
    )
    np.testing.assert_array_equal(result, expected)


def test_concat_members_with_different_fills_densify(xp):
    result = xp.concat([xp.eye(2) + 1.0, xp.eye(2)], axis=0)

    assert isinstance(result.array, np.ndarray)
    np.testing.assert_array_equal(result, np.concatenate([np.eye(2) + 1, np.eye(2)]))


def test_setitem_sparse_value_into_dense_array(xp):
    target = xp.asarray(np.zeros((2, 3)))

    target[0] = xp.eye(2, 3)[0]

    np.testing.assert_array_equal(target, [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])


def test_scalar_conversions_and_array_methods(xp):
    total = xp.sum(xp.eye(3))

    assert float(total) == 3.0
    assert int(xp.eye(3)[1, 1]) == 1
    assert xp.eye(2, 3).T.shape == (3, 2)
    assert xp.eye(2, 3).reshape((3, 2)).shape == (3, 2)
    assert xp.eye(3).astype(np.int32).dtype == np.int32


def test_large_sparse_sum_does_not_densify(xp, monkeypatch):
    array = sp.COO([[0, 1], [7, 999_999]], [2.0, 3.0], shape=(2, 1_000_000))

    def forbid_dense(self):
        pytest.fail("sparse reduction must not allocate a dense output")

    monkeypatch.setattr(sp.COO, "todense", forbid_dense)
    result = xp.sum(xp.wrap(array), axis=0)

    assert isinstance(result.array, sp.COO)
    assert result.shape == (1_000_000,)
    np.testing.assert_array_equal(result.array.coords, [[7, 999_999]])
    np.testing.assert_array_equal(result.array.data, [2.0, 3.0])


@pytest.mark.parametrize("fill", [True, np.inf, -1.0])
def test_sparse_output_preserves_nonzero_fill(xp, monkeypatch, fill):
    array = sp.COO(
        [[7]],
        np.asarray([0], dtype=type(fill)),
        shape=(1_000_000,),
        fill_value=fill,
    )

    def forbid_dense(self):
        pytest.fail("binsparse conversion must preserve sparse output")

    monkeypatch.setattr(sp.COO, "todense", forbid_dense)
    result = xp.from_binsparse(xp.to_binsparse(xp.wrap(array)))

    assert isinstance(result.array, sp.COO)
    assert result.shape == array.shape
    assert result.fill_value == fill
    np.testing.assert_array_equal(result.array.coords, array.coords)
    np.testing.assert_array_equal(result.array.data, array.data)
