"""Shared residual checks for CP and Tucker decompositions of sparse tensors."""

import math

import numpy as np

from binsparse import BinsparseTensor, CustomTensor, ElementLevel, SparseLevel
from binsparse.conversions import to_numpy

from saps.util.error_bounds import summation_error_bound


def _finite(array):
    assert np.isfinite(array).all(), "Tensor contains non-finite values"
    return array


def decomposition_outputs(outputs, shapes):
    """Return outputs as finite float64 arrays after checking their shapes."""
    assert len(outputs) == len(shapes), "Incorrect number of decomposition outputs"
    arrays = []
    for output, shape in zip(outputs, shapes, strict=True):
        assert isinstance(output, BinsparseTensor), "Output must be in binsparse format"
        assert output.shape == shape, f"Expected output shape {shape}, got {output.shape}"
        arrays.append(_finite(np.asarray(to_numpy(output), dtype=np.float64)))
    return arrays


def stored_entries(tensor):
    """Return coordinates and values of stored entries, without densifying COO."""
    if isinstance(tensor, CustomTensor) and isinstance(tensor.level, SparseLevel):
        level = tensor.level
        assert tensor.transpose is None and level.pointers_to_next is None
        assert isinstance(level.level, ElementLevel)
        assert not tensor.fill or tensor.fill_value == 0
        return level.indices, _finite(level.level.values)
    array = _finite(to_numpy(tensor))
    indices = array.nonzero()
    return indices, array[indices]


def input_norms(tensor):
    """Return ||X||² and its absolute-term analogue, summing duplicate coordinates."""
    indices, values = stored_entries(tensor)
    order = np.lexsort(indices[::-1])
    duplicate = np.ones(len(values), dtype=bool)
    duplicate[:1] = False
    for index in indices:
        index = index[order]
        duplicate[1:] &= index[1:] == index[:-1]
    starts = np.flatnonzero(~duplicate)
    values = values[order].astype(np.float64)
    sums = np.add.reduceat(values, starts)
    absolute = np.add.reduceat(np.abs(values), starts)
    return {
        "input_squared_norm": float(sums @ sums),
        "input_absolute_squared_norm": float(absolute @ absolute),
    }


def assert_residual(tensor, ref_meta, model_norm, model_absolute, inner, terms):
    """Assert ||X - Y|| <= rel_error_tol * ||X|| up to float64 roundoff.

    Uses ||X - Y||² = ||X||² + ||Y||² - 2<X,Y>, where model_norm is ||Y||²,
    model_absolute bounds the sum of |Y|'s summands, inner is <X,Y>, and terms
    is the number of products summed to evaluate Y at one coordinate.
    """
    if "input_squared_norm" not in ref_meta:
        ref_meta = {**ref_meta, **input_norms(tensor)}
    input_norm = ref_meta["input_squared_norm"]
    # Cauchy-Schwarz bounds the absolute cross term, so all three terms share one
    # Higham summation bound (Higham 1993, eq. 2.6).
    scale = (math.sqrt(ref_meta["input_absolute_squared_norm"]) + model_absolute) ** 2
    steps = max(sum(tensor.shape), tensor.number_of_stored_values)
    steps += terms**2 + 2 * len(tensor.shape) + 5
    bound = summation_error_bound(np, steps, scale, np.float64)
    error = math.fsum([input_norm, model_norm, -2 * inner])
    assert np.isfinite([model_norm, inner, scale, error]).all(), (
        "Non-finite reconstruction residual"
    )
    # Signed components can cancel in ||Y||²; <X,Y> catches a zero model then.
    assert input_norm == 0 or model_norm > 0 or inner != 0, (
        "Nonzero input has a zero reconstruction"
    )
    limit = ref_meta["rel_error_tol"] ** 2 * input_norm
    assert -bound <= error <= limit + bound, (
        f"Reconstruction residual squared {error:.6g} exceeds "
        f"{limit:.6g} (roundoff allowance {bound:.6g})"
    )
