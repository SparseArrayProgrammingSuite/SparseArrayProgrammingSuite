"""Functional updates for frameworks whose arrays are read-only."""

import numpy as np

import sparse
from binsparse.conversions import from_sparse, to_numpy, to_scipy, to_sparse


def _dense(tensor):
    """A Binsparse tensor as a NumPy array."""
    for convert in (to_numpy, lambda t: to_scipy(t).toarray()):
        try:
            return np.asarray(convert(tensor))
        except (TypeError, ValueError):
            pass
    return to_sparse(tensor).todense()


class WhereAt:
    """xp.at(x, idx) for arrays without __setitem__ (finch's and galley's), as
    array-api-extra updates lazy arrays: set returns where(mask, values, x),
    leaving x unchanged. The mask and the values are sparse tensors built from
    the selected coordinates, so nothing the size of x is densified. Only the
    framework's where, astype, from_binsparse and to_binsparse are used."""

    def __init__(self, xp, x, idx):
        self.xp = xp
        self.x = x
        self.idx = idx

    def _numpy(self, value):
        if isinstance(value, bool | int | float | complex | np.generic | np.ndarray):
            return np.asarray(value)
        return _dense(self.xp.to_binsparse(value))

    def _coordinates(self):
        """The coordinates idx selects, shaped like the selection."""
        shape = self.x.shape
        idx = self.idx if isinstance(self.idx, tuple) else (self.idx,)
        if len(idx) > len(shape):
            raise IndexError(f"too many indices for an array of shape {shape}")
        idx = idx + (slice(None),) * (len(shape) - len(idx))
        parts = [
            index if isinstance(index, slice) else self._numpy(index) for index in idx
        ]
        if any(
            isinstance(part, np.ndarray) and part.dtype == np.bool_ for part in parts
        ):
            raise NotImplementedError(
                "at() on read-only arrays takes integer and slice indices"
            )
        arrays = [part for part in parts if not isinstance(part, slice)]
        if any(part.ndim > 0 for part in arrays) and len(arrays) < len(parts):
            raise NotImplementedError(
                "at() on read-only arrays does not mix index arrays with slices"
            )
        if len(arrays) == len(parts):
            # Integer arrays (and scalars) broadcast together.
            coords = np.broadcast_arrays(*parts)
        else:
            # Slices and integers select an outer product; integers drop axes.
            ranges = [
                np.arange(*part.indices(size)) if isinstance(part, slice) else part
                for part, size in zip(parts, shape, strict=True)
            ]
            coords = np.meshgrid(*ranges, indexing="ij")
            kept = tuple(isinstance(part, slice) for part in parts)
            selection = tuple(
                size for size, keep in zip(coords[0].shape, kept, strict=True) if keep
            )
            coords = [coord.reshape(selection) for coord in coords]
        return [
            np.where(coord < 0, coord + size, coord)
            for coord, size in zip(coords, shape, strict=True)
        ]

    def set(self, value, /):
        x = self.x
        coords = self._coordinates()
        selection = coords[0].shape
        values = np.broadcast_to(self._numpy(value), selection).reshape(-1)
        coords = np.stack([coord.reshape(-1) for coord in coords])
        # Like x[idx] = value, the last write to a repeated coordinate wins.
        linear = np.ravel_multi_index(tuple(coords), x.shape)
        _, last = np.unique(linear[::-1], return_index=True)
        keep = len(linear) - 1 - last
        coords, values = coords[:, keep], values[keep]
        mask = sparse.COO(
            coords, np.ones(len(keep), dtype=np.bool_), shape=x.shape, fill_value=False
        )
        update = sparse.COO(coords, values, shape=x.shape)
        result = self.xp.where(
            self.xp.from_binsparse(from_sparse(mask)),
            self.xp.from_binsparse(from_sparse(update)),
            x,
        )
        if result.dtype != x.dtype:
            result = self.xp.astype(result, x.dtype)
        return result
