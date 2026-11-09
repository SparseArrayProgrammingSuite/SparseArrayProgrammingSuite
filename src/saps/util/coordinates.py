"""Canonical coordinate lists for Binsparse COO and sparse levels."""

import numpy as np


def canonical_coordinates(indices, values):
    """Coordinates sorted with the first axis outermost, repeats summed.

    Binsparse COOR and sparse levels require sorted, duplicate-free
    coordinates, and readers rely on it (binsparse's SciPy conversion marks
    them canonical, and Finch iterates them in order), but downloaded
    coordinate lists (G-CARE edges, FROSTT .tns files) come in file order.
    """
    indices = tuple(np.asarray(axis) for axis in indices)
    values = np.asarray(values)
    if len(values) == 0:
        return indices, values
    order = np.lexsort(indices[::-1])
    indices = tuple(axis[order] for axis in indices)
    values = values[order]
    new = np.ones(len(values), dtype=bool)
    new[1:] = np.any([axis[1:] != axis[:-1] for axis in indices], axis=0)
    if not new.all():
        starts = np.flatnonzero(new)
        indices = tuple(axis[starts] for axis in indices)
        values = np.add.reduceat(values, starts)
    return indices, values
