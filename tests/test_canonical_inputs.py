"""Benchmark inputs must be canonical Binsparse.

Binsparse COO and sparse levels require coordinates sorted with the first axis
outermost and free of duplicates, and frameworks rely on it: binsparse's SciPy
conversion marks COO input canonical without checking, and Finch iterates
coordinates in order. Unsorted inputs made Galley miscount subgraph matches
and Finch miscompute Sobel edges.
"""

import importlib
import json
from pathlib import Path

import pytest

import numpy as np

from binsparse import COORMatrix, CSRMatrix, CustomTensor, SparseLevel

from saps.util.coordinates import canonical_coordinates

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_canonical_coordinates_sorts_and_sums_repeats():
    indices = (np.array([2, 0, 2, 1]), np.array([1, 3, 1, 0]))
    values = np.array([1.0, 2.0, 3.0, 4.0])
    (rows, cols), sums = canonical_coordinates(indices, values)
    np.testing.assert_array_equal(rows, [0, 1, 2])
    np.testing.assert_array_equal(cols, [3, 0, 1])
    np.testing.assert_array_equal(sums, [2.0, 4.0, 4.0])


def test_canonical_coordinates_orders_first_axis_outermost():
    indices = (np.array([1, 0, 1]), np.array([0, 2, 1]), np.array([5, 0, 0]))
    out, _ = canonical_coordinates(indices, np.arange(3))
    assert [tuple(int(axis[k]) for axis in out) for k in range(3)] == [
        (0, 2, 0),
        (1, 0, 5),
        (1, 1, 0),
    ]


def _coordinates(tensor):
    match tensor:
        case COORMatrix():
            return [tensor.indices_0, tensor.indices_1]
        case CSRMatrix():  # CSC subclasses CSR: indices ascend within each slice
            pointers = np.asarray(tensor.pointers_to_1)
            slices = np.repeat(np.arange(len(pointers) - 1), np.diff(pointers))
            return [slices, tensor.indices_1]
        case CustomTensor(level=SparseLevel() as level):
            return list(level.indices)
    return None


def _test_datasets():
    metadata = json.loads((REPO_ROOT / "metadata.json").read_text(encoding="utf-8"))
    for record in metadata["benchmarks"]:
        module_name, class_name, _ = record["asv_ids"]["time"].split(".")
        for generator in record["generators"]:
            for dataset in generator["datasets"]:
                if "suite-test" in dataset["tags"]:
                    param_name = f"{generator['name']}.{dataset['name']}"
                    yield pytest.param(
                        module_name,
                        class_name,
                        param_name,
                        id=f"{record['name']}[{param_name}]",
                    )


@pytest.mark.parametrize(("module_name", "class_name", "param_name"), _test_datasets())
def test_sparse_inputs_are_canonical(module_name, class_name, param_name):
    module = importlib.import_module(f"saps.benchmarks.{module_name}")
    benchmark = getattr(module, class_name)()
    param = next(param for param in benchmark.params if str(param) == param_name)
    generator = param.generator
    instance = (
        generator.cached_generate(param.dataset)
        if generator.cacheable
        else generator.generate(param.dataset)
    )
    for position, tensor in enumerate(instance.inputs):
        coordinates = _coordinates(tensor)
        if coordinates is None or len(coordinates[0]) < 2:
            continue
        coordinates = [np.asarray(axis, dtype=np.int64) for axis in coordinates]
        order = np.lexsort(coordinates[::-1])
        assert (order == np.arange(len(order))).all(), (
            f"input {position} ({type(tensor).__name__}) is not sorted"
        )
        repeated = np.all([axis[1:] == axis[:-1] for axis in coordinates], axis=0)
        assert not repeated.any(), (
            f"input {position} ({type(tensor).__name__}) repeats coordinates"
        )
