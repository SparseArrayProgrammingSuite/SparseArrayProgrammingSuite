import pytest

import numpy as np

from sparseappbench.benchmarks.mri_edge import (
    benchmark_masked_mri_edge,
    dg_masked_mri_1,
    dg_masked_mri_2,
    dg_masked_mri_3,
    dg_masked_mri_4,
)
from sparseappbench.binsparse_format import BinsparseFormat
from sparseappbench.frameworks.numpy_framework import NumpyFramework


def get_framework():
    return NumpyFramework()


def expected_masked_mri_edge(image, roi, t1, t2):
    img_t1 = image > t1
    img_t2 = image > t2
    return (img_t2 & roi) ^ (img_t1 & roi)


@pytest.mark.parametrize(
    "image, roi, t1, t2",
    [
        (
            np.zeros((5, 5), dtype=np.float32),
            np.ones((5, 5), dtype=bool),
            50.0,
            100.0,
        ),
        (
            np.array(
                [
                    [0, 50, 100, 150, 200],
                    [0, 50, 100, 150, 200],
                    [0, 50, 100, 150, 200],
                    [0, 50, 100, 150, 200],
                    [0, 50, 100, 150, 200],
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    [False, False, False, False, False],
                    [False, True, True, True, False],
                    [False, True, True, True, False],
                    [False, True, True, True, False],
                    [False, False, False, False, False],
                ],
                dtype=bool,
            ),
            75.0,
            125.0,
        ),
    ],
)
def test_masked_mri_basic_cases(image, roi, t1, t2):
    xp = get_framework()

    expected = expected_masked_mri_edge(image, roi, t1, t2)

    image_bin = BinsparseFormat.from_numpy(image)
    roi_bin = BinsparseFormat.from_numpy(roi)
    t1_bin = BinsparseFormat.from_numpy(np.array(t1, dtype=np.float32))
    t2_bin = BinsparseFormat.from_numpy(np.array(t2, dtype=np.float32))

    result_bin = benchmark_masked_mri_edge(
        xp, img_bench=image_bin, roi_bench=roi_bin, t1_bench=t1_bin, t2_bench=t2_bin
    )
    result = xp.from_benchmark(result_bin)

    assert result.shape == expected.shape
    assert np.all(result == expected)


@pytest.mark.parametrize(
    "generator", [dg_masked_mri_1, dg_masked_mri_2, dg_masked_mri_3, dg_masked_mri_4]
)
def test_masked_mri_sparse_generators(generator):
    xp = get_framework()
    try:
        image_bin, roi_bin, t1_bin, t2_bin = generator()
    except (FileNotFoundError, ImportError, ValueError) as e:
        pytest.skip(f"Failed to generate data: {e}")

    result_bin = benchmark_masked_mri_edge(
        xp, img_bench=image_bin, roi_bench=roi_bin, t1_bench=t1_bin, t2_bench=t2_bin
    )
    result = xp.from_benchmark(result_bin)

    image = xp.from_benchmark(image_bin)
    roi = xp.from_benchmark(roi_bin)
    t1 = xp.from_benchmark(t1_bin)
    t2 = xp.from_benchmark(t2_bin)

    expected = expected_masked_mri_edge(image, roi, float(t1), float(t2))

    assert np.all(result == expected)
    assert result.dtype == bool
