import pytest

import numpy as np

from sparseappbench.benchmarks.sobel_edge import (
    benchmark_mri_edge,
    dg_mri_sobel_1,
    dg_mri_sobel_2,
    dg_mri_sobel_3,
    dg_mri_sobel_4,
    generate_1d_sobel_matrices,
)
from sparseappbench.binsparse_format import BinsparseFormat
from sparseappbench.frameworks.numpy_framework import NumpyFramework


def get_framework():
    return NumpyFramework()


def expected_sobel_edge(image, threshold):
    img_m1_m1 = np.roll(np.roll(image, 1, axis=0), 1, axis=1)
    img_m1_0 = np.roll(image, 1, axis=0)
    img_m1_p1 = np.roll(np.roll(image, 1, axis=0), -1, axis=1)

    img_p1_m1 = np.roll(np.roll(image, -1, axis=0), 1, axis=1)
    img_p1_0 = np.roll(image, -1, axis=0)
    img_p1_p1 = np.roll(np.roll(image, -1, axis=0), -1, axis=1)

    gx = (img_p1_m1 + 2 * img_p1_0 + img_p1_p1) - (img_m1_m1 + 2 * img_m1_0 + img_m1_p1)

    img_0_m1 = np.roll(image, 1, axis=1)
    img_0_p1 = np.roll(image, -1, axis=1)

    gy = (img_m1_p1 + 2 * img_0_p1 + img_p1_p1) - (img_m1_m1 + 2 * img_0_m1 + img_p1_m1)

    magnitude = np.abs(gx) + np.abs(gy)
    return magnitude > threshold


@pytest.mark.parametrize(
    "image, threshold",
    [
        (np.zeros((5, 5), dtype=np.float32), 10.0),
        (
            np.array(
                [
                    [0, 0, 100, 100, 0],
                    [0, 0, 100, 100, 0],
                    [0, 0, 100, 100, 0],
                    [0, 0, 100, 100, 0],
                    [0, 0, 100, 100, 0],
                ],
                dtype=np.float32,
            ),
            50.0,
        ),
        (
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [100, 100, 100, 100, 100],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                dtype=np.float32,
            ),
            50.0,
        ),
    ],
)
def test_sobel_basic_cases(image, threshold):
    xp = get_framework()

    expected = expected_sobel_edge(image, threshold)

    image_bin = BinsparseFormat.from_numpy(image)
    threshold_bin = BinsparseFormat.from_numpy(np.array(threshold))

    Nx, Ny = image.shape
    dx_bin, sy_bin, sx_bin, dy_bin = generate_1d_sobel_matrices(Nx, Ny)

    result_bin = benchmark_mri_edge(
        xp, image_bin, dx_bin, sy_bin, sx_bin, dy_bin, threshold_bin
    )
    result = xp.from_benchmark(result_bin)

    assert result.shape == expected.shape
    assert np.all(result == expected), (
        "Benchmark MRI Soebel outputdoes not match expected."
    )


@pytest.mark.parametrize(
    "generator", [dg_mri_sobel_1, dg_mri_sobel_2, dg_mri_sobel_3, dg_mri_sobel_4]
)
def test_sobel_sparse_generators(generator):
    xp = get_framework()
    try:
        image_bin, dx_bin, sy_bin, sx_bin, dy_bin, threshold_bin = generator()
    except (FileNotFoundError, ImportError, ValueError) as e:
        pytest.skip(f"Failed to generate data: {e}")

    result_bin = benchmark_mri_edge(
        xp, image_bin, dx_bin, sy_bin, sx_bin, dy_bin, threshold_bin
    )
    result = xp.from_benchmark(result_bin)

    image = xp.from_benchmark(image_bin)
    threshold = xp.from_benchmark(threshold_bin)

    expected = expected_sobel_edge(image, float(threshold))

    assert np.all(result == expected), "Outputs mismatched."
    assert result.dtype == bool
