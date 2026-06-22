import pytest

import numpy as np

import saps.benchmarks.mri_edge as mri_edge
from frameworks.saps_numpy import NumpyFramework


def run_masked_mri_benchmark(xp, data):
    benchmark = mri_edge.MaskedMRIEdgeBenchmark()
    prev_xp = getattr(mri_edge, "xp", None)
    mri_edge.xp = xp
    try:
        (result,) = benchmark.benchmark(data, {})
    finally:
        mri_edge.xp = prev_xp
    return result


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
    xp = NumpyFramework()
    dataset = mri_edge.MaskedMRIDataset(
        "local", "local", "local", t1_val=t1, t2_val=t2, image=image, roi=roi
    )
    problem = mri_edge.MaskedMRIGenerator().generate(dataset)
    data_binsparse = problem.inputs
    meta = problem.meta
    data = [xp.from_binsparse(array) for array in data_binsparse]

    result = run_masked_mri_benchmark(xp, data)
    expected = expected_masked_mri_edge(image, roi, t1, t2)

    assert meta == {}
    assert result.shape == expected.shape
    assert np.all(result == expected)


def test_masked_mri_generator_builds_default_roi():
    xp = NumpyFramework()
    image = np.arange(36, dtype=np.float32).reshape(6, 6)
    dataset = mri_edge.MaskedMRIDataset(
        "tiny", "local", "tiny", t1_val=10.0, t2_val=20.0, image=image
    )

    problem = mri_edge.MaskedMRIGenerator().generate(dataset)
    data_binsparse = problem.inputs
    meta = problem.meta
    image_arr, roi_arr, t1_arr, t2_arr = [
        xp.from_binsparse(array) for array in data_binsparse
    ]

    expected_roi = np.zeros_like(image, dtype=bool)
    expected_roi[1:5, 1:5] = True

    assert meta == {}
    assert np.all(image_arr == image)
    assert np.all(roi_arr == expected_roi)
    assert t1_arr.item() == 10.0
    assert t2_arr.item() == 20.0
