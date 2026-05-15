import pytest

import numpy as np

import saps.benchmarks.sobel_edge as sobel_edge
from frameworks.saps_numpy import NumpyFramework


def run_sobel_benchmark(xp, data):
    benchmark = sobel_edge.MRISobelEdgeBenchmark()
    prev_xp = getattr(sobel_edge, "xp", None)
    sobel_edge.xp = xp
    try:
        (edges,) = benchmark.benchmark(data, {})
    finally:
        sobel_edge.xp = prev_xp
    return edges


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
    xp = NumpyFramework()
    dataset = sobel_edge.MRISobelDataset(
        "local", "local", "local", threshold_val=threshold, image=image
    )
    data_binsparse, meta = sobel_edge.MRISobelGenerator().generate(dataset)
    data = [xp.from_binsparse(array) for array in data_binsparse]

    result = run_sobel_benchmark(xp, data)
    expected = expected_sobel_edge(image, threshold)

    assert meta == {}
    assert result.shape == expected.shape
    assert np.all(result == expected)


def test_sobel_generator_metadata():
    image = np.zeros((3, 4), dtype=np.float32)
    dataset = sobel_edge.MRISobelDataset(
        "tiny", "local", "tiny", threshold_val=7.0, image=image
    )

    data, meta = sobel_edge.MRISobelGenerator().generate(dataset)

    assert len(data) == 6
    assert data[0].data["shape"] == image.shape
    assert data[-1].data["values"].item() == 7.0
    assert meta == {}
