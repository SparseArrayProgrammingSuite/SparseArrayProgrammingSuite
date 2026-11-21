import gzip
import struct

import pytest

import numpy as np

from binsparse.conversions import from_numpy, to_numpy, to_scipy

from frameworks.saps_numpy import NumpyFramework
from saps.benchmark import Param
from saps.benchmarks.mri_masked_edge_detection import (
    MRIMaskedEdgeDetectionBenchmark,
    MRIMaskedEdgeDetectionDataset,
    MRIMaskedEdgeDetectionIXIGenerator,
    MRIMaskedEdgeDetectionKaggleGenerator,
    MRIMaskedEdgeDetectionTestGenerator,
    default_masked_mri_roi,
    reference_masked_mri_edge,
)
from saps.benchmarks.mri_sobel_edge_detection import (
    MRISobelEdgeDetectionBenchmark,
    MRISobelEdgeDetectionDataset,
    MRISobelEdgeDetectionIXIGenerator,
    MRISobelEdgeDetectionKaggleGenerator,
    MRISobelEdgeDetectionTestGenerator,
    generate_1d_sobel_matrices,
    reference_sobel_edge,
)
from saps.downloaders.ixi import load_ixi_scan, read_nifti_int16


def _matrix_sobel_edge(image, threshold):
    """The benchmark's own definition, evaluated with dense NumPy matrices."""
    D_x, S_y, S_x, D_y = (
        to_scipy(m).toarray().astype(np.float64)
        for m in generate_1d_sobel_matrices(*image.shape[-2:])
    )
    image = image.astype(np.float64)
    return np.abs(D_x @ image @ S_y) + np.abs(S_x @ image @ D_y) > threshold


def _random_image(rng, shape):
    return rng.integers(0, 256, size=shape).astype(np.float32)


@pytest.mark.parametrize("shape", [(3, 4), (5, 5), (17, 9), (32, 48)])
@pytest.mark.parametrize("threshold", [0.0, 150.0, 600.0])
def test_sobel_reference_matches_benchmark_matrices(shape, threshold):
    rng = np.random.default_rng(0)
    image = _random_image(rng, shape)
    np.testing.assert_array_equal(
        reference_sobel_edge(image, threshold), _matrix_sobel_edge(image, threshold)
    )


def test_sobel_reference_marks_rows_beside_horizontal_line():
    image = np.zeros((5, 5), dtype=np.float32)
    image[2] = 100
    expected = np.zeros((5, 5), dtype=bool)
    expected[[1, 3]] = True
    np.testing.assert_array_equal(reference_sobel_edge(image, 50.0), expected)


def test_masked_reference_matches_xor_on_random_images():
    rng = np.random.default_rng(0)
    for _ in range(50):
        shape = tuple(int(n) for n in rng.integers(1, 20, size=2))
        image = _random_image(rng, shape)
        roi = rng.random(shape) < 0.5
        # Integer thresholds hit pixels exactly; either order and t1 == t2 occur.
        t1, t2 = (float(t) for t in rng.integers(0, 256, size=2))
        expected = ((image > t2) & roi) ^ ((image > t1) & roi)
        np.testing.assert_array_equal(
            reference_masked_mri_edge(image, roi, t1, t2), expected
        )


def _fake_kaggle(monkeypatch, tmp_path, image):
    from PIL import Image

    (tmp_path / "yes").mkdir()
    Image.fromarray(image.astype(np.uint8)).save(tmp_path / "yes" / "Y6.png")
    monkeypatch.setattr(
        "saps.downloaders.kaggle.download_kaggle_dataset", lambda _: str(tmp_path)
    )


def test_sobel_kaggle_generator_records_reference(monkeypatch, tmp_path):
    image = _random_image(np.random.default_rng(1), (24, 31))
    _fake_kaggle(monkeypatch, tmp_path, image)
    problem = MRISobelEdgeDetectionKaggleGenerator().generate(
        MRISobelEdgeDetectionDataset("yes/Y6", "yes", "Y6.png")
    )
    np.testing.assert_array_equal(
        to_numpy(problem.ref_outputs[0]), _matrix_sobel_edge(image, 150.0)
    )


def test_masked_kaggle_generator_records_reference(monkeypatch, tmp_path):
    image = _random_image(np.random.default_rng(1), (24, 31))
    _fake_kaggle(monkeypatch, tmp_path, image)
    problem = MRIMaskedEdgeDetectionKaggleGenerator().generate(
        MRIMaskedEdgeDetectionDataset("yes/Y6", "yes", "Y6.png")
    )
    roi = default_masked_mri_roi(image)
    expected = ((image > 204.0) & roi) ^ ((image > 191.25) & roi)
    np.testing.assert_array_equal(to_numpy(problem.ref_outputs[0]), expected)


_SCAN = "IXI002-Guys-0828-T1"


def _nifti(volume, datatype=4, slope=1.0):
    """An uncompressed NIfTI-1 file of a C-order volume, slices first."""
    volume = np.asarray(volume)
    header = bytearray(352)
    struct.pack_into("<i", header, 0, 348)
    dim = (volume.ndim, *reversed(volume.shape))
    struct.pack_into("<8h", header, 40, *dim, *[1] * (8 - len(dim)))
    struct.pack_into("<hh", header, 70, datatype, 16)
    struct.pack_into("<ff", header, 108, 352.0, slope)
    header[344:348] = b"n+1\0"
    return bytes(header) + volume.astype("<i2").tobytes()


def _fake_ixi(monkeypatch, tmp_path, volume, slope=1.0):
    """Cache the scan _SCAN as a NIfTI file holding ``volume``."""
    (tmp_path / "ixi").mkdir()
    path = tmp_path / "ixi" / f"{_SCAN}.nii.gz"
    path.write_bytes(gzip.compress(_nifti(volume, slope=slope)))
    monkeypatch.setenv("SAPS_CACHE_DIR", str(tmp_path))


def _random_volume(rng, shape):
    return rng.integers(0, 2000, size=shape).astype(np.int16)


def test_ixi_scan_is_stored_int16_voxels_slices_first(monkeypatch, tmp_path):
    volume = np.arange(24, dtype=np.int16).reshape(2, 3, 4)
    # The scale factor is ignored.
    _fake_ixi(monkeypatch, tmp_path, volume, slope=2.0)
    scan = load_ixi_scan(_SCAN)
    assert scan.dtype == np.int16
    np.testing.assert_array_equal(scan, volume)


def test_read_nifti_requires_int16():
    with pytest.raises(ValueError, match="Expected int16 voxels"):
        read_nifti_int16(_nifti(np.zeros((2, 2)), datatype=16))


def test_sobel_ixi_generator_thresholds_at_99th_percentile(monkeypatch, tmp_path):
    volume = _random_volume(np.random.default_rng(1), (3, 24, 31))
    _fake_ixi(monkeypatch, tmp_path, volume)
    problem = MRISobelEdgeDetectionIXIGenerator().generate(
        MRISobelEdgeDetectionDataset(_SCAN, "T1", f"{_SCAN}.nii.gz", None)
    )
    threshold = round(float(np.percentile(volume, 99)))
    assert to_numpy(problem.inputs[0]).dtype == np.int16
    assert to_numpy(problem.inputs[-1]).item() == threshold
    np.testing.assert_array_equal(
        to_numpy(problem.ref_outputs[0]), _matrix_sobel_edge(volume, threshold)
    )


def test_masked_ixi_generator_thresholds_at_99th_percentile(monkeypatch, tmp_path):
    volume = _random_volume(np.random.default_rng(1), (3, 24, 31))
    _fake_ixi(monkeypatch, tmp_path, volume)
    problem = MRIMaskedEdgeDetectionIXIGenerator().generate(
        MRIMaskedEdgeDetectionDataset(_SCAN, "T1", f"{_SCAN}.nii.gz", None, None)
    )
    p99 = float(np.percentile(volume, 99))
    t1, t2 = round(0.75 * p99), round(0.8 * p99)
    assert [to_numpy(t).item() for t in problem.inputs[2:]] == [t1, t2]
    roi = default_masked_mri_roi(volume)
    expected = ((volume > t2) & roi) ^ ((volume > t1) & roi)
    np.testing.assert_array_equal(to_numpy(problem.ref_outputs[0]), expected)


@pytest.mark.parametrize(
    ("benchmark_cls", "generator_cls"),
    [
        (MRISobelEdgeDetectionBenchmark, MRISobelEdgeDetectionTestGenerator),
        (MRIMaskedEdgeDetectionBenchmark, MRIMaskedEdgeDetectionTestGenerator),
    ],
)
def test_int16_volume_passes_check(benchmark_cls, generator_cls):
    # Its Sobel gradients overflow int16, so this needs the int32 stencils.
    generator = generator_cls()
    (dataset,) = [d for d in generator.datasets if d.name == "int16_volume"]
    benchmark = benchmark_cls()
    param = Param(generator, dataset)
    benchmark.setup(param, use_cache=False, xp=NumpyFramework())
    benchmark.run(param)
    benchmark.check(param)
    assert to_numpy(benchmark._output[0]).any()


_CASES = [
    pytest.param(
        MRISobelEdgeDetectionBenchmark,
        MRISobelEdgeDetectionDataset(_SCAN, "T1", f"{_SCAN}.nii.gz", None),
        "scipy.ndimage Sobel reference",
        id="sobel",
    ),
    pytest.param(
        MRIMaskedEdgeDetectionBenchmark,
        MRIMaskedEdgeDetectionDataset(_SCAN, "T1", f"{_SCAN}.nii.gz", None, None),
        "NumPy threshold-band reference",
        id="masked",
    ),
]


def _checked_benchmark(benchmark_cls, monkeypatch, tmp_path, dataset):
    """Set up and run a benchmark on a fake IXI scan with NumPy."""
    _fake_ixi(monkeypatch, tmp_path, _random_volume(np.random.default_rng(2), (20, 20)))
    benchmark = benchmark_cls()
    param = Param(benchmark.generators[1], dataset)
    benchmark.setup(param, use_cache=False, xp=NumpyFramework())
    benchmark.run(param)
    return benchmark, param


@pytest.mark.parametrize(("benchmark_cls", "dataset", "label"), _CASES)
def test_check_accepts_correct_output(
    benchmark_cls, dataset, label, monkeypatch, tmp_path
):
    benchmark, param = _checked_benchmark(benchmark_cls, monkeypatch, tmp_path, dataset)
    benchmark.check(param)


@pytest.mark.parametrize(("benchmark_cls", "dataset", "label"), _CASES)
def test_check_rejects_flipped_pixel(
    benchmark_cls, dataset, label, monkeypatch, tmp_path
):
    benchmark, param = _checked_benchmark(benchmark_cls, monkeypatch, tmp_path, dataset)
    output = to_numpy(benchmark._output[0]).copy()
    output[7, 3] = ~output[7, 3]
    benchmark._output = [from_numpy(output)]
    with pytest.raises(
        AssertionError, match=f"1 of 400 pixels differ from the {label}"
    ):
        benchmark.check(param)


@pytest.mark.parametrize(("benchmark_cls", "dataset", "label"), _CASES)
def test_check_requires_reference(benchmark_cls, dataset, label, monkeypatch, tmp_path):
    benchmark, param = _checked_benchmark(benchmark_cls, monkeypatch, tmp_path, dataset)
    benchmark._ref_outputs = None
    with pytest.raises(AssertionError, match="regenerate the cached dataset"):
        benchmark.check(param)
