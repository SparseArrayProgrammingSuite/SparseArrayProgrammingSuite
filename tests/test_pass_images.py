import io
import tarfile

import pytest

import numpy as np

from binsparse.conversions import to_numpy
from PIL import Image

from saps.benchmarks import lth_conv2, pass_images
from saps.benchmarks.pass_images import (
    PASSImageGenerator,
    PASSImageShellBenchmark,
    _load_pass_images,
    fetch_pass_images,
)


def _photo(width, height, color):
    """A photo split into thirds along its long side, ``color`` in the middle."""
    pixels = np.zeros((height, width, 3), dtype=np.uint8)
    if width >= height:
        pixels[:, width // 3 : 2 * width // 3] = color
    else:
        pixels[height // 3 : 2 * height // 3] = color
    data = io.BytesIO()
    # PNG keeps the colors exact; Pillow reads a file by its contents, not its name.
    Image.fromarray(pixels).save(data, format="PNG")
    return data.getvalue()


def _shard(members):
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w") as archive:
        directory = tarfile.TarInfo("PASS_dataset/5")
        directory.type = tarfile.DIRTYPE
        archive.addfile(directory)
        for name, content in members:
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    return data.getvalue()


@pytest.fixture
def shard(monkeypatch):
    """Serve a small PASS shard in place of Zenodo's."""
    urls = []

    def serve(members):
        content = _shard(members)

        def fake_urlopen(url, timeout):
            urls.append(url)
            return io.BytesIO(content)

        monkeypatch.setattr(pass_images, "urlopen", fake_urlopen)
        return urls

    return serve


def test_pass_loader_center_crops_and_downsamples_the_first_photos(shard):
    urls = shard(
        [
            ("PASS_dataset/5/README.txt", b"not a photo"),
            ("PASS_dataset/5/aa11.jpg", _photo(24, 8, (255, 0, 0))),
            ("PASS_dataset/4/bb22.jpg", _photo(8, 24, (0, 0, 255))),
            # Never read: the loader stops once it has its photos.
            ("PASS_dataset/7/cc33.jpg", b"not a jpeg"),
        ]
    )

    images, hashes = _load_pass_images(2, 4)

    assert urls == [pass_images._PASS_SHARD_URL]
    assert hashes == ["aa11", "bb22"]
    assert images.shape == (2, 4, 4, 3)
    assert images.dtype == np.uint8
    # The middle square of each photo is all its middle color.
    np.testing.assert_array_equal(images[0], np.broadcast_to([255, 0, 0], (4, 4, 3)))
    np.testing.assert_array_equal(images[1], np.broadcast_to([0, 0, 255], (4, 4, 3)))


def test_pass_loader_rejects_a_shard_with_too_few_photos(shard):
    shard([("PASS_dataset/5/aa11.jpg", _photo(8, 8, (0, 255, 0)))])

    with pytest.raises(ValueError, match="only 1 photos, not 2"):
        _load_pass_images(2, 4)


def test_pass_shell_generator_caches_photos_with_their_hashes(monkeypatch):
    photos = np.arange(2 * 4 * 4 * 3, dtype=np.uint8).reshape(2, 4, 4, 3)
    monkeypatch.setattr(
        pass_images, "_load_pass_images", lambda count, size: (photos, ["aa", "bb"])
    )
    generator = PASSImageGenerator()
    dataset = generator.datasets[0]

    instance = generator.generate(dataset)

    assert generator.cacheable
    assert dataset.suites == []
    assert (dataset.num_images, dataset.size) == (1000, 32)
    assert PASSImageShellBenchmark().generator.name == "pass_images"
    np.testing.assert_array_equal(to_numpy(instance.inputs[0]), photos)
    assert instance.meta["image_hashes"] == ["aa", "bb"]
    assert instance.meta["zenodo_record"] == 6615455


def test_fetch_pass_images_uses_shared_cache(monkeypatch):
    photos = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    monkeypatch.setattr(
        pass_images, "_load_pass_images", lambda count, size: (photos, ["aa", "bb"])
    )
    calls = []

    def fake_cached_generate(self, dataset):
        calls.append((self.name, dataset.name))
        return PASSImageGenerator().generate(dataset)

    monkeypatch.setattr(PASSImageGenerator, "cached_generate", fake_cached_generate)

    images, meta = fetch_pass_images()

    assert calls == [("pass_images", "pass_32x32")]
    np.testing.assert_array_equal(images, photos)
    assert meta["image_hashes"] == ["aa", "bb"]


def test_lth_normalizes_pass_photos_channels_first(monkeypatch):
    photos = np.random.default_rng(0).integers(0, 256, (3, 32, 32, 3), dtype=np.uint8)
    monkeypatch.setattr(pass_images, "fetch_pass_images", lambda: (photos, {}))

    images = lth_conv2._pass_images(2)

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    expected = (photos[:2] / np.float32(255) - mean) / std
    assert images.shape == (2, 3, 32, 32)
    assert images.dtype == np.float32
    np.testing.assert_allclose(images, expected.transpose(0, 3, 1, 2), rtol=1e-6)
    with pytest.raises(ValueError, match="caches 3 photos, not 4"):
        lth_conv2._pass_images(4)
