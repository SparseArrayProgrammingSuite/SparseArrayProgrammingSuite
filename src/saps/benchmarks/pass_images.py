"""The first PASS photos, downsampled to 32x32 and cached once for benchmarks."""

from __future__ import annotations

import tarfile
from io import BytesIO
from pathlib import PurePosixPath
from typing import Any
from urllib.request import urlopen

import numpy as np

from binsparse.conversions import from_numpy, to_numpy

from saps import licenses
from saps.benchmark import (
    Author,
    Contributor,
    DataInstance,
    Dataset,
    Generator,
    License,
    Ref,
    ShellBenchmark,
)

# PASS's first shard. A Zenodo record's files never change, so the order of the
# photos in the shard is fixed, and reading only its first photos streams about
# 130 KB per photo instead of the whole 18.7 GB shard.
_PASS_SHARD_URL = "https://zenodo.org/records/6615455/files/PASS.0.tar?download=1"


class PASSImageDataset(Dataset):
    """The first 1000 photos of PASS's first shard, as 32x32 RGB images."""

    num_images = 1000
    size = 32

    @property
    def name(self) -> str:
        return "pass_32x32"

    @property
    def pretty_name(self) -> str:
        return "PASS 32x32"

    @property
    def description(self) -> str:
        return (
            "The first 1000 photos of PASS's first shard, center-cropped and "
            "downsampled to 32x32 RGB images."
        )

    @property
    def suites(self) -> list[str]:
        return []

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def metadata(self) -> dict[str, Any]:
        data = super().metadata
        data["num_images"] = self.num_images
        data["size"] = self.size
        return data


class PASSImageGenerator(Generator[PASSImageDataset]):
    """Downloads and caches the first PASS photos at 32x32."""

    @property
    def name(self) -> str:
        return "pass_images"

    @property
    def pretty_name(self) -> str:
        return "PASS Images"

    @property
    def description(self) -> str:
        return (
            "Streams the first photos of PASS's first shard from Zenodo, "
            "center-crops and downsamples them to 32x32 RGB images, and caches them "
            "with the hashes that name them in PASS's attribution metadata."
        )

    @property
    def suites(self) -> list[str]:
        # --cache-datasets prepares the shell suite.
        return ["shell"]

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def authors(self) -> list[Contributor]:
        return []

    @property
    def references(self) -> list[Ref]:
        return [
            Ref(
                title=(
                    "PASS: An ImageNet replacement for self-supervised pretraining "
                    "without humans"
                ),
                authors=[
                    Author("Yuki M. Asano"),
                    Author("Christian Rupprecht"),
                    Author("Andrew Zisserman"),
                    Author("Andrea Vedaldi"),
                ],
                conference="NeurIPS Datasets and Benchmarks",
                year=2021,
                url="https://arxiv.org/abs/2109.13228",
            ),
        ]

    @property
    def ai_disclosure(self) -> str:
        return "Generative AI was used to implement this generator."

    @property
    def notes(self) -> str:
        return (
            "The LTH Conv-2 network was trained on 32x32 CIFAR-10 images, which have "
            "no license. PASS's CC-BY photos stand in for them. Streaming only the "
            "first photos of one shard, and caching them once, keeps the download "
            "small."
        )

    @property
    def license(self) -> License | None:
        return licenses.PASS

    @property
    def datasets(self) -> list[PASSImageDataset]:
        return [PASSImageDataset()]

    def generate(self, dataset: PASSImageDataset) -> DataInstance:
        images, hashes = _load_pass_images(dataset.num_images, dataset.size)
        return DataInstance(
            inputs=[from_numpy(images)],
            meta={
                "zenodo_record": 6615455,
                "shard": "PASS.0.tar",
                "num_images": dataset.num_images,
                "size": dataset.size,
                "image_hashes": hashes,
            },
        )


def _load_pass_images(num_images: int, size: int) -> tuple[np.ndarray, list[str]]:
    """The first ``num_images`` PASS photos as (num_images, size, size, 3) uint8."""
    from PIL import Image

    images = np.empty((num_images, size, size, 3), dtype=np.uint8)
    hashes: list[str] = []
    with (
        urlopen(_PASS_SHARD_URL, timeout=60) as response,
        tarfile.open(fileobj=response, mode="r|") as archive,
    ):
        for member in archive:
            file = archive.extractfile(member)
            if file is None or not member.name.endswith(".jpg"):
                continue
            image = Image.open(BytesIO(file.read())).convert("RGB")
            # Crop the middle square, then average each output pixel's box.
            width, height = image.size
            side = min(width, height)
            left, top = (width - side) // 2, (height - side) // 2
            image = image.crop((left, top, left + side, top + side))
            image = image.resize((size, size), Image.Resampling.BOX)
            images[len(hashes)] = np.asarray(image)
            # Each file is named by the hash that keys PASS's pass_metadata.csv.
            hashes.append(PurePosixPath(member.name).stem)
            if len(hashes) == num_images:
                break
    if len(hashes) < num_images:
        raise ValueError(
            f"PASS.0.tar holds only {len(hashes)} photos, not {num_images}."
        )
    return images, hashes


class PASSImageShellBenchmark(ShellBenchmark):
    @property
    def generator(self) -> Generator:
        return PASSImageGenerator()


def fetch_pass_images() -> tuple[np.ndarray, dict[str, Any]]:
    """Fetch (and cache) the PASS photos, as (1000, 32, 32, 3) uint8 RGB images."""
    generator = PASSImageGenerator()
    raw = generator.cached_generate(generator.datasets[0])
    return np.asarray(to_numpy(raw.inputs[0])), raw.meta
