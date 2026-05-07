import os
from typing import Any

import kagglehub
import numpy as np
from PIL import Image

import saps
from saps.benchmark import Benchmark, Contributor, Dataset, Generator, Ref
from saps_framework import BinsparseFormat

xp = saps.xp


class MaskedMRIDataset(Dataset):
    def __init__(
        self,
        name: str,
        category: str,
        filename: str,
        t1_val: float = 191.25,
        t2_val: float = 204.0,
        image: np.ndarray | None = None,
        roi: np.ndarray | None = None,
    ):
        self.source_name = name
        self.category = category
        self.filename = filename
        self.t1_val = t1_val
        self.t2_val = t2_val
        self.image = image
        self.roi = roi

    @property
    def name(self) -> str:
        return self.source_name

    @property
    def pretty_name(self) -> str:
        return f"Masked MRI Edge {self.source_name}"

    @property
    def description(self) -> str:
        return (
            f"MRI image {self.filename} with thresholds {self.t1_val} and "
            f"{self.t2_val}."
        )

    @property
    def tags(self) -> list[str]:
        return ["image-processing", "edge-detection", "mri", "sparse"]

    @property
    def metadata(self) -> dict[str, Any]:
        data = super().metadata
        data["category"] = self.category
        data["filename"] = self.filename
        data["t1_val"] = self.t1_val
        data["t2_val"] = self.t2_val
        return data


class MaskedMRIGenerator(Generator[MaskedMRIDataset]):
    @property
    def name(self) -> str:
        return "masked_mri_inputs"

    @property
    def pretty_name(self) -> str:
        return "Masked MRI Edge Data Generator"

    @property
    def description(self) -> str:
        return (
            "Data Generation: I used MRI image data from this Kaggle set: "
            "https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection. "
            "I used a constant edge threshold of 150.0 with all of the images "
            "that I used."
        )

    @property
    def tags(self) -> list[str]:
        return ["image-processing", "edge-detection", "mri", "sparse"]

    @property
    def authors(self) -> list[Contributor]:
        return MaskedMRIEdgeBenchmark().authors

    @property
    def references(self) -> list[Ref]:
        return MaskedMRIEdgeBenchmark().references

    @property
    def ai_disclosure(self) -> str:
        return MaskedMRIEdgeBenchmark().ai_disclosure

    @property
    def motivation(self) -> str:
        return MaskedMRIEdgeBenchmark().motivation

    @property
    def datasets(self) -> list[MaskedMRIDataset]:
        return [
            MaskedMRIDataset("masked_mri_1", "yes", "Y157.JPG"),
            MaskedMRIDataset("masked_mri_2", "yes", "Y6.jpg"),
            MaskedMRIDataset("masked_mri_3", "yes", "Y194.jpg"),
            MaskedMRIDataset("masked_mri_4", "yes", "Y180.jpg"),
        ]

    def generate(
        self, dataset: MaskedMRIDataset
    ) -> tuple[list[BinsparseFormat], dict[str, Any]]:
        if dataset.image is None:
            path = kagglehub.dataset_download(
                "navoneel/brain-mri-images-for-brain-tumor-detection"
            )
            img_path = os.path.join(path, dataset.category, dataset.filename)
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found at {img_path}")

            img = Image.open(img_path).convert("L")
            img_array = np.array(img, dtype=np.float32)
        else:
            img_array = np.array(dataset.image, dtype=np.float32)

        H, W = img_array.shape
        if dataset.roi is None:
            roi_array = np.zeros_like(img_array, dtype=bool)
            H_val, W_val = H // 4, W // 4
            roi_array[H_val : H - H_val, W_val : W - W_val] = True
        else:
            roi_array = np.array(dataset.roi, dtype=bool)

        image_bin = BinsparseFormat.from_numpy(img_array)
        roi_bin = BinsparseFormat.from_numpy(roi_array)
        t1_bin = BinsparseFormat.from_numpy(np.array(dataset.t1_val, dtype=np.float32))
        t2_bin = BinsparseFormat.from_numpy(np.array(dataset.t2_val, dtype=np.float32))

        return [image_bin, roi_bin, t1_bin, t2_bin], {}


class MaskedMRIEdgeBenchmark(Benchmark):
    @property
    def name(self) -> str:
        return "masked_mri_edge"

    @property
    def pretty_name(self) -> str:
        return "MRI Edge Detection"

    @property
    def description(self) -> str:
        return (
            "What does this code do: This code implements a masked edge detection "
            "algorithm on a 2D MRI image. The benchmark performs boolean threshold "
            "mask operations using t1=75% and t2=80% thresholds and a "
            "Region-of-Interest (ROI) mask."
        )

    @property
    def tags(self) -> list[str]:
        return ["image-processing", "edge-detection", "mri", "sparse"]

    @property
    def authors(self) -> list[Contributor]:
        return [Contributor("Aadharsh Rajkumar", "arajkumar34@gatech.edu")]

    @property
    def motivation(self) -> str:
        return (
            "Motivation: Edge detection is a crucial task that is a part of image "
            "processing pipelines. It is often the case that images and scans in "
            "the medical field rquire post-processing to extract useful "
            "information. In this case, we are using a 2D MRI image to produce "
            "thresholded edge maps. Since medical images are large and often "
            "contain redundant information, it is important to process them "
            "efficiently. The redundancy of MRI makes them a good candidate for "
            "sparse processing."
        )

    @property
    def references(self) -> list[Ref]:
        return [
            Ref(
                title="",
                authors=[],
                url="https://commit.csail.mit.edu/papers/2021/oopsla2021-array-programming.pdf",
            ),
            Ref(
                title="",
                authors=[],
                url="https://www.researchgate.net/publication/310464068_EDGE_DETECTION_OF_MRI_IMAGES_-A_REVIEW",
            ),
            Ref(
                title="",
                authors=[],
                url="https://pmc.ncbi.nlm.nih.gov/articles/PMC4948115/",
            ),
        ]

    @property
    def ai_disclosure(self) -> str:
        return (
            "Statement on the use of Generative AI: No generative AI was used to "
            "construct the benchmark function. Generative AI might have been used "
            "to construct tests. This statement is written by hand."
        )

    @property
    def generators(self) -> list[Generator[Any]]:
        return [MaskedMRIGenerator()]

    def benchmark(self, data: list[Any], meta: dict[str, Any]):
        img, roi, t1, t2 = data

        img_t1 = img > t1
        img_t2 = img > t2

        img_post = (img_t2 & roi) ^ (img_t1 & roi)

        return [img_post]
