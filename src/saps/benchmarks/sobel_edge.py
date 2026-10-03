import os
from typing import Any

import numpy as np

from binsparse import BinsparseTensor, COORMatrix
from binsparse.conversions import from_numpy, to_numpy

from saps.benchmark import Benchmark, Contributor, DataInstance, Dataset, Generator, Ref


def generate_1d_sobel_matrices(Nx, Ny):
    rows = np.arange(Nx)
    cols1 = (rows + 1) % Nx
    cols2 = (rows - 1) % Nx
    D_x_R = np.concatenate([rows, rows])
    D_x_C = np.concatenate([cols1, cols2])
    D_x_V = np.concatenate([np.ones(Nx), -np.ones(Nx)])
    dx_bin = COORMatrix(
        (Nx, Nx),
        len(D_x_V),
        indices_0=D_x_R,
        indices_1=D_x_C,
        values=D_x_V.astype(np.float32),
    )

    rows = np.arange(Ny)
    cols1 = (rows - 1) % Ny
    cols2 = rows
    cols3 = (rows + 1) % Ny
    S_y_R = np.concatenate([rows, rows, rows])
    S_y_C = np.concatenate([cols1, cols2, cols3])
    S_y_V = np.concatenate([np.ones(Ny), 2.0 * np.ones(Ny), np.ones(Ny)])
    sy_bin = COORMatrix(
        (Ny, Ny),
        len(S_y_V),
        indices_0=S_y_R,
        indices_1=S_y_C,
        values=S_y_V.astype(np.float32),
    )

    rows = np.arange(Nx)
    cols1 = (rows - 1) % Nx
    cols2 = rows
    cols3 = (rows + 1) % Nx
    S_x_R = np.concatenate([rows, rows, rows])
    S_x_C = np.concatenate([cols1, cols2, cols3])
    S_x_V = np.concatenate([np.ones(Nx), 2.0 * np.ones(Nx), np.ones(Nx)])
    sx_bin = COORMatrix(
        (Nx, Nx),
        len(S_x_V),
        indices_0=S_x_R,
        indices_1=S_x_C,
        values=S_x_V.astype(np.float32),
    )

    rows = np.arange(Ny)
    cols1 = (rows + 1) % Ny
    cols2 = (rows - 1) % Ny
    D_y_R = np.concatenate([rows, rows])
    D_y_C = np.concatenate([cols1, cols2])
    D_y_V = np.concatenate([np.ones(Ny), -np.ones(Ny)])
    dy_bin = COORMatrix(
        (Ny, Ny),
        len(D_y_V),
        indices_0=D_y_R,
        indices_1=D_y_C,
        values=D_y_V.astype(np.float32),
    )

    return dx_bin, sy_bin, sx_bin, dy_bin


class MRISobelEdgeDetectionDataset(Dataset):
    def __init__(
        self,
        name: str,
        category: str,
        filename: str,
        threshold_val: float = 150.0,
        image: np.ndarray | None = None,
        ref_meta: dict[str, Any] | None = None,
        suites: list[str] | None = None,
        pretty_name: str | None = None,
    ):
        self._suites = suites or []
        self.source_name = name
        self._pretty_name = pretty_name or name
        self.category = category
        self.filename = filename
        self.threshold_val = threshold_val
        self.image = image
        self.ref_meta = ref_meta

    @property
    def name(self) -> str:
        return self.source_name

    @property
    def pretty_name(self) -> str:
        return self._pretty_name

    @property
    def description(self) -> str:
        return f"MRI image {self.filename} with edge threshold {self.threshold_val}."

    @property
    def suites(self) -> list[str]:
        return self._suites

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def metadata(self) -> dict[str, Any]:
        data = super().metadata
        data["category"] = self.category
        data["filename"] = self.filename
        data["threshold_val"] = self.threshold_val
        return data


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


class MRISobelEdgeDetectionTestGenerator(Generator[MRISobelEdgeDetectionDataset]):
    @property
    def name(self) -> str:
        return "mri_sobel_edge_detection_test"

    @property
    def pretty_name(self) -> str:
        return "Magnetic Resonance Imaging Sobel Edge Detection Test"

    @property
    def description(self) -> str:
        return "Small deterministic Sobel edge examples with reference outputs."

    @property
    def suites(self) -> list[str]:
        return ["test"]

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def authors(self) -> list[Contributor]:
        return MRISobelEdgeDetectionBenchmark().authors

    @property
    def references(self) -> list[Ref]:
        return MRISobelEdgeDetectionBenchmark().references

    @property
    def ai_disclosure(self) -> str:
        return MRISobelEdgeDetectionBenchmark().ai_disclosure

    @property
    def motivation(self) -> str:
        return "Provide small Sobel edge examples for benchmark correctness checks."

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def datasets(self) -> list[MRISobelEdgeDetectionDataset]:
        return [
            MRISobelEdgeDetectionDataset(
                "zero_image",
                "local",
                "zero_image",
                pretty_name="Zero Image",
                threshold_val=10.0,
                image=np.zeros((5, 5), dtype=np.float32),
            ),
            MRISobelEdgeDetectionDataset(
                "vertical_edge",
                "local",
                "vertical_edge",
                pretty_name="Vertical Edge",
                threshold_val=50.0,
                image=np.array(
                    [
                        [0, 0, 100, 100, 0],
                        [0, 0, 100, 100, 0],
                        [0, 0, 100, 100, 0],
                        [0, 0, 100, 100, 0],
                        [0, 0, 100, 100, 0],
                    ],
                    dtype=np.float32,
                ),
            ),
            MRISobelEdgeDetectionDataset(
                "horizontal_edge",
                "local",
                "horizontal_edge",
                pretty_name="Horizontal Edge",
                threshold_val=50.0,
                image=np.array(
                    [
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [100, 100, 100, 100, 100],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                    ],
                    dtype=np.float32,
                ),
            ),
            MRISobelEdgeDetectionDataset(
                "zero_3x4",
                "local",
                "zero_3x4",
                pretty_name="Zero 3x4",
                threshold_val=7.0,
                image=np.zeros((3, 4), dtype=np.float32),
                ref_meta={
                    "input_count": 6,
                    "image_shape": (3, 4),
                    "threshold": 7.0,
                },
            ),
        ]

    def generate(self, dataset: MRISobelEdgeDetectionDataset) -> DataInstance:
        problem = MRISobelEdgeDetectionKaggleGenerator().generate(dataset)
        expected = expected_sobel_edge(dataset.image, dataset.threshold_val)
        return DataInstance(
            inputs=problem.inputs,
            meta=problem.meta,
            ref_outputs=[from_numpy(expected)],
            ref_meta=dataset.ref_meta,
        )


class MRISobelEdgeDetectionKaggleGenerator(Generator[MRISobelEdgeDetectionDataset]):
    @property
    def name(self) -> str:
        return "mri_sobel_edge_detection_kaggle"

    @property
    def pretty_name(self) -> str:
        return "Magnetic Resonance Imaging Sobel Edge Detection Kaggle"

    @property
    def description(self) -> str:
        return (
            "Data Generation: I used MRI image data from this Kaggle set:"
            " https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection"
            " . I used a constant edge threshold of 150.0 with all of the images that I"
            " used."
        )

    @property
    def suites(self) -> list[str]:
        return ["standard"]

    @property
    def concepts(self) -> str:
        return """
        <ccs2012>
        <concept>
        <concept_id>10010147.10010371.10010382.10010383</concept_id>
        <concept_desc>Computing methodologies~Image processing</concept_desc>
        <concept_significance>500</concept_significance>
        </concept>
        <concept>
        <concept_id>10010147.10010371.10010382.10010236</concept_id>
        <concept_desc>Computing methodologies~Computational photography</concept_desc>
        <concept_significance>500</concept_significance>
        </concept>
        <concept>
        <concept_id>10010405.10010444.10010087.10010096</concept_id>
        <concept_desc>Applied computing~Imaging</concept_desc>
        <concept_significance>500</concept_significance>
        </concept>
        </ccs2012>
        """

    @property
    def authors(self) -> list[Contributor]:
        return MRISobelEdgeDetectionBenchmark().authors

    @property
    def references(self) -> list[Ref]:
        return MRISobelEdgeDetectionBenchmark().references

    @property
    def ai_disclosure(self) -> str:
        return MRISobelEdgeDetectionBenchmark().ai_disclosure

    @property
    def motivation(self) -> str:
        return MRISobelEdgeDetectionBenchmark().motivation

    @property
    def datasets(self) -> list[MRISobelEdgeDetectionDataset]:
        return [
            MRISobelEdgeDetectionDataset(
                "yes/Y157", "yes", "Y157.JPG", suites=["trace"]
            ),
            MRISobelEdgeDetectionDataset(
                "yes/Y6", "yes", "Y6.jpg", suites=["trace", "train"]
            ),
            MRISobelEdgeDetectionDataset(
                "yes/Y194", "yes", "Y194.jpg", suites=["trace"]
            ),
            MRISobelEdgeDetectionDataset(
                "yes/Y180", "yes", "Y180.jpg", suites=["trace"]
            ),
        ]

    def generate(self, dataset: MRISobelEdgeDetectionDataset) -> DataInstance:
        if dataset.image is None:
            from PIL import Image

            from saps.downloaders.kaggle import download_kaggle_dataset

            path = download_kaggle_dataset(
                "navoneel/brain-mri-images-for-brain-tumor-detection"
            )
            img_path = os.path.join(path, dataset.category, dataset.filename)
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found at {img_path}")

            img = Image.open(img_path).convert("L")
            img_array = np.array(img, dtype=np.float32)
        else:
            img_array = np.array(dataset.image, dtype=np.float32)

        image_bin = from_numpy(img_array)
        threshold_bin = from_numpy(np.array(dataset.threshold_val, dtype=np.float32))

        Nx, Ny = img_array.shape
        dx_bin, sy_bin, sx_bin, dy_bin = generate_1d_sobel_matrices(Nx, Ny)
        return DataInstance(
            inputs=[image_bin, dx_bin, sy_bin, sx_bin, dy_bin, threshold_bin],
            meta={},
        )


class MRISobelEdgeDetectionBenchmark(Benchmark):
    @property
    def name(self) -> str:
        return "mri_sobel_edge_detection"

    @property
    def pretty_name(self) -> str:
        return "Magnetic Resonance Imaging Sobel Edge Detection"

    @property
    def description(self) -> str:
        return (
            "What does this code do: This code implements a simple edge detection"
            " algorithm on a 2D MRI image. The algorithm computes the gradients in the"
            " X and Y directions using the concept of a Sobel operator, which is a"
            " common method for edge detection. The sobel operator was recreated using"
            " array shifts that account for sparse patterns. The magnitude of the"
            " gradients is computed and then masked with a threshold to produce a"
            " binary edge map."
        )

    @property
    def suites(self) -> list[str]:
        return ["standard-spatial"]

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def authors(self) -> list[Contributor]:
        return [Contributor("Aadharsh Rajkumar", "arajkumar34@gatech.edu")]

    @property
    def motivation(self) -> str:
        return (
            "Motivation: Edge detection is a crucial task that is a part of image"
            " processing pipelines. It is often the case that images and scans in the"
            " medical field rquire post-processing to extract useful information. In"
            " this case, we are using a 2D MRI image to produce thresholded edge maps."
            " Since medical images are large and often contain redundant information,"
            " it is important to process them efficiently. The redundancy of MRI makes"
            " them a good candidate for sparse processing."
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
        return [
            MRISobelEdgeDetectionTestGenerator(),
            MRISobelEdgeDetectionKaggleGenerator(),
        ]

    def benchmark(self, xp, data: list[Any], meta: dict[str, Any]):
        image, D_x, S_y, S_x, D_y, threshold = data

        gx = D_x @ image @ S_y
        gy = S_x @ image @ D_y

        magnitude = xp.abs(gx) + xp.abs(gy)
        edges = magnitude > threshold

        return [edges]

    def check(self, param):
        for item in self._output:
            assert isinstance(item, BinsparseTensor), (
                "Output must be in binsparse format"
            )
        if self._ref_outputs is None:
            return

        actual = to_numpy(self._output[0])
        expected = to_numpy(self._ref_outputs[0])

        assert self._meta == {}
        assert actual.shape == expected.shape
        assert np.all(actual == expected)

        if self._ref_meta is None:
            return
        assert len(self._input) == self._ref_meta["input_count"]
        assert tuple(self._input[0].shape) == self._ref_meta["image_shape"]
        assert to_numpy(self._input[-1]).item() == self._ref_meta["threshold"]
