import os
from typing import Any

import numpy as np

from binsparse import BinsparseTensor

from saps.benchmark import Benchmark, Contributor, DataInstance, Dataset, Generator, Ref


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
        suites: list[str] | None = None,
        ref_meta: dict[str, Any] | None = None,
    ):
        self._suites = suites or []
        self.source_name = name
        self.category = category
        self.filename = filename
        self.t1_val = t1_val
        self.t2_val = t2_val
        self.image = image
        self.roi = roi
        self.ref_meta = ref_meta

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
        data["t1_val"] = self.t1_val
        data["t2_val"] = self.t2_val
        return data


def expected_masked_mri_edge(image, roi, t1, t2):
    img_t1 = image > t1
    img_t2 = image > t2
    return (img_t2 & roi) ^ (img_t1 & roi)


def default_masked_mri_roi(image):
    expected_roi = np.zeros_like(image, dtype=bool)
    height, width = image.shape
    h_val, w_val = height // 4, width // 4
    expected_roi[h_val : height - h_val, w_val : width - w_val] = True
    return expected_roi


class MaskedMRITestGenerator(Generator[MaskedMRIDataset]):
    @property
    def name(self) -> str:
        return "masked_mri_test_inputs"

    @property
    def pretty_name(self) -> str:
        return "Masked MRI Edge Test Data Generator"

    @property
    def description(self) -> str:
        return "Small deterministic masked MRI examples."

    @property
    def suites(self) -> list[str]:
        return ["test", "trace"]

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

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
        return "Provide small masked MRI examples for correctness checks."

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def datasets(self) -> list[MaskedMRIDataset]:
        return [
            MaskedMRIDataset(
                "test_masked_mri_zero_image",
                "local",
                "test_masked_mri_zero_image",
                t1_val=50.0,
                t2_val=100.0,
                image=np.zeros((5, 5), dtype=np.float32),
                roi=np.ones((5, 5), dtype=bool),
                suites=["test", "trace"],
            ),
            MaskedMRIDataset(
                "test_masked_mri_basic_roi",
                "local",
                "test_masked_mri_basic_roi",
                t1_val=75.0,
                t2_val=125.0,
                image=np.array(
                    [
                        [0, 50, 100, 150, 200],
                        [0, 50, 100, 150, 200],
                        [0, 50, 100, 150, 200],
                        [0, 50, 100, 150, 200],
                        [0, 50, 100, 150, 200],
                    ],
                    dtype=np.float32,
                ),
                roi=np.array(
                    [
                        [False, False, False, False, False],
                        [False, True, True, True, False],
                        [False, True, True, True, False],
                        [False, True, True, True, False],
                        [False, False, False, False, False],
                    ],
                    dtype=bool,
                ),
                suites=["test", "trace"],
            ),
            MaskedMRIDataset(
                "test_masked_mri_generator_builds_default_roi",
                "local",
                "test_masked_mri_generator_builds_default_roi",
                t1_val=10.0,
                t2_val=20.0,
                image=np.arange(36, dtype=np.float32).reshape(6, 6),
                suites=["test", "trace"],
                ref_meta={"default_roi": True},
            ),
        ]

    def generate(self, dataset: MaskedMRIDataset) -> DataInstance:
        problem = MaskedMRIGenerator().generate(dataset)
        roi = dataset.roi
        if roi is None:
            roi = default_masked_mri_roi(dataset.image)
        expected = expected_masked_mri_edge(
            dataset.image, roi, dataset.t1_val, dataset.t2_val
        )
        return DataInstance(
            inputs=problem.inputs,
            meta=problem.meta,
            ref_outputs=[BinsparseTensor.from_numpy(expected)],
            ref_meta=dataset.ref_meta,
        )


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
            "Data Generation: I used MRI image data from this Kaggle set:"
            " https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection"
            " . I used a constant edge threshold of 150.0 with all of the images that I"
            " used."
        )

    @property
    def suites(self) -> list[str]:
        return []

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

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

    def generate(self, dataset: MaskedMRIDataset) -> DataInstance:
        if dataset.image is None:
            import kagglehub
            from PIL import Image

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

        image_bin = BinsparseTensor.from_numpy(img_array)
        roi_bin = BinsparseTensor.from_numpy(roi_array)
        t1_bin = BinsparseTensor.from_numpy(np.array(dataset.t1_val, dtype=np.float32))
        t2_bin = BinsparseTensor.from_numpy(np.array(dataset.t2_val, dtype=np.float32))

        return DataInstance(inputs=[image_bin, roi_bin, t1_bin, t2_bin], meta={})


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
            "What does this code do: This code implements a masked edge detection"
            " algorithm on a 2D MRI image. The benchmark performs boolean threshold"
            " mask operations using t1=75% and t2=80% thresholds and a"
            " Region-of-Interest (ROI) mask."
        )

    @property
    def suites(self) -> list[str]:
        return []

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
        return [MaskedMRITestGenerator(), MaskedMRIGenerator()]

    def benchmark(self, xp, data: list[Any], meta: dict[str, Any]):
        img, roi, t1, t2 = data

        img_t1 = img > t1
        img_t2 = img > t2

        img_post = (img_t2 & roi) ^ (img_t1 & roi)

        return [img_post]

    def check(self, param):
        for item in self._output:
            assert isinstance(item, BinsparseTensor), (
                "Output must be in binsparse format"
            )
        if self._ref_outputs is None:
            return

        result = self._output[0].data["values"].reshape(self._output[0].data["shape"])
        expected = (
            self._ref_outputs[0]
            .data["values"]
            .reshape(self._ref_outputs[0].data["shape"])
        )

        assert self._meta == {}
        assert result.shape == expected.shape
        assert np.all(result == expected)

        if self._ref_meta is None:
            return
        if self._ref_meta.get("default_roi"):
            image_arr = (
                self._input[0].data["values"].reshape(self._input[0].data["shape"])
            )
            roi_arr = (
                self._input[1].data["values"].reshape(self._input[1].data["shape"])
            )
            expected_roi = default_masked_mri_roi(image_arr)
            assert np.all(roi_arr == expected_roi)
            assert self._input[2].data["values"].item() == 10.0
            assert self._input[3].data["values"].item() == 20.0
