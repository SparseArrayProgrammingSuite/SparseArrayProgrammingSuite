import os
from typing import Any

import numpy as np

from binsparse import BinsparseTensor
from binsparse.conversions import from_numpy, to_numpy

from saps import licenses
from saps.benchmark import (
    Author,
    Benchmark,
    Contributor,
    DataInstance,
    Dataset,
    Generator,
    License,
    Ref,
)
from saps.downloaders.ixi import IXI_REF, load_ixi_scan


class MRIMaskedEdgeDetectionDataset(Dataset):
    def __init__(
        self,
        name: str,
        category: str,
        filename: str,
        t1_val: float | None = 191.25,
        t2_val: float | None = 204.0,
        image: np.ndarray | None = None,
        roi: np.ndarray | None = None,
        suites: list[str] | None = None,
        ref_meta: dict[str, Any] | None = None,
        pretty_name: str | None = None,
    ):
        """``t1_val=t2_val=None`` thresholds at 75% and 80% of the image's 99th
        percentile."""
        self._suites = suites or []
        self.source_name = name
        self._pretty_name = pretty_name or name
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
        return self._pretty_name

    @property
    def description(self) -> str:
        if self.t1_val is None or self.t2_val is None:
            return (
                f"MRI image {self.filename} with thresholds at 75% and 80% of its "
                "99th-percentile voxel value."
            )
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


def reference_masked_mri_edge(image, roi, t1, t2):
    """Mark ROI pixels above exactly one threshold."""
    low, high = sorted((np.float32(t1), np.float32(t2)))
    image = np.asarray(image, dtype=np.float32)
    return np.asarray(roi, dtype=bool) & (low < image) & (image <= high)


def default_masked_mri_roi(image):
    """The middle half of the image along each axis."""
    expected_roi = np.zeros_like(image, dtype=bool)
    expected_roi[tuple(slice(n // 4, n - n // 4) for n in image.shape)] = True
    return expected_roi


def masked_edge_instance(
    dataset: MRIMaskedEdgeDetectionDataset, img_array
) -> DataInstance:
    """The thresholds take the image's dtype, so int16 images compare with
    int16 thresholds."""
    if dataset.roi is None:
        roi_array = default_masked_mri_roi(img_array)
    else:
        roi_array = np.array(dataset.roi, dtype=bool)
    t1_val: float
    t2_val: float
    if dataset.t1_val is None or dataset.t2_val is None:
        p99 = float(np.percentile(img_array, 99))
        t1_val, t2_val = round(0.75 * p99), round(0.8 * p99)
    else:
        t1_val, t2_val = dataset.t1_val, dataset.t2_val

    image_bin = from_numpy(img_array)
    roi_bin = from_numpy(roi_array)
    t1_bin = from_numpy(np.array(t1_val, dtype=img_array.dtype))
    t2_bin = from_numpy(np.array(t2_val, dtype=img_array.dtype))

    # Computed here so it is cached with the dataset; check only compares.
    expected = reference_masked_mri_edge(img_array, roi_array, t1_val, t2_val)
    return DataInstance(
        inputs=[image_bin, roi_bin, t1_bin, t2_bin],
        meta={},
        ref_outputs=[from_numpy(expected)],
    )


class MRIMaskedEdgeDetectionTestGenerator(Generator[MRIMaskedEdgeDetectionDataset]):
    @property
    def name(self) -> str:
        return "mri_masked_edge_detection_test"

    @property
    def pretty_name(self) -> str:
        return "Magnetic Resonance Imaging (MRI) Masked Edge Detection Test"

    @property
    def description(self) -> str:
        return "Small deterministic masked MRI examples."

    @property
    def suites(self) -> list[str]:
        return ["test"]

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def authors(self) -> list[Contributor]:
        return MRIMaskedEdgeDetectionBenchmark().authors

    @property
    def references(self) -> list[Ref]:
        return MRIMaskedEdgeDetectionBenchmark().references

    @property
    def ai_disclosure(self) -> str:
        return MRIMaskedEdgeDetectionBenchmark().ai_disclosure

    @property
    def motivation(self) -> str:
        return "Provide small masked MRI examples for correctness checks."

    @property
    def license(self) -> License | None:
        return licenses.SAPS

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def datasets(self) -> list[MRIMaskedEdgeDetectionDataset]:
        return [
            MRIMaskedEdgeDetectionDataset(
                "zero_image",
                "local",
                "zero_image",
                pretty_name="Zero Image",
                t1_val=50.0,
                t2_val=100.0,
                image=np.zeros((5, 5), dtype=np.float32),
                roi=np.ones((5, 5), dtype=bool),
                suites=["test"],
            ),
            MRIMaskedEdgeDetectionDataset(
                "basic_roi",
                "local",
                "basic_roi",
                pretty_name="Basic Region of Interest (ROI)",
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
                suites=["test"],
            ),
            MRIMaskedEdgeDetectionDataset(
                "default_roi",
                "local",
                "default_roi",
                pretty_name="Default Region of Interest (ROI)",
                t1_val=10.0,
                t2_val=20.0,
                image=np.arange(36, dtype=np.float32).reshape(6, 6),
                suites=["test"],
                ref_meta={"default_roi": True},
            ),
            MRIMaskedEdgeDetectionDataset(
                "int16_volume",
                "local",
                "int16_volume",
                pretty_name="Int16 Volume",
                # The default ROI holds 5000-10000 and 21000-26000.
                t1_val=5000,
                t2_val=21000,
                image=(1000 * np.arange(32, dtype=np.int16)).reshape(2, 4, 4),
                suites=["test"],
            ),
        ]

    def generate(self, dataset: MRIMaskedEdgeDetectionDataset) -> DataInstance:
        problem = masked_edge_instance(dataset, np.asarray(dataset.image))
        return DataInstance(
            inputs=problem.inputs,
            meta=problem.meta,
            ref_outputs=problem.ref_outputs,
            ref_meta=dataset.ref_meta,
        )


class MRIMaskedEdgeDetectionIXIGenerator(Generator[MRIMaskedEdgeDetectionDataset]):
    @property
    def name(self) -> str:
        return "mri_masked_edge_detection_ixi"

    @property
    def pretty_name(self) -> str:
        return "Magnetic Resonance Imaging (MRI) Masked Edge Detection IXI"

    @property
    def description(self) -> str:
        return (
            "3D brain MRI scans from the IXI dataset, one T1, T2 or PD scan per"
            " dataset, as their stored int16 voxels, with thresholds at 75% and"
            " 80% of the scan's 99th-percentile voxel value and a region of"
            " interest covering the middle half of each axis."
        )

    @property
    def suites(self) -> list[str]:
        return ["standard"]

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def authors(self) -> list[Contributor]:
        return MRIMaskedEdgeDetectionBenchmark().authors

    @property
    def references(self) -> list[Ref]:
        return [*MRIMaskedEdgeDetectionBenchmark().references, IXI_REF]

    @property
    def ai_disclosure(self) -> str:
        return "Generative AI was used to implement this generator."

    @property
    def motivation(self) -> str:
        return MRIMaskedEdgeDetectionBenchmark().motivation

    @property
    def license(self) -> License | None:
        return licenses.IXI

    @property
    def datasets(self) -> list[MRIMaskedEdgeDetectionDataset]:
        return [
            MRIMaskedEdgeDetectionDataset(
                "IXI002-Guys-0828-T1",
                "T1",
                "IXI002-Guys-0828-T1.nii.gz",
                t1_val=None,
                t2_val=None,
                suites=["trace", "train"],
            ),
            MRIMaskedEdgeDetectionDataset(
                "IXI012-HH-1211-T2",
                "T2",
                "IXI012-HH-1211-T2.nii.gz",
                t1_val=None,
                t2_val=None,
                suites=["trace"],
            ),
            MRIMaskedEdgeDetectionDataset(
                "IXI035-IOP-0873-PD",
                "PD",
                "IXI035-IOP-0873-PD.nii.gz",
                t1_val=None,
                t2_val=None,
                suites=["trace"],
            ),
            MRIMaskedEdgeDetectionDataset(
                "IXI035-IOP-0873-T1",
                "T1",
                "IXI035-IOP-0873-T1.nii.gz",
                t1_val=None,
                t2_val=None,
                suites=["trace"],
            ),
        ]

    def generate(self, dataset: MRIMaskedEdgeDetectionDataset) -> DataInstance:
        return masked_edge_instance(dataset, load_ixi_scan(dataset.name))


class MRIMaskedEdgeDetectionKaggleGenerator(Generator[MRIMaskedEdgeDetectionDataset]):
    @property
    def name(self) -> str:
        return "mri_masked_edge_detection_kaggle"

    @property
    def pretty_name(self) -> str:
        return "Magnetic Resonance Imaging (MRI) Masked Edge Detection Kaggle"

    @property
    def description(self) -> str:
        return (
            "Data Generation: I used MRI image data from the Kaggle Brain MRI Images"
            " for Brain Tumor Detection set. I used a constant edge threshold of 150.0"
            " with all of the images that I used."
        )

    @property
    def suites(self) -> list[str]:
        return []

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def authors(self) -> list[Contributor]:
        return MRIMaskedEdgeDetectionBenchmark().authors

    @property
    def references(self) -> list[Ref]:
        return [
            *MRIMaskedEdgeDetectionBenchmark().references,
            Ref(
                title="Brain MRI Images for Brain Tumor Detection",
                authors=[Author("Navoneel Chakrabarty")],
                publisher="Kaggle",
                year=2019,
                url="https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection",
            ),
        ]

    @property
    def ai_disclosure(self) -> str:
        return MRIMaskedEdgeDetectionBenchmark().ai_disclosure

    @property
    def motivation(self) -> str:
        return MRIMaskedEdgeDetectionBenchmark().motivation

    @property
    def license(self) -> License | None:
        return licenses.KAGGLE_BRAIN_MRI

    @property
    def datasets(self) -> list[MRIMaskedEdgeDetectionDataset]:
        # De-listed: the Kaggle images have no license (see
        # licenses.KAGGLE_BRAIN_MRI), so they're in no suite and aren't run or
        # cached. MRIMaskedEdgeDetectionIXIGenerator replaces them.
        return [
            MRIMaskedEdgeDetectionDataset("yes/Y157", "yes", "Y157.JPG"),
            MRIMaskedEdgeDetectionDataset("yes/Y6", "yes", "Y6.jpg"),
            MRIMaskedEdgeDetectionDataset("yes/Y194", "yes", "Y194.jpg"),
            MRIMaskedEdgeDetectionDataset("yes/Y180", "yes", "Y180.jpg"),
        ]

    def generate(self, dataset: MRIMaskedEdgeDetectionDataset) -> DataInstance:
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
        return masked_edge_instance(dataset, img_array)


class MRIMaskedEdgeDetectionBenchmark(Benchmark):
    @property
    def name(self) -> str:
        return "mri_masked_edge_detection"

    @property
    def pretty_name(self) -> str:
        return "Magnetic Resonance Imaging (MRI) Masked Edge Detection"

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
        return ["image-processing", "feed-forward"]

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
                title="Compilation of Sparse Array Programming Models",
                authors=[
                    Author("Rawn Henry"),
                    Author("Olivia Hsu"),
                    Author("Rohan Yadav"),
                    Author("Stephen Chou"),
                    Author("Kunle Olukotun"),
                    Author("Saman Amarasinghe"),
                    Author("Fredrik Kjolstad"),
                ],
                journal="Proceedings of the ACM on Programming Languages",
                volume=5,
                number="OOPSLA",
                year=2021,
                url="https://commit.csail.mit.edu/papers/2021/oopsla2021-array-programming.pdf",
                doi="10.1145/3485505",
            ),
            Ref(
                title="Edge Detection of MRI Images - A Review",
                authors=[
                    Author("Madhurima Banerjee"),
                    Author("Samir Kumar Bandyopadhyay"),
                ],
                journal="European Journal of Pharmaceutical and Medical Research",
                volume=3,
                number=10,
                pages="136-140",
                year=2016,
                url="https://www.researchgate.net/publication/310464068_EDGE_DETECTION_OF_MRI_IMAGES_-A_REVIEW",
            ),
            Ref(
                title="Sparse Reconstruction Techniques in Magnetic Resonance Imaging",
                authors=[
                    Author("Alice C. Yang"),
                    Author("Madison Kretzler"),
                    Author("Sonja Sudarski"),
                    Author("Vikas Gulani"),
                    Author("Nicole Seiberlich"),
                ],
                journal="Investigative Radiology",
                volume=51,
                number=6,
                pages="349-364",
                year=2016,
                url="https://pmc.ncbi.nlm.nih.gov/articles/PMC4948115/",
                doi="10.1097/RLI.0000000000000274",
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
            MRIMaskedEdgeDetectionTestGenerator(),
            MRIMaskedEdgeDetectionIXIGenerator(),
            MRIMaskedEdgeDetectionKaggleGenerator(),
        ]

    def benchmark(self, xp, meta: dict[str, Any], img, roi, t1, t2):

        img_t1 = img > t1
        img_t2 = img > t2

        return (img_t2 & roi) ^ (img_t1 & roi)

    def check(self, param):
        for item in self._output:
            assert isinstance(item, BinsparseTensor), (
                "Output must be in binsparse format"
            )
        assert self._ref_outputs is not None, (
            "No NumPy reference edge mask; regenerate the cached dataset"
        )

        result = to_numpy(self._output[0])
        expected = to_numpy(self._ref_outputs[0])

        assert self._meta == {}
        assert result.shape == expected.shape, (
            f"Expected edge mask shape {expected.shape}, got {result.shape}"
        )
        mismatches = np.count_nonzero(result != expected)
        assert mismatches == 0, (
            f"{mismatches} of {expected.size} pixels differ from the "
            "NumPy threshold-band reference"
        )

        if self._ref_meta is None:
            return
        if self._ref_meta.get("default_roi"):
            image_arr = to_numpy(self._input[0])
            roi_arr = to_numpy(self._input[1])
            expected_roi = default_masked_mri_roi(image_arr)
            assert np.all(roi_arr == expected_roi)
            assert to_numpy(self._input[2]).item() == 10.0
            assert to_numpy(self._input[3]).item() == 20.0
