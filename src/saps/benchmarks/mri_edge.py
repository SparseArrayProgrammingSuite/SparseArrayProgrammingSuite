
import os

import numpy as np

try:
    import kagglehub
    from PIL import Image
except ImportError:
    Image = None
    kagglehub = None

from sparseappbench.binsparse_format import BinsparseFormat

"""
Data Generation: I used MRI image data from this Kaggle set:
https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection.
I used a constant edge threshold of 150.0 with all of the images that I used.
"""

def dg_masked_mri_1():
    return generate_masked_mri_data("yes", "Y157.JPG")


def dg_masked_mri_2():
    return generate_masked_mri_data("yes", "Y6.jpg")


def dg_masked_mri_3():
    return generate_masked_mri_data("yes", "Y194.jpg")


def dg_masked_mri_4():
    return generate_masked_mri_data("yes", "Y180.jpg")

def generate_masked_mri_data(category, filename, t1_val=191.25, t2_val=204.0):
    if kagglehub is None or Image is None:
        raise ImportError("kagglehub and Pillow are required.")
    path = kagglehub.dataset_download(
        "navoneel/brain-mri-images-for-brain-tumor-detection"
    )
    img_path = os.path.join(path, category, filename)
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found at {img_path}")

    img = Image.open(img_path).convert("L")
    img_array = np.array(img, dtype=np.float32)

    H, W = img_array.shape
    roi_array = np.zeros_like(img_array, dtype=bool)
    H_val, W_val = H // 4, W // 4
    roi_array[H_val : H - H_val, W_val : W - W_val] = True

    image_bin = BinsparseFormat.from_numpy(img_array)
    roi_bin = BinsparseFormat.from_numpy(roi_array)
    t1_bin = BinsparseFormat.from_numpy(np.array(t1_val, dtype=np.float32))
    t2_bin = BinsparseFormat.from_numpy(np.array(t2_val, dtype=np.float32))

    return image_bin, roi_bin, t1_bin, t2_bin



"""
Name: MRI Edge Detection

Author: Aadharsh Rajkumar

Email: arajkumar34@gatech.edu

What does this code do: This code implements a masked edge detection algorithm
on a 2D MRI image. The benchmark performs boolean threshold mask operations using
t1=75% and t2=80% thresholds and a Region-of-Interest (ROI) mask.


Motivation: Edge detection is a crucial task that is a part of image processing
pipelines. It is often the case that images and scans in the medical field rquire
post-processing to extract useful information. In this case, we are using a 2D
MRI image to produce thresholded edge maps. Since medical images are large and
often contain redundant information, it is important to process them efficiently.
The redundancy of MRI makes them a good candidate for sparse processing.

https://commit.csail.mit.edu/papers/2021/oopsla2021-array-programming.pdf
https://www.researchgate.net/publication/310464068_EDGE_DETECTION_OF_MRI_IMAGES_-A_REVIEW
https://pmc.ncbi.nlm.nih.gov/articles/PMC4948115/


Statement on the use of Generative AI: No generative AI was used to construct
the benchmark function. Generative AI might have been used to construct tests.
This statement is written by hand.
"""
def benchmark_masked_mri_edge(xp, img_bench, roi_bench, t1_bench, t2_bench):
    img = xp.lazy(xp.from_benchmark(img_bench))
    roi = xp.lazy(xp.from_benchmark(roi_bench))
    t1 = xp.lazy(xp.from_benchmark(t1_bench))
    t2 = xp.lazy(xp.from_benchmark(t2_bench))

    img_t1 = img > t1
    img_t2 = img > t2

    img_post = (img_t2 & roi) ^ (img_t1 & roi)

    result = xp.compute(img_post)
    return xp.to_benchmark(result)

