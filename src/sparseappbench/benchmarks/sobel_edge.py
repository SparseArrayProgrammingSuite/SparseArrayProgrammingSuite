"""
Name: MRI Edge Detection

Author: Aadharsh Rajkumar

Email: arajkumar34@gatech.edu

What does this code do: This code implements a simple edge detection algorithm
on a 2D MRI image. The algorithm computes the gradients in the X and Y directions
using the concept of a Sobel operator, which is a common method for edge detection.
The sobel operator was recreated using array shifts that account for sparse patterns.
The magnitude of the gradients is computed and then masked with a threshold to
produce a binary edge map.

Citation for reference implementation:
https://commit.csail.mit.edu/papers/2021/oopsla2021-array-programming.pdf

Motivation: Edge detection is a crucial task that is a part of image processing
pipelines. It is often the case that images and scans in the medical field rquire
post-processing to extract useful information. In this case, we are using a 2D
MRI image to produce thresholded edge maps. Since medical images are large and
often contain redundant information, it is important to process them efficiently.
The redundancy of MRI makes them a good candidate for sparse processing.
https://www.researchgate.net/publication/310464068_EDGE_DETECTION_OF_MRI_IMAGES_-A_REVIEW
https://pmc.ncbi.nlm.nih.gov/articles/PMC4948115/

Data Generation: I used MRI image data from this Kaggle set:
https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection.
I used a constant edge threshold of 150.0 with all of the images that I used.

Statement on the use of Generative AI: No generative AI was used to construct
the benchmark function. Generative AI might have been used to construct tests.
This statement is written by hand.
"""

import os

import numpy as np

try:
    import kagglehub
    from PIL import Image
except ImportError:
    Image = None
    kagglehub = None

from sparseappbench.binsparse_format import BinsparseFormat


def benchmark_mri_edge(xp, image_bench, threshold_bench):
    image = xp.lazy(xp.from_benchmark(image_bench))
    threshold = xp.lazy(xp.from_benchmark(threshold_bench))

    Nx, Ny = image.shape

    # Shifts for X gradient
    D_x = xp.zeros((Nx, Nx))
    for i in range(Nx):
        D_x[i, (i + 1) % Nx] = 1.0
        D_x[i, (i - 1) % Nx] = -1.0

    S_y = xp.zeros((Ny, Ny))
    for j in range(Ny):
        S_y[(j - 1) % Ny, j] = 1.0
        S_y[j, j] = 2.0
        S_y[(j + 1) % Ny, j] = 1.0

    gx = D_x @ image @ S_y

    # Shifts for Y gradient
    S_x = xp.zeros((Nx, Nx))
    for i in range(Nx):
        S_x[i, (i - 1) % Nx] = 1.0
        S_x[i, i] = 2.0
        S_x[i, (i + 1) % Nx] = 1.0

    D_y = xp.zeros((Ny, Ny))
    for j in range(Ny):
        D_y[(j + 1) % Ny, j] = 1.0
        D_y[(j - 1) % Ny, j] = -1.0

    gy = S_x @ image @ D_y

    magnitude = xp.abs(gx) + xp.abs(gy)

    edges = magnitude > threshold

    result = xp.compute(edges)
    return xp.to_benchmark(result)


def generate_mri_sobel_data(category, filename, threshold_val=100.0):
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

    image_bin = BinsparseFormat.from_numpy(img_array)
    threshold_bin = BinsparseFormat.from_numpy(
        np.array(threshold_val, dtype=np.float32)
    )
    return image_bin, threshold_bin


def dg_mri_sobel_1():
    return generate_mri_sobel_data("yes", "Y157.JPG", 150.0)


def dg_mri_sobel_2():
    return generate_mri_sobel_data("yes", "Y6.jpg", 150.0)


def dg_mri_sobel_3():
    return generate_mri_sobel_data("yes", "Y194.jpg", 150.0)


def dg_mri_sobel_4():
    return generate_mri_sobel_data("yes", "Y180.jpg", 150.0)


def generate_sobel_matrix(height, width, axis):
    H = height
    W = width
    row_indices = []
    col_indices = []
    values = []

    r = np.arange(H).reshape(-1, 1).repeat(W, axis=1)
    c = np.arange(W).reshape(1, -1).repeat(H, axis=0)

    r_flat = r.flatten()
    c_flat = c.flatten()
    output_idx = r_flat * W + c_flat

    def add_shift(dy, dx, val):
        input_r = (r_flat - dy) % H
        input_c = (c_flat - dx) % W
        input_idx = input_r * W + input_c
        row_indices.append(output_idx)
        col_indices.append(input_idx)
        values.append(np.full_like(output_idx, val, dtype=np.float32))

    if axis == "x":
        add_shift(-1, 1, 1.0)
        add_shift(-1, 0, 2.0)
        add_shift(-1, -1, 1.0)

        add_shift(1, 1, -1.0)
        add_shift(1, 0, -2.0)
        add_shift(1, -1, -1.0)
    elif axis == "y":
        add_shift(1, -1, 1.0)
        add_shift(0, -1, 2.0)
        add_shift(-1, -1, 1.0)

        add_shift(1, 1, -1.0)
        add_shift(0, 1, -2.0)
        add_shift(-1, 1, -1.0)
    else:
        raise ValueError("axis must be 'x' or 'y'")

    R = np.concatenate(row_indices)
    C = np.concatenate(col_indices)
    V = np.concatenate(values)

    return BinsparseFormat.from_coo((R, C), V, (H * W, H * W))
