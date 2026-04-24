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


def benchmark_mri_edge(
    xp, image_bench, dx_bench, sy_bench, sx_bench, dy_bench, threshold_bench
):
    image = xp.lazy(xp.from_benchmark(image_bench))
    threshold = xp.lazy(xp.from_benchmark(threshold_bench))
    D_x = xp.lazy(xp.from_benchmark(dx_bench))
    S_y = xp.lazy(xp.from_benchmark(sy_bench))
    S_x = xp.lazy(xp.from_benchmark(sx_bench))
    D_y = xp.lazy(xp.from_benchmark(dy_bench))

    gx = D_x @ image @ S_y
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

    Nx, Ny = img_array.shape
    dx_bin, sy_bin, sx_bin, dy_bin = generate_1d_sobel_matrices(Nx, Ny)
    return image_bin, dx_bin, sy_bin, sx_bin, dy_bin, threshold_bin


def dg_mri_sobel_1():
    return generate_mri_sobel_data("yes", "Y157.JPG", 150.0)


def dg_mri_sobel_2():
    return generate_mri_sobel_data("yes", "Y6.jpg", 150.0)


def dg_mri_sobel_3():
    return generate_mri_sobel_data("yes", "Y194.jpg", 150.0)


def dg_mri_sobel_4():
    return generate_mri_sobel_data("yes", "Y180.jpg", 150.0)


def generate_1d_sobel_matrices(Nx, Ny):
    rows = np.arange(Nx)
    cols1 = (rows + 1) % Nx
    cols2 = (rows - 1) % Nx
    D_x_R = np.concatenate([rows, rows])
    D_x_C = np.concatenate([cols1, cols2])
    D_x_V = np.concatenate([np.ones(Nx), -np.ones(Nx)])
    dx_bin = BinsparseFormat.from_coo(
        (D_x_R, D_x_C), D_x_V.astype(np.float32), (Nx, Nx)
    )

    rows = np.arange(Ny)
    cols1 = (rows - 1) % Ny
    cols2 = rows
    cols3 = (rows + 1) % Ny
    S_y_R = np.concatenate([rows, rows, rows])
    S_y_C = np.concatenate([cols1, cols2, cols3])
    S_y_V = np.concatenate([np.ones(Ny), 2.0 * np.ones(Ny), np.ones(Ny)])
    sy_bin = BinsparseFormat.from_coo(
        (S_y_R, S_y_C), S_y_V.astype(np.float32), (Ny, Ny)
    )

    rows = np.arange(Nx)
    cols1 = (rows - 1) % Nx
    cols2 = rows
    cols3 = (rows + 1) % Nx
    S_x_R = np.concatenate([rows, rows, rows])
    S_x_C = np.concatenate([cols1, cols2, cols3])
    S_x_V = np.concatenate([np.ones(Nx), 2.0 * np.ones(Nx), np.ones(Nx)])
    sx_bin = BinsparseFormat.from_coo(
        (S_x_R, S_x_C), S_x_V.astype(np.float32), (Nx, Nx)
    )

    rows = np.arange(Ny)
    cols1 = (rows + 1) % Ny
    cols2 = (rows - 1) % Ny
    D_y_R = np.concatenate([rows, rows])
    D_y_C = np.concatenate([cols1, cols2])
    D_y_V = np.concatenate([np.ones(Ny), -np.ones(Ny)])
    dy_bin = BinsparseFormat.from_coo(
        (D_y_R, D_y_C), D_y_V.astype(np.float32), (Ny, Ny)
    )

    return dx_bin, sy_bin, sx_bin, dy_bin
