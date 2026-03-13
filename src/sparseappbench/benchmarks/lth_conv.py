"""
Name: CIFAR Conv-2 Lottery Ticket Hypothesis Benchmark Inference
Author: Ramya Polaki
Email: rpolaki3@gatech.edu

Motivation:
The goal of this experiment is to generate a pruned neural network from the
OpenLTH framework and convert it into a format that can be used for inference
benchmarking. The CIFAR Conv-2 model is trained and pruned using the Lottery
Ticket workflow, and the resulting model is exported to ONNX in dense format.

Role of sparsity:
The Lottery Ticket Hypothesis shows that dense neural networks contain smaller
sparse subnetworks that can still perform well. In this workflow, the model is
trained and pruned in OpenLTH, then exported to ONNX for inference benchmarking.

Implementation Reference:
J. Frankle and M. Carbin,
“The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks,”
ICLR 2019.
H. Zhou, J. Lan, R. Liu, and J. Yosinski,
“Deconstructing Lottery Tickets: Zeros, Signs, and the Supermask,”
NeurIPS 2019.

Model Definition:
https://github.com/ramyapolaki/open_lth/blob/main/models/cifar_conv.py

Data Sources Used for Training:
Dataset: CIFAR-10

Hyperparameters Used:
Model: cifar_conv_2
Batch size: 60
Optimizer: Adam
Learning rate: 2e-4
Training length: 27 epochs
Pruning strategy: global magnitude pruning
Pruning fraction: 0.2
Pruning levels: 20

Artifacts Generated:
mask.pth
model_ep27_it0.pth
conv2_pruned_dense.onnx
conv2_pruned_dense.onnx.data

Pruning Results:
Total parameters: 4,300,992
Unpruned parameters: 522,887
Sparsity: ~87.84%

Accuracy:
During evaluation the test accuracy remained approximately between 67% and 69%,
with the best observed accuracy around 69%.

Statement on the use of Generative AI:
No generative AI was used to write the benchmark function itself. Generative
AI was used to debug code. This statement was written by hand.
"""

import os
import shutil
import tempfile
import urllib.request

import numpy as np

from ..binsparse_format import BinsparseFormat
from ..utils.onnx_model_params import load_conv2_params
from ..utils.onnx_numpy_ops import conv2d, flatten, gemm, maxpool2d, relu


# Direct download links from Google Drive
DENSE_ONNX_URL = "https://drive.google.com/uc?export=download&id=1jxF9fFXidr9h_p6fEQGiRQXZkqkYfn14"
DENSE_ONNX_DATA_URL = "https://drive.google.com/uc?export=download&id=1FB_CE91DlFbrPWWT7bQ-ETGRXlzIqEcq"


def _fetch_onnx_pair():
    tmpdir = tempfile.mkdtemp(prefix="lth_conv2_")

    onnx_path = os.path.join(tmpdir, "conv2_pruned_dense.onnx")
    data_path = os.path.join(tmpdir, "conv2_pruned_dense.onnx.data")

    urllib.request.urlretrieve(DENSE_ONNX_URL, onnx_path)
    urllib.request.urlretrieve(DENSE_ONNX_DATA_URL, data_path)

    return tmpdir, onnx_path


def _to_xp_params(xp, params):
    out = {}
    for name, value in params.items():
        out[name] = xp.compute(xp.lazy(
            xp.from_benchmark(BinsparseFormat.from_numpy(value))
        ))
    return out


def benchmark_lth_conv(xp, x_bench):
    x = xp.compute(xp.lazy(xp.from_benchmark(x_bench)))
    tmpdir, onnx_file = _fetch_onnx_pair()

    try:
        params = load_conv2_params(onnx_file)
        p = _to_xp_params(xp, params)

        x = conv2d(xp, x, p["conv1_w"], p["conv1_b"], pads=(1, 1, 1, 1), strides=(1, 1))
        x = relu(xp, x)

        x = conv2d(xp, x, p["conv2_w"], p["conv2_b"], pads=(1, 1, 1, 1), strides=(1, 1))
        x = relu(xp, x)

        x = maxpool2d(xp, x, kernel_shape=(2, 2), strides=(2, 2))
        x = flatten(xp, x)

        x = gemm(xp, x, p["fc1_w"], p["fc1_b"])
        x = relu(xp, x)

        x = gemm(xp, x, p["fc2_w"], p["fc2_b"])
        x = relu(xp, x)

        x = gemm(xp, x, p["fc3_w"], p["fc3_b"])

        x = xp.compute(x)
        x = xp.argmax(x, axis=1)
        return xp.to_benchmark(x)

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def dg_lth_conv_dummy():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 3, 32, 32), dtype=np.float32)
    return (BinsparseFormat.from_numpy(x),)

"""
def dg_lth_conv_batch8():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((8, 3, 32, 32), dtype=np.float32)
    return (BinsparseFormat.from_numpy(x),)
"""



