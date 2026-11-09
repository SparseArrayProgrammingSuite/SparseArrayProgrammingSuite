"""End-to-end round-trip test against ``tmp/models/mnist-8.onnx``.

This is the project's definition of done: compile the real ONNX model,
execute the generated function, and require the output to match
onnxruntime within tolerance on a handful of random inputs.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest

from onnxpy import compile_to_source

_MNIST_PATH = Path(__file__).resolve().parents[1] / "tmp" / "models" / "mnist-8.onnx"


@pytest.fixture(scope="module")
def mnist_function():
    if not _MNIST_PATH.exists():
        pytest.skip(f"mnist-8 model not present at {_MNIST_PATH}")
    model = onnx.load(str(_MNIST_PATH))
    source, tensor_inputs = compile_to_source(model)
    scope = {"math": math}
    exec(source, scope)
    return scope["model"], tensor_inputs


@pytest.fixture(scope="module")
def ort_session():
    if not _MNIST_PATH.exists():
        pytest.skip(f"mnist-8 model not present at {_MNIST_PATH}")
    return ort.InferenceSession(str(_MNIST_PATH), providers=["CPUExecutionProvider"])


@pytest.mark.parametrize("seed", [0, 1, 7, 42, 1234])
def test_mnist_matches_ort(mnist_function, ort_session, seed, array_namespace):
    rng = np.random.RandomState(seed)
    x = rng.randn(1, 1, 28, 28).astype(np.float32)
    expected = ort_session.run(["Plus214_Output_0"], {"Input3": x})[0]
    function, tensor_inputs = mnist_function
    actual = np.asarray(function(array_namespace, x, *tensor_inputs.values()))
    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)
    assert int(np.argmax(actual)) == int(np.argmax(expected))
