"""Conv-2 benchmark compiled from ONNX during generator setup."""

from __future__ import annotations

import math
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

import numpy as np

from binsparse.conversions import from_numpy, to_numpy
from filelock import FileLock

from saps.benchmark import (
    Author,
    Benchmark,
    Contributor,
    DataInstance,
    Dataset,
    Generator,
    Ref,
    ReferenceGenerator,
    ShellBenchmark,
)
from saps.codegen import define_function
from saps.downloaders.cache import source_cache_dir

_MODEL_ENV = "LTH_CONV2_ONNX"
_MODEL_FILE_NAME = "conv2_pruned_dense.onnx"
_MODEL_DATA_FILE_NAME = "conv2_pruned_dense.onnx.data"

_MODEL_URL = (
    "https://zenodo.org/records/22650920/files/conv2_pruned_dense.onnx?download=1"
)

_MODEL_DATA_URL = (
    "https://zenodo.org/records/22650920/files/conv2_pruned_dense.onnx.data?download=1"
)

# OpenML's CIFAR-10 lists the 50000 training images before the 10000 test images.
_CIFAR10_TEST_START = 50_000
# OpenLTH normalizes CIFAR-10 images with ImageNet's per-channel statistics.
# These give the model 99.8% training and 67.4% test accuracy, where CIFAR-10's
# own statistics give 99.6% and 66.8%.
_CIFAR10_MEAN = (0.485, 0.456, 0.406)
_CIFAR10_STD = (0.229, 0.224, 0.225)

# The check models each float32 rounding as an independent, mean-zero relative
# error of at most u, so rounding errors add in quadrature rather than in the
# worst case (Higham and Mary, "A New Approach to Probabilistic Rounding Error
# Analysis", 2019). Each output must be within _SIGMAS standard deviations of
# its exact value.
_UNIT_ROUNDOFF = 2.0**-24
_SIGMAS = 8


def _references() -> list[Ref]:
    return [
        Ref(
            title=(
                "The Lottery Ticket Hypothesis: Finding Sparse, "
                "Trainable Neural Networks"
            ),
            authors=[
                Author("Jonathan Frankle"),
                Author("Michael Carbin"),
            ],
            journal="Arxiv",
            volume="arXiv:1803.03635",
            year=2018,
            url="https://arxiv.org/abs/1803.03635",
        ),
        Ref(
            title="Deconstructing Lottery Tickets: Zeros, Signs, and the Supermask",
            authors=[
                Author("Hattie Zhou"),
                Author("Janice Lan"),
                Author("Rosanne Liu"),
                Author("Jason Yosinski"),
            ],
            journal="Arxiv",
            volume="arXiv:1905.01067",
            year=2019,
            url="https://arxiv.org/abs/1905.01067",
        ),
        Ref(
            title="CIFAR Conv-2 Lottery Ticket Hypothesis Experiment",
            authors=[Author("Ramya Polaki")],
            publisher="Zenodo",
            year=2026,
            url="https://zenodo.org/records/22650920",
        ),
    ]


def _cifar10_test_images(count: int) -> np.ndarray:
    """The first ``count`` CIFAR-10 test images, normalized, as (count, 3, 32, 32)."""
    from saps.benchmarks.openml import fetch_openml_features

    features, _ = fetch_openml_features("cifar10")
    rows = features[_CIFAR10_TEST_START : _CIFAR10_TEST_START + count]
    # Each row holds the red, green and blue 32x32 planes, scaled to [0, 1].
    images = rows.reshape(count, 3, 32, 32)
    mean = np.asarray(_CIFAR10_MEAN, dtype=np.float32)[:, None, None]
    std = np.asarray(_CIFAR10_STD, dtype=np.float32)[:, None, None]
    return (images - mean) / std


def _conv(x: np.ndarray, w: np.ndarray, pads, strides, dilations) -> np.ndarray:
    """ONNX Conv of x (N, C, *S) by w (M, C, *K), one kernel offset at a time."""
    rank = w.ndim - 2
    x = np.pad(x, [(0, 0), (0, 0), *zip(pads[:rank], pads[rank:], strict=True)])
    out = [
        (size - (k - 1) * d - 1) // s + 1
        for size, k, d, s in zip(
            x.shape[2:], w.shape[2:], dilations, strides, strict=True
        )
    ]
    y = np.zeros((x.shape[0], *out, w.shape[0]), dtype=np.result_type(x, w))
    for offset in np.ndindex(*w.shape[2:]):
        window: tuple[Any, ...] = (
            ...,
            *(
                slice(o * d, o * d + (n - 1) * s + 1, s)
                for o, d, n, s in zip(offset, dilations, out, strides, strict=True)
            ),
        )
        y += np.tensordot(x[window], w[(..., *offset)], axes=(1, 1))
    return np.moveaxis(y, -1, 1)


def _exact_outputs(model, feed: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the model's output exactly, with the standard deviation of the
    error float32 rounding adds to it.

    A dot product of length n rounds n times, each adding at most u times the
    magnitude sum of |w| |a|, for variance at most u^2 n (sum |w| |a|)^2. Errors
    already in a layer's inputs reach its outputs through w^2. ReLU and reshapes
    keep them, and max pooling keeps at most the sum of its window's.
    """
    from onnx import helper, numpy_helper

    def exact(array):
        array = np.asarray(array)
        return array.astype(np.float64) if array.dtype.kind == "f" else array

    value = {t.name: exact(numpy_helper.to_array(t)) for t in model.graph.initializer}
    value.update((name, exact(array)) for name, array in feed.items())
    variance = {name: np.zeros(np.shape(array)) for name, array in value.items()}
    u2 = _UNIT_ROUNDOFF**2
    (output,) = model.graph.output
    nodes = list(model.graph.node)
    last_use = {name: i for i, node in enumerate(nodes) for name in node.input}
    for i, node in enumerate(nodes):
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        args = [value[name] for name in node.input]
        variances = [variance[name] for name in node.input]
        match node.op_type:
            case "Conv":
                if attrs.get("group", 1) != 1 or attrs.get("auto_pad", b"NOTSET") != (
                    b"NOTSET"
                ):
                    raise NotImplementedError("Only ungrouped, explicitly padded Conv")
                (x, w, *bias_args), vx = args, variances[0]
                rank = w.ndim - 2

                def conv(x, w, rank=rank, attrs=attrs):
                    return _conv(
                        x,
                        w,
                        attrs.get("pads", [0] * 2 * rank),
                        attrs.get("strides", [1] * rank),
                        attrs.get("dilations", [1] * rank),
                    )

                bias = (
                    bias_args[0].reshape(-1, *[1] * rank) if bias_args else np.zeros(())
                )
                y = conv(x, w) + bias
                magnitude = conv(np.abs(x), np.abs(w)) + np.abs(bias)
                v = conv(vx, w * w) + u2 * (w[0].size + len(bias_args)) * magnitude**2
            case "Gemm" | "MatMul":
                (a, b, *c), (va, vb, *_) = args, variances
                if attrs.get("transA"):
                    a, va = a.T, va.T
                if attrs.get("transB"):
                    b, vb = b.T, vb.T
                if attrs.get("alpha", 1.0) != 1.0 or attrs.get("beta", 1.0) != 1.0:
                    raise NotImplementedError("Only Gemm with alpha = beta = 1")
                bias = c[0] if c else np.zeros(())
                y = a @ b + bias
                magnitude = np.abs(a) @ np.abs(b) + np.abs(bias)
                v = (
                    va @ (b * b)
                    + (a * a) @ vb
                    + u2 * (a.shape[-1] + len(c)) * magnitude**2
                )
            case "Relu":
                y, v = np.maximum(args[0], 0), variances[0]
            case "MaxPool":
                if attrs.get("ceil_mode", 0):
                    raise NotImplementedError("Only MaxPool with ceil_mode = 0")
                (x,), (vx,) = args, variances
                kernel = attrs["kernel_shape"]
                rank = len(kernel)
                pads = attrs.get("pads", [0] * 2 * rank)
                steps = (
                    *(slice(None, None, s) for s in attrs.get("strides", [1] * rank)),
                    *(slice(None, None, d) for d in attrs.get("dilations", [1] * rank)),
                )
                spread = [
                    (k - 1) * d + 1
                    for k, d in zip(
                        kernel, attrs.get("dilations", [1] * rank), strict=True
                    )
                ]
                pad_width = [
                    (0, 0),
                    (0, 0),
                    *zip(pads[:rank], pads[rank:], strict=True),
                ]
                axes = tuple(range(2, 2 + rank))
                windows = np.lib.stride_tricks.sliding_window_view(  # type: ignore[call-overload]
                    np.pad(x, pad_width, constant_values=-np.inf), spread, axis=axes
                )[(..., *steps)]
                y = windows.max(axis=tuple(range(-rank, 0)))
                windows = np.lib.stride_tricks.sliding_window_view(  # type: ignore[call-overload]
                    np.pad(vx, pad_width), spread, axis=axes
                )[(..., *steps)]
                v = windows.sum(axis=tuple(range(-rank, 0)))
            case "Reshape":
                (x, shape), vx = args, variances[0]
                shape = [
                    x.shape[axis] if size == 0 and not attrs.get("allowzero") else size
                    for axis, size in enumerate(shape.tolist())
                ]
                y, v = x.reshape(shape), vx.reshape(shape)
            case "Flatten":
                (x,), (vx,) = args, variances
                axis = attrs.get("axis", 1)
                shape = (math.prod(x.shape[:axis]), -1)
                y, v = x.reshape(shape), vx.reshape(shape)
            case "Constant":
                y = exact(numpy_helper.to_array(attrs["value"]))
                v = np.zeros(y.shape)
            case _:
                raise NotImplementedError(f"No rounding model for ONNX {node.op_type}")
        (name,) = node.output
        value[name], variance[name] = y, v
        # Drop activations once no later node reads them.
        for name in node.input:
            if last_use[name] == i and name != output.name:
                del value[name], variance[name]
    return value[output.name], np.sqrt(variance[output.name])


def _default_data_dir() -> Path:
    return source_cache_dir("artifacts/lth").resolve()


def _download_if_missing(url: str, destination: Path) -> None:
    if destination.is_file() and destination.stat().st_size > 0:
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = destination.with_name(destination.name + ".lock")

    with FileLock(lock_path):
        if destination.is_file() and destination.stat().st_size > 0:
            return

        partial = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix=f".{destination.name}.",
                suffix=".part",
                dir=destination.parent,
                delete=False,
            ) as temporary:
                partial = Path(temporary.name)

            urlretrieve(url, partial)
            if partial.stat().st_size == 0:
                raise RuntimeError(f"Downloaded an empty LTH model artifact from {url}")

            partial.replace(destination)
        finally:
            if partial is not None and partial.exists():
                partial.unlink()


def _model_path() -> Path:
    configured = os.environ.get(_MODEL_ENV)

    if configured:
        model_path = Path(configured).expanduser().resolve()

        if not model_path.is_file():
            raise FileNotFoundError(
                f"{_MODEL_ENV} does not point to a file: {model_path}"
            )

        data_path = model_path.with_name(_MODEL_DATA_FILE_NAME)
        if not data_path.is_file():
            raise FileNotFoundError(f"ONNX external-data file not found: {data_path}")

        return model_path

    root = _default_data_dir()
    model_path = root / _MODEL_FILE_NAME
    data_path = root / _MODEL_DATA_FILE_NAME

    _download_if_missing(_MODEL_URL, model_path)
    _download_if_missing(_MODEL_DATA_URL, data_path)

    return model_path.resolve()


class LTHConv2Dataset(Dataset):
    """The pruned model classifying the first ``num_images`` CIFAR-10 test images."""

    def __init__(self, name: str, num_images: int, suites: list[str]):
        self._name = name
        self.num_images = num_images
        self._suites = suites

    @property
    def name(self) -> str:
        return self._name

    @property
    def pretty_name(self) -> str:
        return f"Pruned, {self.num_images} images"

    @property
    def description(self) -> str:
        return (
            "Pruned CIFAR-10 Conv-2 model with approximately 87.84% zero-valued "
            "parameters, archived with its ONNX external data in Zenodo record "
            f"22650920, classifying the first {self.num_images} CIFAR-10 test "
            "images in one batch."
        )

    @property
    def suites(self) -> list[str]:
        return self._suites

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"


def _widened_model(num_images):
    """The pruned model, widened to classify the first `num_images` CIFAR-10
    test images, with its runtime input's value info and those images."""
    import onnx

    model = onnx.load(str(_model_path()), load_external_data=True)

    initializer_names = {tensor.name for tensor in model.graph.initializer}
    real_inputs = [
        value_info
        for value_info in model.graph.input
        if value_info.name not in initializer_names
    ]

    if len(real_inputs) != 1:
        raise ValueError(f"Expected one runtime input, found {len(real_inputs)}.")

    input_info = real_inputs[0]

    shape = []
    for dim in input_info.type.tensor_type.shape.dim:
        if not dim.HasField("dim_value") or dim.dim_value <= 0:
            raise ValueError(f"Input {input_info.name!r} must have a static shape.")
        shape.append(int(dim.dim_value))

    dtype = np.dtype(
        onnx.helper.tensor_dtype_to_np_dtype(input_info.type.tensor_type.elem_type)
    )

    model_input = _cifar10_test_images(num_images).astype(dtype)
    if model_input.shape[1:] != tuple(shape[1:]):
        raise ValueError(
            f"Input {input_info.name!r} of shape {tuple(shape)} doesn't take "
            f"images of shape {model_input.shape[1:]}."
        )
    # The model is exported for one image. Widen its batch, including the
    # constant shape its flatten reshapes to, and drop the stale shapes
    # inferred for one image.
    batch = shape[0]
    for value_info in (input_info, *model.graph.output):
        value_info.type.tensor_type.shape.dim[0].dim_value = num_images
    del model.graph.value_info[:]
    reshapes = {node.input[1] for node in model.graph.node if node.op_type == "Reshape"}
    for tensor in model.graph.initializer:
        if tensor.name in reshapes:
            target = onnx.numpy_helper.to_array(tensor).copy()
            if target[0] == batch:
                target[0] = num_images
                tensor.CopyFrom(onnx.numpy_helper.from_array(target, tensor.name))
    return model, input_info, model_input


class LTHConv2ONNXGenerator(Generator[LTHConv2Dataset]):
    @property
    def name(self) -> str:
        return "lth_conv2_onnx"

    @property
    def pretty_name(self) -> str:
        return "Lottery Ticket Hypothesis (LTH) Conv-2 ONNX"

    @property
    def description(self) -> str:
        return (
            "Loads the Conv-2 ONNX artifact from the runner's shared SAPS cache, "
            "imports its trained parameters and pruning-induced zeros, widens it "
            "to a batch of CIFAR-10 test images from the shared OpenML copy, and "
            "generates an inline NumPy benchmark function with ONNXPY during setup."
        )

    @property
    def suites(self) -> list[str]:
        return ["lth"]

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def authors(self) -> list[Contributor]:
        return [
            Contributor("Ramya Polaki", "rpolaki3@gatech.edu"),
            Contributor("Michael Wang", "mwang764@gatech.edu"),
        ]

    @property
    def references(self) -> list[Ref]:
        return _references()

    @property
    def ai_disclosure(self) -> str:
        return (
            "Generative AI was used to assist with adapting existing benchmark "
            "integration code to the current SAPS and ONNXPY APIs. The model "
            "training, pruning workflow, benchmark objective, and model artifacts "
            "were created by the contributors."
        )

    @property
    def motivation(self) -> str:
        return (
            "This generator turns a reproducible lottery-ticket training artifact "
            "into fixed benchmark inputs. The runner-managed shared cache and "
            "locked atomic downloads allow concurrent and chunked SAPS runs to "
            "reuse the same ONNX files safely."
        )

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def datasets(self) -> list[LTHConv2Dataset]:
        return [
            LTHConv2Dataset("pruned", 1000, ["standard", "train"]),
            LTHConv2Dataset("pruned_test", 16, ["test"]),
        ]

    def generate(self, dataset: LTHConv2Dataset) -> DataInstance:
        problem = self.generate_inputs(dataset)
        reference = ReferenceGenerator(self).cached_generate(dataset)
        problem.ref_outputs = reference.ref_outputs
        problem.ref_meta = reference.ref_meta
        return problem

    def generate_inputs(self, dataset: LTHConv2Dataset) -> DataInstance:
        from onnxpy import compile_to_source

        model, input_info, model_input = _widened_model(dataset.num_images)
        model_source, tensor_inputs = compile_to_source(
            model, function_name="benchmark", extra_parameters=("meta",)
        )
        return DataInstance(
            inputs=[
                from_numpy(model_input),
                *(from_numpy(np.asarray(value)) for value in tensor_inputs.values()),
            ],
            meta={
                "onnx_input_name": input_info.name,
                "onnx_initializer_names": tuple(tensor_inputs),
                "benchmark_source": model_source,
            },
        )

    def generate_reference(self, dataset: LTHConv2Dataset) -> DataInstance:
        model, input_info, model_input = _widened_model(dataset.num_images)
        exact, sigma = _exact_outputs(model, {input_info.name: model_input})
        return DataInstance(
            inputs=[],
            meta={},
            ref_outputs=[from_numpy(exact), from_numpy(sigma)],
            ref_meta={"sigmas": _SIGMAS},
        )

    def generate_benchmark_function(
        self,
        dataset: LTHConv2Dataset,
        problem: DataInstance,
        benchmark: Callable[..., Any],
    ) -> Callable[..., Any]:
        return define_function(
            problem.meta["benchmark_source"],
            f"<saps-generated {self.name}.{dataset.name}>",
            {"math": math},
        )


class LTHConv2ONNXReferenceShellBenchmark(ShellBenchmark):
    @property
    def generator(self) -> Generator:
        return ReferenceGenerator(LTHConv2ONNXGenerator())


class LTHConv2Benchmark(Benchmark):
    @property
    def name(self) -> str:
        return "lth_conv2"

    @property
    def pretty_name(self) -> str:
        return "Lottery Ticket Hypothesis (LTH) Conv-2"

    @property
    def description(self) -> str:
        return (
            "Benchmarks inference for a sparse CIFAR-10 Conv-2 network derived "
            "from the Lottery Ticket Hypothesis workflow. Using OpenLTH's training "
            "and pruning infrastructure, and motivated by the original LTH and "
            "Uber AI lottery-ticket studies, the CIFAR Conv-2 architecture was "
            "trained on CIFAR-10 and subjected to 20 levels of iterative global "
            "magnitude pruning. The selected checkpoint and pruning mask were "
            "combined and exported as a dense ONNX model whose zero-valued "
            "parameters preserve approximately 87.84% sparsity. The model and its "
            "external tensor data are archived in Zenodo record 22650920. ONNXPY "
            "translates that graph into inline array-based Python during setup, which "
            "the SAPS runner executes across supported array frameworks."
        )

    @property
    def suites(self) -> list[str]:
        return ["lth", "standard-machine-learning"]

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def authors(self) -> list[Contributor]:
        return [
            Contributor("Ramya Polaki", "rpolaki3@gatech.edu"),
            Contributor("Michael Wang", "mwang764@gatech.edu"),
        ]

    @property
    def references(self) -> list[Ref]:
        return _references()

    @property
    def ai_disclosure(self) -> str:
        return (
            "Generative AI was used to assist with adapting existing benchmark "
            "integration code to the current SAPS and ONNXPY APIs. The model "
            "training, pruning workflow, benchmark objective, and model artifacts "
            "were created by the contributors."
        )

    @property
    def motivation(self) -> str:
        return (
            "This benchmark connects a reproducible lottery-ticket training "
            "artifact to framework-independent sparse inference measurement. "
            "The archived ONNX model is the source of truth, ONNXPY preserves "
            "its graph as array operations, and the SAPS runner supplies framework "
            "selection, timing, diagnostics, shared caching, and result recording."
        )

    @property
    def generators(self) -> list[Generator[Any]]:
        return [LTHConv2ONNXGenerator()]

    def benchmark(self, xp, meta):
        raise NotImplementedError(
            "Conv-2 functions are compiled from ONNX by the generator's "
            "generate_benchmark_function."
        )

    def check(self, param):
        (output,) = self._output
        exact, sigma = (to_numpy(array) for array in self._ref_outputs)
        sigmas = self._ref_meta["sigmas"]
        error = np.abs(to_numpy(output).astype(np.float64) - exact)
        worst = np.unravel_index(np.argmax(error - sigmas * sigma), error.shape)
        assert error[worst] <= sigmas * sigma[worst], (
            f"Output {tuple(map(int, worst))} is off by {error[worst]:.3g}, more "
            f"than {sigmas} standard deviations ({sigma[worst]:.3g}) of float32 "
            "rounding error"
        )
