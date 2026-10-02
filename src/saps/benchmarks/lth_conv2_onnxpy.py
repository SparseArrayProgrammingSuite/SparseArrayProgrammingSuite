"""Generated Conv-2 ONNXPY benchmark wrapper for SAPS.

Regenerate from the ONNXPY checkout with:
    poetry run python scripts/print_conv2lth_numpy.py --write-saps-benchmark
"""

from __future__ import annotations

import os
import tempfile
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
)
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

model_inputs = (
    "input",
    "layers.0.conv1.weight",
    "layers.0.conv1.bias",
    "layers.0.conv2.weight",
    "layers.0.conv2.bias",
    "fc1.weight",
    "fc1.bias",
    "fc2.weight",
    "fc2.bias",
    "fc3.weight",
    "fc3.bias",
    "val_4",
)
tensor_inputs = (
    "layers.0.conv1.weight",
    "layers.0.conv1.bias",
    "layers.0.conv2.weight",
    "layers.0.conv2.bias",
    "fc1.weight",
    "fc1.bias",
    "fc2.weight",
    "fc2.bias",
    "fc3.weight",
    "fc3.bias",
    "val_4",
)


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
    @property
    def name(self) -> str:
        return "conv2_pruned"

    @property
    def pretty_name(self) -> str:
        return "Lottery Ticket Conv-2 Pruned Model"

    @property
    def description(self) -> str:
        return (
            "Pruned CIFAR-10 Conv-2 model with approximately 87.84% zero-valued "
            "parameters, archived with its ONNX external data in Zenodo record "
            "22650920."
        )

    @property
    def suites(self) -> list[str]:
        return ["lth"]

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"


class LTHConv2ONNXPYGenerator(Generator[LTHConv2Dataset]):
    @property
    def name(self) -> str:
        return "lth_conv2_onnxpy_inputs"

    @property
    def pretty_name(self) -> str:
        return "Lottery Ticket Conv-2 ONNXPY Inputs"

    @property
    def description(self) -> str:
        return (
            "Loads the Conv-2 ONNX artifact from the runner's shared SAPS cache, "
            "imports its trained parameters and pruning-induced zeros, and "
            "generates a deterministic runtime input for SAPS execution."
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
        return [LTHConv2Dataset()]

    def generate(self, _dataset: LTHConv2Dataset) -> DataInstance:
        import onnx
        from onnx import numpy_helper

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

        rng = np.random.default_rng(0)
        model_input = rng.standard_normal(tuple(shape)).astype(dtype)
        initializers = {tensor.name: tensor for tensor in model.graph.initializer}
        missing = [name for name in tensor_inputs if name not in initializers]
        if missing:
            raise ValueError(
                "Expected ONNX initializers not found: " + ", ".join(missing)
            )

        return DataInstance(
            inputs=[
                from_numpy(model_input),
                *(
                    from_numpy(np.asarray(numpy_helper.to_array(initializers[name])))
                    for name in tensor_inputs
                ),
            ],
            meta={
                "onnx_input_name": input_info.name,
                "onnx_initializer_names": tensor_inputs,
            },
        )


class LTHConv2ONNXPYBenchmark(Benchmark):
    @property
    def name(self) -> str:
        return "lth_conv2_onnxpy"

    @property
    def pretty_name(self) -> str:
        return "Lottery Ticket Conv-2 via ONNXPY"

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
            "translated that fixed graph into array-based Python, which the current "
            "SAPS runner executes across supported array frameworks."
        )

    @property
    def suites(self) -> list[str]:
        return ["lth", "group-machine-learning"]

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
        return [LTHConv2ONNXPYGenerator()]

    def setup(self, param, *, use_cache: bool = True, xp=None):
        import onnx
        from onnx.reference import ReferenceEvaluator

        model_path = _model_path()

        super().setup(param, use_cache=use_cache, xp=xp)

        dense_input = to_numpy(self._input[0])
        input_name = self._meta["onnx_input_name"]

        model = onnx.load(str(model_path), load_external_data=True)
        outputs = ReferenceEvaluator(model).run(
            None,
            {input_name: dense_input},
        )
        if isinstance(outputs, dict):
            raise TypeError("Expected ReferenceEvaluator outputs as a list")
        expected = outputs[0]
        self._ref_outputs = [from_numpy(np.asarray(expected))]
        self._ref_meta = {
            "rtol": 1e-4,
            "atol": 1e-4,
        }

    def benchmark(
        self,
        xp,
        meta: dict[str, Any],
        input,
        layers_0_conv1_weight,
        layers_0_conv1_bias,
        layers_0_conv2_weight,
        layers_0_conv2_bias,
        fc1_weight,
        fc1_bias,
        fc2_weight,
        fc2_bias,
        fc3_weight,
        fc3_bias,
        val_4,
    ):
        # conv1: Conv(strides=1, pads=1, dilations=1, group=1) + ReLU
        x = input
        w = layers_0_conv1_weight
        kernel = tuple(int(k) for k in w.shape[2:])
        out_spatial = tuple(
            max(int(x.shape[2 + i]) + 2 - kernel[i] + 1, 0) for i in range(2)
        )
        c_in = x.shape[1]
        m_out = w.shape[0]
        kernel_size = kernel[0] * kernel[1]
        try:
            zero = xp.zeros((), dtype=x.dtype, device=getattr(x, "device", None))
        except TypeError:
            zero = xp.zeros((), dtype=x.dtype)
        patches = xp.unfold(
            x,
            kernel,
            axes=(2, 3),
            strides=(1, 1),
            dilations=(1, 1),
            padding=((1, 1), (1, 1)),
            fill_value=zero,
        )
        cols = xp.reshape(
            xp.permute_dims(patches, (0, 2, 3, 1, 4, 5)), (-1, c_in * kernel_size)
        )
        rows = xp.reshape(w, (m_out, c_in * kernel_size))
        conv2d = cols @ xp.permute_dims(rows, (1, 0))
        conv2d = xp.reshape(conv2d, (x.shape[0], *out_spatial, m_out))
        conv2d = xp.permute_dims(conv2d, (0, 3, 1, 2))
        conv2d = conv2d + xp.reshape(layers_0_conv1_bias, (1, m_out, 1, 1))
        relu = xp.maximum(conv2d, 0)

        # conv2: Conv(strides=1, pads=1, dilations=1, group=1) + ReLU
        x = relu
        w = layers_0_conv2_weight
        kernel = tuple(int(k) for k in w.shape[2:])
        out_spatial = tuple(
            max(int(x.shape[2 + i]) + 2 - kernel[i] + 1, 0) for i in range(2)
        )
        c_in = x.shape[1]
        m_out = w.shape[0]
        kernel_size = kernel[0] * kernel[1]
        try:
            zero = xp.zeros((), dtype=x.dtype, device=getattr(x, "device", None))
        except TypeError:
            zero = xp.zeros((), dtype=x.dtype)
        patches = xp.unfold(
            x,
            kernel,
            axes=(2, 3),
            strides=(1, 1),
            dilations=(1, 1),
            padding=((1, 1), (1, 1)),
            fill_value=zero,
        )
        cols = xp.reshape(
            xp.permute_dims(patches, (0, 2, 3, 1, 4, 5)), (-1, c_in * kernel_size)
        )
        rows = xp.reshape(w, (m_out, c_in * kernel_size))
        conv2d_1 = cols @ xp.permute_dims(rows, (1, 0))
        conv2d_1 = xp.reshape(conv2d_1, (x.shape[0], *out_spatial, m_out))
        conv2d_1 = xp.permute_dims(conv2d_1, (0, 3, 1, 2))
        conv2d_1 = conv2d_1 + xp.reshape(layers_0_conv2_bias, (1, m_out, 1, 1))
        relu_1 = xp.maximum(conv2d_1, 0)

        # max_pool: MaxPool(kernel=2, strides=2, pads=0, ceil_mode=0)
        x = relu_1
        try:
            integral = xp.isdtype(x.dtype, "integral")
        except (AttributeError, TypeError):
            integral = False
        min_value = xp.iinfo(x.dtype).min if integral else float("-inf")
        try:
            device = getattr(x, "device", None)
            fill = xp.asarray(min_value, dtype=x.dtype, device=device)
        except TypeError:
            fill = xp.asarray(min_value, dtype=x.dtype)
        patches = xp.unfold(
            x,
            (2, 2),
            axes=(2, 3),
            strides=(2, 2),
            dilations=(1, 1),
            padding=((0, 0), (0, 0)),
            fill_value=fill,
        )
        max_pool2d = xp.max(patches, axis=(4, 5))

        # view: Reshape(val_4, allowzero=1)
        if len(val_4.shape) == 0:
            view_shape: tuple[int, ...] = (int(val_4),)
        else:
            view_shape = tuple(int(val_4[i]) for i in range(val_4.shape[0]))
        view = xp.reshape(max_pool2d, view_shape)

        # fc1, fc2, fc3: Gemm(alpha=1, beta=1, transB=1)
        linear = 1.0 * (view @ xp.permute_dims(fc1_weight, (1, 0)))
        linear = linear + 1.0 * fc1_bias
        relu_2 = xp.maximum(linear, 0)
        linear_1 = 1.0 * (relu_2 @ xp.permute_dims(fc2_weight, (1, 0)))
        linear_1 = linear_1 + 1.0 * fc2_bias
        relu_3 = xp.maximum(linear_1, 0)
        linear_2 = 1.0 * (relu_3 @ xp.permute_dims(fc3_weight, (1, 0)))
        return linear_2 + 1.0 * fc3_bias

    def check(self, param):
        actual = to_numpy(self._output[0])
        expected = to_numpy(self._ref_outputs[0])

        np.testing.assert_allclose(
            actual,
            expected,
            rtol=self._ref_meta["rtol"],
            atol=self._ref_meta["atol"],
        )

    def teardown(self, param):
        super().teardown(param)
