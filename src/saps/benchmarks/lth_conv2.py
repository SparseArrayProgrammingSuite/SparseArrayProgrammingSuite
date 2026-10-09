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
        return "pruned"

    @property
    def pretty_name(self) -> str:
        return "Pruned"

    @property
    def description(self) -> str:
        return (
            "Pruned CIFAR-10 Conv-2 model with approximately 87.84% zero-valued "
            "parameters, archived with its ONNX external data in Zenodo record "
            "22650920."
        )

    @property
    def suites(self) -> list[str]:
        return ["lth", "standard"]

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"


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
            "imports its trained parameters and pruning-induced zeros, and "
            "generates a deterministic input and inline NumPy benchmark function "
            "with ONNXPY during setup."
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
        from onnx.reference import ReferenceEvaluator
        from onnxpy import compile_to_source

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
        model_source, tensor_inputs = compile_to_source(
            model, function_name="benchmark", extra_parameters=("meta",)
        )
        outputs = ReferenceEvaluator(model).run(None, {input_info.name: model_input})
        if isinstance(outputs, dict):
            raise TypeError("Expected ReferenceEvaluator outputs as a list")

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
            ref_outputs=[from_numpy(np.asarray(output)) for output in outputs],
            ref_meta={"rtol": 1e-4, "atol": 1e-4},
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
        for actual, expected in zip(self._output, self._ref_outputs, strict=True):
            np.testing.assert_allclose(
                to_numpy(actual),
                to_numpy(expected),
                rtol=self._ref_meta["rtol"],
                atol=self._ref_meta["atol"],
            )
