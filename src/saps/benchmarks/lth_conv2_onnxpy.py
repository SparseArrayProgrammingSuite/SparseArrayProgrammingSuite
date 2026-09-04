"""Lottery Ticket Conv-2 benchmark executed through ONNXPY and SAPS.

The ONNX model is the source of truth. If the model artifacts are not already
available, they are downloaded from Google Drive. ONNXPY compiles the complete
Conv-2 graph during benchmark setup, outside the timed region.

The generated ONNXPY model loads its weights internally, while SAPS supplies
the model's runtime input.
"""

from __future__ import annotations

import importlib.util
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

import gdown
import onnx
from binsparse.conversions import from_numpy, to_numpy
from onnx.reference import ReferenceEvaluator
from onnxpy import compile_model

from saps.benchmark import (
    Author,
    Benchmark,
    Contributor,
    DataInstance,
    Dataset,
    Generator,
    Ref,
)

_MODEL_ENV = "LTH_CONV2_ONNX"
_MODEL_FILE_NAME = "conv2_pruned_dense.onnx"
_MODEL_DATA_FILE_NAME = "conv2_pruned_dense.onnx.data"

_MODEL_URL = (
    "https://drive.google.com/uc?export=download&id=1jxF9fFXidr9h_p6fEQGiRQXZkqkYfn14"
)

_MODEL_DATA_URL = (
    "https://drive.google.com/uc?export=download&id=1FB_CE91DlFbrPWWT7bQ-ETGRXlzIqEcq"
)


def _references() -> list[Ref]:
    return [
        Ref(
            title=(
                "The Lottery Ticket Hypothesis: Finding Sparse, "
                "Trainable Neural Networks"
            ),
            authors=[Author("Jonathan Frankle"), Author("Michael Carbin")],
            conference="ICLR",
            year=2019,
        ),
        Ref(
            title="Deconstructing Lottery Tickets: Zeros, Signs, and the Supermask",
            authors=[
                Author("Hattie Zhou"),
                Author("Janice Lan"),
                Author("Rosanne Liu"),
                Author("Jason Yosinski"),
            ],
            conference="NeurIPS",
            year=2019,
        ),
    ]


def _default_data_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "lth"


def _download_if_missing(url: str, destination: Path) -> None:
    if destination.is_file() and destination.stat().st_size > 0:
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")

    if partial.exists():
        partial.unlink()

    result = gdown.download(url=url, output=str(partial), quiet=False)

    if result is None or not partial.is_file() or partial.stat().st_size == 0:
        if partial.exists():
            partial.unlink()
        raise RuntimeError(f"Failed to download LTH model artifact from {url}")

    partial.replace(destination)


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


def _load_generated_model(path: Path):
    module_name = f"_saps_onnxpy_{abs(hash(path.resolve()))}"
    spec = importlib.util.spec_from_file_location(module_name, path)

    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import ONNXPY generated file: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "model"):
        raise RuntimeError(f"ONNXPY generated module does not define model(): {path}")

    return module.model


class LTHConv2Dataset(Dataset):
    @property
    def name(self) -> str:
        return "conv2_pruned"

    @property
    def pretty_name(self) -> str:
        return "Lottery Ticket Conv-2 Pruned Model"

    @property
    def description(self) -> str:
        return "Pruned CIFAR-10 Conv-2 Lottery Ticket model exported to ONNX."

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
        return "Generates a deterministic runtime input for the Conv-2 model."

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
            "Lottery-ticket pruning produces neural-network parameters containing "
            "many exact zeros. This benchmark establishes the complete "
            "ONNX-to-ONNXPY-to-SAPS inference path for correctness and timing."
        )

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def datasets(self) -> list[LTHConv2Dataset]:
        return [LTHConv2Dataset()]

    def generate(self, _dataset: LTHConv2Dataset) -> DataInstance:
        model = onnx.load(str(_model_path()), load_external_data=False)

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

        return DataInstance(
            inputs=[from_numpy(model_input)],
            meta={"onnx_input_name": input_info.name},
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
        return "Runs the pruned Conv-2 ONNX graph through ONNXPY-generated Python."

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
            "The ONNX model remains the source of truth and ONNXPY translates "
            "the complete graph instead of manually reimplementing Conv-2."
        )

    @property
    def generators(self) -> list[Generator[Any]]:
        return [LTHConv2ONNXPYGenerator()]

    def setup(self, param, *, use_cache: bool = True, xp=None):
        self._onnxpy_tmpdir = tempfile.TemporaryDirectory(prefix="saps_lth_onnxpy_")
        generated_path = Path(self._onnxpy_tmpdir.name) / "conv2_generated.py"

        try:
            model_path = _model_path()

            compile_model(model_path, generated_path)
            self._onnxpy_model = _load_generated_model(generated_path)

            super().setup(param, use_cache=use_cache, xp=xp)

            dense_input = to_numpy(self._input[0])
            input_name = self._meta["onnx_input_name"]

            model = onnx.load(str(model_path), load_external_data=True)
            expected = ReferenceEvaluator(model).run(
                None,
                {input_name: dense_input},
            )[0]

            self._ref_outputs = [from_numpy(np.asarray(expected))]
            self._ref_meta = {
                "rtol": 1e-4,
                "atol": 1e-4,
            }

        except Exception:
            if hasattr(self, "_onnxpy_model"):
                del self._onnxpy_model

            if hasattr(self, "_onnxpy_tmpdir"):
                self._onnxpy_tmpdir.cleanup()
                del self._onnxpy_tmpdir

            raise

    def benchmark(
        self,
        xp,
        data: list[Any],
        meta: dict[str, Any],
    ):
        return [
            self._onnxpy_model(
                data[0],
                xp=xp,
            )
        ]

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
        try:
            super().teardown(param)
        finally:
            if hasattr(self, "_onnxpy_model"):
                del self._onnxpy_model

            if hasattr(self, "_onnxpy_tmpdir"):
                self._onnxpy_tmpdir.cleanup()
                del self._onnxpy_tmpdir
