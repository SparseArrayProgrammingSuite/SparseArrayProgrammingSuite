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
        # return model(input, ..., val_4, xp=xp)

        # conv2d = _onnxpy_conv(...)
        x_c1 = input
        w_c1 = layers_0_conv1_weight
        b_c1 = layers_0_conv1_bias
        kernel_shape_c1 = None
        strides_c1 = (1, 1)
        pads_c1 = (1, 1, 1, 1)
        dilations_c1 = (1, 1)
        group_c1 = 1
        auto_pad_c1 = "NOTSET"
        rank_c1 = x_c1.ndim - 2
        kernel_c1 = tuple(int(k) for k in (kernel_shape_c1 or w_c1.shape[2:]))
        strides_t_c1 = tuple(
            int(s) for s in (strides_c1 if strides_c1 is not None else [1] * rank_c1)
        )
        dilations_t_c1 = tuple(
            int(d)
            for d in (dilations_c1 if dilations_c1 is not None else [1] * rank_c1)
        )
        group_i_c1 = int(group_c1)
        ceil_mode_c1 = 0

        # pad_pairs_c1 = _onnxpy_resolve_pads(...)
        input_spatial_c1rp = x_c1.shape[2:]
        rank_c1rp = len(kernel_c1)
        mode_c1rp = (auto_pad_c1 or "NOTSET").upper()
        if mode_c1rp == "NOTSET":
            if pads_c1 is None:
                pad_pairs_c1 = tuple((0, 0) for _ in range(rank_c1rp))
            else:
                flat_c1rp = list(pads_c1)
                pad_pairs_c1 = tuple(
                    (int(flat_c1rp[i]), int(flat_c1rp[i + rank_c1rp]))
                    for i in range(rank_c1rp)
                )
        elif mode_c1rp == "VALID":
            pad_pairs_c1 = tuple((0, 0) for _ in range(rank_c1rp))
        else:
            pairs_c1rp = []
            for i_c1rp in range(rank_c1rp):
                in_size_c1rp = int(input_spatial_c1rp[i_c1rp])
                s_c1rp = int(strides_t_c1[i_c1rp])
                d_c1rp = int(dilations_t_c1[i_c1rp])
                k_c1rp = int(kernel_c1[i_c1rp])
                out_size_c1rp = -(-in_size_c1rp // s_c1rp)
                eff_k_c1rp = (k_c1rp - 1) * d_c1rp + 1
                total_c1rp = max(
                    (out_size_c1rp - 1) * s_c1rp + eff_k_c1rp - in_size_c1rp, 0
                )
                if mode_c1rp == "SAME_UPPER":
                    pairs_c1rp.append((total_c1rp // 2, total_c1rp - total_c1rp // 2))
                else:
                    pairs_c1rp.append((total_c1rp - total_c1rp // 2, total_c1rp // 2))
            pad_pairs_c1 = tuple(pairs_c1rp)

        # out_spatial_c1 = _onnxpy_output_spatial(...)
        input_spatial_c1os = x_c1.shape[2:]
        rank_c1os = len(kernel_c1)
        output_c1os = []
        for i_c1os in range(rank_c1os):
            in_size_c1os = int(input_spatial_c1os[i_c1os])
            k_c1os = int(kernel_c1[i_c1os])
            s_c1os = int(strides_t_c1[i_c1os])
            d_c1os = int(dilations_t_c1[i_c1os])
            pad_before_c1os, pad_after_c1os = pad_pairs_c1[i_c1os]
            effective_kernel_c1os = (k_c1os - 1) * d_c1os + 1
            numerator_c1os = (
                in_size_c1os + pad_before_c1os + pad_after_c1os - effective_kernel_c1os
            )
            if int(ceil_mode_c1):
                out_size_c1os = numerator_c1os // s_c1os + 1
                if numerator_c1os % s_c1os:
                    out_size_c1os += 1
            else:
                out_size_c1os = numerator_c1os // s_c1os + 1
            output_c1os.append(max(out_size_c1os, 0))
        out_spatial_c1 = tuple(output_c1os)

        # pad_pairs_c1 = _onnxpy_pads_for_output(...)
        input_spatial_c1po = x_c1.shape[2:]
        expanded_c1po = []
        for i_c1po, out_size_c1po in enumerate(out_spatial_c1):
            in_size_c1po = int(input_spatial_c1po[i_c1po])
            k_c1po = int(kernel_c1[i_c1po])
            s_c1po = int(strides_t_c1[i_c1po])
            d_c1po = int(dilations_t_c1[i_c1po])
            before_c1po, after_c1po = pad_pairs_c1[i_c1po]
            effective_kernel_c1po = (k_c1po - 1) * d_c1po + 1
            needed_c1po = max(
                (int(out_size_c1po) - 1) * s_c1po
                + effective_kernel_c1po
                - in_size_c1po,
                0,
            )
            existing_c1po = before_c1po + after_c1po
            expanded_c1po.append(
                (before_c1po, after_c1po + max(needed_c1po - existing_c1po, 0))
            )
        pad_pairs_c1 = tuple(expanded_c1po)

        c_per_group_c1 = x_c1.shape[1] // group_i_c1
        m_out_c1 = w_c1.shape[0]
        m_per_group_c1 = m_out_c1 // group_i_c1

        kernel_size_c1 = 1
        for size_c1 in kernel_c1:
            kernel_size_c1 *= int(size_c1)

        unfold_axes_c1 = tuple(range(2, x_c1.ndim))

        # unfold_fill_value_c1 = _onnxpy_zeros(...)
        shape_c1z: tuple[int, ...] = ()
        dtype_c1z = getattr(x_c1, "dtype", None)
        # device_c1z = _onnxpy_device(x_c1)
        device_c1z = getattr(x_c1, "device", None)
        try:
            unfold_fill_value_c1 = xp.zeros(
                tuple(int(dim) for dim in shape_c1z), dtype=dtype_c1z, device=device_c1z
            )
        except TypeError:
            unfold_fill_value_c1 = xp.zeros(
                tuple(int(dim) for dim in shape_c1z), dtype=dtype_c1z
            )

        # patches_c1 = _onnxpy_unfold(...)
        if hasattr(xp, "unfold"):
            patches_c1 = xp.unfold(
                x_c1,
                kernel_c1,
                axes=unfold_axes_c1,
                strides=strides_t_c1,
                dilations=dilations_t_c1,
                padding=pad_pairs_c1,
                fill_value=unfold_fill_value_c1,
            )
        else:
            # UNAVOIDABLE DEVIATION: the original pure-Python fallback of
            # _onnxpy_unfold builds the windows with nested recursive functions
            # (element/window/output_at/base_output and _onnxpy_stack_nested), which
            # cannot be inlined without nested defs or recursion.
            raise NotImplementedError(
                "Array namespace does not provide unfold; the pure-Python unfold "
                "fallback is not available in the inlined benchmark."
            )

        spatial_axes_c1 = tuple(range(2, 2 + rank_c1))
        kernel_axes_c1 = tuple(range(2 + rank_c1, 2 + 2 * rank_c1))
        groups_c1 = []
        for group_index_c1 in range(group_i_c1):
            channel_start_c1 = group_index_c1 * c_per_group_c1
            channel_end_c1 = channel_start_c1 + c_per_group_c1
            output_start_c1 = group_index_c1 * m_per_group_c1
            output_end_c1 = output_start_c1 + m_per_group_c1
            group_patches_c1 = patches_c1[:, channel_start_c1:channel_end_c1, ...]
            group_weights_c1 = w_c1[output_start_c1:output_end_c1, ...]
            cols_c1 = xp.reshape(
                xp.permute_dims(
                    group_patches_c1,
                    (0, *spatial_axes_c1, 1, *kernel_axes_c1),
                ),
                (-1, c_per_group_c1 * kernel_size_c1),
            )
            rows_c1 = xp.reshape(
                group_weights_c1, (m_per_group_c1, c_per_group_c1 * kernel_size_c1)
            )
            group_out_c1 = cols_c1 @ xp.permute_dims(rows_c1, (1, 0))
            group_out_c1 = xp.reshape(
                group_out_c1, (x_c1.shape[0], *out_spatial_c1, m_per_group_c1)
            )
            groups_c1.append(
                xp.permute_dims(group_out_c1, (0, rank_c1 + 1, *range(1, rank_c1 + 1)))
            )

        out_c1 = (
            groups_c1[0] if len(groups_c1) == 1 else xp.concat(tuple(groups_c1), axis=1)
        )
        if b_c1 is not None:
            out_c1 = out_c1 + xp.reshape(b_c1, (1, m_out_c1, *((1,) * rank_c1)))
        conv2d = out_c1

        relu = xp.maximum(conv2d, 0)

        # conv2d_1 = _onnxpy_conv(...)
        x_c2 = relu
        w_c2 = layers_0_conv2_weight
        b_c2 = layers_0_conv2_bias
        kernel_shape_c2 = None
        strides_c2 = (1, 1)
        pads_c2 = (1, 1, 1, 1)
        dilations_c2 = (1, 1)
        group_c2 = 1
        auto_pad_c2 = "NOTSET"
        rank_c2 = x_c2.ndim - 2
        kernel_c2 = tuple(int(k) for k in (kernel_shape_c2 or w_c2.shape[2:]))
        strides_t_c2 = tuple(
            int(s) for s in (strides_c2 if strides_c2 is not None else [1] * rank_c2)
        )
        dilations_t_c2 = tuple(
            int(d)
            for d in (dilations_c2 if dilations_c2 is not None else [1] * rank_c2)
        )
        group_i_c2 = int(group_c2)
        ceil_mode_c2 = 0

        # pad_pairs_c2 = _onnxpy_resolve_pads(...)
        input_spatial_c2rp = x_c2.shape[2:]
        rank_c2rp = len(kernel_c2)
        mode_c2rp = (auto_pad_c2 or "NOTSET").upper()
        if mode_c2rp == "NOTSET":
            if pads_c2 is None:
                pad_pairs_c2 = tuple((0, 0) for _ in range(rank_c2rp))
            else:
                flat_c2rp = list(pads_c2)
                pad_pairs_c2 = tuple(
                    (int(flat_c2rp[i]), int(flat_c2rp[i + rank_c2rp]))
                    for i in range(rank_c2rp)
                )
        elif mode_c2rp == "VALID":
            pad_pairs_c2 = tuple((0, 0) for _ in range(rank_c2rp))
        else:
            pairs_c2rp = []
            for i_c2rp in range(rank_c2rp):
                in_size_c2rp = int(input_spatial_c2rp[i_c2rp])
                s_c2rp = int(strides_t_c2[i_c2rp])
                d_c2rp = int(dilations_t_c2[i_c2rp])
                k_c2rp = int(kernel_c2[i_c2rp])
                out_size_c2rp = -(-in_size_c2rp // s_c2rp)
                eff_k_c2rp = (k_c2rp - 1) * d_c2rp + 1
                total_c2rp = max(
                    (out_size_c2rp - 1) * s_c2rp + eff_k_c2rp - in_size_c2rp, 0
                )
                if mode_c2rp == "SAME_UPPER":
                    pairs_c2rp.append((total_c2rp // 2, total_c2rp - total_c2rp // 2))
                else:
                    pairs_c2rp.append((total_c2rp - total_c2rp // 2, total_c2rp // 2))
            pad_pairs_c2 = tuple(pairs_c2rp)

        # out_spatial_c2 = _onnxpy_output_spatial(...)
        input_spatial_c2os = x_c2.shape[2:]
        rank_c2os = len(kernel_c2)
        output_c2os = []
        for i_c2os in range(rank_c2os):
            in_size_c2os = int(input_spatial_c2os[i_c2os])
            k_c2os = int(kernel_c2[i_c2os])
            s_c2os = int(strides_t_c2[i_c2os])
            d_c2os = int(dilations_t_c2[i_c2os])
            pad_before_c2os, pad_after_c2os = pad_pairs_c2[i_c2os]
            effective_kernel_c2os = (k_c2os - 1) * d_c2os + 1
            numerator_c2os = (
                in_size_c2os + pad_before_c2os + pad_after_c2os - effective_kernel_c2os
            )
            if int(ceil_mode_c2):
                out_size_c2os = numerator_c2os // s_c2os + 1
                if numerator_c2os % s_c2os:
                    out_size_c2os += 1
            else:
                out_size_c2os = numerator_c2os // s_c2os + 1
            output_c2os.append(max(out_size_c2os, 0))
        out_spatial_c2 = tuple(output_c2os)

        # pad_pairs_c2 = _onnxpy_pads_for_output(...)
        input_spatial_c2po = x_c2.shape[2:]
        expanded_c2po = []
        for i_c2po, out_size_c2po in enumerate(out_spatial_c2):
            in_size_c2po = int(input_spatial_c2po[i_c2po])
            k_c2po = int(kernel_c2[i_c2po])
            s_c2po = int(strides_t_c2[i_c2po])
            d_c2po = int(dilations_t_c2[i_c2po])
            before_c2po, after_c2po = pad_pairs_c2[i_c2po]
            effective_kernel_c2po = (k_c2po - 1) * d_c2po + 1
            needed_c2po = max(
                (int(out_size_c2po) - 1) * s_c2po
                + effective_kernel_c2po
                - in_size_c2po,
                0,
            )
            existing_c2po = before_c2po + after_c2po
            expanded_c2po.append(
                (before_c2po, after_c2po + max(needed_c2po - existing_c2po, 0))
            )
        pad_pairs_c2 = tuple(expanded_c2po)

        c_per_group_c2 = x_c2.shape[1] // group_i_c2
        m_out_c2 = w_c2.shape[0]
        m_per_group_c2 = m_out_c2 // group_i_c2

        kernel_size_c2 = 1
        for size_c2 in kernel_c2:
            kernel_size_c2 *= int(size_c2)

        unfold_axes_c2 = tuple(range(2, x_c2.ndim))

        # unfold_fill_value_c2 = _onnxpy_zeros(...)
        shape_c2z: tuple[int, ...] = ()
        dtype_c2z = getattr(x_c2, "dtype", None)
        # device_c2z = _onnxpy_device(x_c2)
        device_c2z = getattr(x_c2, "device", None)
        try:
            unfold_fill_value_c2 = xp.zeros(
                tuple(int(dim) for dim in shape_c2z), dtype=dtype_c2z, device=device_c2z
            )
        except TypeError:
            unfold_fill_value_c2 = xp.zeros(
                tuple(int(dim) for dim in shape_c2z), dtype=dtype_c2z
            )

        # patches_c2 = _onnxpy_unfold(...)
        if hasattr(xp, "unfold"):
            patches_c2 = xp.unfold(
                x_c2,
                kernel_c2,
                axes=unfold_axes_c2,
                strides=strides_t_c2,
                dilations=dilations_t_c2,
                padding=pad_pairs_c2,
                fill_value=unfold_fill_value_c2,
            )
        else:
            # UNAVOIDABLE DEVIATION: the original pure-Python fallback of
            # _onnxpy_unfold builds the windows with nested recursive functions
            # (element/window/output_at/base_output and _onnxpy_stack_nested), which
            # cannot be inlined without nested defs or recursion.
            raise NotImplementedError(
                "Array namespace does not provide unfold; the pure-Python unfold "
                "fallback is not available in the inlined benchmark."
            )

        spatial_axes_c2 = tuple(range(2, 2 + rank_c2))
        kernel_axes_c2 = tuple(range(2 + rank_c2, 2 + 2 * rank_c2))
        groups_c2 = []
        for group_index_c2 in range(group_i_c2):
            channel_start_c2 = group_index_c2 * c_per_group_c2
            channel_end_c2 = channel_start_c2 + c_per_group_c2
            output_start_c2 = group_index_c2 * m_per_group_c2
            output_end_c2 = output_start_c2 + m_per_group_c2
            group_patches_c2 = patches_c2[:, channel_start_c2:channel_end_c2, ...]
            group_weights_c2 = w_c2[output_start_c2:output_end_c2, ...]
            cols_c2 = xp.reshape(
                xp.permute_dims(
                    group_patches_c2,
                    (0, *spatial_axes_c2, 1, *kernel_axes_c2),
                ),
                (-1, c_per_group_c2 * kernel_size_c2),
            )
            rows_c2 = xp.reshape(
                group_weights_c2, (m_per_group_c2, c_per_group_c2 * kernel_size_c2)
            )
            group_out_c2 = cols_c2 @ xp.permute_dims(rows_c2, (1, 0))
            group_out_c2 = xp.reshape(
                group_out_c2, (x_c2.shape[0], *out_spatial_c2, m_per_group_c2)
            )
            groups_c2.append(
                xp.permute_dims(group_out_c2, (0, rank_c2 + 1, *range(1, rank_c2 + 1)))
            )

        out_c2 = (
            groups_c2[0] if len(groups_c2) == 1 else xp.concat(tuple(groups_c2), axis=1)
        )
        if b_c2 is not None:
            out_c2 = out_c2 + xp.reshape(b_c2, (1, m_out_c2, *((1,) * rank_c2)))
        conv2d_1 = out_c2

        relu_1 = xp.maximum(conv2d_1, 0)

        # max_pool2d = _onnxpy_max_pool(...)
        x_mp = relu_1
        kernel_shape_mp = (2, 2)
        strides_mp = (2, 2)
        pads_mp = (0, 0, 0, 0)
        dilations_mp = (1, 1)
        ceil_mode_mp = 0
        auto_pad_mp = "NOTSET"
        rank_mp = x_mp.ndim - 2
        kernel_mp = tuple(int(k) for k in kernel_shape_mp)
        strides_t_mp = tuple(
            int(s) for s in (strides_mp if strides_mp is not None else [1] * rank_mp)
        )
        dilations_t_mp = tuple(
            int(d)
            for d in (dilations_mp if dilations_mp is not None else [1] * rank_mp)
        )

        # pad_pairs_mp = _onnxpy_resolve_pads(...)
        input_spatial_mprp = x_mp.shape[2:]
        rank_mprp = len(kernel_mp)
        mode_mprp = (auto_pad_mp or "NOTSET").upper()
        if mode_mprp == "NOTSET":
            if pads_mp is None:
                pad_pairs_mp = tuple((0, 0) for _ in range(rank_mprp))
            else:
                flat_mprp = list(pads_mp)
                pad_pairs_mp = tuple(
                    (int(flat_mprp[i]), int(flat_mprp[i + rank_mprp]))
                    for i in range(rank_mprp)
                )
        elif mode_mprp == "VALID":
            pad_pairs_mp = tuple((0, 0) for _ in range(rank_mprp))
        else:
            pairs_mprp = []
            for i_mprp in range(rank_mprp):
                in_size_mprp = int(input_spatial_mprp[i_mprp])
                s_mprp = int(strides_t_mp[i_mprp])
                d_mprp = int(dilations_t_mp[i_mprp])
                k_mprp = int(kernel_mp[i_mprp])
                out_size_mprp = -(-in_size_mprp // s_mprp)
                eff_k_mprp = (k_mprp - 1) * d_mprp + 1
                total_mprp = max(
                    (out_size_mprp - 1) * s_mprp + eff_k_mprp - in_size_mprp, 0
                )
                if mode_mprp == "SAME_UPPER":
                    pairs_mprp.append((total_mprp // 2, total_mprp - total_mprp // 2))
                else:
                    pairs_mprp.append((total_mprp - total_mprp // 2, total_mprp // 2))
            pad_pairs_mp = tuple(pairs_mprp)

        # out_spatial_mp = _onnxpy_output_spatial(...)
        input_spatial_mpos = x_mp.shape[2:]
        rank_mpos = len(kernel_mp)
        output_mpos = []
        for i_mpos in range(rank_mpos):
            in_size_mpos = int(input_spatial_mpos[i_mpos])
            k_mpos = int(kernel_mp[i_mpos])
            s_mpos = int(strides_t_mp[i_mpos])
            d_mpos = int(dilations_t_mp[i_mpos])
            pad_before_mpos, pad_after_mpos = pad_pairs_mp[i_mpos]
            effective_kernel_mpos = (k_mpos - 1) * d_mpos + 1
            numerator_mpos = (
                in_size_mpos + pad_before_mpos + pad_after_mpos - effective_kernel_mpos
            )
            if int(ceil_mode_mp):
                out_size_mpos = numerator_mpos // s_mpos + 1
                if numerator_mpos % s_mpos:
                    out_size_mpos += 1
            else:
                out_size_mpos = numerator_mpos // s_mpos + 1
            output_mpos.append(max(out_size_mpos, 0))
        out_spatial_mp = tuple(output_mpos)

        # pad_pairs_mp = _onnxpy_pads_for_output(...)
        input_spatial_mppo = x_mp.shape[2:]
        expanded_mppo = []
        for i_mppo, out_size_mppo in enumerate(out_spatial_mp):
            in_size_mppo = int(input_spatial_mppo[i_mppo])
            k_mppo = int(kernel_mp[i_mppo])
            s_mppo = int(strides_t_mp[i_mppo])
            d_mppo = int(dilations_t_mp[i_mppo])
            before_mppo, after_mppo = pad_pairs_mp[i_mppo]
            effective_kernel_mppo = (k_mppo - 1) * d_mppo + 1
            needed_mppo = max(
                (int(out_size_mppo) - 1) * s_mppo
                + effective_kernel_mppo
                - in_size_mppo,
                0,
            )
            existing_mppo = before_mppo + after_mppo
            expanded_mppo.append(
                (before_mppo, after_mppo + max(needed_mppo - existing_mppo, 0))
            )
        pad_pairs_mp = tuple(expanded_mppo)

        unfold_axes_mp = tuple(range(2, x_mp.ndim))

        # unfold_fill_value_mp = _onnxpy_min_value(...)
        min_value_done_mpmv = False
        try:
            if xp.isdtype(x_mp.dtype, "integral"):
                value_mpmv = xp.iinfo(x_mp.dtype).min

                # unfold_fill_value_mp = _onnxpy_asarray_like(...)
                dtype_mpmvi = getattr(x_mp, "dtype", None)
                # device_mpmvi = _onnxpy_device(x_mp)
                device_mpmvi = getattr(x_mp, "device", None)
                try:
                    unfold_fill_value_mp = xp.asarray(
                        value_mpmv, dtype=dtype_mpmvi, device=device_mpmvi
                    )
                except TypeError:
                    unfold_fill_value_mp = xp.asarray(value_mpmv, dtype=dtype_mpmvi)
                min_value_done_mpmv = True
        except (AttributeError, TypeError):
            pass
        if not min_value_done_mpmv:
            # unfold_fill_value_mp = _onnxpy_asarray_like(...)
            dtype_mpmvf = getattr(x_mp, "dtype", None)
            # device_mpmvf = _onnxpy_device(x_mp)
            device_mpmvf = getattr(x_mp, "device", None)
            try:
                unfold_fill_value_mp = xp.asarray(
                    float("-inf"), dtype=dtype_mpmvf, device=device_mpmvf
                )
            except TypeError:
                unfold_fill_value_mp = xp.asarray(float("-inf"), dtype=dtype_mpmvf)

        # patches_mp = _onnxpy_unfold(...)
        if hasattr(xp, "unfold"):
            patches_mp = xp.unfold(
                x_mp,
                kernel_mp,
                axes=unfold_axes_mp,
                strides=strides_t_mp,
                dilations=dilations_t_mp,
                padding=pad_pairs_mp,
                fill_value=unfold_fill_value_mp,
            )
        else:
            # UNAVOIDABLE DEVIATION: the original pure-Python fallback of
            # _onnxpy_unfold builds the windows with nested recursive functions
            # (element/window/output_at/base_output and _onnxpy_stack_nested), which
            # cannot be inlined without nested defs or recursion.
            raise NotImplementedError(
                "Array namespace does not provide unfold; the pure-Python unfold "
                "fallback is not available in the inlined benchmark."
            )

        max_pool2d = xp.max(patches_mp, axis=tuple(range(2 + rank_mp, 2 + 2 * rank_mp)))

        # view = _onnxpy_reshape(...)
        allow_zero_rs = 1
        # shape_py_rs = _onnxpy_to_python(val_4)
        shape_py_rs: Any
        if isinstance(val_4, list | tuple):
            shape_py_rs = list(val_4)
        else:
            to_python_done_rs = False
            if hasattr(val_4, "shape"):
                if len(val_4.shape) == 0:
                    shape_py_rs = int(val_4)
                    to_python_done_rs = True
                elif len(val_4.shape) == 1:
                    shape_py_rs = [int(val_4[i]) for i in range(val_4.shape[0])]
                    to_python_done_rs = True
            if not to_python_done_rs:
                shape_py_rs = val_4
        if not isinstance(shape_py_rs, list | tuple):
            shape_py_rs = [int(shape_py_rs)]
        target_rs = []
        in_shape_rs = tuple(int(d) for d in max_pool2d.shape)
        for i_rs, dim_rs in enumerate(shape_py_rs):
            d_rs = int(dim_rs)
            if d_rs == 0 and not int(allow_zero_rs):
                target_rs.append(in_shape_rs[i_rs])
            else:
                target_rs.append(d_rs)
        view = xp.reshape(max_pool2d, tuple(target_rs))

        # linear = _onnxpy_gemm(...)
        alpha_g1 = 1.0
        beta_g1 = 1.0
        trans_a_g1 = 0
        trans_b_g1 = 1
        a2_g1 = xp.permute_dims(view, (1, 0)) if int(trans_a_g1) else view
        b2_g1 = xp.permute_dims(fc1_weight, (1, 0)) if int(trans_b_g1) else fc1_weight
        out_g1 = float(alpha_g1) * (a2_g1 @ b2_g1)
        if fc1_bias is not None:
            out_g1 = out_g1 + float(beta_g1) * fc1_bias
        linear = out_g1

        relu_2 = xp.maximum(linear, 0)

        # linear_1 = _onnxpy_gemm(...)
        alpha_g2 = 1.0
        beta_g2 = 1.0
        trans_a_g2 = 0
        trans_b_g2 = 1
        a2_g2 = xp.permute_dims(relu_2, (1, 0)) if int(trans_a_g2) else relu_2
        b2_g2 = xp.permute_dims(fc2_weight, (1, 0)) if int(trans_b_g2) else fc2_weight
        out_g2 = float(alpha_g2) * (a2_g2 @ b2_g2)
        if fc2_bias is not None:
            out_g2 = out_g2 + float(beta_g2) * fc2_bias
        linear_1 = out_g2

        relu_3 = xp.maximum(linear_1, 0)

        # return _onnxpy_gemm(...)
        alpha_g3 = 1.0
        beta_g3 = 1.0
        trans_a_g3 = 0
        trans_b_g3 = 1
        a2_g3 = xp.permute_dims(relu_3, (1, 0)) if int(trans_a_g3) else relu_3
        b2_g3 = xp.permute_dims(fc3_weight, (1, 0)) if int(trans_b_g3) else fc3_weight
        out_g3 = float(alpha_g3) * (a2_g3 @ b2_g3)
        if fc3_bias is not None:
            out_g3 = out_g3 + float(beta_g3) * fc3_bias
        return out_g3

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
