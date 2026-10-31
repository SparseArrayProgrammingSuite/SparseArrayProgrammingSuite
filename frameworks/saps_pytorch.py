import os

import numpy as np

import array_api_compat.torch as torch_xp
import torch
import torch._dynamo
import torch.nn.functional as F
from binsparse.conversions import from_numpy, from_torch, to_sparse, to_torch

from saps_framework import (
    Framework,
    normalize_pad_width,
    normalize_unfold_args,
)
from saps_framework.einsum import Access, nary_ops, parse_einsum, reduction_ops

torch._dynamo.config.suppress_errors = True
torch_xp.power = torch.pow  # type: ignore[attr-defined]


# Keep parsing and subscript translation outside Dynamo; tensor operations remain
# traceable, including the native torch.einsum call.
@torch.compiler.disable
def _parse_einsum(prgm):
    parsed = parse_einsum(prgm)
    return parsed, parsed.native_contraction()


# torch can't promote these with bool (or run matmul on them), so bool operands
# are cast to the unsigned dtype first.
_WIDE_UNSIGNED = frozenset({torch.uint16, torch.uint32, torch.uint64})
_BOOL_PROMOTED_OPS = frozenset(
    {"add", "subtract", "multiply", "bitwise_and", "bitwise_or", "bitwise_xor"}
)


def _cast_bool_to_wide_unsigned(x1, x2):
    if torch.is_tensor(x1) and torch.is_tensor(x2):
        if x1.dtype == torch.bool and x2.dtype in _WIDE_UNSIGNED:
            x1 = x1.to(x2.dtype)
        elif x2.dtype == torch.bool and x1.dtype in _WIDE_UNSIGNED:
            x2 = x2.to(x1.dtype)
    return x1, x2


def _is_sparse_tensor(array):
    return array.layout in {
        torch.sparse_coo,
        torch.sparse_csr,
        torch.sparse_csc,
    }


def _dense(array):
    if torch.is_tensor(array) and _is_sparse_tensor(array):
        return array.to_dense()
    return array


def _dense_if_mixed(x1, x2):
    if any(torch.is_tensor(x) and x.layout == torch.strided for x in (x1, x2)):
        return _dense(x1), _dense(x2)
    return x1, x2


class PytorchLinalg:
    @staticmethod
    def _dense(array):
        return array.to_dense() if _is_sparse_tensor(array) else array

    @staticmethod
    def solve(A, b, **kwargs):
        return torch.linalg.solve(PytorchLinalg._dense(A), b, **kwargs)

    @staticmethod
    def norm(x, **kwargs):
        return torch.linalg.norm(PytorchLinalg._dense(x), **kwargs)

    @staticmethod
    def lstsq(A, b, **kwargs):
        result = torch.linalg.lstsq(PytorchLinalg._dense(A), b, **kwargs)
        return result.solution, result.residuals, result.rank, result.singular_values

    @staticmethod
    def pinv(A, **kwargs):
        return torch.linalg.pinv(PytorchLinalg._dense(A), **kwargs)

    @staticmethod
    def svd(A, **kwargs):
        return torch.linalg.svd(PytorchLinalg._dense(A), **kwargs)


def _device() -> str:
    # The competition config marks GPU frameworks with SAPS_DEVICE=gpu.
    if os.environ.get("SAPS_DEVICE", "cpu") != "gpu":
        return "cpu"
    if not torch.cuda.is_available():
        raise RuntimeError("SAPS_DEVICE=gpu, but PyTorch cannot see a CUDA device")
    return "cuda"


def _f_pad(x, pad_width, mode):
    """Pad x with F.pad in a non-constant mode, or return None if F.pad can't.

    These modes only pad the last 1-3 axes of an input with 1 or 2 leading
    axes, and reflect and circular padding can't wrap past the other end.
    """
    padded = [axis for axis, pair in enumerate(pad_width) if pair != (0, 0)]
    if not padded:
        return x
    trailing = x.ndim - padded[0]
    if trailing > 3 or x.ndim - trailing not in (1, 2):
        return None
    for axis in padded:
        if mode == "reflect" and max(pad_width[axis]) >= x.shape[axis]:
            return None
        if mode == "circular" and max(pad_width[axis]) > x.shape[axis]:
            return None
    pad = [width for pair in reversed(pad_width[padded[0] :]) for width in pair]
    return F.pad(x, pad, mode=mode)


class PytorchFramework(Framework):
    def __init__(self, sparse_layout: str = "COO"):
        self.sparse_layout = sparse_layout
        # Inputs from to_torch and arrays created by benchmarks both follow the
        # default device, so this alone moves the whole benchmark to the GPU.
        self.device = _device()
        torch.set_default_device(self.device)
        # Benchmarks are written against NumPy, whose default float is float64.
        # Torch's float32 would break matmuls with float64 inputs and round
        # values stored into arrays created without a dtype.
        torch.set_default_dtype(torch.float64)

    @property
    def linalg(self):
        return PytorchLinalg

    def from_binsparse(self, array):
        if array.fill and array.fill_value != 0:
            return torch.asarray(np.asarray(to_sparse(array).todense()))
        result = to_torch(array)
        if self.sparse_layout == "CSR" and result.layout == torch.sparse_coo:
            if result.ndim != 2:
                raise ValueError("PyTorch CSR only works for 2D matrices")
            return result.to_sparse_csr()
        return result

    def to_binsparse(self, array):
        if not torch.is_tensor(array):
            # Benchmarks may return NumPy arrays or Python scalars.
            return from_numpy(np.asarray(array))
        return from_torch(array.detach().cpu())

    def lazy(self, array):
        return array

    def compute(self, array):
        return array

    def compile(self, func):
        return torch.compile(func)

    def einsum(self, prgm, **kwargs):
        parsed, contraction = _parse_einsum(prgm)
        if contraction is not NotImplemented:
            result = self._native_einsum(parsed.op, contraction, kwargs)
            if result is not NotImplemented:
                return result
        return parsed.run(self, kwargs)

    @staticmethod
    def _native_einsum(reduction, contraction, kwargs):
        equation, leaves, bitwise = contraction
        arrays = [kwargs[leaf.tns] for leaf in leaves if isinstance(leaf, Access)]
        # Torch's einsum does not support sparse layouts. Keep their existing
        # evaluator rather than densifying the tensor to enter the fast path.
        if any(array.layout != torch.strided for array in arrays):
            return NotImplemented
        device = arrays[0].device if arrays else None
        operands = [
            kwargs[leaf.tns]
            if isinstance(leaf, Access)
            else torch.tensor(leaf.value, device=device)
            for leaf in leaves
        ]
        boolean = all(operand.dtype == torch.bool for operand in operands)
        logical_reduction = reduction in {"|", "or"}
        if (bitwise or logical_reduction) and not boolean:
            return NotImplemented
        dtype = operands[0].dtype
        for operand in operands[1:]:
            if dtype in _WIDE_UNSIGNED or operand.dtype in _WIDE_UNSIGNED:
                return NotImplemented
            dtype = torch.promote_types(dtype, operand.dtype)
        if dtype in _WIDE_UNSIGNED:
            return NotImplemented
        if not dtype.is_floating_point and not dtype.is_complex:
            if not boolean and len(operands) > 1 and dtype != torch.int64:
                # Preserve overflow in narrow integer products before summing.
                return NotImplemented
            dtype = torch.int64
        # Boolean bmm is unsupported; integer contraction counts matches, and
        # comparison with zero implements an OR reduction without changing +=.
        # CUDA has no integer bmm either, so count in float64 there (exact below
        # 2**53) and return int64 as on the CPU.
        compute_dtype = dtype
        if dtype == torch.int64 and operands[0].is_cuda:
            compute_dtype = torch.float64
        result = torch.einsum(
            equation, *(operand.to(compute_dtype) for operand in operands)
        ).to(dtype)
        return result != 0 if logical_reduction else result

    def pad_constant(self, x, pad_width, *, fill_value=0):
        x = x.to_dense() if _is_sparse_tensor(x) else x
        pad_width = normalize_pad_width(x.ndim, pad_width)
        if torch.is_tensor(fill_value):
            fill_value = fill_value.detach().cpu().item()
        # F.pad lists (before, after) pairs starting from the last axis.
        pad = [width for pair in reversed(pad_width) for width in pair]
        return F.pad(x, pad, mode="constant", value=float(fill_value))

    def pad_reflect(self, x, pad_width):
        x = x.to_dense() if _is_sparse_tensor(x) else x
        padded = _f_pad(x, normalize_pad_width(x.ndim, pad_width), "reflect")
        return super().pad_reflect(x, pad_width) if padded is None else padded

    def pad_edge(self, x, pad_width):
        x = x.to_dense() if _is_sparse_tensor(x) else x
        padded = _f_pad(x, normalize_pad_width(x.ndim, pad_width), "replicate")
        return super().pad_edge(x, pad_width) if padded is None else padded

    def pad_wrap(self, x, pad_width):
        x = x.to_dense() if _is_sparse_tensor(x) else x
        padded = _f_pad(x, normalize_pad_width(x.ndim, pad_width), "circular")
        return super().pad_wrap(x, pad_width) if padded is None else padded

    def unfold(self, x, kernel_shape, *, axes=None, strides=None, dilations=None):
        x = x.to_dense() if _is_sparse_tensor(x) else x
        axes_t, strides_t, dilations_t = normalize_unfold_args(
            x.ndim,
            kernel_shape,
            axes,
            strides,
            dilations,
        )
        kernel_t = tuple(int(size) for size in kernel_shape)

        if x.ndim == 4 and axes_t == (2, 3):
            patches = F.unfold(
                x,
                kernel_size=kernel_t,
                dilation=dilations_t,
                padding=0,
                stride=strides_t,
            )
            out_spatial = tuple(
                (int(x.shape[axis]) - ((kernel - 1) * dilation + 1)) // step + 1
                for axis, kernel, dilation, step in zip(
                    axes_t, kernel_t, dilations_t, strides_t, strict=True
                )
            )
            return patches.reshape(
                x.shape[0],
                x.shape[1],
                *kernel_t,
                *out_spatial,
            ).permute(0, 1, 4, 5, 2, 3)

        effective_kernel = tuple(
            (kernel - 1) * dilation + 1
            for kernel, dilation in zip(kernel_t, dilations_t, strict=True)
        )
        windows = x
        for axis, size, step in zip(axes_t, effective_kernel, strides_t, strict=True):
            windows = windows.unfold(axis, size, step)

        slices = [slice(None)] * windows.ndim
        for window_axis, dilation in enumerate(dilations_t, start=x.ndim):
            slices[window_axis] = slice(None, None, dilation)
        return windows[tuple(slices)]

    def conv(
        self,
        x,
        w,
        *,
        strides=None,
        dilations=None,
        padding=None,
        padding_mode="constant",
        fill_value=0,
        group=1,
        combine="*",
        reduce="+",
    ):
        if (
            nary_ops[combine] == "multiply"
            and reduction_ops[reduce] == "sum"
            and 1 <= x.ndim - 2 <= 3
            and x.ndim == w.ndim
            and x.dtype == w.dtype
            and x.dtype.is_floating_point
        ):
            x = x.to_dense() if _is_sparse_tensor(x) else x
            w = w.to_dense() if _is_sparse_tensor(w) else w
            axes = tuple(range(2, x.ndim))
            _, strides_t, dilations_t = normalize_unfold_args(
                x.ndim, w.shape[2:], axes, strides, dilations
            )
            x = self._pad(x, padding, axes, padding_mode, fill_value)
            conv = (F.conv1d, F.conv2d, F.conv3d)[x.ndim - 3]
            return conv(x, w, stride=strides_t, dilation=dilations_t, groups=group)
        return super().conv(
            x,
            w,
            strides=strides,
            dilations=dilations,
            padding=padding,
            padding_mode=padding_mode,
            fill_value=fill_value,
            group=group,
            combine=combine,
            reduce=reduce,
        )

    def topk(self, x, k, *, axis=-1, largest=True):
        x = x.to_dense() if _is_sparse_tensor(x) else x
        # torch.topk orders ties arbitrarily; a stable sort keeps index order.
        values, indices = torch.sort(x, dim=axis, descending=largest, stable=True)
        return values.narrow(axis, 0, k), indices.narrow(axis, 0, k)

    def pool(
        self,
        x,
        kernel_shape,
        *,
        reduce,
        strides=None,
        dilations=None,
        padding=None,
        padding_mode="constant",
        fill_value=0,
    ):
        rank = len(kernel_shape)
        operation = reduction_ops[reduce]
        if (
            operation in ("max", "min", "mean")
            and 1 <= rank <= 3
            and x.ndim == rank + 2
            and x.dtype.is_floating_point
        ):
            axes = tuple(range(2, x.ndim))
            _, strides_t, dilations_t = normalize_unfold_args(
                x.ndim, kernel_shape, axes, strides, dilations
            )
            # avg_pool has no dilation.
            if operation != "mean" or all(dilation == 1 for dilation in dilations_t):
                x = x.to_dense() if _is_sparse_tensor(x) else x
                x = self._pad(x, padding, axes, padding_mode, fill_value)
                kernel_t = tuple(int(size) for size in kernel_shape)
                max_pool = (F.max_pool1d, F.max_pool2d, F.max_pool3d)[rank - 1]
                if operation == "max":
                    return max_pool(x, kernel_t, stride=strides_t, dilation=dilations_t)
                if operation == "min":
                    return -max_pool(
                        -x, kernel_t, stride=strides_t, dilation=dilations_t
                    )
                avg_pool = (F.avg_pool1d, F.avg_pool2d, F.avg_pool3d)[rank - 1]
                return avg_pool(x, kernel_t, stride=strides_t)
        return super().pool(
            x,
            kernel_shape,
            reduce=reduce,
            strides=strides,
            dilations=dilations,
            padding=padding,
            padding_mode=padding_mode,
            fill_value=fill_value,
        )

    def replace(self, arr, old, new):
        if _is_sparse_tensor(arr):
            if old == 0 and new != 0:
                arr = arr.to_dense()
            else:
                coo = arr.to_sparse_coo().coalesce()
                values = coo.values()
                values = torch.where(
                    torch.isnan(values) if old != old else values == old, new, values
                )
                result = torch.sparse_coo_tensor(
                    coo.indices(),
                    values,
                    coo.shape,
                    device=arr.device,
                    is_coalesced=True,
                )
                if arr.layout == torch.sparse_csr:
                    return result.to_sparse_csr()
                if arr.layout == torch.sparse_csc:
                    return result.to_sparse_csc()
                return result
        return torch.where(torch.isnan(arr) if old != old else arr == old, new, arr)

    def matmul(self, x1, x2, /, **kwargs):
        x1, x2 = _cast_bool_to_wide_unsigned(x1, x2)
        out_dtype = torch.result_type(x1, x2)
        if x1.is_cuda and not (out_dtype.is_floating_point or out_dtype.is_complex):
            # CUDA has no integer GEMM. Float64 is exact below 2**53, and going
            # back through int64 wraps narrow dtypes like the CPU kernels do.
            result = torch_xp.matmul(
                x1.to(torch.float64), x2.to(torch.float64), **kwargs
            )
            return result.to(torch.int64).to(out_dtype)
        if x1.dtype in _WIDE_UNSIGNED or x2.dtype in _WIDE_UNSIGNED:
            out_dtype = torch.result_type(x1, x2)
            return torch_xp.matmul(x1.to(torch.int64), x2.to(torch.int64), **kwargs).to(
                out_dtype
            )
        return torch_xp.matmul(x1, x2, **kwargs)

    # Torch sparse has no where or logical ops, so sparse operands are densified.
    # A binary op densifies only next to a dense operand, so it allocates no more
    # than that operand does; two sparse operands still fail rather than densify.
    def where(self, condition, x1, x2, /):
        # With a zero else value, the result is sparse wherever the condition
        # is: x1 at the condition's stored True entries.
        if (
            torch.is_tensor(condition)
            and _is_sparse_tensor(condition)
            and not torch.is_tensor(x2)
            and x2 == 0
            and (
                not torch.is_tensor(x1)
                or (x1.layout == torch.strided and x1.shape == condition.shape)
            )
        ):
            mask = condition.to_sparse_coo().coalesce()
            indices = mask.indices()[:, mask.values().to(torch.bool)]
            if torch.is_tensor(x1):
                values = x1[tuple(indices)]
            else:
                values = torch.full((indices.shape[1],), x1, device=indices.device)
            return torch.sparse_coo_tensor(
                indices,
                values.to(torch.result_type(x1, x2)),
                condition.shape,
                is_coalesced=True,
            )
        return torch_xp.where(_dense(condition), _dense(x1), _dense(x2))

    def logical_not(self, x, /):
        return torch.logical_not(_dense(x))

    def logical_and(self, x1, x2, /):
        return torch.logical_and(*_dense_if_mixed(x1, x2))

    def logical_or(self, x1, x2, /):
        return torch.logical_or(*_dense_if_mixed(x1, x2))

    def logical_xor(self, x1, x2, /):
        return torch.logical_xor(*_dense_if_mixed(x1, x2))
    # NumPy-style array(), which torch lacks. Code from saps.codegen calls
    # xp.array(value, dtype=np.<dtype>), and torch only takes torch dtypes, so
    # NumPy builds the value and torch.asarray converts it.
    def array(self, obj, *args, **kwargs):
        return torch.asarray(np.asarray(obj, *args, **kwargs))

    # NumPy's transpose(x, axes); torch.transpose swaps two given dims instead.
    def transpose(self, x, axes=None):
        return x.permute(*(reversed(range(x.ndim)) if axes is None else axes))

    def maximum(self, x, y):
        if not torch.is_tensor(y):
            y = torch.as_tensor(y, dtype=x.dtype, device=x.device)
        return torch.maximum(x, y)

    def minimum(self, x, y):
        if not torch.is_tensor(y):
            y = torch.as_tensor(y, dtype=x.dtype, device=x.device)
        return torch.minimum(x, y)

    def __getattr__(self, name):
        attr = (
            getattr(torch_xp, name) if hasattr(torch_xp, name) else getattr(torch, name)
        )
        if name in _BOOL_PROMOTED_OPS:

            def op(x1, x2, /, **kwargs):
                return attr(*_cast_bool_to_wide_unsigned(x1, x2), **kwargs)

            return op
        return attr


xp = PytorchFramework()
