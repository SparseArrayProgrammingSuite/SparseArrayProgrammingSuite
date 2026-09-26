import array_api_compat.torch as torch_xp
import torch
import torch._dynamo
import torch.nn.functional as F
from binsparse.conversions import from_torch, to_torch

from saps_framework import Framework, normalize_unfold_args
from saps_framework.einsum import parse_einsum

torch._dynamo.config.suppress_errors = True
torch_xp.power = torch.pow  # type: ignore[attr-defined]
# Keep the Lark parser outside Dynamo; the parsed tensor operations remain traceable.
_parse_einsum = torch.compiler.disable(parse_einsum)


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


class PytorchFramework(Framework):
    def __init__(self, sparse_layout: str = "COO"):
        self.sparse_layout = sparse_layout

    @property
    def linalg(self):
        return PytorchLinalg

    def from_binsparse(self, array):
        result = to_torch(array)
        if self.sparse_layout == "CSR" and result.layout == torch.sparse_coo:
            if result.ndim != 2:
                raise ValueError("PyTorch CSR only works for 2D matrices")
            return result.to_sparse_csr()
        return result

    def to_binsparse(self, array):
        return from_torch(array.detach().cpu())

    def lazy(self, array):
        return array

    def compute(self, array):
        return array

    def compile(self, func):
        return torch.compile(func)

    def einsum(self, prgm, **kwargs):
        return _parse_einsum(prgm).run(self, kwargs)

    def unfold(
        self,
        x,
        kernel_shape,
        *,
        axes=None,
        strides=None,
        dilations=None,
        padding=None,
        fill_value=0,
    ):
        x = x.to_dense() if _is_sparse_tensor(x) else x
        axes_t, strides_t, dilations_t, padding_t = normalize_unfold_args(
            x.ndim,
            kernel_shape,
            axes,
            strides,
            dilations,
            padding,
        )
        kernel_t = tuple(int(size) for size in kernel_shape)
        x = self._pad_for_unfold(x, axes_t, padding_t, fill_value)

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

    @staticmethod
    def _pad_for_unfold(x, axes, padding, fill_value):
        if not any(pair != (0, 0) for pair in padding):
            return x

        full_padding = [(0, 0)] * x.ndim
        for axis, pad_pair in zip(axes, padding, strict=True):
            full_padding[axis] = pad_pair

        pad = []
        for before, after in reversed(full_padding):
            pad.extend((before, after))

        if torch.is_tensor(fill_value):
            fill_value = fill_value.detach().cpu().item()
        return F.pad(x, tuple(pad), mode="constant", value=float(fill_value))

    def with_fill_value(self, array, value):
        return array

    def matmul(self, x1, x2, /, **kwargs):
        x1, x2 = _cast_bool_to_wide_unsigned(x1, x2)
        if x1.dtype in _WIDE_UNSIGNED or x2.dtype in _WIDE_UNSIGNED:
            out_dtype = torch.result_type(x1, x2)
            return torch_xp.matmul(x1.to(torch.int64), x2.to(torch.int64), **kwargs).to(
                out_dtype
            )
        return torch_xp.matmul(x1, x2, **kwargs)

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
