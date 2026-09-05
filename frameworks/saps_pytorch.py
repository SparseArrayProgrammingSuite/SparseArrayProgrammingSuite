import array_api_compat
import array_api_compat.torch as torch_xp
import torch
import torch._dynamo
import torch.nn.functional as F
from binsparse.conversions import from_torch, to_torch

from saps_framework import Framework, einsum, normalize_unfold_args

torch._dynamo.config.suppress_errors = True


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
        xp = array_api_compat.array_namespace(*kwargs.values(), use_compat=True)
        if not hasattr(xp, "power"):
            xp.power = torch.pow
        return einsum(xp, prgm, **kwargs)

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

    def maximum(self, x, y):
        if not torch.is_tensor(y):
            y = torch.as_tensor(y, dtype=x.dtype, device=x.device)
        return torch.maximum(x, y)

    def minimum(self, x, y):
        if not torch.is_tensor(y):
            y = torch.as_tensor(y, dtype=x.dtype, device=x.device)
        return torch.minimum(x, y)

    def __getattr__(self, name):
        if hasattr(torch_xp, name):
            return getattr(torch_xp, name)
        return getattr(torch, name)


xp = PytorchFramework()
