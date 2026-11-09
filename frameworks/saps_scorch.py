"""SAPS adapter for Scorch (https://github.com/bobbyyyan/scorch).

Scorch is used as a drop-in PyTorch namespace (``import scorch as torch``):
arrays stay ``torch.Tensor`` objects (dense, COO or CSR, as binsparse provides
them), ``scorch.matmul`` and ``scorch.einsum`` run Scorch's sparse kernels, and
every other name falls through to PyTorch through Scorch's module
``__getattr__``. Scorch returns its own ``STensor`` from sparse kernels; the
only conversion it provides back to PyTorch is ``STensor.to_torch()``, which
yields a dense tensor.

Install from an editable source checkout, as ``pixi run -e scorch
install-scorch`` does (scripts/install_editable_git.py explains why; PyPI's
``scorch`` is an unrelated package). It builds a C++ extension against the
installed torch and JIT-compiles kernels with a C++ compiler and OpenMP at run
time. Scorch has no GPU backend.
"""

import array_api_compat.torch as torch_xp
import scorch
import torch
import torch.nn.functional as F
from binsparse.conversions import from_torch, to_torch

from saps_framework import Framework, normalize_unfold_args
from saps_framework.einsum import Access, parse_einsum

torch_xp.power = torch.pow  # type: ignore[attr-defined]

# Value dtypes Scorch's kernel generator supports (scorch/utils.py).
_SCORCH_DTYPES = frozenset(
    {torch.float32, torch.float64, torch.int32, torch.int64, torch.int8, torch.uint8}
)

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


# Scorch's code generator is built around float32: it has no float64 pointer
# type or int64 workspaces, so float64 results can't leave Scorch and int64
# kernels don't compile. Scorch's generated sparse kernels therefore run in
# float32, and results are cast back to the dtype the Array API expects.
_KERNEL_DTYPE = torch.float32

# Value dtypes of Scorch's prebuilt CSR matmul kernels (CSR times a dense
# vector, a dense matrix or a CSR matrix; scorch/prebuilt_kernels.py), which
# don't go through the code generator.
_PREBUILT_CSR_DTYPES = frozenset(
    {torch.float32, torch.float64, torch.int32, torch.int64}
)


def _to_torch(result):
    """Scorch kernels may return an STensor; continue with dense PyTorch tensors.

    STensor.to_torch() densifies with a generated kernel, which can't handle
    float64, so PyTorch densifies the CSR results of the prebuilt kernels.
    """
    if not isinstance(result, scorch.STensor):
        return result
    if str(result.format) == "d,s" and result.storage.index.mode_order == [0, 1]:
        crow_indices, col_indices = result.index.mode_indices[1]
        return torch.sparse_csr_tensor(
            crow_indices, col_indices, result.values, result.shape
        ).to_dense()
    return result.to_torch()


class ScorchFramework(Framework):
    def from_binsparse(self, array):
        return to_torch(array)

    def to_binsparse(self, array):
        return from_torch(_to_torch(array).detach().cpu())

    def lazy(self, array):
        return array

    def compute(self, array):
        return array

    def einsum(self, prgm, **kwargs):
        parsed = parse_einsum(prgm)
        contraction = parsed.native_contraction()
        if contraction is not NotImplemented:
            result = self._native_einsum(parsed.op, contraction, kwargs)
            if result is not NotImplemented:
                return result
        return parsed.run(self, kwargs)

    @staticmethod
    def _native_einsum(reduction, contraction, kwargs):
        """Run a sum-of-products contraction with scorch.einsum.

        scorch.einsum multiplies all operands and sums the dropped indices. It
        needs a nonempty output, operands without repeated indices, one shared
        value dtype from _SCORCH_DTYPES, and COO, CSR or dense layouts.
        """
        equation, leaves, bitwise = contraction
        if not all(isinstance(leaf, Access) for leaf in leaves):
            return NotImplemented
        inputs, output = equation.split("->")
        terms = inputs.split(",")
        if not output or any(not t or len(set(t)) != len(t) for t in terms):
            return NotImplemented
        operands = [kwargs[leaf.tns] for leaf in leaves]
        if any(op.layout == torch.sparse_csc for op in operands):
            return NotImplemented
        sizes: dict[str, int] = {}
        for operand, term in zip(operands, terms, strict=True):
            for idx, size in zip(term, operand.shape, strict=True):
                if sizes.setdefault(idx, size) != size:
                    return NotImplemented
        boolean = all(operand.dtype == torch.bool for operand in operands)
        logical_reduction = reduction in {"|", "or"}
        if (bitwise or logical_reduction) and not boolean:
            return NotImplemented
        dtype = operands[0].dtype
        for operand in operands[1:]:
            if dtype in _WIDE_UNSIGNED or operand.dtype in _WIDE_UNSIGNED:
                return NotImplemented
            dtype = torch.promote_types(dtype, operand.dtype)
        if not dtype.is_floating_point and not dtype.is_complex:
            if not boolean and len(operands) > 1 and dtype != torch.int64:
                # Preserve overflow in narrow integer products before summing.
                return NotImplemented
            # Summing booleans counts matches, as the Array API sum does; a
            # nonzero count implements an OR reduction.
            dtype = torch.int64
        if dtype not in _SCORCH_DTYPES:
            return NotImplemented
        if all(op.layout == torch.strided for op in operands):
            kernel_dtype = dtype
        else:
            kernel_dtype = _KERNEL_DTYPE
        result = _to_torch(
            scorch.einsum(equation, *(operand.to(kernel_dtype) for operand in operands))
        ).to(dtype)
        return result != 0 if logical_reduction else result

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

    def matmul(self, x1, x2, /):
        if not (_is_sparse_tensor(x1) or _is_sparse_tensor(x2)):
            return _to_torch(scorch.matmul(x1, x2))
        dtype = torch.promote_types(x1.dtype, x2.dtype)
        if (
            dtype in _PREBUILT_CSR_DTYPES
            and x1.layout in {torch.sparse_coo, torch.sparse_csr}
            and x1.ndim == 2
            and (
                (x2.layout == torch.strided and x2.ndim in {1, 2})
                or (x2.layout in {torch.sparse_coo, torch.sparse_csr} and x2.ndim == 2)
            )
        ):
            # Scorch's prebuilt kernels take CSR, so they run in dtype.
            x1 = x1.to_sparse_csr().to(dtype)
            x2 = (
                x2.to(dtype)
                if x2.layout == torch.strided
                else x2.to_sparse_csr().to(dtype)
            )
            return _to_torch(scorch.matmul(x1, x2))
        result = scorch.matmul(x1.to(_KERNEL_DTYPE), x2.to(_KERNEL_DTYPE))
        return _to_torch(result).to(dtype)

    def maximum(self, x, y):
        if not torch.is_tensor(y):
            y = torch.as_tensor(y, dtype=x.dtype, device=x.device)
        return torch.maximum(x, y)

    def minimum(self, x, y):
        if not torch.is_tensor(y):
            y = torch.as_tensor(y, dtype=x.dtype, device=x.device)
        return torch.minimum(x, y)

    def __getattr__(self, name):
        # Array API spellings first, then Scorch's namespace, which falls
        # through to torch for everything Scorch doesn't define.
        attr = (
            getattr(torch_xp, name)
            if hasattr(torch_xp, name)
            else getattr(scorch, name)
        )
        if name in _BOOL_PROMOTED_OPS:

            def op(x1, x2, /, **kwargs):
                return attr(*_cast_bool_to_wide_unsigned(x1, x2), **kwargs)

            return op
        return attr


xp = ScorchFramework()
