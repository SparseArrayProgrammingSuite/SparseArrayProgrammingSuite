"""Scorch (https://github.com/bobbyyyan/scorch) framework wrapper.

Scorch is a sparse compiler for PyTorch. Sparse ``matmul`` calls and einsum
programs that are a sum of products of tensor accesses are compiled by Scorch;
everything else, including any operation Scorch rejects (for example, its
kernel compiler only supports float32), runs through the PyTorch wrapper.
Rejected operation signatures are remembered so they are not retried.
"""

import string

import scorch
import torch
from saps_pytorch import PytorchFramework, _parse_einsum, torch_xp

from saps_framework.einsum import Access, Call

PRODUCT_OPS = {"*", "mul", "multiply"}
SUM_OPS = {None, "+", "add", "sum"}
SCORCH_ERRORS = (
    AssertionError,
    IndexError,
    KeyError,
    NotImplementedError,
    RuntimeError,
    TypeError,
    ValueError,
)


@torch.compiler.disable
def scorch_call(op, *args, **kwargs):
    """Run a Scorch op eagerly and return its result as a strided or COO tensor."""
    args = tuple(
        arg.to_sparse_csr()
        if torch.is_tensor(arg) and arg.layout == torch.sparse_coo and arg.ndim == 2
        else arg
        for arg in args
    )
    result = op(*args, **kwargs)
    if torch.is_tensor(result):
        return result if result.layout == torch.strided else result.to_sparse_coo()
    if result.format.is_dense():
        return result.to_torch()
    if str(result.format) == "d,s" and result.index.mode_order == [0, 1]:
        crow, col = result.index.mode_indices[1]
        nnz = int(crow[-1])
        return torch.sparse_csr_tensor(
            crow.long(), col[:nnz].long(), result.values[:nnz], result.shape
        ).to_sparse_coo()
    return result.to_torch().to_sparse_coo()


class ScorchFramework(PytorchFramework):
    def __init__(self):
        super().__init__(sparse_layout="COO")
        self.unsupported = set()

    def einsum(self, prgm, **kwargs):
        einsum = _parse_einsum(prgm)

        factors = []
        pending = [einsum.arg]
        while pending and factors is not None:
            expr = pending.pop()
            if isinstance(expr, Call) and expr.func in PRODUCT_OPS:
                pending.extend(reversed(expr.args))
            elif isinstance(expr, Access) and len(set(expr.idxs)) == len(expr.idxs):
                factors.append(expr)
            else:
                factors = None
        tensors = [kwargs[factor.tns] for factor in factors or []]
        idxs = dict.fromkeys(idx for factor in factors or [] for idx in factor.idxs)

        if (
            einsum.op in SUM_OPS
            and factors
            and all(torch.is_tensor(tensor) for tensor in tensors)
            and any(tensor.layout != torch.strided for tensor in tensors)
            and len(idxs) <= len(string.ascii_letters)
            and idxs.keys() >= set(einsum.idxs)
            # Scorch crashes or returns wrong results for sparse operands that
            # are not 2-D, outputs above 2-D, and broadcast (size-1) indices.
            and all(t.ndim == 2 for t in tensors if t.layout != torch.strided)
            and len(einsum.idxs) <= 2
            and all(
                t.ndim == len(f.idxs) for f, t in zip(factors, tensors, strict=True)
            )
            and len(idxs)
            == len(
                {
                    (idx, t.shape[axis])
                    for f, t in zip(factors, tensors, strict=True)
                    for axis, idx in enumerate(f.idxs)
                }
            )
        ):
            letters = dict(zip(idxs, string.ascii_letters, strict=False))
            inputs = ",".join(
                "".join(letters[idx] for idx in factor.idxs) for factor in factors
            )
            output = "".join(letters[idx] for idx in einsum.idxs)
            sparse_idxs = {
                idx
                for factor, tensor in zip(factors, tensors, strict=True)
                if tensor.layout != torch.strided
                for idx in factor.idxs
            }
            if len(output) == 2 and sparse_idxs.issuperset(einsum.idxs):
                output_format = "ds"
            else:
                output_format = "d" * len(output)

            expression = f"{inputs}->{output}"
            key = (
                "einsum",
                expression,
                output_format,
                tuple((t.layout, t.dtype, t.ndim) for t in tensors),
            )
            # Letters follow first appearance, so sorted index strings mean no
            # operand or output is transposed; Scorch gets transposes wrong.
            transposed = any(
                list(part) != sorted(part) for part in [*inputs.split(","), output]
            )
            if not transposed and key not in self.unsupported:
                try:
                    return scorch_call(
                        scorch.einsum, expression, *tensors, format=output_format
                    )
                except SCORCH_ERRORS:
                    self.unsupported.add(key)

        return einsum.run(torch_xp, kwargs)

    def matmul(self, x1, x2):
        if (
            torch.is_tensor(x1)
            and torch.is_tensor(x2)
            and (x1.layout != torch.strided or x2.layout != torch.strided)
            and x1.ndim == 2
            and x2.ndim in (1, 2)
        ):
            key = ("matmul", tuple((t.layout, t.dtype, t.ndim) for t in (x1, x2)))
            if key not in self.unsupported:
                try:
                    return scorch_call(scorch.matmul, x1, x2)
                except SCORCH_ERRORS:
                    self.unsupported.add(key)

        return torch_xp.matmul(x1, x2)


xp = ScorchFramework()
