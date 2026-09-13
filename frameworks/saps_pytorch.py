import array_api_compat.torch as torch_xp
import torch
import torch._dynamo
from binsparse.conversions import from_torch, to_torch

from saps_framework import Framework
from saps_framework.einsum import parse_einsum

torch._dynamo.config.suppress_errors = True
torch_xp.power = torch.pow  # type: ignore[attr-defined]
# Keep the Lark parser outside Dynamo; the parsed tensor operations remain traceable.
_parse_einsum = torch.compiler.disable(parse_einsum)


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
        return _parse_einsum(prgm).run(torch_xp, kwargs)

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
