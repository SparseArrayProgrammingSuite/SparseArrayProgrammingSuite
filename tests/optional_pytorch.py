"""PyTorch for tests that cover it alongside other frameworks.

Only pixi's pytorch environment installs PyTorch (`pixi run test-pytorch`); in
the default environment its cases skip, and `PytorchFramework` is a stand-in
that no array is an instance of.
"""

import pytest

try:
    import torch

    from frameworks.saps_pytorch import PytorchFramework
except ImportError:
    torch = None

    class PytorchFramework:  # type: ignore[no-redef]
        """Stand-in for frameworks.saps_pytorch.PytorchFramework."""


requires_torch = pytest.mark.skipif(
    torch is None, reason="PyTorch runs in pixi's pytorch environment"
)


def pytorch_param(*values, id=None):
    """A parametrize case that needs PyTorch."""
    return pytest.param(*values, id=id, marks=requires_torch)
