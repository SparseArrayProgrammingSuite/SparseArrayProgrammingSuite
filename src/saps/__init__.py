from .benchmark_runner import main as main
from .frameworks import einsum
import os
import numpy as np

xp = os.environ.get("SAPS_FRAMEWORK", "np")

__all__ = [
    "xp",
    "einsum",
    "main",
]