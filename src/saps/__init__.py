from .frameworks import einsum
import os
import numpy as np
from .benchmark import Author, Ref, Benchmark

xp = os.environ.get("SAPS_FRAMEWORK", "np")

__all__ = [
    "xp",
    "einsum",
    "Author",
    "Ref",
    "Benchmark",
]