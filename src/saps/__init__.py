from .frameworks import einsum
import os
import numpy as np
from .benchmark import Author, Ref, Benchmark, Contributor

xp = os.environ.get("SAPS_FRAMEWORK", "np")

__all__ = [
    "xp",
    "einsum",
    "Author",
    "Contributor",
    "Ref",
    "Benchmark",
]