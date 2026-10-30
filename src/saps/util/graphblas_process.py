"""Run SuiteSparse:GraphBLAS work where its OpenMP runtime can start.

On macOS, torch and scikit-learn bundle their own LLVM OpenMP runtime, and a
second copy, such as the one GraphBLAS bundles, aborts the process when it
initializes. Linux and Windows wheels use GNU OpenMP, which tolerates several
copies, so there GraphBLAS always runs in the calling process.
"""

from __future__ import annotations

import multiprocessing
import sys
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from typing import Any

_LLVM_OPENMP_MODULES = ("torch", "sklearn")


def call_graphblas(function: Callable[..., Any], *args: Any) -> Any:
    """Call `function(*args)`, in a fresh process if GraphBLAS cannot start here.

    `function` must be importable at module level so a child process can find it.
    """
    if sys.platform == "darwin" and any(
        name in sys.modules for name in _LLVM_OPENMP_MODULES
    ):
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=1, mp_context=context) as pool:
            return pool.submit(function, *args).result()
    return function(*args)
