"""Build benchmark functions from source at run time.

Generators whose datasets each need a different benchmark signature override
``Generator.generate_benchmark_function`` and build that dataset's function with
these helpers. Source registered through ``define_function`` stays visible to
``inspect.getsource``, which source-level compilers such as ``finch_fused.jit``
rely on.
"""

import builtins
import linecache
from collections.abc import Callable, Sequence
from typing import Any


def define_function(
    source: str, filename: str, namespace: dict[str, Any] | None = None
) -> Callable[..., Any]:
    """Compile ``source`` (a single ``def``) under ``filename`` and return it.

    ``filename`` should be unique per generated function, e.g.
    ``"<saps-generated generator.dataset>"``; it appears in tracebacks and is
    registered in ``linecache`` so ``inspect.getsource`` can find the source.
    ``namespace`` supplies the function's globals.
    """
    code = compile(source, filename, "exec")
    lines = source.splitlines(keepends=True)
    linecache.cache[filename] = (len(source), None, lines, filename)
    scope: dict[str, Any] = {"__builtins__": builtins, **(namespace or {})}
    local: dict[str, Any] = {}
    exec(code, scope, local)  # noqa: S102
    functions = [value for value in local.values() if callable(value)]
    if len(functions) != 1:
        raise ValueError(f"{filename}: expected one function, got {sorted(local)}")
    return functions[0]


def _signature(params: Sequence[str]) -> str:
    for name in params:
        if not name.isidentifier():
            raise ValueError(f"Invalid parameter name {name!r}")
    return f"def benchmark({', '.join(['xp', 'meta', *params])}):\n"


def einsum_function_source(expr: str, params: Sequence[str]) -> str:
    """Source for ``benchmark(xp, meta, *params)`` evaluating one einsum."""
    kwargs = ", ".join(f"{name}={name}" for name in params)
    return _signature(params) + f"    return xp.einsum({expr!r}, {kwargs})\n"


def constant_function_source(value: Any, dtype: str, params: Sequence[str]) -> str:
    """Source for ``benchmark(xp, meta, *params)`` returning a constant array.

    The function refers to ``np``, so define it with ``{"np": numpy}``.
    """
    return _signature(params) + f"    return xp.array({value!r}, dtype=np.{dtype})\n"
