"""Rendering helpers for generated benchmark modules.

The helpers emit code in the layout ``ruff format`` would produce, so a
generated module is stable under the formatter and ``--check`` can compare a
fresh render against the committed file byte for byte.
"""

import difflib
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

LINE_LENGTH = 88
_BENCHMARK_ARGS = ("self", "xp", "meta")


def fits(line: str) -> bool:
    return len(line) <= LINE_LENGTH


def render_string(
    text: str, indent: str, suffix: str = "", delimiter: str = " "
) -> list[str]:
    """Render a string literal, split into implicit concatenation if too long.

    Splits only after occurrences of ``delimiter``, which stays at the end of
    each piece.
    """
    if fits(f'{indent}"{text}"{suffix}'):
        return [f'{indent}"{text}"{suffix}']
    terms = text.split(delimiter)
    pieces = [term + delimiter for term in terms[:-1]] + [terms[-1]]
    lines, current = [], ""
    for piece in pieces:
        if current and not fits(f'{indent}"{current}{piece}"'):
            lines.append(f'{indent}"{current}"')
            current = ""
        current += piece
    lines.append(f'{indent}"{current}"{suffix}')
    return lines


def render_signature(params: Sequence[str]) -> list[str]:
    """Render ``def benchmark(self, xp, meta, *params):`` inside a class body."""
    args = [*_BENCHMARK_ARGS, *params]
    one_line = f"    def benchmark({', '.join(args)}):"
    if fits(one_line):
        return [one_line]
    hugged = f"        {', '.join(args)}"
    if fits(hugged):
        return ["    def benchmark(", hugged, "    ):"]
    return ["    def benchmark(", *(f"        {arg}," for arg in args), "    ):"]


def render_einsum_return(
    expr: str, params: Sequence[str], delimiter: str = " "
) -> list[str]:
    """Render ``return xp.einsum(expr, p=p, ...)`` inside a method body."""
    kwargs = [f"{name}={name}" for name in params]
    one_line = f'        return xp.einsum("{expr}", {", ".join(kwargs)})'
    if fits(one_line):
        return [one_line]
    hugged = f'            "{expr}", {", ".join(kwargs)}'
    if fits(hugged):
        return ["        return xp.einsum(", hugged, "        )"]
    return [
        "        return xp.einsum(",
        *render_string(expr, "            ", ",", delimiter),
        *(f"            {kwarg}," for kwarg in kwargs),
        "        )",
    ]


def require_unique(names: Iterable[str], what: str) -> None:
    names = list(names)
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate generated {what}: {duplicates}")


def write_or_check(path: Path, rendered: str, check: bool) -> int:
    """Write ``rendered`` to ``path``, or with ``check`` diff it and return 1."""
    if check:
        current = path.read_text() if path.exists() else ""
        if current == rendered:
            print(f"{path} is up to date")
            return 0
        sys.stdout.writelines(
            difflib.unified_diff(
                current.splitlines(keepends=True),
                rendered.splitlines(keepends=True),
                str(path),
                "rendered",
            )
        )
        return 1
    path.write_text(rendered)
    print(f"wrote {path}")
    return 0
