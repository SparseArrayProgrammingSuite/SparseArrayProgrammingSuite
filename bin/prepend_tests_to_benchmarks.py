#!/usr/bin/env python3
"""Prepend benchmark test files to their corresponding benchmark modules."""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = ROOT / "tests"
BENCHMARKS_DIR = ROOT / "src" / "saps" / "benchmarks"
BENCHMARK_IMPORT_PREFIX = "saps.benchmarks."
BEGIN_PREFIX = "# BEGIN COPIED TEST FILE:"
END_PREFIX = "# END COPIED TEST FILE:"


@dataclass(frozen=True)
class CopyPlan:
    test_path: Path
    benchmark_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy each benchmark-specific tests/test_*.py file into the top of "
            "the corresponding src/saps/benchmarks/*.py file."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the files that would be updated without changing them.",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help=(
            "Paste test files as executable Python instead of a commented context "
            "block. Commented blocks are the default to avoid import side effects."
        ),
    )
    parser.add_argument(
        "--tests-dir",
        type=Path,
        default=TESTS_DIR,
        help=f"Directory containing test files. Defaults to {TESTS_DIR}.",
    )
    parser.add_argument(
        "--benchmarks-dir",
        type=Path,
        default=BENCHMARKS_DIR,
        help=f"Directory containing benchmark modules. Defaults to {BENCHMARKS_DIR}.",
    )
    return parser.parse_args()


def benchmark_modules_from_imports(test_path: Path) -> set[str]:
    tree = ast.parse(test_path.read_text(), filename=str(test_path))
    modules: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(BENCHMARK_IMPORT_PREFIX):
                    module = alias.name.removeprefix(BENCHMARK_IMPORT_PREFIX)
                    if module:
                        modules.add(module.split(".", 1)[0])
                elif alias.name == "saps.benchmarks":
                    continue
        elif (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.startswith(BENCHMARK_IMPORT_PREFIX)
        ):
            module = node.module.removeprefix(BENCHMARK_IMPORT_PREFIX)
            if module:
                modules.add(module.split(".", 1)[0])

    return modules


def choose_benchmark_module(test_path: Path, modules: set[str]) -> str | None:
    if not modules:
        return None

    test_stem = test_path.stem.removeprefix("test_").lower()
    if len(modules) == 1:
        return next(iter(modules))

    for module in sorted(modules):
        module_key = module.lower().removesuffix("_benchmark")
        if module_key == test_stem:
            return module

    return None


def discover_copy_plan(tests_dir: Path, benchmarks_dir: Path) -> list[CopyPlan]:
    plans: list[CopyPlan] = []
    skipped: list[tuple[Path, str]] = []

    for test_path in sorted(tests_dir.glob("test_*.py")):
        modules = benchmark_modules_from_imports(test_path)
        module = choose_benchmark_module(test_path, modules)
        if module is None:
            skipped.append((test_path, "no unique benchmark import"))
            continue

        benchmark_path = benchmarks_dir / f"{module}.py"
        if not benchmark_path.exists():
            skipped.append((test_path, f"missing benchmark file for {module!r}"))
            continue

        plans.append(CopyPlan(test_path=test_path, benchmark_path=benchmark_path))

    if skipped:
        print("Skipped:")
        for test_path, reason in skipped:
            print(f"  {test_path.relative_to(ROOT)} ({reason})")
        print()

    return plans


def comment_test_content(test_path: Path) -> str:
    body = test_path.read_text().rstrip()
    commented_lines = [f"# {line}" if line else "#" for line in body.splitlines()]
    rel_path = test_path.relative_to(ROOT)
    return "\n".join(
        [
            f"{BEGIN_PREFIX} {rel_path}",
            *commented_lines,
            f"{END_PREFIX} {rel_path}",
            "",
            "",
        ]
    )


def raw_test_content(test_path: Path) -> str:
    rel_path = test_path.relative_to(ROOT)
    body = test_path.read_text().rstrip()
    return "\n".join(
        [
            f"# {BEGIN_PREFIX.removeprefix('# ')} {rel_path}",
            body,
            f"# {END_PREFIX.removeprefix('# ')} {rel_path}",
            "",
            "",
        ]
    )


def strip_existing_copied_test(content: str) -> str:
    lines = content.splitlines(keepends=True)
    if not lines:
        return content

    first_line = lines[0].rstrip("\n")
    if not first_line.startswith(BEGIN_PREFIX):
        return content

    for index, line in enumerate(lines):
        if line.rstrip("\n").startswith(END_PREFIX):
            remainder = "".join(lines[index + 1 :])
            return remainder.lstrip("\n")

    raise ValueError("found copied-test start marker without end marker")


def prepend_test(plan: CopyPlan, *, dry_run: bool, raw: bool) -> bool:
    benchmark_content = plan.benchmark_path.read_text()
    cleaned_content = strip_existing_copied_test(benchmark_content)
    if raw:
        copied_content = raw_test_content(plan.test_path)
    else:
        copied_content = comment_test_content(plan.test_path)
    new_content = copied_content + cleaned_content

    rel_test = plan.test_path.relative_to(ROOT)
    rel_benchmark = plan.benchmark_path.relative_to(ROOT)
    if new_content == benchmark_content:
        print(f"unchanged {rel_benchmark} <= {rel_test}")
        return False

    action = "would update" if dry_run else "updated"
    print(f"{action} {rel_benchmark} <= {rel_test}")
    if not dry_run:
        plan.benchmark_path.write_text(new_content)
    return True


def main() -> int:
    args = parse_args()
    plans = discover_copy_plan(args.tests_dir, args.benchmarks_dir)
    if not plans:
        print("No benchmark test files found.")
        return 1

    changed = 0
    for plan in plans:
        try:
            changed += prepend_test(plan, dry_run=args.dry_run, raw=args.raw)
        except ValueError as error:
            rel_path = plan.benchmark_path.relative_to(ROOT)
            print(f"error: {rel_path}: {error}", file=sys.stderr)
            return 1

    print()
    action = "Would update" if args.dry_run else "Updated"
    print(f"{action} {changed} benchmark file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
