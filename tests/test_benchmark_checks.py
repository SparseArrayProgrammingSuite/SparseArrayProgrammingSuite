from __future__ import annotations

import ast
import importlib
import inspect
import pkgutil
from collections.abc import Iterator
from pathlib import Path

import pytest

import saps
import saps.benchmarks
from frameworks.saps_numpy import NumpyFramework
from frameworks.saps_sparse import PyDataSparseFramework
from saps.benchmark import Benchmark


def _benchmark_classes() -> Iterator[type[Benchmark]]:
    for module_info in pkgutil.iter_modules(saps.benchmarks.__path__):
        module = importlib.import_module(f"saps.benchmarks.{module_info.name}")
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__module__ != module.__name__:
                continue
            if (
                issubclass(cls, Benchmark)
                and cls is not Benchmark
                and not inspect.isabstract(cls)
                and cls.check is not Benchmark.check
            ):
                yield cls


def _framework_params():
    return [
        pytest.param(NumpyFramework, id="numpy"),
        pytest.param(PyDataSparseFramework, id="sparse"),
    ]


def _test_params():
    for cls in _benchmark_classes():
        benchmark = cls()
        for param in benchmark.params:
            suites = set(param.generator.suites) | set(param.dataset.suites)
            if "test" in suites:
                yield pytest.param(cls, param, id=f"{cls.__name__}[{param}]")


def test_saps_does_not_export_global_xp():
    assert not hasattr(saps, "xp")


def test_benchmark_methods_accept_explicit_xp():
    for cls in _benchmark_classes():
        signature = inspect.signature(cls.benchmark)
        assert list(signature.parameters)[:4] == ["self", "xp", "data", "meta"]


def test_benchmark_modules_do_not_define_global_xp():
    root = Path(__file__).parents[1]
    violations = []
    for path in (root / "src" / "saps" / "benchmarks").glob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "xp":
                        violations.append(f"{path.relative_to(root)}:{node.lineno}")

    assert not violations, "benchmark module global xp definitions found: " + ", ".join(
        violations
    )


@pytest.mark.parametrize(("benchmark_cls", "param"), list(_test_params()))
@pytest.mark.parametrize("framework_cls", _framework_params())
def test_benchmark_check(benchmark_cls: type[Benchmark], param, framework_cls):
    xp = framework_cls()
    benchmark = benchmark_cls()
    benchmark.setup(param, use_cache=False, xp=xp)
    benchmark.run(param)
    benchmark.teardown(param)
