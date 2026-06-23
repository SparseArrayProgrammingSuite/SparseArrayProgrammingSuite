from __future__ import annotations

import importlib
import inspect
import pkgutil
from collections.abc import Iterator

import pytest

import saps
import saps.benchmarks
from frameworks.saps_numpy import NumpyFramework
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


def _test_params():
    for cls in _benchmark_classes():
        benchmark = cls()
        for param in benchmark.params:
            suites = set(param.generator.suites) | set(param.dataset.suites)
            if "test" in suites:
                yield pytest.param(cls, param, id=f"{cls.__name__}[{param}]")


@pytest.mark.parametrize(("benchmark_cls", "param"), list(_test_params()))
def test_benchmark_check(benchmark_cls: type[Benchmark], param):
    saps.xp = NumpyFramework()
    benchmark = benchmark_cls()
    try:
        benchmark.setup(param, use_cache=False)
        benchmark.run(param)
        benchmark.teardown(param)
    finally:
        saps.xp = None
