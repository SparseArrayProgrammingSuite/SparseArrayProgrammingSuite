from collections import Counter

import pytest

from saps.benchmarks.cp_als import CPNFactorizeableGenerator, CPNFrosttGenerator
from saps.benchmarks.HOSVD import (
    HOSVDDenseGenerator,
    HOSVDFrosttGenerator,
    HOSVDSparseGenerator,
)
from saps.metadata import _benchmark_instances


@pytest.mark.parametrize(
    ("name", "generator_classes"),
    [
        ("cp_als", [CPNFactorizeableGenerator, CPNFrosttGenerator]),
        (
            "hosvd",
            [HOSVDDenseGenerator, HOSVDSparseGenerator, HOSVDFrosttGenerator],
        ),
    ],
)
def test_dimension_cases_partition_datasets(name, generator_classes):
    benchmarks = {
        benchmark.name: benchmark
        for benchmark in _benchmark_instances()
        if benchmark.name.startswith(name)
    }
    assert set(benchmarks) == {f"{name}_{n}d" for n in (3, 4, 5)}

    expected = Counter(
        (generator.name, dataset.name)
        for cls in generator_classes
        for generator in [cls()]
        for dataset in generator.datasets
    )
    actual = Counter()
    for n in (3, 4, 5):
        benchmark = benchmarks[f"{name}_{n}d"]
        assert {type(generator) for generator in benchmark.generators} == set(
            generator_classes
        )
        assert benchmark.params
        for param in benchmark.params:
            assert param.dataset.n == n
            actual[param.generator.name, param.dataset.name] += 1
        assert any("test" in param.dataset.suites for param in benchmark.params)
        assert any("trace" in param.dataset.suites for param in benchmark.params)
        for metric in ("time", "peakmem"):
            assert [
                method for method in dir(benchmark) if method.startswith(f"{metric}_")
            ] == [f"{metric}_{name}_{n}d"]

    assert actual == expected
