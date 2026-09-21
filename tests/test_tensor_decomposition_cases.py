from collections import Counter

import pytest

import numpy as np

from binsparse.conversions import from_numpy, to_numpy

from frameworks.saps_numpy import NumpyFramework
from saps.benchmarks import HOSVD, cp_als
from saps.benchmarks.cp_als import CPNFactorizeableGenerator, CPNFrosttGenerator
from saps.benchmarks.HOSVD import (
    HOSVDDenseGenerator,
    HOSVDFrosttGenerator,
    HOSVDSparseGenerator,
)
from saps.metadata import _benchmark_instances

_DECOMPOSITION_CLASSES = [
    cp_als.CP_ALS_3D,
    cp_als.CP_ALS_4D,
    cp_als.CP_ALS_5D,
    HOSVD.HOSVD3DBenchmark,
    HOSVD.HOSVD4DBenchmark,
    HOSVD.HOSVD5DBenchmark,
]


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


@pytest.mark.parametrize("benchmark_cls", _DECOMPOSITION_CLASSES)
@pytest.mark.parametrize("corruption", ["values", "shape", "format"])
def test_decomposition_check_rejects_invalid_output(benchmark_cls, corruption):
    benchmark = benchmark_cls()
    param = next(param for param in benchmark.params if "test" in param.dataset.suites)
    benchmark.setup(param, use_cache=False, xp=NumpyFramework())
    benchmark.run(param)
    benchmark.check(param)

    first_output = to_numpy(benchmark._output[0])
    if corruption == "values":
        benchmark._output[0] = from_numpy(np.zeros_like(first_output))
    elif corruption == "shape":
        benchmark._output[0] = from_numpy(first_output[..., :-1])
    else:
        benchmark._output[0] = first_output

    with pytest.raises(AssertionError):
        benchmark.check(param)


@pytest.mark.parametrize("benchmark_cls", _DECOMPOSITION_CLASSES)
@pytest.mark.parametrize("ref_meta", [None, {}, {"check_reconstruction": False}])
def test_decomposition_check_skips_dense_reconstruction(
    benchmark_cls, ref_meta, monkeypatch
):
    benchmark = benchmark_cls()
    benchmark._ref_meta = ref_meta
    benchmark._output = [from_numpy(np.ones(1)) for _ in range(benchmark.n + 1)]

    def fail_to_numpy(_):
        pytest.fail("reconstruction-disabled checks must not densify tensors")

    monkeypatch.setattr(f"{benchmark_cls.__module__}.to_numpy", fail_to_numpy)
    benchmark.check(None)
