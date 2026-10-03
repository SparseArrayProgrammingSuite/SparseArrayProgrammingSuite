import pytest

import numpy as np

from binsparse.conversions import to_numpy

from frameworks.saps_numpy import NumpyFramework
from frameworks.saps_sparse import PyDataSparseFramework
from saps.benchmarks.MSBFS import (
    MultiSourceBreadthFirstSearchBenchmark,
    MultiSourceBreadthFirstSearchTestGenerator,
    reference_levels,
)

_DATASETS = MultiSourceBreadthFirstSearchTestGenerator().datasets


@pytest.mark.parametrize("dataset", _DATASETS, ids=lambda d: d.name)
def test_msbfs_test_expectations_match_scipy_reference(dataset):
    problem = MultiSourceBreadthFirstSearchTestGenerator().generate(dataset)
    sources = problem.meta["sources"]
    expected = to_numpy(problem.ref_outputs[0])
    assert expected.shape == (len(sources), dataset.A.shape[0])
    np.testing.assert_array_equal(reference_levels(dataset.A != 0, sources), expected)


@pytest.mark.parametrize("framework", [NumpyFramework, PyDataSparseFramework])
@pytest.mark.parametrize("dataset", _DATASETS, ids=lambda d: d.name)
def test_msbfs_test_problems(framework, dataset):
    problem = MultiSourceBreadthFirstSearchTestGenerator().generate(dataset)
    xp = framework()
    output = MultiSourceBreadthFirstSearchBenchmark().benchmark(
        xp, [xp.from_binsparse(x) for x in problem.inputs], problem.meta
    )[0]
    np.testing.assert_array_equal(
        NumpyFramework().from_binsparse(xp.to_binsparse(output)),
        to_numpy(problem.ref_outputs[0]),
    )
