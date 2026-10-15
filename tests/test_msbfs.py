import pytest

import numpy as np
from scipy.sparse.csgraph import shortest_path

from binsparse.conversions import to_numpy

from frameworks.saps_numpy import NumpyFramework
from frameworks.saps_sparse import PyDataSparseFramework
from saps.benchmarks.msbfs import (
    MSBFSBenchmark,
    MSBFSTestGenerator,
)

_DATASETS = MSBFSTestGenerator().datasets


@pytest.mark.parametrize("dataset", _DATASETS, ids=lambda d: d.name)
def test_msbfs_test_expectations_match_scipy_reference(dataset):
    problem = MSBFSTestGenerator().generate(dataset)
    sources = problem.meta["sources"]
    expected = to_numpy(problem.ref_outputs[0])
    assert expected.shape == (len(sources), dataset.A.shape[0])
    distances = shortest_path(
        dataset.A != 0, directed=True, unweighted=True, indices=sources
    )
    levels = np.where(np.isfinite(distances), distances + 1, 0).astype(int)
    np.testing.assert_array_equal(levels, expected)


@pytest.mark.parametrize("framework", [NumpyFramework, PyDataSparseFramework])
@pytest.mark.parametrize("dataset", _DATASETS, ids=lambda d: d.name)
def test_msbfs_test_problems(framework, dataset):
    problem = MSBFSTestGenerator().generate(dataset)
    xp = framework()
    output = MSBFSBenchmark().benchmark(
        xp, problem.meta, *(xp.from_binsparse(x) for x in problem.inputs)
    )
    np.testing.assert_array_equal(
        NumpyFramework().from_binsparse(xp.to_binsparse(output)),
        to_numpy(problem.ref_outputs[0]),
    )
