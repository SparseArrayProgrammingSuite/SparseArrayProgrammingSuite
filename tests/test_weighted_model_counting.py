import numpy as np

from frameworks.saps_numpy import NumpyFramework

import saps.benchmarks.weighted_model_counting as wmc


def test_weighted_model_counting_datasets():
    xp = NumpyFramework()
    wmc.xp = xp

    generator = wmc.WMCGenerator()
    benchmark = wmc.WeightedModelCounting()

    for dataset in generator.datasets:
        raw_matrices, meta = generator.generate(dataset)

        input_arrays = [xp.from_binsparse(m) for m in raw_matrices]

        results = benchmark.benchmark(input_arrays, meta)

        res = float(results[0])
        expected = meta["expected_result"]

        msg = f"Test '{dataset.name}' failed: Expected {expected}, got {res}"
        assert np.isclose(res, expected, rtol=10e-8), msg
