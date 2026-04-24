from sparseappbench.frameworks.numpy_framework import NumpyFramework

import sparseappbench.benchmarks.model_counting as mc


def test_model_counting_datasets():
    xp = NumpyFramework()
    mc.xp = xp

    generator = mc.MCGenerator()
    benchmark = mc.ModelCounting()

    for dataset in generator.datasets:
        raw_matrices, meta = generator.generate(dataset)

        input_arrays = [xp.from_binsparse(m) for m in raw_matrices]

        results = benchmark.benchmark(input_arrays, meta)

        res = int(results[0])
        expected = meta["expected_result"]

        msg = f"Test '{dataset.name}' failed: Expected {expected}, got {res}"

        assert res == expected, msg
