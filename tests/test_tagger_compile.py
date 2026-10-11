import copy
import json
from pathlib import Path

import numpy as np

from binsparse.conversions import to_numpy

from frameworks.saps_numpy import NumpyFramework
from saps.benchmarks.matrix_vector_multiplication import (
    MatrixVectorMultiplicationBenchmark,
)
from saps.framework import load_framework


def tagger_framework():
    path = Path(__file__).resolve().parents[1] / "frameworks/saps_tagger.py"
    return type(load_framework(path))(wrapped=NumpyFramework())


def test_compiled_tagger_resets_stats_and_preserves_input_tags():
    xp = tagger_framework()

    def identity(array):
        return array

    compiled = xp.compile(identity)
    array = xp.asarray(np.zeros((2, 2, 2, 2, 2)))
    xp.sin(array)  # An earlier operation must not contaminate this invocation.
    assert compiled(array) is array
    assert "feature-high-dimensional" in xp.tags
    assert "feature-transcendental-ops" not in xp.tags
    first_stats = copy.deepcopy(xp.stats)
    compiled(array)
    assert xp.stats == first_stats


def test_compiled_tagger_writes_dataset_statistics(tmp_path, monkeypatch):
    stats_dir = tmp_path / "tags"
    statistics_path = tmp_path / "statistics.json"
    monkeypatch.setenv("SAPS_TAGGER_STATS_DIR", str(stats_dir))
    monkeypatch.setenv("SAPS_STATISTICS_PATH", str(statistics_path))
    xp = tagger_framework()
    benchmark = MatrixVectorMultiplicationBenchmark()
    param = next(p for p in benchmark.params if p.dataset.name == "small")
    benchmark.setup(param, xp=xp, use_cache=False)
    benchmark.run(param)
    A, b = (to_numpy(array) for array in benchmark._input)
    np.testing.assert_allclose(to_numpy(benchmark._output[0]), A @ b)
    first_stats = copy.deepcopy(xp.stats)
    benchmark.run(param)
    assert xp.stats == first_stats
    (record_path,) = stats_dir.glob("*.json")
    record = json.loads(record_path.read_text())
    assert record == {
        "benchmark_name": benchmark.name,
        "generator_name": param.generator.name,
        "dataset_name": param.dataset.name,
        "tags": xp.tags,
    }
    document = json.loads(statistics_path.read_text())
    dataset = document["benchmarks"][0]["generators"][0]["datasets"][0]
    assert dataset["name"] == param.dataset.name
    assert dataset["freshness"] == param.dataset.freshness
    assert dataset["statistics"] == xp.tags
    benchmark.teardown(param)
