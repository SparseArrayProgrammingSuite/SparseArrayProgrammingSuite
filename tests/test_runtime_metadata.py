import json
from types import SimpleNamespace

import saps.benchmark as benchmark_module
from saps.benchmark import Benchmark


class _RuntimeDataset:
    name = "dataset"

    @property
    def file(self):
        raise AssertionError("runtime source metadata must not be evaluated")

    @property
    def freshness(self):
        raise AssertionError("runtime freshness must not be evaluated")

    @property
    def dependencies(self):
        raise AssertionError("runtime dependencies must not be evaluated")


def test_statistics_use_committed_dataset_metadata(monkeypatch, tmp_path):
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "benchmarks": [
                    {
                        "name": "benchmark",
                        "generators": [
                            {
                                "name": "generator",
                                "datasets": [
                                    {
                                        "name": "dataset",
                                        "file": "src/dataset.py",
                                        "freshness": "committed-freshness",
                                        "dependencies": ["numpy"],
                                    }
                                ],
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SAPS_METADATA_PATH", str(metadata_path))
    monkeypatch.setenv("SAPS_STATISTICS_PATH", str(tmp_path / "statistics.json"))
    captured = {}

    def capture(*args):
        captured["args"] = args

    monkeypatch.setattr(benchmark_module, "_write_statistics_tags", capture)
    benchmark = SimpleNamespace(name="benchmark")
    param = SimpleNamespace(
        generator=SimpleNamespace(name="generator"), dataset=_RuntimeDataset()
    )

    Benchmark._write_tagger_stats(benchmark, param, SimpleNamespace(tags={"sparse"}))

    assert captured["args"][4:7] == (
        "src/dataset.py",
        "committed-freshness",
        ["numpy"],
    )
