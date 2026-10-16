from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

from saps.metadata import metadata_document


def test_metadata_generation_does_not_import_suitesparse():
    # A fresh interpreter avoids hiding an eager import behind the module cache.
    # Downloader tests supply fakes; normal CI metadata must not import ssgetpy,
    # whose package initialization can download the SuiteSparse index.
    script = textwrap.dedent(
        """
        import builtins
        import sys

        original_import = builtins.__import__

        def no_suitesparse_import(name, *args, **kwargs):
            if name == "ssgetpy" or name.startswith("ssgetpy."):
                raise AssertionError("Metadata attempted a SuiteSparse import: " + name)
            return original_import(name, *args, **kwargs)

        builtins.__import__ = no_suitesparse_import
        from saps.metadata import metadata_document
        from saps.freshness import _freshness_inputs

        document = metadata_document()
        assert document["benchmarks"]
        _, dependencies = _freshness_inputs("saps.benchmarks.cg")
        assert "ssgetpy.matrix" in dependencies
        assert "ssgetpy" not in sys.modules
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_statistics_tags_apply_to_whole_benchmark(tmp_path: Path):
    benchmark = metadata_document()["benchmarks"][0]
    generator = benchmark["generators"][0]
    dataset = generator["datasets"][0]
    statistics_path = tmp_path / "statistics.json"
    statistics_path.write_text(
        json.dumps(
            {
                "benchmarks": [
                    {
                        "name": benchmark["name"],
                        "generators": [
                            {
                                "name": generator["name"],
                                "datasets": [
                                    {
                                        "name": dataset["name"],
                                        "freshness": dataset["freshness"],
                                        "statistics": ["feature-fresh"],
                                    },
                                    {
                                        "name": "stale",
                                        "freshness": "stale",
                                        "statistics": ["feature-stale"],
                                    },
                                ],
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    tagged = metadata_document([statistics_path])["benchmarks"][0]
    assert tagged["name"] == benchmark["name"]
    records = [
        tagged,
        *tagged["generators"],
        *(
            dataset
            for generator in tagged["generators"]
            for dataset in generator["datasets"]
        ),
    ]
    for record in records:
        assert "feature-fresh" in record["tags"]
        assert "feature-stale" not in record["tags"]
