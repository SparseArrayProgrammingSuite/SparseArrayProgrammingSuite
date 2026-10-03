from __future__ import annotations

import json
from pathlib import Path

from saps.metadata import metadata_document


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
