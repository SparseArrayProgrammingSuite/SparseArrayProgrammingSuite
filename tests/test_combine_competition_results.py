from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_combiner():
    module_path = REPO_ROOT / "bin/combine_competition_results.py"
    spec = importlib.util.spec_from_file_location(
        "combine_competition_results", module_path
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_combine_results_ignores_non_asv_json(tmp_path):
    combiner = _load_combiner()
    results_dir = tmp_path / "results" / "machine"
    results_dir.mkdir(parents=True)
    (tmp_path / "results" / "benchmarks_meta.json").write_text(
        json.dumps({"benchmark": "metadata"}),
        encoding="utf-8",
    )
    (results_dir / "result.json").write_text(
        json.dumps(
            {
                "commit_hash": "abc123",
                "date": 1,
                "env_name": "env",
                "env_vars": {"SAPS_FRAMEWORK": "/repo/frameworks/saps_numpy.py"},
                "params": {"machine": "machine"},
                "requirements": {"numpy": "2.3"},
                "result_columns": ["result"],
                "results": {"bench.time": [[1.0]]},
            }
        ),
        encoding="utf-8",
    )

    combined = combiner.combine_results(tmp_path)

    assert combined["result_file_count"] == 1
    assert combined["results"][0]["framework"] == "numpy"
    assert combined["results"][0]["results"] == {"bench.time": [[1.0]]}
