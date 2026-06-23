from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


def _extract_result_json(output: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, char in enumerate(output):
        if char != "{":
            continue
        try:
            document, end = decoder.raw_decode(output[index:])
        except json.JSONDecodeError:
            continue
        if output[index + end :].strip():
            continue
        if isinstance(document, dict) and "results" in document:
            return document
    raise AssertionError(f"No result JSON found in harness output:\n{output}")


def _test_dataset_slots(benchmark_metadata: dict[str, Any]) -> list[tuple[int, str]]:
    slots = []
    index = 0
    benchmark_is_test = "test" in benchmark_metadata.get("suites", [])
    for generator in benchmark_metadata["generators"]:
        generator_is_test = benchmark_is_test or "test" in generator.get("suites", [])
        for dataset in generator["datasets"]:
            if generator_is_test or "test" in dataset.get("suites", []):
                slots.append((index, f"{generator['name']}.{dataset['name']}"))
            index += 1
    return slots


def _is_passing_result(value: Any) -> bool:
    return value is not None and not (isinstance(value, float) and math.isnan(value))


def test_harness_test_suite_outputs_pass_for_all_test_datasets(tmp_path):
    config_path = tmp_path / "saps.conf.json"
    config_path.write_text(
        json.dumps(
            {
                "environment_type": "existing:same",
                "matrix": {
                    "env_nobuild": {
                        "SAPS_FRAMEWORK": ["frameworks/saps_numpy.py"],
                        "SAPS_REPO_ROOT": [str(REPO_ROOT)],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": os.pathsep.join(
                path
                for path in (str(REPO_ROOT), env.get("PYTHONPATH", ""))
                if path
            ),
            "SAPS_FRAMEWORK": str(REPO_ROOT / "frameworks/saps_numpy.py"),
        }
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "bin/run_benchmark.py"),
            "--config",
            str(config_path),
            "--tag",
            "test",
            "--check-suite",
            "--metric",
            "time",
            "--quick",
            "--timeout",
            "30",
            "--remote-storage-backend",
            "local",
            "--remote-storage-bucket",
            str(tmp_path / "remote-storage"),
        ],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=600,
    )

    assert completed.returncode == 0, completed.stdout
    result_json = _extract_result_json(completed.stdout)
    assert result_json["result_count"] > 0

    metadata_path = tmp_path / ".saps/outputs/results/benchmarks_meta.json"
    benchmark_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    failures = []
    checked = 0
    for benchmark_name, result in result_json["results"].items():
        errcode = result["errcode"]
        if errcode not in (None, 0):
            failures.append(f"{benchmark_name}: errcode={errcode}\n{result['stderr']}")

        values = result["result"]
        test_slots = _test_dataset_slots(benchmark_metadata[benchmark_name])
        assert test_slots, f"{benchmark_name} did not include any test datasets"
        for index, dataset_name in test_slots:
            checked += 1
            if not _is_passing_result(values[index]):
                failures.append(
                    f"{benchmark_name}[{dataset_name}] did not produce a passing "
                    f"result value: {values[index]!r}"
                )

    assert checked > 0
    assert not failures, "\n\n".join(failures)
