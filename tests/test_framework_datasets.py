"""Run every suite-test dataset under the framework named by SAPS_FRAMEWORK.

The harness test runs NumPy through ASV, one process per dataset. This runs
the same datasets in one process, which frameworks that start Julia or compile
kernels need. Skipped unless SAPS_FRAMEWORK is set, as the pixi environments in
pixi.toml do.
"""

import importlib
import json
import os
from pathlib import Path

import pytest

from saps.framework import load_framework

if "SAPS_FRAMEWORK" not in os.environ:
    pytest.skip("SAPS_FRAMEWORK is not set", allow_module_level=True)

REPO_ROOT = Path(__file__).resolve().parents[1]
METADATA = json.loads((REPO_ROOT / "metadata.json").read_text(encoding="utf-8"))


def _test_datasets():
    for record in METADATA["benchmarks"]:
        module_name, class_name, _ = record["asv_ids"]["time"].split(".")
        for generator in record["generators"]:
            for dataset in generator["datasets"]:
                if "suite-test" not in dataset["tags"]:
                    continue
                param_name = f"{generator['name']}.{dataset['name']}"
                yield pytest.param(
                    module_name,
                    class_name,
                    param_name,
                    id=f"{record['name']}[{param_name}]",
                )


@pytest.mark.parametrize(("module_name", "class_name", "param_name"), _test_datasets())
def test_dataset_passes(module_name, class_name, param_name):
    module = importlib.import_module(f"saps.benchmarks.{module_name}")
    benchmark = getattr(module, class_name)()
    param = next(param for param in benchmark.params if str(param) == param_name)
    benchmark.setup(param, use_cache=False, xp=load_framework())
    benchmark.run(param)
    benchmark.teardown(param)
