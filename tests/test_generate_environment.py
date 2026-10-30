from __future__ import annotations

import importlib.metadata
import re
import tomllib
from pathlib import Path

from saps.generate_environment import generate_environment_mismatches

ROOT = Path(__file__).parents[1]


def _canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _write_pyproject(tmp_path: Path, dependencies: str) -> Path:
    path = tmp_path / "pyproject.toml"
    path.write_text(
        "[tool.poetry.group.generate.dependencies]\n" + dependencies,
        encoding="utf-8",
    )
    return path


def test_matching_pins_pass(tmp_path):
    version = importlib.metadata.version("packaging")
    path = _write_pyproject(tmp_path, f'packaging = "=={version}"\n')
    assert generate_environment_mismatches(path) == []


def test_reports_wrong_and_missing_packages(tmp_path):
    path = _write_pyproject(
        tmp_path,
        'packaging = "==0.0.1"\nsaps-not-a-real-package = "==1.0"\n',
    )
    mismatches = generate_environment_mismatches(path)
    assert len(mismatches) == 2
    assert mismatches[0].startswith("packaging: installed ")
    assert mismatches[1] == "saps-not-a-real-package: not installed"


def test_skips_pins_whose_markers_do_not_apply(tmp_path):
    path = _write_pyproject(
        tmp_path,
        "saps-not-a-real-package = "
        '{version = "==1.0", markers = "python_version < \'3\'"}\n',
    )
    assert generate_environment_mismatches(path) == []


def test_generate_group_pins_every_locked_package():
    """Every package the generate environment installs is pinned at its locked
    version, so freshness hashes only change when a pin does."""
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    pins = {
        _canonical(name): spec
        for name, spec in pyproject["tool"]["poetry"]["group"]["generate"][
            "dependencies"
        ].items()
    }
    lock = tomllib.loads((ROOT / "poetry.lock").read_text(encoding="utf-8"))
    for package in lock["package"]:
        if not {"main", "generate"} & set(package["groups"]):
            continue
        spec = pins.get(_canonical(package["name"]))
        assert spec is not None, f"{package['name']} is not pinned"
        if isinstance(spec, str):
            spec = {"version": spec}
        if "git" in spec:
            assert spec["rev"] == package["source"]["resolved_reference"]
        else:
            assert spec["version"] == f"=={package['version']}", package["name"]
