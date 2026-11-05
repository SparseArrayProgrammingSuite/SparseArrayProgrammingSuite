from __future__ import annotations

import importlib.metadata
import re
import sys
from pathlib import Path

import pytest

import yaml  # type: ignore[import-untyped]

from saps.generate_environment import generate_environment_mismatches, generate_pins

ROOT = Path(__file__).parents[1]


def _canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _write_manifest(tmp_path: Path, dependencies: str, targets: str = "") -> Path:
    path = tmp_path / "pixi.toml"
    path.write_text(
        "[feature.generate.pypi-dependencies]\n" + dependencies + targets,
        encoding="utf-8",
    )
    return path


def test_matching_pins_pass(tmp_path):
    version = importlib.metadata.version("packaging")
    path = _write_manifest(tmp_path, f'packaging = "=={version}"\n')
    assert generate_environment_mismatches(path) == []


def test_reports_wrong_and_missing_packages(tmp_path):
    path = _write_manifest(
        tmp_path,
        'packaging = "==0.0.1"\nsaps-not-a-real-package = "==1.0"\n',
    )
    mismatches = generate_environment_mismatches(path)
    assert len(mismatches) == 2
    assert mismatches[0].startswith("packaging: installed ")
    assert mismatches[1] == "saps-not-a-real-package: not installed"


def test_skips_pins_for_other_platforms(tmp_path):
    other = "linux" if sys.platform == "win32" else "win"
    path = _write_manifest(
        tmp_path,
        "",
        f"[feature.generate.target.{other}.pypi-dependencies]\n"
        'saps-not-a-real-package = "==1.0"\n',
    )
    assert generate_environment_mismatches(path) == []


_LOCK_PLATFORMS = {"darwin": "macos-arm64", "linux": "linux-64", "win32": "win-64"}


def test_generate_feature_pins_every_locked_package():
    """Every package the generate environment installs is pinned at its locked
    version, so freshness hashes only change when a pin does."""
    lock_path = ROOT / "pixi.lock"
    if not lock_path.exists():
        pytest.skip("pixi.lock is written by `pixi install`")
    lock = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    platform = _LOCK_PLATFORMS[sys.platform]
    urls = {
        entry["pypi"]
        for entry in lock["environments"]["generate"]["packages"][platform]
        if "pypi" in entry
    }
    pins = {_canonical(name): spec for name, spec in generate_pins().items()}
    for package in lock["packages"]:
        if package.get("pypi") not in urls or package["name"] == "saps":
            continue
        spec = pins.get(_canonical(package["name"]))
        assert spec is not None, f"{package['name']} is not pinned"
        if isinstance(spec, str):
            spec = {"version": spec}
        if "git" in spec:
            assert package["pypi"].endswith("#" + spec["rev"]), package["name"]
        else:
            assert spec["version"] == f"=={package['version']}", package["name"]
