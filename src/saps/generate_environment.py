"""Check that installed packages match the generate feature's exact pins.

Freshness hashes include installed dependency versions, so generating data or
metadata from a drifted environment would silently mark datasets stale.
"""

from __future__ import annotations

import importlib.metadata
import json
import sys
import tomllib
from pathlib import Path
from typing import Any

from packaging.version import Version

from saps.freshness import repo_root


def _platform_targets() -> tuple[str, ...]:
    """pixi target selectors that apply to this platform."""
    if sys.platform == "win32":
        return ("win",)
    if sys.platform == "darwin":
        return ("unix", "osx")
    return ("unix", "linux")


def generate_pins(manifest_path: Path | None = None) -> dict[str, Any]:
    """The generate feature's PyPI pins that apply to this platform."""
    manifest_path = manifest_path or repo_root() / "pixi.toml"
    manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    feature = manifest["feature"]["generate"]
    pins = dict(feature.get("pypi-dependencies", {}))
    for target in _platform_targets():
        pins.update(
            feature.get("target", {}).get(target, {}).get("pypi-dependencies", {})
        )
    return pins


def _installed_commit(distribution: importlib.metadata.Distribution) -> str | None:
    direct_url = distribution.read_text("direct_url.json")
    if direct_url is None:
        return None
    return json.loads(direct_url).get("vcs_info", {}).get("commit_id")


def generate_environment_mismatches(manifest_path: Path | None = None) -> list[str]:
    mismatches = []
    for name, spec in generate_pins(manifest_path).items():
        if isinstance(spec, str):
            spec = {"version": spec}
        try:
            distribution = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            mismatches.append(f"{name}: not installed")
            continue
        if "git" in spec:
            commit = _installed_commit(distribution)
            if commit != spec["rev"]:
                installed = f"commit {commit}" if commit else distribution.version
                mismatches.append(
                    f"{name}: installed {installed}, pinned commit {spec['rev']}"
                )
        elif Version(distribution.version) != Version(
            spec["version"].removeprefix("==")
        ):
            mismatches.append(
                f"{name}: installed {distribution.version}, pinned {spec['version']}"
            )
    return mismatches


def check_generate_environment() -> None:
    mismatches = generate_environment_mismatches()
    if mismatches:
        raise SystemExit(
            "Installed packages do not match the generate feature's pins in "
            "pixi.toml:\n  "
            + "\n  ".join(mismatches)
            + "\nRun it in the default pixi environment: `pixi run ...`."
        )
