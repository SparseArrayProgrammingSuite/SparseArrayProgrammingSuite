"""Check that installed packages match the generate group's exact pins.

Freshness hashes include installed dependency versions, so generating data or
metadata from a drifted environment would silently mark datasets stale.
"""

from __future__ import annotations

import importlib.metadata
import json
import tomllib
from pathlib import Path

from packaging.markers import Marker
from packaging.version import Version

from saps.freshness import repo_root


def _installed_commit(distribution: importlib.metadata.Distribution) -> str | None:
    direct_url = distribution.read_text("direct_url.json")
    if direct_url is None:
        return None
    return json.loads(direct_url).get("vcs_info", {}).get("commit_id")


def generate_environment_mismatches(pyproject_path: Path | None = None) -> list[str]:
    pyproject_path = pyproject_path or repo_root() / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    pins = pyproject["tool"]["poetry"]["group"]["generate"]["dependencies"]

    mismatches = []
    for name, spec in pins.items():
        if isinstance(spec, str):
            spec = {"version": spec}
        if "markers" in spec and not Marker(spec["markers"]).evaluate():
            continue
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
            "Installed packages do not match the generate group pins in "
            "pyproject.toml:\n  "
            + "\n  ".join(mismatches)
            + "\nRun `poetry sync --with generate` first."
        )
