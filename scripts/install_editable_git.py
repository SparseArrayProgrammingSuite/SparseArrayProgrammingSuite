#!/usr/bin/env python3
"""Install a package from a git checkout, editable, into the running Python.

Some frameworks only work from a source tree. Scorch, for one, looks for its
``setup.py`` above its package directory to find the C++ sources it compiles
kernels from, so a regular install never finds them (and its search never ends
once it reaches ``/``). ASV requirements cannot express an editable git install
(pip needs the package name, which ASV drops), so competition.config.json runs
this script as an install command, and pixi's install-scorch task runs it too.

``SAPS_EDITABLE_GIT`` names the repository and commit as ``URL@REV``. Without it
the script does nothing, so every environment can run it. The checkout goes in
``<environment>/src/<repository name>``, next to the environment it serves.
"""

import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path


def _installed(checkout: Path, rev: str) -> bool:
    """Whether some distribution is installed editable from checkout at rev.

    Checked without importing the package: a broken install (such as Scorch's
    regular install) may never finish importing.
    """
    if not (checkout / ".git").is_dir():
        return False
    head = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    if not head.startswith(rev) and not rev.startswith(head or "-"):
        return False
    target = checkout.resolve().as_uri()
    for dist in importlib.metadata.distributions():
        direct_url = dist.read_text("direct_url.json")
        if direct_url:
            info = json.loads(direct_url)
            editable = info.get("dir_info", {}).get("editable", False)
            if editable and info.get("url", "").rstrip("/") == target:
                return True
    return False


def main() -> int:
    spec = os.environ.get("SAPS_EDITABLE_GIT", "").strip()
    if not spec:
        return 0
    url, sep, rev = spec.rpartition("@")
    if not sep or not url or not rev:
        sys.exit(f"SAPS_EDITABLE_GIT must be URL@REV, got {spec!r}")
    name = url.rstrip("/").rsplit("/", 1)[-1].removesuffix(".git")
    checkout = Path(sys.prefix) / "src" / name
    if _installed(checkout, rev):
        return 0

    if not (checkout / ".git").is_dir():
        checkout.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init", "-q", str(checkout)], check=True)
        subprocess.run(
            ["git", "-C", str(checkout), "remote", "add", "origin", url], check=True
        )
    subprocess.run(
        ["git", "-C", str(checkout), "fetch", "-q", "--depth", "1", "origin", rev],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(checkout), "checkout", "-q", "--force", "FETCH_HEAD"],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-build-isolation",
            "--editable",
            str(checkout),
        ],
        check=True,
    )
    print(f"install_editable_git: installed {name} at {rev} from {checkout}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
