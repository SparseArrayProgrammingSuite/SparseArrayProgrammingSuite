#!/usr/bin/env python3
"""Install extra pip requirements into one ASV benchmark environment.

ASV's ``req`` entries cannot express editable VCS installs, and they drop the
``#egg=`` fragment pip needs to name one. Frameworks that must be installed that
way (for example, packages that read their own source tree at runtime) set
``SAPS_ENV_REQUIREMENTS`` in their include entry's ``env`` to a requirements
file relative to the repository root. Environments without it are left alone.

Requirements are installed with ``--no-build-isolation`` so that packages whose
build imports an already-installed dependency (such as ``torch``) build against
the environment's pinned version.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    requirements = os.environ.get("SAPS_ENV_REQUIREMENTS")
    if not requirements:
        return 0
    return subprocess.call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-build-isolation",
            "-r",
            str(REPO_ROOT / requirements),
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
