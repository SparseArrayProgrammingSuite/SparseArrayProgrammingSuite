import sysconfig
from importlib.metadata import version
from pathlib import Path

import donfig  # type: ignore[import]

"""
Benchmark Configuration Module

This module manages configuration settings for the benchmark.
The harness stores its settings and data in the `SAPS_PATH` directory, which
defaults to `~/.sparse_array_programming_suite` but can be customized using the
`SAPS_PATH` environment variable.

Configuration details:
- Settings are stored in a `config.json` file within the `SAPS_PATH` directory.
- Values can be set via environment variables, the `config.json` file,
    or the `set_config` function.
- Configuration values are loaded automatically when the module is imported
    and can be accessed using the `get_config` function.

Use this module to easily manage and retrieve saps-specific settings.
"""

default = {
    "data_path": str(Path(sysconfig.get_path("data")) / "sparse_array_programming_suite"),
}

config = donfig.Config("saps", defaults=[default])


def get_version():
    """
    Get the version of Sparse Array Programming Suite.
    """

    return version("saps")
