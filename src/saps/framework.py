import os
import sys
from pathlib import Path

from saps_framework import Framework

framework_path = os.environ.get("SAPS_FRAMEWORK")
if framework_path is not None:
    import importlib.util

    resolved_path = Path(framework_path)

    # Verify the file exists
    if not resolved_path.exists():
        raise FileNotFoundError(f"Framework file not found: {resolved_path}")

    spec = importlib.util.spec_from_file_location(
        "custom_framework", str(resolved_path)
    )
    assert spec is not None, "Failed to create module spec"
    custom_framework = importlib.util.module_from_spec(spec)
    assert spec.loader is not None, "Module spec has no loader"
    # Register before exec so the module is importable by name
    sys.modules["custom_framework"] = custom_framework
    spec.loader.exec_module(custom_framework)
    xp = custom_framework.xp
    assert isinstance(xp, Framework), (
        "The custom framework must define an 'xp' variable referring to an "
        "instance of a saps.Framework subclass."
    )
else:
    xp = None
