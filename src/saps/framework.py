import os
import sys
from pathlib import Path

from saps_framework import Framework

_xp: Framework | None = None
_xp_path: Path | None = None


def load_framework(framework_path: str | os.PathLike[str] | None = None) -> Framework:
    global _xp, _xp_path

    framework_path = framework_path or os.environ.get("SAPS_FRAMEWORK")
    if framework_path is None:
        if _xp is not None:
            return _xp
        raise RuntimeError(
            "No SAPS framework was supplied. Pass an xp framework explicitly or set "
            "SAPS_FRAMEWORK."
        )

    import importlib.util

    resolved_path = Path(framework_path)
    if _xp is not None and _xp_path == resolved_path.resolve():
        return _xp

    # Verify the file exists
    if not resolved_path.exists():
        raise FileNotFoundError(f"Framework file not found: {resolved_path}")

    framework_dir = str(resolved_path.parent)
    if framework_dir not in sys.path:
        sys.path.insert(0, framework_dir)

    spec = importlib.util.spec_from_file_location(
        "custom_framework", str(resolved_path)
    )
    assert spec is not None, "Failed to create module spec"
    custom_framework = importlib.util.module_from_spec(spec)
    assert spec.loader is not None, "Module spec has no loader"
    # Register before exec so the module is importable by name
    sys.modules["custom_framework"] = custom_framework
    spec.loader.exec_module(custom_framework)
    framework = custom_framework.xp
    assert isinstance(framework, Framework), (
        "The custom framework must define an 'xp' variable referring to an "
        "instance of a saps.Framework subclass."
    )
    _xp = framework
    _xp_path = resolved_path.resolve()
    return framework
