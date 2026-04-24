import os

from saps_framework import Framework

framework_path = os.environ.get("SAPS_FRAMEWORK")
if framework_path is not None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("custom_framework", framework_path)
    custom_framework = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(custom_framework)
    xp = custom_framework.xp
    assert isinstance(xp, Framework), (
        "The custom framework must define an 'xp' variable that is an "
        "instance of Framework."
    )
else:
    xp = None
