#!/usr/bin/env python3
"""Start Julia once in a freshly installed competition environment.

finch-tensor and galley-jl-python install Julia and their Julia packages, then
precompile them, the first time a process starts Julia. That takes minutes, far
longer than a benchmark's process timeout, so competition.config.json runs this
script as an install command: environments without either package skip it, and
the others pay the startup once, under ASV's install timeout.

Failures are reported but not fatal: an environment that fails here would fail
its benchmarks anyway, and they record the error where the results show it.
"""

import importlib.util
import sys
import time
import traceback


def main() -> int:
    if importlib.util.find_spec("galley_jl_python") is not None:
        name = "galley-jl-python"

        def start():
            import galley_jl_python  # noqa: F401

    elif (
        importlib.util.find_spec("finch") is not None
        and importlib.util.find_spec("juliacall") is not None
    ):
        name = "finch-tensor"

        def start():
            from finch.compile_jl.julia import init_julia

            init_julia()

    else:
        return 0

    started = time.perf_counter()
    print(f"warm_julia_env: starting Julia for {name}", flush=True)
    try:
        start()
    except Exception:  # noqa: BLE001 - benchmarks report the failure instead
        traceback.print_exc()
        print(f"warm_julia_env: could not start Julia for {name}", file=sys.stderr)
        return 0
    elapsed = time.perf_counter() - started
    print(f"warm_julia_env: Julia ready for {name} in {elapsed:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
