import sys
from importlib.machinery import ModuleSpec
from pathlib import Path

import pytest

from saps import freshness
from saps.benchmarks.suitesparse import SuiteSparseDataset


@pytest.mark.parametrize("origin", ["frozen", "built-in", "/extension.so", None])
def test_non_python_origins_do_not_resolve_filesystem_paths(monkeypatch, origin):
    monkeypatch.setattr(
        freshness.importlib.util,
        "find_spec",
        lambda name: ModuleSpec(name, loader=None, origin=origin),
    )

    def unavailable_cwd(self, *args, **kwargs):
        raise FileNotFoundError("working directory no longer exists")

    monkeypatch.setattr(Path, "resolve", unavailable_cwd)
    assert freshness._module_path("example") is None


@pytest.mark.parametrize("external", [False, True])
@pytest.mark.parametrize("namespace", [False, True])
def test_module_lookup_does_not_execute_package_initializers(
    monkeypatch, tmp_path, external, namespace
):
    repo = tmp_path / "repo"
    root = tmp_path / "external" if external else repo
    package = root / "freshness_no_import"
    subpackage = package / "nested"
    subpackage.mkdir(parents=True)
    if not namespace:
        (package / "__init__.py").write_text(
            "raise AssertionError('Package initializer executed')\n", encoding="utf-8"
        )
    (subpackage / "__init__.py").write_text(
        "raise AssertionError('Nested package initializer executed')\n",
        encoding="utf-8",
    )
    source = subpackage / "module.py"
    source.write_text("value = 1\n", encoding="utf-8")
    monkeypatch.setenv("SAPS_REPO_ROOT", str(repo))
    monkeypatch.syspath_prepend(str(root))

    assert freshness._module_path("freshness_no_import.nested.module") == (
        None if external else source
    )
    assert not any(name.startswith("freshness_no_import") for name in sys.modules)


@pytest.mark.parametrize(
    "location", ["src", "env/site-packages", "../task/site-packages"]
)
def test_dataset_file_is_canonical_in_checkout_and_installed_envs(
    monkeypatch, tmp_path, location
):
    package = (tmp_path / "repo" / location / "saps").resolve()
    monkeypatch.setenv("SAPS_REPO_ROOT", str(tmp_path / "repo"))
    monkeypatch.setattr("saps.benchmark.__file__", str(package / "benchmark.py"))
    monkeypatch.setattr(
        "saps.benchmark.inspect.getfile",
        lambda cls: package / "benchmarks/suitesparse.py",
    )

    assert (
        SuiteSparseDataset("HB/bcsstk01").file == "src/saps/benchmarks/suitesparse.py"
    )


def test_installed_harness_can_locate_benchmark_loaded_from_checkout(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("SAPS_REPO_ROOT", str(tmp_path / "repo"))
    monkeypatch.setattr(
        "saps.benchmark.__file__", str(tmp_path / "env/site-packages/saps/benchmark.py")
    )
    monkeypatch.setattr(
        "saps.benchmark.inspect.getfile",
        lambda cls: tmp_path / "repo/src/saps/benchmarks/suitesparse.py",
    )

    assert (
        SuiteSparseDataset("HB/bcsstk01").file == "src/saps/benchmarks/suitesparse.py"
    )
