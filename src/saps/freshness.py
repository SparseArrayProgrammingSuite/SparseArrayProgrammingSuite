import ast
import hashlib
import importlib.util
import os
import sys
from functools import cache
from pathlib import Path

from saps.dependencies import dependency_versions


def repo_root() -> Path:
    root = os.environ.get("SAPS_REPO_ROOT")
    if root:
        return Path(root).resolve()
    return Path(__file__).resolve().parents[2]


_FIRST_PARTY_ROOTS = frozenset({"saps", "saps_framework"})


def _is_under_venv(path: Path) -> bool:
    for prefix in (sys.prefix, sys.base_prefix):
        try:
            path.relative_to(Path(prefix).resolve())
            return True
        except ValueError:
            continue
    return False


def _module_path(module_name: str) -> Path | None:
    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, ModuleNotFoundError, ValueError):
        return None
    if spec is None or spec.origin is None:
        return None
    path = Path(spec.origin).resolve()
    if path.suffix != ".py":
        return None
    try:
        path.relative_to(repo_root())
    except ValueError:
        return None
    # Exclude third-party packages (e.g. numpy installed in the venv) so their
    # source isn't hashed and their relative imports aren't walked. First-party
    # packages are kept even when installed under site-packages so that changes
    # to them still invalidate dataset caches.
    if module_name.split(".", 1)[0] not in _FIRST_PARTY_ROOTS and (
        "site-packages" in path.parts or _is_under_venv(path)
    ):
        return None
    return path


def _imported_modules(path: Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError:
        return set()

    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                raise RuntimeError(
                    f"Relative import in {path}:{node.lineno}. "
                    "Use an absolute import so freshness hashes are stable."
                )
            if node.module:
                modules.add(node.module)
    return modules


@cache
def _freshness_inputs(
    module_name: str,
) -> tuple[tuple[Path, ...], tuple[str, ...]]:
    pending = [module_name]
    seen_modules: set[str] = set()
    seen_files: set[Path] = set()
    external_modules: set[str] = set()
    while pending:
        current = pending.pop()
        if current in seen_modules:
            continue
        seen_modules.add(current)

        path = _module_path(current)
        if path is None:
            if (
                current.split(".", 1)[0] not in sys.stdlib_module_names
                and _module_path(current.split(".", 1)[0]) is None
            ):
                external_modules.add(current)
            continue
        if path in seen_files:
            continue
        seen_files.add(path)
        pending.extend(
            dependency
            for dependency in _imported_modules(path.resolve())
            if dependency not in seen_modules
        )

    return tuple(sorted(seen_files)), tuple(sorted(external_modules))


@cache
def _source_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@cache
def source_freshness(module_name: str, source_path: Path) -> str:
    files, dependencies = _freshness_inputs(module_name)
    if not files:
        files = (source_path,)
    digest = hashlib.sha256()
    for value in (
        *sorted(_source_digest(path) for path in files),
        *sorted(dependencies),
        *(
            f"{record['name']}=={record['version']}"
            for record in dependency_versions(list(dependencies))
        ),
    ):
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()
