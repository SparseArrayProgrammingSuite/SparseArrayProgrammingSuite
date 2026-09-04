import importlib.metadata
from functools import cache


@cache
def _packages_distributions() -> dict[str, list[str]]:
    return dict(importlib.metadata.packages_distributions())


def dependency_versions(dependencies: list[str]) -> list[dict[str, str]]:
    resolved: set[tuple[str, str]] = set()
    for dependency in dependencies:
        package_name = dependency.split(".", 1)[0]
        distributions = _packages_distributions().get(package_name)
        if not distributions:
            distributions = [package_name]

        for distribution in distributions:
            try:
                version = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                continue
            resolved.add((distribution, version))

    return [
        {
            "name": name,
            "version": version,
        }
        for name, version in sorted(resolved)
    ]
