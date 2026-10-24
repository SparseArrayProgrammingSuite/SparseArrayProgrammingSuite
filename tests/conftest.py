from pathlib import Path

import pytest

from numpy.random import default_rng

from saps.benchmark import ReferenceGenerator


@pytest.fixture
def uncached_references(monkeypatch):
    """Compute reference generators' outputs instead of reading the dataset cache."""
    monkeypatch.setattr(
        ReferenceGenerator, "cached_generate", ReferenceGenerator.generate
    )


@pytest.fixture(scope="session")
def lazy_datadir() -> Path:
    return Path(__file__).parent / "reference"


@pytest.fixture(scope="session")
def original_datadir() -> Path:
    return Path(__file__).parent / "reference"


@pytest.fixture
def rng():
    return default_rng(42)


@pytest.fixture
def random_wrapper(rng):
    def _random_wrapper_applier(arrays, wrapper):
        """
        Applies a wrapper to each array in the list with a 50% chance.
        Args:
            arrays: A list of NumPy arrays.
            wrapper: A function to apply to the arrays.
        """
        return [wrapper(arr) if rng.random() > 0.5 else arr for arr in arrays]

    return _random_wrapper_applier
