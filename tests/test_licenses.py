from __future__ import annotations

import datetime
import re

import pytest

# Not `from saps import licenses`, which makes mypy lose track of the
# saps.benchmarks submodules that tests/test_suitesparse_rhs.py imports.
import saps.licenses as licenses
from saps.benchmark import INHERIT, Author, Dataset, Generator, License, Ref
from saps.metadata import _benchmark_instances

# SPDX license identifiers used so far; extend this when a source adds one.
KNOWN_SPDX_IDS = {
    "MIT",
    "CC-BY-2.5",
    "CC-BY-4.0",
    "CC-BY-SA-3.0",
    "CC0-1.0",
    "ODC-By-1.0",
    "GFDL-1.1-or-later",
    "GPL-2.0-or-later",
}


def _generators() -> dict[str, Generator]:
    generators = {}
    for benchmark in _benchmark_instances():
        for generator in benchmark.generators:
            cls = type(generator)
            generators[f"{cls.__module__}.{cls.__qualname__}"] = generator
    return generators


GENERATORS = _generators()


def _check_license(license: License) -> None:
    assert isinstance(license.date_accessed, datetime.date)
    assert license.date_accessed <= datetime.date.today()
    assert license.ref.url, "a license must say where it was found"
    assert isinstance(license.redistributable, bool)
    assert isinstance(license.notes, str)
    tokens = re.split(r"\s+|[()]", license.spdx)
    for token in filter(None, tokens):
        if token in ("AND", "OR", "WITH"):
            continue
        assert token in KNOWN_SPDX_IDS or re.fullmatch(
            r"LicenseRef-[A-Za-z0-9.-]+", token
        ), f"{token!r} in {license.spdx!r} is not a known SPDX id or LicenseRef"


def test_benchmarks_have_generators():
    assert len(GENERATORS) > 100


@pytest.mark.parametrize("key", sorted(GENERATORS))
def test_every_generator_documents_its_license(key):
    generator = GENERATORS[key]
    license = generator.license
    assert license is None or isinstance(license, License)
    if license is not None:
        _check_license(license)
    expected = None if license is None else license.metadata
    assert generator.metadata["license"] == expected
    for dataset, dataset_metadata in zip(
        generator.datasets, generator.metadata["datasets"], strict=True
    ):
        own = dataset.license
        assert own is INHERIT or own is None or isinstance(own, License)
        if isinstance(own, License):
            _check_license(own)
        if own is INHERIT:
            assert "license" not in dataset_metadata
            assert generator.dataset_license(dataset) is license
        else:
            own_expected = None if own is None else own.metadata
            assert dataset_metadata["license"] == own_expected
            assert generator.dataset_license(dataset) is own


LICENSE_CONSTANTS = [
    name for name in dir(licenses) if name.isupper() and not name.startswith("_")
]


@pytest.mark.parametrize("name", LICENSE_CONSTANTS)
def test_license_constants_are_valid(name):
    license = getattr(licenses, name)
    assert license is None or isinstance(license, License)
    if license is not None:
        _check_license(license)


def test_license_metadata_is_json_ready():
    license = License(
        spdx="CC-BY-4.0",
        ref=Ref(title="About", authors=[Author("A. Person")], url="https://x.test"),
        date_accessed=datetime.date(2026, 11, 18),
        redistributable=True,
        notes="The About page says so.",
    )
    assert license.metadata == {
        "spdx": "CC-BY-4.0",
        "ref": 'A. Person. "About", URL: https://x.test.',
        "date_accessed": "2026-11-18",
        "redistributable": True,
        "notes": "The About page says so.",
    }


def test_reference_generators_share_their_sources_license():
    for key, generator in GENERATORS.items():
        if key.endswith("ReferenceGenerator"):
            source = GENERATORS[key.removesuffix("ReferenceGenerator") + "Generator"]
            assert generator.license is source.license, key
            for reference, dataset in zip(
                generator.datasets, source.datasets, strict=True
            ):
                assert generator.dataset_license(reference) is source.dataset_license(
                    dataset
                ), (key, dataset.name)


_OTHER = License(
    spdx="CC0-1.0",
    ref=Ref(title="Other", authors=[Author("B. Person")], url="https://y.test"),
    date_accessed=datetime.date(2026, 11, 18),
    redistributable=False,
)


class _LicensedDataset(Dataset):
    def __init__(self, name, license=INHERIT):
        self._name = name
        self._license = license

    @property
    def name(self):
        return self._name

    @property
    def pretty_name(self):
        return self._name

    @property
    def description(self):
        return self._name

    @property
    def suites(self):
        return []

    @property
    def concepts(self):
        return "<ccs2012></ccs2012>"

    @property
    def license(self):
        return self._license


@pytest.mark.parametrize(
    ("own", "generator_license", "expected"),
    [
        (INHERIT, licenses.SAPS, licenses.SAPS),
        (INHERIT, None, None),
        (_OTHER, licenses.SAPS, _OTHER),
        (_OTHER, None, _OTHER),
        (None, licenses.SAPS, None),
    ],
)
def test_dataset_license_overrides_generator_license(own, generator_license, expected):
    # dataset_license reads only self.license, so a stand-in suffices.
    generator = type("G", (), {"license": generator_license})()
    dataset = _LicensedDataset("d", own)
    assert Generator.dataset_license(generator, dataset) is expected
    metadata = dataset.metadata
    if own is INHERIT:
        assert "license" not in metadata
    else:
        assert metadata["license"] == (None if own is None else own.metadata)
