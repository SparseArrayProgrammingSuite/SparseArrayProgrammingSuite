from __future__ import annotations

import zipfile

import pytest

import numpy as np

from saps.downloaders import slicot


def test_slicot_problem_registry_matches_published_table():
    assert len(slicot.list_slicot_problems()) == 18
    assert slicot.list_slicot_problems()[0] == "eady.mat"

    mna_5 = slicot.slicot_problem_metadata("MNA_5")

    assert mna_5.title == "MNA example - 5"
    assert mna_5.order == 10913
    assert mna_5.inputs == 9
    assert mna_5.outputs == 9


@pytest.mark.parametrize(
    ("source_name", "expected"),
    [
        ("eady", "eady.mat"),
        ("eady.mat", "eady.mat"),
        ("eady.zip", "eady.mat"),
        ("slicot://CDPLAYER", "CDplayer.mat"),
        ("Orr-Som", "Orr-Som.mat"),
    ],
)
def test_normalize_slicot_source_name_accepts_common_aliases(source_name, expected):
    assert slicot.normalize_slicot_source_name(source_name) == expected


def test_normalize_slicot_source_name_rejects_paths():
    with pytest.raises(ValueError, match="Invalid SLICOT problem name"):
        slicot.normalize_slicot_source_name("../eady")


def test_slicot_urls_use_official_archive_location():
    assert (
        slicot.slicot_source_url("heat-cont")
        == "https://www.slicot.org/objects/software/shared/bench-data/heat-cont.zip"
    )
    assert (
        slicot.slicot_collection_url()
        == "https://www.slicot.org/objects/software/shared/bench-data/All-Data.zip"
    )


def test_download_slicot_problem_uses_cached_mat_file(monkeypatch, tmp_path):
    mat_path = tmp_path / "eady.mat"
    mat_path.write_bytes(b"cached")

    monkeypatch.setattr(
        slicot.urllib.request,
        "urlretrieve",
        lambda url, path: pytest.fail("cached MAT file should not download"),
    )

    assert slicot.download_slicot_problem("eady", data_dir=tmp_path) == mat_path
    assert mat_path.read_bytes() == b"cached"


def test_download_slicot_problem_downloads_and_extracts_archive(monkeypatch, tmp_path):
    def fake_urlretrieve(url, path):
        assert url == slicot.slicot_source_url("eady")
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("nested/eady.mat", b"mat contents")

    monkeypatch.setattr(slicot.urllib.request, "urlretrieve", fake_urlretrieve)

    mat_path = slicot.download_slicot_problem("eady", data_dir=tmp_path)

    assert mat_path == tmp_path / "eady.mat"
    assert mat_path.read_bytes() == b"mat contents"
    assert (tmp_path / "eady.zip").exists()


def test_download_slicot_problem_extracts_cached_archive(tmp_path):
    with zipfile.ZipFile(tmp_path / "tline.zip", "w") as archive:
        archive.writestr("tline.mat", b"cached archive contents")

    mat_path = slicot.download_slicot_problem("tline", data_dir=tmp_path)

    assert mat_path == tmp_path / "tline.mat"
    assert mat_path.read_bytes() == b"cached archive contents"


def test_download_slicot_collection_uses_cached_archive(monkeypatch, tmp_path):
    archive_path = tmp_path / slicot.SLICOT_ALL_DATA_ARCHIVE
    archive_path.write_bytes(b"all")

    monkeypatch.setattr(
        slicot.urllib.request,
        "urlretrieve",
        lambda url, path: pytest.fail("cached collection should not download"),
    )

    assert slicot.download_slicot_collection(data_dir=tmp_path) == archive_path


def test_load_slicot_problem_returns_variables_and_metadata(monkeypatch, tmp_path):
    from scipy.io import savemat

    mat_path = tmp_path / "build.mat"
    savemat(mat_path, {"A": np.eye(2), "B": np.ones((2, 1))})
    monkeypatch.setattr(
        slicot,
        "download_slicot_problem",
        lambda source_name, data_dir=None: mat_path,
    )

    variables, meta = slicot.load_slicot_problem("build", data_dir=tmp_path)

    assert sorted(variables) == ["A", "B"]
    np.testing.assert_array_equal(variables["A"], np.eye(2))
    assert meta["dataset_name"] == "build.mat"
    assert meta["order"] == 48
    assert meta["source_page_url"] == slicot.SLICOT_BENCHMARK_PAGE_URL


def test_extract_slicot_mat_archive_rejects_ambiguous_archives(tmp_path):
    archive_path = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("one.mat", b"1")
        archive.writestr("two.mat", b"2")

    with pytest.raises(ValueError, match="Expected one MAT file"):
        slicot._extract_slicot_mat_archive(archive_path, tmp_path, "missing.mat")
