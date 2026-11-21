from __future__ import annotations

import hashlib
import io
import lzma
import tarfile
import urllib.request
from pathlib import Path

import pytest

import numpy as np

from binsparse.conversions import to_numpy

from frameworks.saps_numpy import NumpyFramework
from saps.benchmark import DataInstance, ReferenceDataset
from saps.benchmarks import (
    mccomp_pool_instance,
    model_counting,
    weighted_model_counting,
)
from saps.benchmarks.mccomp_pool_instance import (
    MCCOMP_POOL_WEIGHTED_INSTANCES,
    MCCompPoolDataset,
    MCCompPoolInstanceGenerator,
)
from saps.benchmarks.model_counting import (
    MCCompInstanceGenerator,
    ModelCountingMCCompGenerator,
    ModelCountingMCCompPoolGenerator,
    ModelCountingMCCompPoolReferenceGenerator,
    fetch_mccomp_instance,
    parse_mccomp_exact,
)
from saps.benchmarks.model_counting import (
    clauses_to_einsum as mc_clauses_to_einsum,
)
from saps.benchmarks.weighted_model_counting import (
    WeightedModelCountingMCCompGenerator,
    WeightedModelCountingMCCompPoolGenerator,
    WeightedModelCountingMCCompPoolReferenceGenerator,
    parse_format,
)
from saps.benchmarks.weighted_model_counting import (
    clauses_to_einsums as wmc_clauses_to_einsums,
)
from saps.downloaders import mccomp
from saps.downloaders.mccomp import (
    MCCOMP_POOL_ARCHIVES,
    download_mccomp_instance,
    download_mccomp_pool_archive,
    download_mccomp_pool_instance,
    list_mccomp_instances,
    mccomp_raw_url,
    mccomp_source_url,
    normalize_mccomp_source_path,
    parse_dimacs,
)


def test_shared_dimacs_parser_handles_model_counting_instance():
    num_vars, clauses = parse_dimacs(
        """\
        p cnf 3 2
        c t mc
        1 -3 0
        2 3 -1 0
        c c s exact arb int 6
        """
    )

    assert num_vars == 3
    assert clauses == [[1, -3], [2, 3, -1]]


def test_shared_dimacs_parser_accepts_whitespace_and_multiline_clauses():
    num_vars, clauses = parse_dimacs(
        """\
        c comments may appear before the header
        p   cnf   4   2
        1 -2
        3 0
        4
        -1 0
        %
        """
    )

    assert num_vars == 4
    assert clauses == [[1, -2, 3], [4, -1]]


def test_shared_dimacs_parser_rejects_missing_problem_line():
    with pytest.raises(ValueError, match="missing a 'p cnf'"):
        parse_dimacs("c no header\n1 0\n")


def test_shared_dimacs_parser_rejects_clause_count_mismatch():
    with pytest.raises(ValueError, match="declared 2 clauses but parsed 1"):
        parse_dimacs("p cnf 1 2\n1 0\n")


def test_shared_dimacs_parser_rejects_unterminated_clause():
    with pytest.raises(ValueError, match="missing a terminating 0"):
        parse_dimacs("p cnf 1 1\n1\n")


def test_shared_dimacs_parser_rejects_out_of_range_literals():
    with pytest.raises(ValueError, match="exceeds declared variable count"):
        parse_dimacs("p cnf 1 1\n2 0\n")


def test_empty_dimacs_clause_builds_false_formula():
    assert parse_dimacs("p cnf 2 1\n0\n") == (2, [[]])
    assert mc_clauses_to_einsum([[]]) == "s[] += False"
    assert wmc_clauses_to_einsums([[]], 2, [1, 2]) == ["s[] += False"]


def test_weighted_parser_reuses_dimacs_clause_shape():
    num_vars, clauses, weights = parse_format(
        """\
        p cnf 2 1
        c t pwmc
        c p show 2 1 0
        c p weight -1 0.25 0
        c p weight 1 0.75 0
        1 -2 0
        """
    )

    assert num_vars == 2
    assert clauses == [[1, -2]]
    assert weights[-1] == 0.25
    assert weights[1] == 0.75


def test_mccomp_downloader_uses_cached_source_file(tmp_path):
    path = tmp_path / "Track1_MC" / "random_mc_1.cnf"
    path.parent.mkdir()
    path.write_text("p cnf 1 1\nc t mc\n1 0\nc c s exact arb int 1\n")

    local_path = download_mccomp_instance("random_mc_1", data_dir=tmp_path)

    assert local_path == path
    assert local_path.read_text() == "p cnf 1 1\nc t mc\n1 0\nc c s exact arb int 1\n"


def test_mccomp_url_and_name_helpers():
    assert normalize_mccomp_source_path("random_mc_1") == "Track1_MC/random_mc_1.cnf"
    assert (
        mccomp_source_url("Track1_MC/random_mc_1.cnf")
        == "https://github.com/arijitsh/mccomp-test-instances/blob/main/"
        "Track1_MC/random_mc_1.cnf"
    )
    assert (
        mccomp_raw_url("Track1_MC/random_mc_1.cnf")
        == "https://raw.githubusercontent.com/arijitsh/mccomp-test-instances/main/"
        "Track1_MC/random_mc_1.cnf"
    )


def test_model_counting_mccomp_track1_generator_parses_downloaded_source(
    monkeypatch, tmp_path
):
    source_path = tmp_path / "Track1_MC" / "random_mc_1.cnf"
    source_path.parent.mkdir()
    source_path.write_text("p cnf 2 1\nc t mc\n1 2 0\nc c s exact arb int 3\n")
    calls = []

    def fake_cached_generate(self, dataset):
        calls.append((self.name, dataset.source_path))
        return DataInstance(
            inputs=[],
            meta={"source_path": dataset.source_path},
        )

    monkeypatch.setattr(
        MCCompInstanceGenerator, "cached_generate", fake_cached_generate
    )
    monkeypatch.setattr(
        "saps.benchmarks.model_counting.download_mccomp_instance",
        lambda _: source_path,
    )
    generator = ModelCountingMCCompGenerator()

    instance = generator.generate(generator.datasets[0])

    assert calls == [("mccomp_instance", "Track1_MC/random_mc_1.cnf")]
    assert instance.meta["expr"] == "s[] += (B[v1] or B[v2])"
    assert instance.meta["expected_result"] == 3
    np.testing.assert_array_equal(to_numpy(instance.inputs[0]), np.array([0, 1]))
    np.testing.assert_array_equal(to_numpy(instance.ref_outputs[0]), np.array(3))


def test_weighted_mccomp_track4_generator_parses_downloaded_source(
    monkeypatch, tmp_path
):
    source_path = tmp_path / "Track4_PWMC" / "random_pwmc_1.cnf"
    source_path.parent.mkdir()
    source_path.write_text(
        """\
        p cnf 2 2
        c t pwmc
        c p show 1 0
        c p weight -1 0.25 0
        c p weight 1 0.75 0
        1 2 0
        1 -2 0
        c c s exact arb float 7.5E-01
        """
    )

    def fake_download_mccomp_instance(source_path_name):
        assert source_path_name == "Track4_PWMC/random_pwmc_1.cnf"
        return source_path

    monkeypatch.setattr(
        weighted_model_counting,
        "download_mccomp_instance",
        fake_download_mccomp_instance,
    )
    generator = WeightedModelCountingMCCompGenerator()

    instance = generator.generate(generator.datasets[0])

    # Only x1 = True extends to a model, so the projected count is w(x1).
    assert instance.meta["shown"] == [1]
    assert to_numpy(instance.ref_outputs[0]) == 0.75
    assert instance.ref_meta["reference_count"] == 0.75
    assert instance.meta["source_problem_type"] == "pwmc"
    assert instance.meta["source_exact_type"] == "float"
    assert instance.meta["source_exact_value"] == "7.5E-01"
    assert instance.meta["source_exact_is_projected"]
    np.testing.assert_array_equal(to_numpy(instance.inputs[0]), np.array([0, 1]))
    np.testing.assert_allclose(to_numpy(instance.inputs[1]), np.array([0.25, 0.75]))


def test_parse_mccomp_exact_reads_model_counting_answer():
    assert parse_mccomp_exact("c c s exact arb int 6\n") == ("int", "6")


def test_list_mccomp_instances_filters_by_track():
    assert len(list_mccomp_instances()) == 40
    assert list_mccomp_instances("mc") == [
        f"Track1_MC/random_mc_{index}.cnf" for index in range(1, 11)
    ]
    assert list_mccomp_instances("Track4_PWMC")[0] == "Track4_PWMC/random_pwmc_1.cnf"


def test_fetch_mccomp_instance_accepts_basename(monkeypatch):
    calls = []

    def fake_cached_generate(self, dataset):
        calls.append(dataset.source_path)
        return DataInstance(
            inputs=[],
            meta={"source_path": dataset.source_path},
        )

    monkeypatch.setattr(
        MCCompInstanceGenerator, "cached_generate", fake_cached_generate
    )

    instance = fetch_mccomp_instance("random_mc_2.cnf")

    assert calls == ["Track1_MC/random_mc_2.cnf"]
    assert instance.meta["source_path"] == "Track1_MC/random_mc_2.cnf"


def test_fetch_mccomp_instance_accepts_suffixless_basename(monkeypatch):
    calls = []

    def fake_cached_generate(self, dataset):
        calls.append(dataset.source_path)
        return DataInstance(
            inputs=[],
            meta={"source_path": dataset.source_path},
        )

    monkeypatch.setattr(
        MCCompInstanceGenerator, "cached_generate", fake_cached_generate
    )

    instance = fetch_mccomp_instance("random_mc_2")

    assert calls == ["Track1_MC/random_mc_2.cnf"]
    assert instance.meta["source_path"] == "Track1_MC/random_mc_2.cnf"


def _tar_bytes(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


_POOL_ARCHIVE = "mc2099-track1-mc_collected"
_POOL_CNF = b"c t mc\np cnf 3 2\n1 -3 0\n2 3 -1 0\n"


def test_mccomp_pool_archive_unpacks_nested_set_tars(monkeypatch, tmp_path):
    # Like the 2020-2023 pools: one tar per benchmark set, beside a macOS
    # AppleDouble file that is not a tar.
    archive = _tar_bytes(
        {
            f"{_POOL_ARCHIVE}/Set1.tar": _tar_bytes(
                {"Set1/plain/a.cnf.xz": lzma.compress(_POOL_CNF)}
            ),
            f"{_POOL_ARCHIVE}/._Set1.tar": b"\x00\x05\x16\x07AppleDouble",
        }
    )
    md5 = hashlib.md5(archive, usedforsecurity=False).hexdigest()
    monkeypatch.setitem(mccomp.MCCOMP_POOL_ARCHIVES, _POOL_ARCHIVE, ("1", md5))
    calls = []

    def fake_urlretrieve(url, path):
        calls.append(url)
        Path(path).write_bytes(archive)

    monkeypatch.setattr(urllib.request, "urlretrieve", fake_urlretrieve)
    source_path = f"{_POOL_ARCHIVE}/Set1/plain/a.cnf.xz"

    path = download_mccomp_pool_instance(source_path, data_dir=tmp_path)

    assert path == tmp_path / "zenodo" / source_path
    assert lzma.decompress(path.read_bytes()) == _POOL_CNF
    assert not (tmp_path / "zenodo" / _POOL_ARCHIVE / "Set1.tar").exists()
    # Unpacked once, the archive is not downloaded again.
    assert download_mccomp_pool_instance(source_path, data_dir=tmp_path) == path
    assert calls == [
        f"https://zenodo.org/api/records/1/files/{_POOL_ARCHIVE}.tar/content"
    ]


def test_mccomp_pool_archive_rejects_wrong_md5(monkeypatch, tmp_path):
    monkeypatch.setitem(mccomp.MCCOMP_POOL_ARCHIVES, _POOL_ARCHIVE, ("1", "0" * 32))
    monkeypatch.setattr(
        urllib.request, "urlretrieve", lambda url, path: Path(path).write_bytes(b"x")
    )

    with pytest.raises(ValueError, match="does not match Zenodo's MD5"):
        download_mccomp_pool_archive(_POOL_ARCHIVE, data_dir=tmp_path)

    assert not (tmp_path / "zenodo" / f"{_POOL_ARCHIVE}.tar").exists()
    assert not (tmp_path / "zenodo" / f"{_POOL_ARCHIVE}.tar.tmp").exists()


def test_mccomp_pool_dataset_names_follow_source_paths():
    datasets = MCCompPoolInstanceGenerator().datasets
    names = [dataset.name for dataset in datasets]
    assert len(set(names)) == len(names)
    assert not [name for name in names if "." in name]
    assert {dataset.source_path.split("/", 1)[0] for dataset in datasets} <= set(
        MCCOMP_POOL_ARCHIVES
    )
    assert (
        MCCompPoolDataset(
            "mc2021-track1-mc_collected/MC2021_Track1_Set3_satlib_Hoos/cnf/uf/"
            "uf20-0661.cnf.xz"
        ).name
        == "MC2021_Track1_Set3_satlib_Hoos/cnf/uf/uf20-0661"
    )
    assert (
        MCCompPoolDataset("mc2020-track1-mc_competition/mc2020_track1_000.cnf.xz").name
        == "mc2020_track1_000"
    )
    assert (
        MCCompPoolDataset(
            "mc2020-track1-mc_collected/MC2020_Track1_Set1_VLSAT_Garavel-Bouvier_"
            "2020-03-10/plain/vlsat_18_21.cnf.bpe.xz"
        ).name
        == "MC2020_Track1_Set1_VLSAT_Garavel-Bouvier_2020-03-10/plain/"
        "vlsat_18_21_cnf_bpe"
    )
    assert (
        MCCompPoolDataset(
            "mc2024-track4-pwmc_competition/mc2024_track4_182.cnf.xz"
        ).name
        == "mc2024_track4_182"
    )


def test_mccomp_pool_generators_build_formula_and_count(monkeypatch, tmp_path):
    path = tmp_path / "a.cnf.xz"
    path.write_bytes(lzma.compress(_POOL_CNF))
    monkeypatch.setattr(
        mccomp_pool_instance, "download_mccomp_pool_instance", lambda _: path
    )
    dataset = MCCompPoolInstanceGenerator().datasets[0]

    raw = MCCompPoolInstanceGenerator().generate(dataset)

    assert to_numpy(raw.inputs[0]).tobytes() == _POOL_CNF

    monkeypatch.setattr(model_counting, "fetch_mccomp_pool_instance", lambda _: raw)
    reference = ModelCountingMCCompPoolReferenceGenerator().generate(
        ReferenceDataset(dataset)
    )
    assert reference.ref_meta == {"reference_count": 5}

    monkeypatch.setattr(
        ModelCountingMCCompPoolReferenceGenerator,
        "cached_generate",
        lambda self, reference_dataset: reference,
    )
    problem = ModelCountingMCCompPoolGenerator().generate(dataset)

    assert problem.meta["expr"] == (
        "s[] += (B[v1] or not B[v3]) and (B[v2] or B[v3] or not B[v1])"
    )
    assert problem.meta["expected_result"] == 5
    assert to_numpy(problem.ref_outputs[0]) == 5


def test_mccomp_pool_generators_handle_a_formula_without_clauses(monkeypatch, tmp_path):
    path = tmp_path / "a.cnf.xz"
    path.write_bytes(lzma.compress(b"p cnf 2 0\n"))
    monkeypatch.setattr(
        mccomp_pool_instance, "download_mccomp_pool_instance", lambda _: path
    )
    dataset = MCCompPoolInstanceGenerator().datasets[0]
    raw = MCCompPoolInstanceGenerator().generate(dataset)
    monkeypatch.setattr(model_counting, "fetch_mccomp_pool_instance", lambda _: raw)

    reference = ModelCountingMCCompPoolReferenceGenerator().generate(
        ReferenceDataset(dataset)
    )
    problem = ModelCountingMCCompPoolGenerator().generate_inputs(dataset)

    assert reference.ref_meta == {"reference_count": 4}
    assert problem.meta["expr"] is None
    assert problem.meta["default_total"] == 4


def test_weighted_mccomp_pool_generators_count_the_projection(monkeypatch, tmp_path):
    # Of x1 and x2, the shown variables, (x1 or x3) and (not x3 or x2) let every
    # assignment but x1 = x2 = False extend to a model, so the projected count is
    # 1 - w(-x1) * w(-x2) = 1 - 0.75 * 0.4. x3 has no weights, so they're 1.
    path = tmp_path / "a.cnf.xz"
    path.write_bytes(
        lzma.compress(
            b"c t pwmc\np cnf 3 2\nc p show 1 2 0\nc p weight 1 0.25 0\n"
            b"c p weight -2 0.4 0\n1 3 0\n-3 2 0\n"
        )
    )
    monkeypatch.setattr(
        mccomp_pool_instance, "download_mccomp_pool_instance", lambda _: path
    )
    dataset = MCCOMP_POOL_WEIGHTED_INSTANCES[0]
    raw = MCCompPoolInstanceGenerator().generate(dataset)
    monkeypatch.setattr(
        weighted_model_counting, "fetch_mccomp_pool_instance", lambda _: raw
    )

    reference = WeightedModelCountingMCCompPoolReferenceGenerator().generate(
        ReferenceDataset(dataset)
    )
    assert np.isclose(reference.ref_meta["reference_count"], 0.7, rtol=0, atol=1e-15)

    monkeypatch.setattr(
        WeightedModelCountingMCCompPoolReferenceGenerator,
        "cached_generate",
        lambda self, reference_dataset: reference,
    )
    generator = WeightedModelCountingMCCompPoolGenerator()
    problem = generator.generate(dataset)
    assert problem.meta["shown"] == [1, 2]
    function = generator.generate_benchmark_function(dataset, problem, None)
    xp = NumpyFramework()
    result = function(xp, problem.meta, *map(xp.from_binsparse, problem.inputs))
    assert np.isclose(
        float(result),
        problem.ref_meta["reference_count"],
        rtol=0,
        atol=problem.ref_meta["error_bound"],
    )


@pytest.mark.parametrize(
    ("generator", "traced"),
    [
        (
            ModelCountingMCCompPoolGenerator(),
            "MC2024_Track1_Set6_Pehousek_2024-02-16/plain/m3_d5_n24_jdp_0",
        ),
        (WeightedModelCountingMCCompPoolGenerator(), "mc2024_track4_182"),
    ],
    ids=lambda value: getattr(value, "name", value),
)
def test_mccomp_pool_traces_and_trains_one_instance(generator, traced):
    assert {
        dataset.name: dataset.suites for dataset in generator.datasets if dataset.suites
    } == {traced: ["trace", "train"]}
