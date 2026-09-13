from __future__ import annotations

import pytest

import numpy as np

from binsparse.conversions import to_numpy

from saps.benchmark import DataInstance
from saps.benchmarks import weighted_model_counting
from saps.benchmarks.model_counting import (
    MCCompGenerator,
    MCCompMCGenerator,
    fetch_mccomp_instance,
    parse_mccomp_exact,
)
from saps.benchmarks.model_counting import (
    clauses_to_einsum as mc_clauses_to_einsum,
)
from saps.benchmarks.weighted_model_counting import (
    MCCompPWMCGenerator,
    parse_format,
)
from saps.benchmarks.weighted_model_counting import (
    clauses_to_einsum as wmc_clauses_to_einsum,
)
from saps.downloaders.mccomp import (
    download_mccomp_instance,
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
    assert wmc_clauses_to_einsum([[]], 2) == "s[] += False"


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
            meta={
                "source_path": dataset.source_path,
                "local_path": str(source_path),
            },
        )

    monkeypatch.setattr(MCCompGenerator, "cached_generate", fake_cached_generate)
    generator = MCCompMCGenerator()

    instance = generator.generate(generator.datasets[0])

    assert calls == [("mccomp", "Track1_MC/random_mc_1.cnf")]
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
        p cnf 2 1
        c t pwmc
        c p show 1 0
        c p weight -1 0.25 0
        c p weight 1 0.75 0
        1 2 0
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
    generator = MCCompPWMCGenerator()

    instance = generator.generate(generator.datasets[0])

    assert instance.ref_outputs is None
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
            meta={
                "source_path": dataset.source_path,
                "local_path": "/tmp/source.cnf",
            },
        )

    monkeypatch.setattr(MCCompGenerator, "cached_generate", fake_cached_generate)

    instance = fetch_mccomp_instance("random_mc_2.cnf")

    assert calls == ["Track1_MC/random_mc_2.cnf"]
    assert instance.meta["source_path"] == "Track1_MC/random_mc_2.cnf"


def test_fetch_mccomp_instance_accepts_suffixless_basename(monkeypatch):
    calls = []

    def fake_cached_generate(self, dataset):
        calls.append(dataset.source_path)
        return DataInstance(
            inputs=[],
            meta={
                "source_path": dataset.source_path,
                "local_path": "/tmp/source.cnf",
            },
        )

    monkeypatch.setattr(MCCompGenerator, "cached_generate", fake_cached_generate)

    instance = fetch_mccomp_instance("random_mc_2")

    assert calls == ["Track1_MC/random_mc_2.cnf"]
    assert instance.meta["source_path"] == "Track1_MC/random_mc_2.cnf"
