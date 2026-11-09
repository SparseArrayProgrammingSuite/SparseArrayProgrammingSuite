from __future__ import annotations

import math
import urllib.request

import pytest

from saps.benchmarks import qasmbench
from saps.downloaders.qasmbench import (
    QASMBENCH_CIRCUITS,
    QASMBENCH_COMMIT,
    download_qasmbench_circuit,
    parse_transpiled_qasm,
    qasmbench_raw_url,
    qasmbench_source_path,
    qasmbench_source_url,
)

CIRCUIT = """\
OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];
rz(-3*pi/4) q[0];
sx q[1];
x q[2];
cx q[0],q[2];
barrier q[0],q[1],q[2];
measure q[0] -> c[0];
"""


def test_parser_reads_the_transpiled_gate_set():
    nqubits, gates = parse_transpiled_qasm(CIRCUIT)

    assert nqubits == 3
    assert gates == [
        ("rz", (0,), -3 * math.pi / 4),
        ("sx", (1,), None),
        ("x", (2,), None),
        ("cx", (0, 2), None),
    ]


def test_parser_lays_registers_out_in_declaration_order():
    nqubits, gates = parse_transpiled_qasm(
        "qreg carry[2];\nqreg a[3];\ncx carry[1],a[2];\n"
    )

    assert nqubits == 5
    assert gates == [("cx", (1, 4), None)]


@pytest.mark.parametrize(
    "line",
    [
        "h q[0];",  # Not in the transpiled basis.
        "rz(__import__('os')) q[0];",
        "rz(pi*theta) q[0];",
        "x q[3];",
        "x r[0];",
        "cx q[0];",
    ],
)
def test_parser_rejects_unsupported_statements(line):
    with pytest.raises(ValueError):
        parse_transpiled_qasm(f"qreg q[3];\n{line}\n")


def test_urls_pin_the_commit():
    path = qasmbench_source_path("ghz_state_n23")

    assert path == "medium/ghz_state_n23/ghz_state_n23_transpiled.qasm"
    assert qasmbench_raw_url("ghz_state_n23").endswith(f"{QASMBENCH_COMMIT}/{path}")
    assert f"/blob/{QASMBENCH_COMMIT}/" in qasmbench_source_url("ghz_state_n23")
    with pytest.raises(ValueError, match="Unknown QASMBench circuit"):
        qasmbench_source_path("not_a_circuit")


def test_download_caches_one_file(monkeypatch, tmp_path):
    urls = []

    def fake_urlretrieve(url, destination):
        urls.append(url)
        destination.write_text(CIRCUIT, encoding="utf-8")

    monkeypatch.setattr(urllib.request, "urlretrieve", fake_urlretrieve)
    first = download_qasmbench_circuit("qram_n20", data_dir=tmp_path)
    second = download_qasmbench_circuit("qram_n20", data_dir=tmp_path)

    assert first == second == tmp_path / qasmbench_source_path("qram_n20")
    assert urls == [qasmbench_raw_url("qram_n20")]
    assert first.read_text(encoding="utf-8") == CIRCUIT


def test_shell_generator_stores_the_parsed_circuit(monkeypatch, tmp_path):
    path = tmp_path / "circuit.qasm"
    path.write_text(CIRCUIT, encoding="utf-8")
    monkeypatch.setattr(qasmbench, "download_qasmbench_circuit", lambda name: path)
    generator = qasmbench.QASMBenchCircuitGenerator()

    assert [d.name for d in generator.datasets] == list(QASMBENCH_CIRCUITS)
    meta = generator.generate(qasmbench.qasmbench_circuit("qram_n20")).meta

    assert meta["nqubits"] == 3
    assert meta["gates"][3] == ["cx", [0, 2], None]
    assert meta["source_url"] == qasmbench_source_url("qram_n20")
