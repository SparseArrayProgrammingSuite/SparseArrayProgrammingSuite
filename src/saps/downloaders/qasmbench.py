"""Downloader utilities for QASMBench quantum circuits."""

from __future__ import annotations

import math
import re
import urllib.request
from pathlib import Path

from saps.downloaders.cache import download_lock, source_cache_dir

QASMBENCH_REPOSITORY_URL = "https://github.com/pnnl/QASMBench"
# Circuits are fetched from a pinned commit, so their contents never change.
QASMBENCH_COMMIT = "357b942396d5c2b7cbc1c229c585a6ef5ccaebac"
QASMBENCH_RAW_BASE_URL = (
    f"https://raw.githubusercontent.com/pnnl/QASMBench/{QASMBENCH_COMMIT}"
)

# Each listed circuit with its QASMBench size class (the directory it lives in).
# These keep their statevectors sparse in floating point. Others, such as
# bigadder_n18 and seca_n11, are sparse in exact arithmetic, but rounding in
# their transpiled rz-sx-rz gates leaves residues that fill the state.
QASMBENCH_CIRCUITS = {
    "cat_state_n22": "medium",
    "ghz_state_n23": "medium",
    "multiplier_n15": "medium",
    "multiply_n13": "medium",
    "qec9xz_n17": "medium",
    "qram_n20": "medium",
}

# Gates of the transpiled circuits: IBM's native basis.
_ONE_QUBIT_GATES = {"sx", "x"}
_GATE_RE = re.compile(
    r"^(?P<name>[a-z]+)(?:\((?P<angle>[^)]*)\))?\s+(?P<qubits>[^;]+);$"
)
_QUBIT_RE = re.compile(r"^(?P<register>\w+)\[(?P<index>\d+)\]$")
# Angles are arithmetic on decimal numbers and pi, such as "-3*pi/4".
_ANGLE_RE = re.compile(r"^(?:pi|\d+(?:\.\d*)?(?:e[-+]?\d+)?|\.\d+|[-+*/()\s])+$")


def parse_transpiled_qasm(text: str) -> tuple[int, list[tuple]]:
    """Parse a transpiled QASMBench circuit into (nqubits, gates).

    Transpiled circuits use only rz, sx, x, and cx. Quantum registers are laid
    out in declaration order, so the second register's qubits follow the first's.
    Each gate is ("rz", (qubit,), angle), ("sx", (qubit,), None),
    ("x", (qubit,), None), or ("cx", (control, target), None). Measurements and
    barriers end or annotate the circuit and are skipped.
    """
    offsets: dict[str, tuple[int, int]] = {}
    nqubits = 0
    gates: list[tuple] = []
    for raw_line in text.splitlines():
        line = raw_line.split("//", 1)[0].strip()
        if not line or line.startswith(("OPENQASM", "include", "creg", "barrier")):
            continue
        if line.startswith("measure"):
            continue
        if line.startswith("qreg"):
            match = re.fullmatch(r"qreg\s+(\w+)\[(\d+)\];", line)
            if match is None or match.group(1) in offsets:
                raise ValueError(f"Malformed qreg: {line!r}")
            offsets[match.group(1)] = (nqubits, int(match.group(2)))
            nqubits += int(match.group(2))
            continue
        match = _GATE_RE.match(line)
        if match is None:
            raise ValueError(f"Unsupported QASM statement: {line!r}")
        name, angle_text = match.group("name"), match.group("angle")
        qubits = []
        for operand in match.group("qubits").split(","):
            qubit = _QUBIT_RE.match(operand.strip())
            if qubit is None or qubit.group("register") not in offsets:
                raise ValueError(f"Unsupported operand in {line!r}")
            offset, size = offsets[qubit.group("register")]
            index = int(qubit.group("index"))
            if index >= size:
                raise ValueError(f"Qubit out of range in {line!r}")
            qubits.append(offset + index)
        if name == "rz" and angle_text is not None and len(qubits) == 1:
            gates.append(("rz", (qubits[0],), _parse_angle(angle_text)))
        elif name in _ONE_QUBIT_GATES and angle_text is None and len(qubits) == 1:
            gates.append((name, (qubits[0],), None))
        elif name == "cx" and angle_text is None and len(qubits) == 2:
            gates.append(("cx", (qubits[0], qubits[1]), None))
        else:
            raise ValueError(f"Unsupported QASM gate: {line!r}")
    if not offsets:
        raise ValueError("QASM circuit declares no quantum register")
    return nqubits, gates


def _parse_angle(text: str) -> float:
    if not _ANGLE_RE.match(text):
        raise ValueError(f"Unsupported angle expression: {text!r}")
    return float(eval(text, {"__builtins__": {}}, {"pi": math.pi}))  # noqa: S307


def qasmbench_source_path(name: str) -> str:
    """The circuit's transpiled QASM path inside the QASMBench repository."""
    if name not in QASMBENCH_CIRCUITS:
        raise ValueError(f"Unknown QASMBench circuit: {name!r}")
    return f"{QASMBENCH_CIRCUITS[name]}/{name}/{name}_transpiled.qasm"


def qasmbench_source_url(name: str) -> str:
    return f"{QASMBENCH_REPOSITORY_URL}/blob/{QASMBENCH_COMMIT}/" + (
        qasmbench_source_path(name)
    )


def qasmbench_raw_url(name: str) -> str:
    return f"{QASMBENCH_RAW_BASE_URL}/{qasmbench_source_path(name)}"


def download_qasmbench_circuit(
    name: str, *, data_dir: str | Path | None = None
) -> Path:
    """Download one transpiled QASMBench circuit and return its local path."""
    root = Path(data_dir) if data_dir is not None else source_cache_dir("qasmbench")
    dest_path = root / qasmbench_source_path(name)
    with download_lock(dest_path):
        if dest_path.exists():
            return dest_path
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = dest_path.with_name(dest_path.name + ".tmp")
        try:
            urllib.request.urlretrieve(qasmbench_raw_url(name), tmp_path)  # noqa: S310
            tmp_path.replace(dest_path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
        return dest_path
