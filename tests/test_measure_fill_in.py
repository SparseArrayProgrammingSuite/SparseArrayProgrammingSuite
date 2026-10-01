import importlib.util
import json
from pathlib import Path

import pytest

import numpy as np
import scipy.sparse as sps

import h5py


@pytest.fixture
def fill_in():
    path = Path(__file__).parents[1] / "scripts/measure_fill_in.py"
    spec = importlib.util.spec_from_file_location("measure_fill_in", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def degrees_from_dense(module, matrix, chunk=1_000_000):
    coo = sps.coo_array(matrix)
    coo.sum_duplicates()  # Also sorts by row, then column.
    return module.row_degrees(len(matrix), coo.row, coo.col, coo.data, chunk=chunk)


def reference_counts(matrix, squarings):
    """Nonzeros of (I | A)**(2**s) for s = 0, ..., squarings, by dense squaring."""
    reachability = (matrix != 0) | np.eye(len(matrix), dtype=bool)
    counts = []
    for _ in range(squarings + 1):
        counts.append(int(np.count_nonzero(reachability)))
        reachability = (reachability.astype(int) @ reachability.astype(int)) > 0
    return counts


def dense_graph(kind):
    n = 7
    matrix = np.zeros((n, n), dtype=int)
    if kind in ("chain", "cycle"):
        matrix[np.arange(n - 1), np.arange(1, n)] = 1
        if kind == "cycle":
            matrix[n - 1, 0] = 1
    elif kind == "star":
        matrix[0, 1:] = matrix[1:, 0] = 1
    elif kind == "full":
        matrix[:] = 1
    elif kind == "random":
        matrix = (np.random.default_rng(0).random((40, 40)) < 0.04).astype(int)
    return matrix


@pytest.mark.parametrize("target", [0.05, 0.3, 1.0])
@pytest.mark.parametrize(
    "kind", ["isolated", "chain", "cycle", "star", "full", "random"]
)
def test_bounds_hold_for_boolean_squaring(fill_in, kind, target):
    matrix = dense_graph(kind)
    n = len(matrix)
    result = fill_in.bound_fill_in(
        n, degrees_from_dense(fill_in, matrix), target_density=target
    )
    stages = result["stages"]
    actual = reference_counts(matrix, len(stages) + 5)
    assert stages[0]["nnz_bound"] == result["nnz"] == actual[0]
    assert stages[0]["growth_factor_bound"] == 1
    for stage in stages:
        assert stage["nnz_bound"] >= actual[stage["squarings"]]
        assert stage["max_path_length"] == 2 ** stage["squarings"]
    for stage in stages[:-1]:
        assert stage["density_bound"] < target
    # The reported step count never exceeds the true first squaring at target.
    reached = [s for s, count in enumerate(actual) if count >= target * n * n]
    if result["squarings"] is None:
        assert result["stop_reason"] == "never"
        assert not reached
    else:
        assert result["stop_reason"] == "target"
        assert result["squarings"] == len(stages) - 1
        assert not reached or result["squarings"] <= reached[0]


def test_chain_bound_uses_row_degrees_and_max_degree(fill_in):
    degree = degrees_from_dense(fill_in, dense_graph("chain"))
    assert degree.tolist() == [1, 1, 1, 1, 1, 1, 0]
    result = fill_in.bound_fill_in(7, degree, target_density=0.5)
    assert result["max_degree"] == 1
    assert [stage["nnz_bound"] for stage in result["stages"]] == [13, 19, 31]
    assert [stage["row_nnz_bound"] for stage in result["stages"]] == [2, 3, 5]
    assert result["squarings"] == 2
    assert result["stages"][-1]["growth_factor_bound"] == 31 / 13


def test_bound_that_stops_growing_below_target_never_reaches_it(fill_in):
    result = fill_in.bound_fill_in(
        200, np.zeros(200, dtype=np.int64), target_density=0.01
    )
    assert result["squarings"] is None
    assert result["stop_reason"] == "never"
    assert [stage["nnz_bound"] for stage in result["stages"]] == [200]


def test_empty_matrix_never_reaches_target(fill_in):
    result = fill_in.bound_fill_in(0, np.zeros(0, dtype=np.int64))
    assert result["stop_reason"] == "never"
    assert result["nnz"] == 0


def test_stored_zeros_and_diagonal_are_not_edges(fill_in):
    degree = fill_in.row_degrees(
        3,
        np.array([0, 0, 0, 1]),
        np.array([0, 1, 2, 2]),
        np.array([1, 0, -2, 0]),
    )
    assert degree.tolist() == [1, 0, 0]
    assert fill_in.bound_fill_in(3, degree, target_density=1.0)["nnz"] == 4


def test_degrees_cross_chunk_boundaries(fill_in):
    matrix = dense_graph("random")
    expected = degrees_from_dense(fill_in, matrix)
    assert degrees_from_dense(fill_in, matrix, chunk=3).tolist() == expected.tolist()


@pytest.mark.parametrize(
    "rows,columns",
    [
        ((1, 0), (2, 1)),  # rows out of order
        ((0, 0), (2, 1)),  # columns out of order within a row
        ((0, 0), (1, 1)),  # duplicate coordinate
        ((-1, 1), (1, 2)),
        ((0, 1), (1, 3)),
    ],
)
@pytest.mark.parametrize("chunk", [1, 1_000_000])
def test_rejects_invalid_coordinates(fill_in, rows, columns, chunk):
    with pytest.raises(ValueError):
        fill_in.row_degrees(
            3, np.array(rows), np.array(columns), np.ones(2), chunk=chunk
        )


def write_coor(path, *, chunked=False, rows=(0, 1), columns=(1, 2)):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as file:
        tensor = file.create_group("inputs/0")
        tensor.attrs["binsparse"] = json.dumps(
            {"binsparse": {"format": "COOR", "shape": [3, 3]}}
        )
        options = {"chunks": True, "compression": "gzip"} if chunked else {}
        for name, values in (
            ("indices_0", rows),
            ("indices_1", columns),
            ("values", [1, 1]),
        ):
            tensor.create_dataset(name, data=np.array(values), **options)


@pytest.mark.parametrize("chunked", [False, True])
def test_reads_prepared_coor_without_materializing_matrix(fill_in, tmp_path, chunked):
    path = tmp_path / "graph.bsp.h5"
    write_coor(path, chunked=chunked)
    with fill_in.open_matrix(path) as (n, rows, columns, values):
        assert isinstance(rows, h5py.Dataset if chunked else np.memmap)
        degree = fill_in.row_degrees(n, rows, columns, values)
    assert degree.tolist() == [1, 1, 0]


@pytest.fixture
def cache(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "suitesparse_matrix.GAP/GAP-road": {"digest": "road"},
                "suitesparse_matrix.GAP/GAP-web": {"digest": "web"},
                "suitesparse_matrix.SNAP/ca-GrQc": {"digest": "grqc"},
                "suitesparse_matrix.HB/bcsstk01": {"digest": "bcsstk01"},
            }
        )
    )
    write_coor(tmp_path / "suitesparse_matrix/GAP/GAP-road/road.bsp.h5")
    write_coor(
        tmp_path / "suitesparse_matrix/SNAP/ca-GrQc/grqc.bsp.h5",
        rows=(0, 1),
        columns=(1, 0),
    )
    return ["--manifest", str(manifest), "--cache-dir", str(tmp_path)]


def test_default_datasets_are_gap_and_snap_groups(fill_in, cache, capsys):
    assert fill_in.main([*cache, "--list-datasets"]) == 0
    assert capsys.readouterr().out.split() == [
        "GAP/GAP-road",
        "GAP/GAP-web",
        "SNAP/ca-GrQc",
    ]


def test_cli_writes_steps_and_reports_missing_matrices(fill_in, cache, tmp_path):
    output = tmp_path / "results.json"
    code = fill_in.main([*cache, "--output", str(output), "--target-density", "0.6"])
    result = json.loads(output.read_text())
    assert code == 1
    # Both graphs have row degrees [1, 1, 0]: 5 nonzeros, then 3 + 3 + 1.
    assert [
        (entry["dataset"], entry["squarings"], entry["stop_reason"])
        for entry in result["steps"]
    ] == [
        ("GAP/GAP-road", 1, "target"),
        ("GAP/GAP-web", None, "missing"),
        ("SNAP/ca-GrQc", 1, "target"),
    ]
    assert result["matrices"][0]["stages"][-1]["nnz_bound"] == 7
    assert result["matrices"][1]["status"] == "missing"


def test_merge_combines_per_task_results(fill_in, cache, tmp_path):
    parts = []
    for name in ("SNAP/ca-GrQc", "GAP/GAP-road"):
        parts.append(tmp_path / f"{name.replace('/', '_')}.json")
        assert (
            fill_in.main([*cache, "--datasets", name, "--output", str(parts[-1])]) == 0
        )
    output = tmp_path / "merged.json"
    assert fill_in.main(["--merge", *map(str, parts), "--output", str(output)]) == 0
    result = json.loads(output.read_text())
    assert [entry["dataset"] for entry in result["steps"]] == [
        "SNAP/ca-GrQc",
        "GAP/GAP-road",
    ]
    assert [entry["squarings"] for entry in result["steps"]] == [0, 0]
