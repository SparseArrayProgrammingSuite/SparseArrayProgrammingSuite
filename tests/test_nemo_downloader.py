from __future__ import annotations

import gzip

import numpy as np

from binsparse import BinsparseTensor
from binsparse.conversions import from_numpy, to_numpy

import saps.benchmarks.particle_sim as ps
from frameworks.saps_numpy import NumpyFramework
from saps.downloaders.nemo import download_nemo_dataset, parse_nemo_snapshot


def test_parse_nemo_row_table(tmp_path):
    path = tmp_path / "plummer.dat.gz"
    with gzip.open(path, "wt", encoding="utf-8") as file:
        file.write("0.5 1.0 2.0 3.0 0.1 0.2 0.3\n0.5 4.0 6.0 8.0 0.4 0.6 0.8\n")

    x, y, z, vx, vy, vz = parse_nemo_snapshot(
        path,
        columns=("mass", "x", "y", "z", "vx", "vy", "vz"),
        expected_particles=2,
    )

    np.testing.assert_allclose(x, [1.0, 4.0])
    np.testing.assert_allclose(y, [2.0, 6.0])
    np.testing.assert_allclose(z, [3.0, 8.0])
    np.testing.assert_allclose(vx, [0.1, 0.4])
    np.testing.assert_allclose(vy, [0.2, 0.6])
    np.testing.assert_allclose(vz, [0.3, 0.8])


def test_parse_nemo_wrapped_phase_space_stream(tmp_path):
    path = tmp_path / "stars.dat"
    path.write_text(
        "3.0 2.0\n1.0 0.3 0.2 0.1\n8.0 7.0 6.0\n0.8 0.7 0.6\n",
        encoding="utf-8",
    )

    x, y, z, vx, vy, vz = parse_nemo_snapshot(
        path,
        columns=("z", "y", "x", "vz", "vy", "vx"),
        wrap=True,
        expected_particles=2,
    )

    np.testing.assert_allclose(x, [1.0, 6.0])
    np.testing.assert_allclose(y, [2.0, 7.0])
    np.testing.assert_allclose(z, [3.0, 8.0])
    np.testing.assert_allclose(vx, [0.1, 0.6])
    np.testing.assert_allclose(vy, [0.2, 0.7])
    np.testing.assert_allclose(vz, [0.3, 0.8])


def test_parse_nemo_rejects_partial_snapshot(tmp_path):
    path = tmp_path / "partial.dat"
    path.write_text("1 2 3 4 5 6 7\n", encoding="utf-8")

    try:
        parse_nemo_snapshot(
            path,
            columns=("mass", "x", "y", "z", "vx", "vy", "vz"),
            expected_particles=2,
        )
    except ValueError as exc:
        assert "Expected 2 particles" in str(exc)
    else:
        raise AssertionError("partial NEMO snapshot should fail row-count check")


def test_download_nemo_dataset_preserves_archive_scale(tmp_path):
    source = tmp_path / "source.dat"
    source.write_text(
        "0.5 1.0 2.0 3.0 0.1 0.2 0.3\n0.5 4.0 6.0 8.0 0.4 0.6 0.8\n",
        encoding="utf-8",
    )
    archive = tmp_path / "plummer" / "tab2.gz"
    archive.parent.mkdir()
    with gzip.open(archive, "wb") as output_file:
        output_file.write(source.read_bytes())

    bins, meta = download_nemo_dataset(
        "plummer/tab2.gz",
        columns=("mass", "x", "y", "z", "vx", "vy", "vz"),
        expected_particles=2,
        data_dir=tmp_path,
        num_steps=7,
    )

    assert all(isinstance(item, BinsparseTensor) for item in bins)
    assert meta["n_particles"] == 2
    assert meta["steps"] == 7
    assert meta["size"] == 5.0
    assert "position_scale" not in meta

    x = to_numpy(bins[0])
    y = to_numpy(bins[1])
    z = to_numpy(bins[2])
    vx = to_numpy(bins[3])
    vy = to_numpy(bins[4])
    vz = to_numpy(bins[5])
    np.testing.assert_allclose(x, [0.0, 3.0])
    np.testing.assert_allclose(y, [0.0, 4.0])
    np.testing.assert_allclose(z, [0.0, 5.0])
    np.testing.assert_allclose(vx, [0.1, 0.4])
    np.testing.assert_allclose(vy, [0.2, 0.6])
    np.testing.assert_allclose(vz, [0.3, 0.8])


def test_download_nemo_dataset_can_include_particle_mass(tmp_path):
    source = tmp_path / "source.dat"
    source.write_text(
        "0.25 1.0 2.0 3.0 0.1 0.2 0.3\n0.75 4.0 6.0 8.0 0.4 0.6 0.8\n",
        encoding="utf-8",
    )
    archive = tmp_path / "plummer" / "tab2.gz"
    archive.parent.mkdir()
    with gzip.open(archive, "wb") as output_file:
        output_file.write(source.read_bytes())

    bins, meta = download_nemo_dataset(
        "plummer/tab2.gz",
        columns=("mass", "x", "y", "z", "vx", "vy", "vz"),
        expected_particles=2,
        data_dir=tmp_path,
        include_mass=True,
    )

    assert len(bins) == 7
    np.testing.assert_allclose(to_numpy(bins[6]), [0.25, 0.75])
    assert meta["source_mass_min"] == 0.25
    assert meta["source_mass_max"] == 0.75
    assert meta["source_mass_sum"] == 1.0


def test_particle_sim_real_generator_uses_nemo(monkeypatch):
    calls = []

    def fake_download(path, **kwargs):
        calls.append((path, kwargs))
        n = kwargs["expected_particles"]
        values = np.arange(n, dtype=np.float64)
        inputs = [
            from_numpy(values),
            from_numpy(values),
            from_numpy(values),
            from_numpy(values),
            from_numpy(values),
            from_numpy(values),
        ]
        if kwargs["include_mass"]:
            inputs.append(from_numpy(np.ones(n, dtype=np.float64) / n))
        return inputs, {"size": 1.0, "steps": kwargs["num_steps"], "n_particles": n}

    monkeypatch.setattr(ps, "download_nemo_dataset", fake_download)

    generator = ps.ParticleSimGenerator()
    datasets = generator.datasets

    assert {dataset.name for dataset in datasets} == {
        "nemo_plummer_128",
        "nemo_plummer_1024",
        "nemo_dubinski_m31",
    }
    assert all(dataset.suites == ["standard"] for dataset in datasets)
    assert all(dataset.n_particles > 0 for dataset in datasets)
    for dataset in datasets:
        assert dataset.parameters["force_model"] == "newtonian_gravity"
        assert dataset.parameters["boundary_model"] == "unbounded"
        assert dataset.parameters["gravitational_constant"] == 1.0
        assert dataset.parameters["dt"] == 1.0 / 32.0
        assert dataset.parameters["softening"] == 0.05
        assert dataset.parameters["cutoff"] > 0.0
        assert "particle_mass" not in dataset.parameters
        assert "mass" not in dataset.parameters
    assert {dataset.name: dataset.parameters["cutoff"] for dataset in datasets} == {
        "nemo_plummer_128": 1.0,
        "nemo_plummer_1024": 1.0,
        "nemo_dubinski_m31": 20.0,
    }
    assert all("mass" in dataset.source_columns for dataset in datasets)

    instance = generator.generate(datasets[0])
    assert len(instance.inputs) == 7
    assert instance.meta["n_particles"] == 128
    assert instance.meta["parameters"] == datasets[0].parameters
    assert instance.meta["parameters"]["gravitational_constant"] == 1.0
    assert calls[0][0] == "plummer/tab128.gz"
    assert calls[0][1]["columns"] == ("mass", "x", "y", "z", "vx", "vy", "vz")
    assert calls[0][1]["include_mass"] is True


def test_particle_sim_benchmark_runs_newtonian_gravity_with_particle_masses():
    xp = NumpyFramework()
    data = [
        np.array([0.0, 1.0]),
        np.array([0.0, 0.0]),
        np.array([0.0, 0.0]),
        np.array([0.0, 0.0]),
        np.array([0.0, 0.0]),
        np.array([0.0, 0.0]),
        np.array([0.25, 0.75]),
    ]
    meta = {
        "size": 1.0,
        "steps": 1,
        "parameters": {
            "force_model": "newtonian_gravity",
            "boundary_model": "unbounded",
            "cutoff": 2.0,
            "dt": 1.0 / 32.0,
            "softening": 0.05,
            "gravitational_constant": 1.0,
        },
    }

    result = ps.ParticleSimBenchmark().benchmark(xp, data, meta)

    assert len(result) == 6
    assert result[3][0] > 0.0
    assert result[3][1] < 0.0
