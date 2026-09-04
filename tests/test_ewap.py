import pytest

import numpy as np

from binsparse import BinsparseTensor
from binsparse.conversions import from_numpy

import saps.benchmarks.particle_sim as ps
from frameworks.saps_numpy import NumpyFramework
from saps.downloaders.ewap import load_toy_ewap_dataset


def _toy_parameters():
    return {
        "force_model": "cs267_repulsive",
        "boundary_model": "reflective_box",
        "cutoff": 0.01,
        "softening": 0.0001,
        "dt": 0.0005,
        "gravitational_constant": 1.0,
    }


def test_toy_ewap_dataset_shape():
    """Parser returns 4 particles (one per unique pedestrian in the toy data)."""
    bins, meta = load_toy_ewap_dataset(num_steps=10)
    assert len(bins) == 7  # x, y, z, vx, vy, vz, mass
    n = meta["n_particles"]
    assert n == 4
    for b in bins:
        assert isinstance(b, BinsparseTensor)


def test_toy_ewap_positions_preserve_dataset_coordinates():
    """Positions are loaded from the EWAP columns without rescaling."""
    xp = NumpyFramework()
    bins, meta = load_toy_ewap_dataset()
    x = xp.from_binsparse(bins[0])
    y = xp.from_binsparse(bins[1])
    z = xp.from_binsparse(bins[2])
    np.testing.assert_allclose(x, np.array([0.0, 3.0, 0.0, 3.0]))
    np.testing.assert_allclose(y, np.array([0.0, 0.0, 4.0, 4.0]))
    np.testing.assert_allclose(z, np.zeros(4))
    assert meta["size"] == 4.0
    assert "parameters" not in meta
    assert meta["source_dimensions"] == 2
    assert meta["simulation_dimensions"] == 3


def test_toy_ewap_velocities_preserve_dataset_values():
    """Velocities are loaded from the EWAP columns without rescaling."""
    xp = NumpyFramework()
    bins, _meta = load_toy_ewap_dataset()
    vx = xp.from_binsparse(bins[3])
    vy = xp.from_binsparse(bins[4])
    vz = xp.from_binsparse(bins[5])
    np.testing.assert_allclose(vx, np.array([0.1, -0.1, 0.0, 0.0]))
    np.testing.assert_allclose(vy, np.array([0.0, 0.0, 0.2, -0.2]))
    np.testing.assert_allclose(vz, np.zeros(4))


def test_toy_ewap_mass_is_scalar_tensor():
    """Uniform pedestrian mass is data, represented as a scalar tensor."""
    xp = NumpyFramework()
    bins, _meta = load_toy_ewap_dataset()
    mass = xp.from_binsparse(bins[6])
    np.testing.assert_allclose(mass, np.asarray(0.01))
    assert mass.shape == ()


def test_benchmark_runs_with_toy_ewap_data():
    """Full benchmark pipeline executes without error on toy EWAP data."""
    xp = NumpyFramework()
    ps.xp = xp
    bins, meta = load_toy_ewap_dataset(num_steps=5)
    meta["parameters"] = _toy_parameters()
    data = [xp.from_binsparse(b) for b in bins]
    result = ps.ParticleSimBenchmark().benchmark(xp, data, meta)
    assert len(result) == 6  # x, y, z, vx, vy, vz
    for arr in result:
        assert arr.shape == (meta["n_particles"],)


def test_ewap_particle_sim_generator_uses_downloader(monkeypatch):
    calls = []

    def fake_download(scene, *, num_steps):
        calls.append((scene, num_steps))
        inputs = [
            from_numpy(np.array([0.0, 1.0], dtype=np.float64)),
            from_numpy(np.array([0.0, 0.0], dtype=np.float64)),
            from_numpy(np.zeros(2, dtype=np.float64)),
            from_numpy(np.zeros(2, dtype=np.float64)),
            from_numpy(np.zeros(2, dtype=np.float64)),
            from_numpy(np.zeros(2, dtype=np.float64)),
            from_numpy(np.asarray(0.01, dtype=np.float64)),
        ]
        return inputs, {"size": 1.0, "steps": num_steps, "n_particles": 2}

    monkeypatch.setattr(ps, "download_ewap_dataset", fake_download)

    generator = ps.EWAPParticleSimGenerator()
    datasets = generator.datasets

    assert generator.cacheable is False
    assert [dataset.name for dataset in datasets] == [
        "ewap_seq_eth",
        "ewap_seq_hotel",
    ]
    assert all(dataset.suites == ["standard"] for dataset in datasets)
    assert all(
        dataset.parameters["force_model"] == "cs267_repulsive"
        for dataset in datasets
    )
    assert all(dataset.parameters["softening"] == 0.0001 for dataset in datasets)
    assert all("mass" not in dataset.parameters for dataset in datasets)

    instance = generator.generate(datasets[0])

    assert calls == [("seq_eth", 50)]
    assert len(instance.inputs) == 7
    assert instance.meta["parameters"] == datasets[0].parameters


@pytest.mark.slow
def test_ewap_seq_eth_download():
    """Download seq_eth and verify it produces a non-empty particle dataset."""
    from saps.downloaders.ewap import download_ewap_dataset

    bins, meta = download_ewap_dataset("seq_eth", num_steps=10)
    assert len(bins) == 7
    n = meta["n_particles"]
    assert n > 0
    xp = NumpyFramework()
    x = xp.from_binsparse(bins[0])
    z = xp.from_binsparse(bins[2])
    assert meta["source_x_min"] == float(x.min())
    assert meta["source_x_max"] == float(x.max())
    assert meta["source_dimensions"] == 2
    assert meta["simulation_dimensions"] == 3
    assert np.allclose(z, 0.0)
