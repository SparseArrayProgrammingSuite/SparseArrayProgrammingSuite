import pytest

import numpy as np

from binsparse import BinsparseTensor

import saps.benchmarks.particle_sim as ps
from frameworks.saps_numpy import NumpyFramework
from saps.downloaders.ewap import load_toy_ewap_dataset


def test_toy_ewap_dataset_shape():
    """Parser returns 4 particles (one per unique pedestrian in the toy data)."""
    bins, meta = load_toy_ewap_dataset(num_steps=10)
    assert len(bins) == 6  # x, y, z, vx, vy, vz
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


def test_benchmark_runs_with_toy_ewap_data():
    """Full benchmark pipeline executes without error on toy EWAP data."""
    xp = NumpyFramework()
    ps.xp = xp
    bins, meta = load_toy_ewap_dataset(num_steps=5)
    data = [xp.from_binsparse(b) for b in bins]
    result = ps.ParticleSimBenchmark().benchmark(xp, data, meta)
    assert len(result) == 6  # x, y, z, vx, vy, vz
    for arr in result:
        assert arr.shape == (meta["n_particles"],)


@pytest.mark.slow
def test_ewap_seq_eth_download():
    """Download seq_eth and verify it produces a non-empty particle dataset."""
    from saps.downloaders.ewap import download_ewap_dataset

    bins, meta = download_ewap_dataset("seq_eth", num_steps=10)
    assert len(bins) == 6
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
