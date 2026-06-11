import numpy as np
import pytest

import saps.benchmarks.particle_sim as ps
from frameworks.saps_numpy import NumpyFramework
from saps.downloaders.ewap import load_toy_ewap_dataset
from saps_framework.binsparse_format import BinsparseFormat


def test_toy_ewap_dataset_shape():
    """Parser returns 4 particles (one per unique pedestrian in the toy data)."""
    bins, meta = load_toy_ewap_dataset(num_steps=10)
    assert len(bins) == 4  # x, y, vx, vy
    n = meta["n_particles"]
    assert n == 4
    for b in bins:
        assert isinstance(b, BinsparseFormat)


def test_toy_ewap_positions_in_box():
    """Rescaled positions lie within [0, box_size]."""
    xp = NumpyFramework()
    bins, meta = load_toy_ewap_dataset()
    box_size = meta["size"]
    x = xp.from_binsparse(bins[0])
    y = xp.from_binsparse(bins[1])
    assert float(x.min()) >= 0.0
    assert float(x.max()) <= box_size + 1e-9
    assert float(y.min()) >= 0.0
    assert float(y.max()) <= box_size + 1e-9


def test_toy_ewap_velocity_scaling():
    """Velocities are scaled by the same factor as positions."""
    xp = NumpyFramework()
    bins, meta = load_toy_ewap_dataset()
    box_size = meta["size"]
    # Raw toy positions span 3 m in x and 4 m in y; scale = box_size / 4
    scale = box_size / 4.0
    vx = xp.from_binsparse(bins[2])
    vy = xp.from_binsparse(bins[3])
    # Pedestrian 1 raw v_x = 0.1 → scaled = 0.1 * scale
    assert np.isclose(float(vx[0]), 0.1 * scale, rtol=1e-6)
    # Pedestrian 3 raw v_y = 0.2 → scaled = 0.2 * scale
    assert np.isclose(float(vy[2]), 0.2 * scale, rtol=1e-6)


def test_benchmark_runs_with_toy_ewap_data():
    """Full benchmark pipeline executes without error on toy EWAP data."""
    xp = NumpyFramework()
    ps.xp = xp
    bins, meta = load_toy_ewap_dataset(num_steps=5)
    data = [xp.from_binsparse(b) for b in bins]
    result = ps.ParticleSimBenchmark().benchmark(data, meta)
    assert len(result) == 4  # x, y, vx, vy
    for arr in result:
        assert arr.shape == (meta["n_particles"],)


@pytest.mark.slow
def test_ewap_seq_eth_download():
    """Download seq_eth and verify it produces a non-empty particle dataset."""
    from saps.downloaders.ewap import download_ewap_dataset

    bins, meta = download_ewap_dataset("seq_eth", num_steps=10)
    assert len(bins) == 4
    n = meta["n_particles"]
    assert n > 0
    xp = NumpyFramework()
    x = xp.from_binsparse(bins[0])
    box_size = meta["size"]
    assert float(x.min()) >= 0.0
    assert float(x.max()) <= box_size + 1e-9
