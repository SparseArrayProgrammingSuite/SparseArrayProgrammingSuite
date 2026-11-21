import zipfile

import pytest

import numpy as np

from binsparse.conversions import to_numpy

import saps.benchmarks.particle_simulation as ps
from saps.downloaders.lyon import download_lyon_dataset, parse_crowd_snapshot

_TOP_HEADER = "# id frame x/m y/m z/m id_global t/s x_RGF y_RGF\n"
# The wide CCTV views have no id_global column.
_LARGE_HEADER = "# id frame x/m y/m z/m t/s x_RGF y_RGF\n"


def _row(ped, frame, x, y, large=False):
    t = f"{frame / 30:.6f}"
    middle = t if large else f"{110000 + ped} {t}"
    return f"{ped} {frame} {x} {y} 1.8 {middle} 0 0\n"


def _trajectories(large=False):
    """At 30 fps, pedestrian 1 walks 1 m/s along y in frames 0-19, pedestrian
    2 walks 0.3 m/s along x in frames 5-9, and pedestrian 3 stands at (0, 0)
    in frames 15-30. Frames 5-9 and 15-19 each hold two pedestrians, so the
    snapshot is frame 5."""
    rows = [_row(3, f, 0.0, 0.0, large) for f in range(15, 31)]
    rows += [_row(1, f, 4.0, 1.0 + f / 30, large) for f in range(20)]
    rows += [_row(2, f, -2.0 + 0.3 * (f - 5) / 30, 5.0, large) for f in range(5, 10)]
    return (_LARGE_HEADER if large else _TOP_HEADER) + "".join(rows)


@pytest.mark.parametrize("large", [False, True])
def test_snapshot_is_the_first_most_crowded_frame(large):
    frame, x, y, vx, vy = parse_crowd_snapshot(_trajectories(large))
    assert frame == 5
    # Velocities span frames 0-11 for pedestrian 1 and, clipped to its
    # trajectory, frames 5-9 for pedestrian 2.
    np.testing.assert_allclose(x, [4.0, -2.0])
    np.testing.assert_allclose(y, [1.0 + 5 / 30, 5.0])
    np.testing.assert_allclose(vx, [0.0, 0.3], atol=1e-5)
    np.testing.assert_allclose(vy, [1.0, 0.0], atol=1e-5)


def test_snapshot_keeps_the_first_of_repeated_rows():
    # A repeat of pedestrian 1's frame 5 row, listed after it, is dropped.
    _, x, y, _, _ = parse_crowd_snapshot(_trajectories() + _row(1, 5, 9.0, 9.0))
    np.testing.assert_allclose(x, [4.0, -2.0])
    np.testing.assert_allclose(y, [1.0 + 5 / 30, 5.0])


def test_snapshot_needs_two_observations_per_pedestrian():
    text = _TOP_HEADER + _row(1, 0, 0.0, 0.0) + _row(2, 0, 1.0, 0.0)
    with pytest.raises(ValueError, match="one observation"):
        parse_crowd_snapshot(text)


def _fake_zip(tmp_path):
    with zipfile.ZipFile(tmp_path / "Data_Madras.zip", "w") as archive:
        archive.writestr(
            "Data_Madras/TopView_trajectories/TopView_2B.txt", _trajectories()
        )
        archive.writestr(
            "Data_Madras/LargeView_trajectories/LargeView_zoom_O.txt",
            _trajectories(large=True),
        )
    return tmp_path


@pytest.mark.parametrize("sequence", ["TopView_2B", "LargeView_zoom_O"])
def test_download_lyon_dataset_is_ewap_shaped(tmp_path, sequence):
    bins, meta = download_lyon_dataset(
        sequence, data_dir=_fake_zip(tmp_path), num_steps=7
    )
    x, y, z, vx, vy, vz, mass = (to_numpy(b) for b in bins)
    np.testing.assert_allclose(x, [4.0, -2.0])
    np.testing.assert_array_equal(z, [0.0, 0.0])
    np.testing.assert_array_equal(vz, [0.0, 0.0])
    assert mass.shape == () and mass.item() == 0.01
    assert meta["n_particles"] == 2
    assert meta["steps"] == 7
    assert meta["size"] == pytest.approx(6.0)
    assert meta["source_sequence"] == sequence
    assert meta["source_frame"] == 5


def test_download_lyon_dataset_rejects_unknown_sequence(tmp_path):
    with pytest.raises(ValueError, match="Unknown Lyon sequence"):
        download_lyon_dataset("LargeView_zoom_A", data_dir=tmp_path)


def test_lyon_generator_records_bounds(monkeypatch, tmp_path):
    data_dir = _fake_zip(tmp_path)
    monkeypatch.setattr(
        ps,
        "download_lyon_dataset",
        lambda sequence, num_steps: download_lyon_dataset(
            sequence, data_dir=data_dir, num_steps=num_steps
        ),
    )
    generator = ps.ParticleSimulationLyonGenerator()
    assert {d.name: d.suites for d in generator.datasets} == {
        "LargeView_zoom_O": ["standard", "trace"],
        "TopView_2B": ["standard", "trace"],
    }
    instance = generator.generate(generator.datasets[0])
    assert len(instance.inputs) == 7
    assert instance.meta["parameters"]["force_model"] == "cs267_repulsive"
    # Every output's lower bound, then every output's upper bound.
    assert len(instance.ref_outputs) % 2 == 0
