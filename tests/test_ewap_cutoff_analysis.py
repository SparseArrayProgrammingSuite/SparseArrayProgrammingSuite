import numpy as np

from scripts.analyze_ewap_cutoff import analyze_scene, force_curve


def test_exponential_cutoff_two_particle_boundary():
    errors, pairs, error_energy, full_energy = force_curve(
        np.array([[0.0, 0.0], [2.0, 0.0]]), np.array([1.0, 2.0, 3.0])
    )
    np.testing.assert_allclose(errors, [1, 0, 0])
    np.testing.assert_array_equal(pairs, [0, 2, 2])
    np.testing.assert_allclose(full_energy, 2 * np.exp(-4))
    np.testing.assert_allclose(error_energy, [full_energy, 0, 0])


def test_ewap_analysis_uses_simultaneous_frames_and_exposes_weak_forces(tmp_path):
    path = tmp_path / "obsmat.txt"
    path.write_text(
        """1 1 0 0 0 0 0 0
1 2 1 0 0 0 0 0
2 3 0 0 0 0 0 0
2 4 10 0 0 0 0 0
"""
    )
    result = analyze_scene(path, np.array([2.0, 10.0]))
    assert result["unique_pedestrians"] == 4
    assert result["simultaneous_pedestrians_max"] == 2
    assert result["first_passing_cutoff"] == 10
    assert result["first_passing_pooled_cutoff"] == 2
    np.testing.assert_allclose(result["pair_fraction_over_frames"], [0.5, 1.0])
