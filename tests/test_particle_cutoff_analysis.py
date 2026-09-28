"""Independent checks of the force-error analysis, without downloading data."""

import numpy as np

from scripts.analyze_particle_cutoff import acceleration_curve, error_metrics


def test_cutoff_curve_matches_scalar_force_with_masses_and_softening():
    # Includes a softened pair, a pair exactly at the cutoff, and unequal masses.
    position = np.array([[0, 0, 0], [0.01, 0, 0], [1, 0, 0], [0, 2, 0]])
    mass = np.array([1.0, 2.0, 3.0, 4.0])
    radii = np.array([0.005, 0.01, 1.0, 3.0])
    full, truncated, density = acceleration_curve(position, mass, 0.05, 1.7, radii)
    for index, radius in enumerate(radii):
        expected = np.zeros_like(position, dtype=float)
        for i, point in enumerate(position):
            for j, neighbor in enumerate(position):
                delta = neighbor - point
                distance2 = float(delta @ delta)
                if distance2 <= radius**2:
                    expected[i] += (
                        1.7 * mass[j] * delta / max(distance2, 0.05**2) ** 1.5
                    )
        np.testing.assert_allclose(truncated[index], expected, atol=1e-12)
        single_full, single_truncated, single_density = acceleration_curve(
            position, mass, 0.05, 1.7, np.array([radius])
        )
        np.testing.assert_allclose(single_full, full, atol=1e-12)
        np.testing.assert_allclose(single_truncated[0], expected, atol=1e-12)
        np.testing.assert_allclose(single_density[0], density[index])
    np.testing.assert_allclose(full, truncated[-1], atol=1e-12)
    np.testing.assert_allclose(density, [0, 2 / 12, 6 / 12, 1])


def test_metrics_expose_large_relative_error_on_weak_force_particle():
    full = np.array([[100.0, 0, 0], [1.0, 0, 0]])
    truncated = np.array([[[100.0, 0, 0], [0, 0, 0]]])
    metrics = error_metrics(full, truncated, np.ones(2))
    np.testing.assert_allclose(metrics["relative_force_l2"], [1 / np.sqrt(10001)])
    np.testing.assert_allclose(metrics["particle_relative_max"], [1])
    np.testing.assert_allclose(metrics["fraction_particles_above_2pct"], [0.5])
