#!/usr/bin/env python3
"""Measure Plummer cutoff error against direct, untruncated benchmark gravity.

Run with ``poetry run python scripts/analyze_particle_cutoff.py --output PATH``.
Downloads the two small NEMO Plummer snapshots through the normal source cache.
This measures spherical force truncation error, not time integration or softening
error. It does not model the benchmark's neighboring-bucket interaction region.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from binsparse.conversions import to_numpy

from saps.benchmarks.particle_sim import ParticleSimulationNEMOGenerator


def acceleration_curve(position, mass, softening, gravitational_constant, radii):
    """Return exact all-pairs acceleration and accelerations at each radius."""
    delta = position[None, :, :] - position[:, None, :]
    distance2 = np.sum(delta * delta, axis=2)
    coefficient = (
        gravitational_constant
        * mass[None, :]
        / np.maximum(distance2, softening**2) ** 1.5
    )
    contributions = delta * coefficient[:, :, None]
    full = contributions.sum(axis=1)
    if len(radii) == 1:
        within = distance2 <= radii[0] ** 2
        truncated = np.sum(contributions * within[:, :, None], axis=1)[None, :, :]
        density = np.array(
            [(within.sum() - len(position)) / (len(position) * (len(position) - 1))]
        )
        return full, truncated, density
    order = np.argsort(distance2, axis=1)
    distances = np.take_along_axis(distance2, order, axis=1)
    cumulative = np.take_along_axis(contributions, order[:, :, None], axis=1)
    cumulative = np.cumsum(cumulative, axis=1)
    # Leading zero also makes an empty neighborhood well-defined.
    cumulative = np.pad(cumulative, ((0, 0), (1, 0), (0, 0)))
    counts = np.array(
        [np.searchsorted(row, radii**2, side="right") for row in distances]
    )
    truncated = cumulative[np.arange(len(position))[:, None], counts]
    # Exclude self-pairs, whose force is zero, from interaction density.
    density = (counts.sum(axis=0) - len(position)) / (
        len(position) * (len(position) - 1)
    )
    return full, truncated.transpose(1, 0, 2), density


def error_metrics(full, truncated, mass):
    """Global force L2 error, plus relative acceleration error per particle."""
    error = np.linalg.norm(truncated - full[None, :, :], axis=2)
    magnitude = np.linalg.norm(full, axis=1)
    relative = np.divide(
        error, magnitude, out=np.full_like(error, np.inf), where=magnitude > 0
    )
    relative[(magnitude == 0)[None, :] & (error == 0)] = 0
    return {
        "relative_force_l2": np.linalg.norm(error * mass, axis=1)
        / np.linalg.norm(magnitude * mass),
        "particle_relative_rms": np.sqrt(np.mean(relative**2, axis=1)),
        "particle_relative_p50": np.median(relative, axis=1),
        "particle_relative_p95": np.quantile(relative, 0.95, axis=1, method="higher"),
        "particle_relative_max": np.max(relative, axis=1),
        "fraction_particles_above_2pct": np.mean(relative > 0.02, axis=1),
    }


def trajectory(position, velocity, mass, parameters, radii, steps, evolve_cutoff=None):
    """Evaluate errors on identical positions, then advance full or cutoff gravity."""
    position, velocity = position.copy(), velocity.copy()
    initial = None
    worst = {}
    density_min = np.ones(len(radii))
    density_max = np.zeros(len(radii))
    history = []
    for step in range(steps + 1):
        full, truncated, density = acceleration_curve(
            position,
            mass,
            parameters["softening"],
            parameters["gravitational_constant"],
            radii,
        )
        metrics = error_metrics(full, truncated, mass)
        if initial is None:
            initial = {key: value.tolist() for key, value in metrics.items()}
        for key, value in metrics.items():
            worst[key] = np.maximum(worst.get(key, np.zeros_like(value)), value)
        density_min = np.minimum(density_min, density)
        density_max = np.maximum(density_max, density)
        history.append(metrics["relative_force_l2"].tolist())
        if step == steps:
            break
        acceleration = full if evolve_cutoff is None else truncated[evolve_cutoff]
        velocity += acceleration * parameters["dt"]
        position += velocity * parameters["dt"]
    return {
        "initial": initial,
        "worst": {key: value.tolist() for key, value in worst.items()},
        "interaction_density_min": density_min.tolist(),
        "interaction_density_max": density_max.tolist(),
        "force_l2_by_step": history,
    }


def first_passing(radii, values, target):
    passing = np.flatnonzero(np.asarray(values) <= target)
    return float(radii[passing[0]]) if len(passing) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # A grid avoids assuming that vector force error is monotone in radius.
    radii = np.round(np.arange(0.05, 30.0001, 0.05), 2)
    generator = ParticleSimulationNEMOGenerator()
    result = {
        "target": 0.02,
        "metric": "norm(F_cutoff - F_full) / norm(F_full), over all particles",
        "radii": radii.tolist(),
        "datasets": {},
    }
    for dataset in generator.datasets:
        if not dataset.name.startswith("plummer_"):
            continue
        instance = generator.generate(dataset)
        arrays = [to_numpy(value) for value in instance.inputs]
        position = np.column_stack(arrays[:3])
        velocity = np.column_stack(arrays[3:6])
        mass = arrays[6]
        print(f"Sweeping {dataset.name}, {dataset.num_steps} steps", flush=True)
        full = trajectory(
            position, velocity, mass, dataset.parameters, radii, dataset.num_steps
        )
        candidates = {
            metric: first_passing(radii, values, 0.02)
            for metric, values in full["worst"].items()
            if metric
            in {
                "relative_force_l2",
                "particle_relative_rms",
                "particle_relative_p95",
                "particle_relative_max",
            }
        }
        # Each criterion gets a rounded radius checked on both trajectories.
        recommendations = {}
        for metric, candidate in candidates.items():
            if candidate is None:
                raise RuntimeError(f"No radius meets the {metric} target")
            chosen = np.ceil(candidate * 2) / 2
            while chosen <= radii[-1]:
                index = int(np.flatnonzero(radii == chosen)[0])
                cutoff = trajectory(
                    position,
                    velocity,
                    mass,
                    dataset.parameters,
                    np.array([chosen]),
                    dataset.num_steps,
                    evolve_cutoff=0,
                )
                if (
                    max(full["worst"][metric][index], cutoff["worst"][metric][0])
                    <= 0.02
                ):
                    break
                chosen += 0.5
            else:
                raise RuntimeError(f"No cutoff trajectory meets the {metric} target")
            recommendations[metric] = {
                "radius": chosen,
                "cutoff_trajectory": cutoff,
            }
        record = {
            "parameters": dataset.parameters,
            "steps": dataset.num_steps,
            "source_url": instance.meta["source_url"],
            "input_sha256": hashlib.sha256(
                b"".join(array.tobytes() for array in arrays)
            ).hexdigest(),
            "full_trajectory": full,
            "first_passing_radius_on_grid": candidates,
            "recommendations": recommendations,
            "configured_cutoff_trajectory": trajectory(
                position,
                velocity,
                mass,
                dataset.parameters,
                np.array([dataset.parameters["cutoff"]]),
                dataset.num_steps,
                evolve_cutoff=0,
            ),
        }
        result["datasets"][dataset.name] = record
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "dataset": dataset.name,
                    "candidates": candidates,
                    "recommendations": {
                        metric: {
                            "radius": item["radius"],
                            "worst": item["cutoff_trajectory"]["worst"],
                        }
                        for metric, item in recommendations.items()
                    },
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
