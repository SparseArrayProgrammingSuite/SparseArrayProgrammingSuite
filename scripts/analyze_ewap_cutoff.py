#!/usr/bin/env python3
"""Validate truncation of an assumed exponential interaction on EWAP frames.

This is NOT calibration of the current CS267 force or behavioral validation.
The circular interaction law is proportional to exp(-distance / B), B=1 metre,
motivated by Johansson et al. https://arxiv.org/html/0810.4587 (Eq. 6, Fig. 11).
A common amplitude and body-radius factor cancel in the relative error metric.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def force_curve(position, radii, decay_length=1.0):
    delta = position[:, None, :] - position[None, :, :]
    distance = np.linalg.norm(delta, axis=2)
    coefficient = np.divide(
        np.exp(-distance / decay_length),
        distance,
        out=np.zeros_like(distance),
        where=distance > 0,
    )
    pair_force = delta * coefficient[:, :, None]
    full = pair_force.sum(axis=1)
    truncated = np.array(
        [
            np.sum(pair_force * (distance <= radius)[:, :, None], axis=1)
            for radius in radii
        ]
    )
    error_norm = np.linalg.norm(truncated - full, axis=(1, 2))
    full_norm = np.linalg.norm(full)
    errors = error_norm / full_norm
    pairs = np.array(
        [np.count_nonzero(distance <= radius) - len(position) for radius in radii]
    )
    return errors, pairs, error_norm**2, full_norm**2


def analyze_scene(path, radii):
    rows = np.loadtxt(path)
    frames, counts = np.unique(rows[:, 0], return_counts=True)
    worst = np.zeros(len(radii))
    total_pairs = np.zeros(len(radii), dtype=np.int64)
    possible_pairs = 0
    checked_frames = 0
    frame_errors = []
    error_energy = np.zeros(len(radii))
    full_energy = 0.0
    for frame in frames:
        current = rows[rows[:, 0] == frame]
        if len(current) < 2:
            continue
        position = current[:, [2, 4]]
        errors, pairs, squared_error, squared_full = force_curve(position, radii)
        if not np.all(np.isfinite(errors)):
            raise ValueError(f"Undefined relative force error in frame {frame}")
        worst = np.maximum(worst, errors)
        total_pairs += pairs
        possible_pairs += len(current) * (len(current) - 1)
        checked_frames += 1
        frame_errors.append(errors)
        error_energy += squared_error
        full_energy += squared_full
    passing = np.flatnonzero(worst <= 0.02)
    pooled_error = np.sqrt(error_energy / full_energy)
    pooled_passing = np.flatnonzero(pooled_error <= 0.02)
    return {
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "total_frames": len(frames),
        "checked_frames": checked_frames,
        "unique_pedestrians": len(np.unique(rows[:, 1])),
        "simultaneous_pedestrians_max": int(counts.max()),
        "simultaneous_pedestrians_median": float(np.median(counts)),
        "worst_frame_relative_l2": worst.tolist(),
        "median_frame_relative_l2": np.median(frame_errors, axis=0).tolist(),
        "pooled_relative_l2": pooled_error.tolist(),
        "pair_fraction_over_frames": (total_pairs / possible_pairs).tolist(),
        "first_passing_cutoff": float(radii[passing[0]]) if len(passing) else None,
        "first_passing_pooled_cutoff": float(radii[pooled_passing[0]])
        if len(pooled_passing)
        else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache", type=Path, default=Path(".saps/outputs/cache/ewap/ewap")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    radii = np.r_[0.01, np.round(np.arange(0.05, 15.001, 0.05), 2)]
    result = {
        "assumed_model": "isotropic pair repulsion proportional to exp(-r/B)",
        "decay_length_metres": 1.0,
        "target_global_relative_force_l2": 0.02,
        "radii": radii.tolist(),
        "scenes": {},
        "limitation": (
            "Numerical truncation validation only; model not fitted to EWAP motion"
        ),
    }
    for scene in ("seq_eth", "seq_hotel"):
        record = analyze_scene(args.cache / scene / "obsmat.txt", radii)
        result["scenes"][scene] = record
        print(scene, record["first_passing_cutoff"], "metres", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
