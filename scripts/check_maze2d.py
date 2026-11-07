#!/usr/bin/env python3
"""Check the legacy simulator and optionally verify the cached D4RL dataset.

This script never downloads data. The optional --dataset argument checks an
existing HDF5 file, defaulting to the standard D4RL cache location.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform

from inverter.envs.maze2d import DATASET_FILENAME, DATASET_SHA256, make_umaze


RECORDED_VERSIONS = {
    "gym": "0.23.1", "d4rl": "1.1", "mujoco-py": "2.1.2.14",
    "numpy": "1.24.4", "torch": "2.8.0", "scipy": "1.11.4",
    "h5py": "3.14.0", "Cython": "0.29.37",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", nargs="?", const="default", metavar="PATH",
                        help="also checksum an existing dataset (default: D4RL cache)")
    parser.add_argument("--strict-versions", action="store_true",
                        help="require the audited direct-dependency versions")
    args = parser.parse_args()
    import numpy as np

    versions = {name: importlib.metadata.version(name) for name in RECORDED_VERSIONS}
    mismatches = {name: {"installed": versions[name], "recorded": expected}
                  for name, expected in RECORDED_VERSIONS.items()
                  if versions[name].split("+")[0] != expected}
    if args.strict_versions and (mismatches or platform.python_version_tuple()[:2] != ("3", "9")):
        raise SystemExit("Version mismatch: " + json.dumps({"python": platform.python_version(),
                                                           "packages": mismatches}))
    env = make_umaze()
    try:
        env.seed(777)
        obs = env.reset()
        expected = np.array([0.9765633180901656, 1.0200141050898004,
                             -0.15836366914181446, 0.06324575844117478])
        if not np.allclose(obs, expected, atol=1e-10, rtol=0):
            raise RuntimeError("Seed-777 reset differs from the recorded environment: " + repr(obs))
        next_obs, reward, done, info = env.step(np.zeros(2))
        if reward != 1.0 or done or not np.isfinite(next_obs).all():
            raise RuntimeError("Unexpected zero-action smoke-check transition.")
        report = {"status": "passed", "python": platform.python_version(),
                  "packages": versions, "version_differences": mismatches,
                  "environment": env.spec.id, "native_mujoco": "2.1.0",
                  "goal": np.asarray(env.unwrapped._target, dtype=float).tolist(), "horizon": int(env._max_episode_steps),
                  "seed_777_observation": obs.tolist(), "zero_action_reward": reward}
    finally:
        env.close()
    if args.dataset:
        if args.dataset == "default":
            path = Path(os.environ.get("D4RL_DATASET_DIR", "~/.d4rl/datasets")).expanduser() / DATASET_FILENAME
        else:
            path = Path(args.dataset).expanduser()
        if not path.is_file():
            raise SystemExit("Dataset is absent; this check does not download it: " + str(path))
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != DATASET_SHA256:
            raise RuntimeError("Dataset checksum differs from the paper's archived input.")
        import h5py
        with h5py.File(path, "r") as data:
            assert data["observations"].shape == (1_000_000, 4)
            assert data["actions"].shape == (1_000_000, 2)
            boundaries = np.logical_or(data["timeouts"][:], data["terminals"][:])
            episodes = int(boundaries.sum()) + int(not boundaries[-1])
            assert episodes == 12_460
        report["dataset"] = {"path": str(path), "sha256": digest.hexdigest(),
                             "transitions": 1_000_000, "episodes": episodes}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
