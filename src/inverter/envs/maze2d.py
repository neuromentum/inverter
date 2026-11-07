"""The legacy D4RL environment used for the paper's Maze2D U-Maze runs.

No simulator or dataset is imported/downloaded until ``make_umaze`` is called.
The returned object uses Gym's old reset/step API and is not reseeded here.
"""
from __future__ import annotations

import os

ENV_ID = "maze2d-umaze-v1"
DATASET_FILENAME = "maze2d-umaze-sparse-v1.hdf5"
DATASET_SHA256 = "f2d327d95067d53e274ac2fe876b740ed7883d0a9738bd5b447635cdb786bfb7"
DATASET_URL = (
    "https://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/"
    + DATASET_FILENAME
)


def make_umaze():
    """Return the unmodified sparse-reward D4RL v1 U-Maze environment.

    Import PointMaze explicitly: D4RL 1.1 groups it with optional suites in
    its package initializer, so missing dependencies can otherwise skip its
    registration. This does not patch D4RL or change its physics.
    """
    if os.environ.get("MAZE2D_BODY_POS") or os.environ.get("MAZE2D_CENTERED") == "1":
        raise RuntimeError(
            "Unset MAZE2D_BODY_POS and MAZE2D_CENTERED: paper evaluation "
            "uses the original D4RL body origin (1.2, 1.2, 0)."
        )
    os.environ.setdefault("D4RL_SUPPRESS_IMPORT_ERROR", "1")
    import gym
    import numpy as np
    import mujoco_py
    import d4rl.pointmaze  # noqa: F401 -- register after optional-suite failures

    if mujoco_py.functions.mj_version() != 210:
        raise RuntimeError("Paper evaluation requires native MuJoCo 2.1.0.")
    env = gym.make(ENV_ID)
    try:
        base = env.unwrapped
        body_id = base.model.body_name2id("particle")
        if not np.array_equal(base.model.body_pos[body_id], [1.2, 1.2, 0.0]):
            raise RuntimeError("Unexpected Maze2D body origin; use stock D4RL 1.1.")
        if not np.array_equal(base._target, [1.0, 1.0]):
            raise RuntimeError("Unexpected U-Maze benchmark goal.")
        if base.reward_type != "sparse" or base.reset_target:
            raise RuntimeError("Expected sparse rewards and a fixed benchmark goal.")
        if env._max_episode_steps != 300:
            raise RuntimeError("Expected the 300-step D4RL U-Maze time limit.")
        if base.frame_skip != 1 or base.model.opt.timestep != 0.01:
            raise RuntimeError("Unexpected Maze2D simulator timestep/frame skip.")
    except Exception:
        env.close()
        raise
    # Use the HTTPS transport for the same archived dataset bytes.
    base.dataset_url = base._dataset_url = DATASET_URL
    return env
