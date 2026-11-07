"""Protocol checks independent of MuJoCo, GPU availability, and model weights."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

path = Path(__file__).resolve().parents[1] / "examples/maze2d_umaze/evaluate.py"
spec = importlib.util.spec_from_file_location("umaze_evaluate", path)
evaluate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluate)


class FixedStateEnv:
    def __init__(self, distance=0.0, terminate_at=300):
        self.unwrapped = self
        self._target = np.array([1.0, 1.0])
        self.state = np.array([1.0 + distance, 1.0, 0.0, 0.0])
        self.terminate_at = terminate_at
        self.actions = []

    def seed(self, seed):
        self.observed_seed = seed

    def reset(self):
        self.actions.clear()
        return self.state.copy()

    def step(self, action):
        self.actions.append(action.copy())
        reward = float(np.linalg.norm(self.state[:2] - self._target) <= 0.5)
        return self.state.copy(), reward, len(self.actions) >= self.terminate_at, {}


class EvaluationProtocolTests(unittest.TestCase):
    def test_success_does_not_stop_episode_or_replace_inverter(self):
        env = FixedStateEnv()
        calls = []
        def plan(state, goal):
            calls.append(state.copy())
            np.testing.assert_array_equal(goal, [1.0, 1.0, 0.0, 0.0])
            return np.full((128, 2), 2.0)
        row = evaluate.evaluate_episode(env, plan, 777, 128)
        self.assertEqual(env.observed_seed, 777)
        self.assertEqual(len(calls), 3)
        self.assertEqual(len(env.actions), 300)
        self.assertEqual(row["return"], 300)
        self.assertEqual(row["steps_to_goal"], 0)
        np.testing.assert_array_equal(env.actions, np.ones((300, 2)))

    def test_reward_radius_and_success_radius_are_distinct(self):
        env = FixedStateEnv(distance=0.47)
        row = evaluate.evaluate_episode(env, lambda s, g: np.zeros((128, 2)), 778, 64)
        self.assertEqual(row["return"], 300)
        self.assertFalse(row["success"])
        self.assertIsNone(row["steps_to_goal"])
        self.assertEqual(row["controller_calls"], 5)

    def test_unexpected_early_termination_rejects_different_protocol(self):
        env = FixedStateEnv(terminate_at=10)
        with self.assertRaisesRegex(RuntimeError, "Unexpected early termination"):
            evaluate.evaluate_episode(env, lambda s, g: np.zeros((128, 2)), 777, 16)


if __name__ == "__main__":
    unittest.main()
