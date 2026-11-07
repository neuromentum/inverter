#!/usr/bin/env python3
"""Evaluate released Inverters with the paper's fixed-goal D4RL U-Maze protocol.

This evaluator uses only the Inverter at deployment. Its optional timing output
is a new measurement: the historical timing table subtracted visualization FoM
work from a larger evaluation harness. No offline dataset or FoM is required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LABELS = ("default", "fmseed123", "fmseed456", "fmseed789")
EPISODE_STEPS = 300
GOAL_THRESHOLD = 0.45


def evaluate_episode(env, plan, seed, replan_every, save_trajectory=False):
    """Keep replanning for all 300 steps, including after reaching the goal."""
    env.seed(seed)
    state = np.asarray(env.reset(), dtype=np.float32)
    goal = np.asarray(env.unwrapped._target, dtype=np.float64).copy()
    if state.shape != (4,) or not np.array_equal(goal, [1.0, 1.0]):
        raise RuntimeError("Expected the four-state, fixed-goal maze2d-umaze-v1 environment.")
    initial_state = state.copy()
    goal_state = np.array([goal[0], goal[1], 0.0, 0.0], dtype=np.float32)
    first_goal_step = 0 if np.linalg.norm(state[:2] - goal) < GOAL_THRESHOLD else None
    episode_return = 0.0
    steps = calls = 0
    states = [state.tolist()] if save_trajectory else None
    actions = [] if save_trajectory else None
    while steps < EPISODE_STEPS:
        proposed = np.asarray(plan(state, goal_state), dtype=np.float32)
        if proposed.shape != (128, 2) or not np.isfinite(proposed).all():
            raise RuntimeError("The U-Maze Inverter must emit 128 finite two-dimensional actions.")
        calls += 1
        for action in proposed[:min(replan_every, EPISODE_STEPS - steps)]:
            action = np.clip(action, -1.0, 1.0).astype(np.float32)
            observation, reward, done, _ = env.step(action)
            state = np.asarray(observation, dtype=np.float32)
            episode_return += float(reward)
            steps += 1
            if first_goal_step is None and np.linalg.norm(state[:2] - goal) < GOAL_THRESHOLD:
                first_goal_step = steps
            if save_trajectory:
                states.append(state.tolist())
                actions.append(action.tolist())
            if done and steps != EPISODE_STEPS:
                raise RuntimeError("Unexpected early termination: this benchmark runs 300 steps.")
    result = {
        "seed": seed, "initial_state": initial_state.tolist(), "goal_xy": goal.tolist(),
        "return": episode_return, "steps": steps, "controller_calls": calls,
        "success": first_goal_step is not None, "steps_to_goal": first_goal_step,
        "final_distance": float(np.linalg.norm(state[:2] - goal)),
    }
    if save_trajectory:
        result.update(states=states, actions=actions)
    return result


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_inverter(path, device):
    import torch
    from inverter.models.iwm_transformer import IWM_Transformer

    # The released checkpoints contain model configuration and training metadata.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint.get("args", {})
    horizon = checkpoint.get("horizon", config.get("horizon", 128))
    if horizon != 128:
        raise ValueError("This tutorial evaluates the paper's 128-action Inverter.")
    model = IWM_Transformer(
        state_dim=4, goal_dim=4, action_dim=2, horizon=horizon,
        d_model=config.get("d_model", 192), n_heads=config.get("n_heads", 6),
        n_layers=config.get("n_layers", 4), d_ff=config.get("d_ff", 576),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, {"epoch": checkpoint.get("epoch"), "horizon": horizon}


def run_condition(model, env, device, episodes, eval_seed, replan_every, save_trajectory):
    import torch

    forward_seconds = []
    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def plan(state, goal):
        s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        g = torch.tensor(goal, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            synchronize()
            started = time.perf_counter()
            actions = model(s, g)
            synchronize()
            forward_seconds.append(time.perf_counter() - started)
            return actions[0].cpu().numpy()

    for _ in range(10):
        plan(np.zeros(4, dtype=np.float32), np.zeros(4, dtype=np.float32))
    forward_seconds.clear()
    started = time.perf_counter()
    rows = [evaluate_episode(env, plan, eval_seed + i, replan_every, save_trajectory)
            for i in range(episodes)]
    elapsed = time.perf_counter() - started
    mean_return = float(np.mean([row["return"] for row in rows]))
    return {
        "episodes": rows, "mean_return": mean_return,
        "return_std_across_episodes": float(np.std([row["return"] for row in rows])),
        "d4rl_score": float(100.0 * env.get_normalized_score(mean_return)),
        "success_percent": float(100.0 * np.mean([row["success"] for row in rows])),
        "fresh_timing": {
            "device": str(device), "evaluation_wall_seconds": elapsed,
            "mean_inverter_forward_ms": 1000.0 * float(np.mean(forward_seconds)),
            "controller_calls": len(forward_seconds),
            "note": "Fresh measurement; excludes 10 warm-up calls. Forward timing excludes tensor preparation, device-to-host transfer, and simulator stepping. This is not a reproduction of the historical end-to-end timing table.",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=(*LABELS, "all"), default="default")
    parser.add_argument("--checkpoint", type=Path, help="Optional trusted custom checkpoint; incompatible with --model all.")
    parser.add_argument("--replan-every", type=int, nargs="+", choices=(16, 32, 64, 128), default=[64])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--eval-seed", type=int, default=777)
    parser.add_argument("--device", default="cpu", help="cpu or an available CUDA device, e.g. cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "maze2d_umaze")
    parser.add_argument("--save-trajectories", action="store_true")
    args = parser.parse_args()
    if args.episodes < 1 or args.cpu_threads < 1:
        parser.error("--episodes and --cpu-threads must be positive.")
    if args.checkpoint is not None and args.model == "all":
        parser.error("Use a single --model with --checkpoint.")
    import torch
    from inverter.envs.maze2d import make_umaze

    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        parser.error("Use cpu or cuda[:device_index].")
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu or install a compatible PyTorch build.")
    torch.set_num_threads(args.cpu_threads)
    reference = json.loads((ROOT / "reference_results/maze2d_umaze/scores.json").read_text())
    protocol = json.loads((ROOT / "reference_results/maze2d_umaze/protocol.json").read_text())
    labels = LABELS if args.model == "all" else (args.model,)
    args.output.mkdir(parents=True, exist_ok=True)
    summaries = []
    env = make_umaze()
    try:
        for label in labels:
            path = args.checkpoint or Path(__file__).resolve().parent / "checkpoints" / label / "inverter.pt"
            if not path.is_file():
                parser.error(f"Checkpoint not found: {path}")
            model_hash = sha256(path)
            if args.checkpoint is None and model_hash != reference["models"][label]["checkpoint_sha256"]:
                parser.error(f"Bundled checkpoint checksum mismatch: {path}")
            model, checkpoint_info = load_inverter(path, device)
            for k in dict.fromkeys(args.replan_every):
                result = run_condition(model, env, device, args.episodes, args.eval_seed, k, args.save_trajectories)
                result.update(
                    model_label=label, checkpoint_sha256=model_hash, checkpoint=checkpoint_info,
                    replan_every=k, eval_seed=args.eval_seed, episode_count=args.episodes,
                    protocol=protocol, software={"torch": torch.__version__, "numpy": np.__version__},
                    cpu_threads=args.cpu_threads,
                )
                if (args.episodes == 100 and args.eval_seed == 777
                        and model_hash == reference["models"][label]["checkpoint_sha256"]):
                    expected = reference["models"][label]["scores"][str(k)]
                    result["historical_comparison"] = {
                        "expected_d4rl_score": expected,
                        "difference": result["d4rl_score"] - expected,
                        "note": "Numerical kernels and simulator builds can affect trajectories; no bitwise reproducibility is assumed.",
                    }
                out = args.output / label / f"K{k}"
                out.mkdir(parents=True, exist_ok=True)
                (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
                summaries.append({"model_label": label, "K": k, "d4rl_score": result["d4rl_score"],
                                  "success_percent": result["success_percent"],
                                  "result_file": str((out / "results.json").relative_to(args.output))})
                comparison = result.get("historical_comparison")
                suffix = f"; historical delta {comparison['difference']:+.4f}" if comparison else ""
                print(f"{label} K={k}: score {result['d4rl_score']:.4f}; success {result['success_percent']:.1f}%{suffix}", flush=True)
    finally:
        env.close()
    aggregate = {"conditions": summaries, "four_model_aggregates": {}}
    if len(labels) == 4:
        for k in dict.fromkeys(args.replan_every):
            scores = [row["d4rl_score"] for row in summaries if row["K"] == k]
            aggregate["four_model_aggregates"][str(k)] = {
                "mean": float(np.mean(scores)), "population_std": float(np.std(scores)),
                "unit_of_replication": "four released checkpoint families",
            }
    (args.output / "summary.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    print(f"Results: {args.output / 'summary.json'}")


if __name__ == "__main__":
    main()
