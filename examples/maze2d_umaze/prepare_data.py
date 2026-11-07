"""Download and extract the original D4RL maze2d-umaze-v1 offline data.

The five NumPy arrays preserve the archived extraction convention. Their
SHA-256 hashes are checked against configs/data.json after extraction.
"""

import os
import argparse
import numpy as np
import json
import hashlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Suppress noisy d4rl import warnings
os.environ['D4RL_SUPPRESS_IMPORT_ERROR'] = '1'

from inverter.envs.maze2d import make_umaze

MAZE_CONFIGS = {
    'umaze':  'maze2d-umaze-v1',
}


def extract(maze: str, data_dir: Path):
    env_name = MAZE_CONFIGS[maze]
    save_dir = str(data_dir)
    os.makedirs(save_dir, exist_ok=True)

    print(f'\n{"="*60}')
    print(f'Extracting d4rl v1 data: {env_name}')
    print(f'{"="*60}')

    env = make_umaze()
    dataset = env.get_dataset()

    obs = dataset['observations']        # (N, 4) float32
    acts = dataset['actions']            # (N, 2) float32
    timeouts = dataset['timeouts']       # (N,) bool
    terminals = dataset['terminals']     # (N,) bool
    goals = dataset['infos/goal']        # (N, 2) float32

    N = len(obs)
    print(f'  Total transitions: {N:,}')

    # Build next_states: obs[i+1] for non-boundary steps, obs[i] for last step of episode
    # (same convention as v3 pipeline)
    next_states = np.empty_like(obs)
    next_states[:-1] = obs[1:]
    next_states[-1] = obs[-1]  # last transition — will be overwritten below

    # Fix episode boundaries: at timeout/terminal, next_state should not bleed into next episode
    boundary_mask = timeouts | terminals
    boundary_indices = np.where(boundary_mask)[0]
    next_states[boundary_indices] = obs[boundary_indices]  # self-loop at boundary

    # Compute episode lengths
    episode_ends = np.where(boundary_mask)[0]
    episode_starts = np.concatenate([[0], episode_ends[:-1] + 1])
    episode_lengths = episode_ends - episode_starts + 1

    # If last transition is not a boundary, add the tail episode
    if not boundary_mask[-1]:
        tail_start = episode_ends[-1] + 1 if len(episode_ends) > 0 else 0
        tail_len = N - tail_start
        episode_lengths = np.append(episode_lengths, tail_len)

    print(f'  Episodes: {len(episode_lengths):,}')
    print(f'  Episode lengths: min={episode_lengths.min()}, max={episode_lengths.max()}, '
          f'mean={episode_lengths.mean():.1f}')

    # Verify shapes
    assert obs.shape == (N, 4), f'obs shape {obs.shape}'
    assert acts.shape == (N, 2), f'acts shape {acts.shape}'
    assert next_states.shape == (N, 4), f'next_states shape {next_states.shape}'
    assert goals.shape == (N, 2), f'goals shape {goals.shape}'
    assert episode_lengths.sum() == N, f'episode_lengths sum {episode_lengths.sum()} != {N}'

    # Save
    np.save(os.path.join(save_dir, 'states.npy'), obs.astype(np.float32))
    np.save(os.path.join(save_dir, 'actions.npy'), acts.astype(np.float32))
    np.save(os.path.join(save_dir, 'next_states.npy'), next_states.astype(np.float32))
    np.save(os.path.join(save_dir, 'goals.npy'), goals.astype(np.float32))
    np.save(os.path.join(save_dir, 'episode_lengths.npy'), episode_lengths.astype(np.int64))

    print(f'\n  Saved to {save_dir}/')
    print(f'  States:     {obs.shape} float32')
    print(f'  Actions:    {acts.shape} float32')
    print(f'  NextStates: {next_states.shape} float32')
    print(f'  Goals:      {goals.shape} float32')
    print(f'  EpLengths:  {episode_lengths.shape} int64')

    # Print coordinate ranges for verification
    print(f'\n  Coordinate ranges:')
    print(f'    x:  [{obs[:,0].min():.4f}, {obs[:,0].max():.4f}]')
    print(f'    y:  [{obs[:,1].min():.4f}, {obs[:,1].max():.4f}]')
    print(f'    vx: [{obs[:,2].min():.4f}, {obs[:,2].max():.4f}]')
    print(f'    vy: [{obs[:,3].min():.4f}, {obs[:,3].max():.4f}]')
    print(f'    actions: [{acts.min():.4f}, {acts.max():.4f}]')
    print(f'    goal_x: [{goals[:,0].min():.4f}, {goals[:,0].max():.4f}]')
    print(f'    goal_y: [{goals[:,1].min():.4f}, {goals[:,1].max():.4f}]')

    env.close()
    expected = json.loads((PROJECT_ROOT / 'examples/maze2d_umaze/configs/data.json').read_text())
    for filename, metadata in expected['arrays'].items():
        actual = hashlib.sha256((Path(save_dir) / filename).read_bytes()).hexdigest()
        if actual != metadata['sha256']:
            raise ValueError(f'{filename}: checksum differs from the archived paper data ({actual})')
    print('Verified: extracted arrays match the archived paper dataset.')
    return save_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--maze', type=str, default='umaze',
                        choices=['umaze'])
    parser.add_argument('--data_dir', type=Path, default=PROJECT_ROOT / 'data/maze2d_v1/umaze')
    args = parser.parse_args()
    extract(args.maze, args.data_dir)
