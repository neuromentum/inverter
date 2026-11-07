"""Train a U-Maze action-sequence Inverter through a frozen forward model.

The archived recipe uses --no_episode_aware (flat window sampling), a
128-step output horizon, and dense/occupancy weights 5/5. --seed controls
NumPy window sampling and live-evaluation seeds; it did not seed PyTorch in
the archived training runs. Live per-epoch diagnostics retain the historical
protocol; the separate evaluate.py implements the reported benchmark.
"""

import sys, os, argparse, json, time
os.environ['D4RL_SUPPRESS_IMPORT_ERROR'] = '1'

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ─── Maze configs (d4rl v1) ───────────────────────────────────────────

MAZE_CONFIGS = {
    'umaze': {
        'env_id': 'maze2d-umaze-v1',
        'horizon': 128,
        'eval_max_steps': 300,
        'fm_path': 'examples/maze2d_umaze/checkpoints/default/forward.pt',
        'data_dir': 'data/maze2d_v1/umaze',
    },
}

# ─── Args ───────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument('--maze', type=str, default='umaze', choices=['umaze'])
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--epochs', type=int, default=2000)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--segment_length', type=int, default=16)
parser.add_argument('--n_train', type=int, default=100_000)
parser.add_argument('--n_val', type=int, default=10_000)
parser.add_argument('--lambda_terminal', type=float, default=0.0)
parser.add_argument('--lambda_dense', type=float, default=5.0)
parser.add_argument('--lambda_boundary', type=float, default=5.0)
parser.add_argument('--boundary_mode', type=str, default='binary',
                    choices=['binary', 'frequency'],
                    help="Occupancy grid mode: 'binary' (any visit=1) or "
                         "'frequency' (log-scaled visit counts, penalises "
                         "low-coverage regions)")
parser.add_argument('--lambda_fidelity', type=float, default=0.0,
                    help='Fidelity loss weight: MSE(FM_traj, data_traj). '
                         'Replaces boundary when >0 and boundary=0.')
parser.add_argument('--lambda_action_boundary', type=float, default=0.0,
                    help='Action boundary loss: penalise (position, action) pairs '
                         'outside training data support. Keeps IWM actions '
                         'in-distribution for the FM.')
parser.add_argument('--latent_dim', type=int, default=256)
# transformer_large defaults
parser.add_argument('--d_model', type=int, default=192)
parser.add_argument('--n_heads', type=int, default=6)
parser.add_argument('--n_layers', type=int, default=4)
parser.add_argument('--d_ff', type=int, default=576)
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--name', type=str, default='umaze_inverter')
parser.add_argument('--n_eval_episodes', type=int, default=10,
                    help='Number of eval episodes per epoch for monitor')
parser.add_argument('--goal_threshold', type=float, default=0.45)
parser.add_argument('--K_p', type=float, default=10.0)
parser.add_argument('--K_d', type=float, default=5.0)
parser.add_argument('--out_dir', type=str, default=str(PROJECT_ROOT / 'outputs/maze2d_umaze/inverter'),
                    help='Override output directory (used by experiment launcher)')
parser.add_argument('--data_dir', type=str, default=str(PROJECT_ROOT / 'data/maze2d_v1/umaze'))
parser.add_argument('--fm_path', type=str, default=None,
                    help='FM checkpoint path (relative to project root). '
                         'Falls back to per-maze default if omitted.')
parser.add_argument('--horizon', type=int, default=None,
                    help='Override planning horizon (default: per-maze setting)')
parser.add_argument('--no_episode_aware', action='store_true',
                    help='Disable episode-aware sampling (use flat random indexing like original code)')
parser.add_argument('--n_eval_episodes_override', type=int, default=None,
                    help='Override n_eval_episodes (use 10 to match original code)')
parser.add_argument('--save_every', type=int, default=0,
                    help='Save a checkpoint every N epochs (0=disabled). '
                         'Saved as checkpoint_epoch{N}.pt.')
parser.add_argument('--checkpoint_subdir', type=str, default=None,
                    help='Save periodic checkpoints into this subdirectory (e.g. "checkpoints")')
parser.add_argument('--zscore_loss', action='store_true',
                    help='Normalise each loss component by its running σ (EMA) '
                         'before applying λ weights. Makes λ operate on '
                         'unit-variance signals.')
parser.add_argument('--zscore_momentum', type=float, default=0.99,
                    help='EMA momentum for running σ estimation (default: 0.99)')
parser.add_argument('--zscore_warmup', type=int, default=100,
                    help='Number of batches before zscore normalisation activates '
                         '(default: 100, returns σ=1.0 during warmup)')
args = parser.parse_args()

maze_cfg = MAZE_CONFIGS[args.maze]
HORIZON = args.horizon if args.horizon is not None else maze_cfg['horizon']
ENV_ID = maze_cfg['env_id']
EVAL_MAX_STEPS = maze_cfg['eval_max_steps']

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

from inverter.models.iwm_transformer import IWM_Transformer
from inverter.models.iwm_v2 import MultipleShooting, build_occupancy_grid, build_state_action_occupancy
from inverter.models.chunk_forward_model import ChunkForwardModelFull
from inverter.models.transformer_forward_model import TransformerForwardModel
from inverter.models.universal_transformer_fm import UniversalTransformerFM

from inverter.envs.maze2d import make_umaze


if args.out_dir:
    OUT_DIR = Path(args.out_dir)
else:
    OUT_DIR = PROJECT_ROOT / 'results' / 'iwm_v1' / args.name
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Save config.json ─────────────────────────────────────────────────
config_file = OUT_DIR / 'config.json'
if not config_file.exists():
    from datetime import datetime
    _training_keys = [
        'maze', 'epochs', 'batch_size', 'lr', 'segment_length',
        'n_train', 'n_val', 'lambda_terminal',
        'lambda_dense', 'lambda_boundary', 'boundary_mode', 'lambda_fidelity',
        'lambda_action_boundary', 'latent_dim', 'd_model', 'n_heads',
        'n_layers', 'd_ff', 'seed', 'fm_path', 'horizon',
        'n_eval_episodes', 'goal_threshold', 'K_p', 'K_d', 'no_episode_aware', 'data_dir',
        'zscore_loss', 'zscore_momentum', 'zscore_warmup',
    ]
    _cfg = {
        'experiment': {
            'name': args.name,
            'created': datetime.now().isoformat(timespec='seconds'),
            'script': 'examples/maze2d_umaze/train_inverse.py',
            'gpu': args.gpu,
            'python': sys.executable,
            'project_root': str(PROJECT_ROOT),
        },
        'training': {k: getattr(args, k) for k in _training_keys
                     if getattr(args, k, None) is not None},
        'raw_args': sys.argv[1:],
    }
    # Always record the resolved fm_path (even if it came from MAZE_CONFIGS default)
    _cfg['training']['fm_path'] = args.fm_path if args.fm_path else maze_cfg['fm_path']
    config_file.write_text(json.dumps(_cfg, indent=2))

# ─── Running σ estimator for zscore loss ──────────────────────────────

class RunningStd:
    """EMA-based running standard deviation estimator.

    During the first `warmup` updates, returns 1.0 (no normalisation)
    so the early noisy estimates don't create instability.
    After warmup, returns the EMA-tracked σ.
    """

    def __init__(self, momentum: float = 0.99, warmup: int = 100):
        self.momentum = momentum
        self.warmup = warmup
        self.n = 0
        self._mean = 0.0
        self._var = 0.0

    def update(self, x: float) -> float:
        """Update with new observation and return current σ estimate."""
        self.n += 1
        alpha = min(1.0 - self.momentum, 1.0 / self.n)
        delta = x - self._mean
        self._mean += alpha * delta
        self._var = (1.0 - alpha) * (self._var + alpha * delta * delta)
        if self.n < self.warmup:
            return 1.0
        return max(self._var ** 0.5, 1e-8)

    @property
    def std(self) -> float:
        if self.n < self.warmup:
            return 1.0
        return max(self._var ** 0.5, 1e-8)

    def state_dict(self) -> dict:
        return {'n': self.n, 'mean': self._mean, 'var': self._var,
                'momentum': self.momentum, 'warmup': self.warmup}

    def load_state_dict(self, d: dict):
        self.n = d['n']
        self._mean = d['mean']
        self._var = d['var']
        self.momentum = d['momentum']
        self.warmup = d['warmup']


# ─── Dataset ───────────────────────────────────────────────────────────

class WindowDataset(Dataset):
    """Sample (s_0, goal=s_{t+horizon}) pairs, episode-aware.
    When return_trajectory=True, also returns the full ground truth
    trajectory s_{t+1} ... s_{t+horizon} for fidelity loss."""

    def __init__(self, all_states, horizon, n_samples, seed=42,
                 valid_starts=None, return_trajectory=False):
        self.states = all_states
        self.horizon = horizon
        self.n_samples = n_samples
        self.return_trajectory = return_trajectory

        rng = np.random.RandomState(seed)
        if valid_starts is not None and len(valid_starts) > 0:
            # Episode-aware: only sample starts where s[t+horizon] is in same episode
            self.indices = valid_starts[rng.randint(0, len(valid_starts), size=n_samples)]
        else:
            # Fallback: flat indexing (single-episode data like umaze)
            max_start = len(all_states) - horizon - 1
            self.indices = rng.randint(0, max_start, size=n_samples)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        t = self.indices[idx]
        s0 = self.states[t]
        goal = self.states[t + self.horizon]
        if self.return_trajectory:
            # Ground truth trajectory: states[t+1 .. t+horizon] (H steps)
            traj = self.states[t + 1 : t + self.horizon + 1]  # (H, 4)
            return (
                torch.from_numpy(s0.astype(np.float32)),
                torch.from_numpy(goal.astype(np.float32)),
                torch.from_numpy(traj.astype(np.float32)),
            )
        return (
            torch.from_numpy(s0.astype(np.float32)),
            torch.from_numpy(goal.astype(np.float32)),
        )


# ─── Load data ─────────────────────────────────────────────────────────

data_dir = Path(args.data_dir)
print(f'Loading data from {data_dir}/ ({args.maze})...', flush=True)
all_states = np.load(data_dir / 'states.npy').astype(np.float32)
print(f'  States: {all_states.shape}', flush=True)
print(f'  x=[{all_states[:,0].min():.2f}, {all_states[:,0].max():.2f}] '
      f'y=[{all_states[:,1].min():.2f}, {all_states[:,1].max():.2f}]', flush=True)

# ── Episode-aware valid starts ──────────────────────────────────────
if args.n_eval_episodes_override is not None:
    args.n_eval_episodes = args.n_eval_episodes_override
ep_lens_file = data_dir / 'episode_lengths.npy'
valid_starts = None
if args.no_episode_aware:
    print(f'  Episode-aware sampling DISABLED (--no_episode_aware)', flush=True)
elif ep_lens_file.exists():
    ep_lens_raw = np.load(ep_lens_file)
    # episode_lengths may count observations (not transitions)
    if ep_lens_raw.sum() > len(all_states):
        ep_lens = ep_lens_raw - 1
        print(f'  Adjusted episode lengths: obs-based → transition-based '
              f'(sum {ep_lens_raw.sum()} → {ep_lens.sum()})', flush=True)
    else:
        ep_lens = ep_lens_raw
    # For each episode, valid starts are indices t where t+HORIZON < episode end
    vs = []
    ep_start = 0
    for ep_len in ep_lens:
        max_t = ep_start + ep_len - HORIZON  # exclusive
        if max_t > ep_start:
            vs.append(np.arange(ep_start, max_t))
        ep_start += ep_len
    valid_starts = np.concatenate(vs) if vs else None
    n_valid = len(valid_starts) if valid_starts is not None else 0
    print(f'  Episode-aware sampling: {len(ep_lens_raw):,} episodes, '
          f'{n_valid:,} valid starts (horizon={HORIZON})', flush=True)
else:
    print(f'  No episode_lengths.npy found, using flat sampling', flush=True)

need_traj = args.lambda_fidelity > 0
train_ds = WindowDataset(all_states, HORIZON, args.n_train, seed=args.seed,
                         valid_starts=valid_starts, return_trajectory=need_traj)
val_ds = WindowDataset(all_states, HORIZON, args.n_val, seed=args.seed + 1,
                       valid_starts=valid_starts, return_trajectory=need_traj)

train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=4, pin_memory=True, persistent_workers=True)
val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                        num_workers=4, pin_memory=True, persistent_workers=True)

print(f'  Train: {len(train_ds):,} samples, Val: {len(val_ds):,} samples', flush=True)
print(f'  Horizon: {HORIZON}, Env: {ENV_ID}, Eval max steps: {EVAL_MAX_STEPS}', flush=True)


# ─── Load chunk forward model (frozen) ────────────────────────────────

if args.fm_path:
    fm_path_str = args.fm_path
else:
    fm_path_str = maze_cfg['fm_path']
    print(f'  WARNING: --fm_path not set, using default: {fm_path_str}', flush=True)
fm_path = PROJECT_ROOT / fm_path_str
print(f'Loading FM from {fm_path}...', flush=True)
fwd_ckpt = torch.load(fm_path, map_location='cpu', weights_only=False)

fm_type = fwd_ckpt.get('model_type', 'cnn_film')
if fm_type == 'universal_transformer':
    fwd_model = UniversalTransformerFM(
        state_dim=fwd_ckpt['state_dim'],
        action_dim=fwd_ckpt['action_dim'],
        chunk_length=fwd_ckpt['chunk_length'],
        output_dim=fwd_ckpt.get('output_dim', fwd_ckpt['state_dim']),
        d_model=fwd_ckpt.get('d_model', 128),
        n_heads=fwd_ckpt.get('n_heads', 4),
        n_layers=fwd_ckpt.get('n_layers', 4),
        ff_mult=fwd_ckpt.get('ff_mult', 4),
        dropout=fwd_ckpt.get('dropout', 0.1),
    )
    print(f'  FM type: UniversalTransformerFM', flush=True)
elif fm_type == 'transformer_causal':
    fwd_model = TransformerForwardModel(
        state_dim=fwd_ckpt['state_dim'],
        action_dim=fwd_ckpt['action_dim'],
        chunk_length=fwd_ckpt['chunk_length'],
        d_model=fwd_ckpt['d_model'],
        n_heads=fwd_ckpt['n_heads'],
        n_layers=fwd_ckpt['n_layers'],
        d_ff=fwd_ckpt.get('d_ff', 0),
    )
    print(f'  FM type: Transformer (causal)', flush=True)
else:
    fwd_model = ChunkForwardModelFull(
        state_dim=fwd_ckpt['state_dim'],
        action_dim=fwd_ckpt['action_dim'],
        chunk_length=fwd_ckpt['chunk_length'],
        hidden_dim=fwd_ckpt['hidden_dim'],
        n_layers=fwd_ckpt['n_layers'],
    )
    print(f'  FM type: CNN+FiLM', flush=True)
fwd_model.load_state_dict(fwd_ckpt['model_state_dict'])
fwd_model.to(DEVICE)
fwd_model.eval()
for p in fwd_model.parameters():
    p.requires_grad = False

print(f'  FM: val_loss={fwd_ckpt.get("val_loss", "?"):.6f} @epoch {fwd_ckpt.get("epoch", "?")}', flush=True)


# ─── Build IWM ────────────────────────────────────────────────────────

iwm = IWM_Transformer(
    state_dim=4, goal_dim=4, action_dim=2,
    horizon=HORIZON,
    d_model=args.d_model, n_heads=args.n_heads,
    n_layers=args.n_layers, d_ff=args.d_ff,
).to(DEVICE)

n_params = sum(p.numel() for p in iwm.parameters())
print(f'  IWM: {n_params:,} params (horizon={HORIZON})', flush=True)

# Occupancy grid from maze-specific data
states_path = str(data_dir / 'states.npy')
occupancy = build_occupancy_grid(
    states_path, resolution=64, sigma=1.5, mode=args.boundary_mode
) if args.lambda_boundary > 0 else None

# State-action occupancy grid (4D)
action_occupancy = None
if args.lambda_action_boundary > 0:
    actions_path = str(data_dir / 'actions.npy')
    action_occupancy = build_state_action_occupancy(
        states_path, actions_path, pos_resolution=16, act_resolution=16, sigma=1.5)

shooting = MultipleShooting(
    iwm=iwm, forward_model=fwd_model,
    segment_length=args.segment_length,
    lambda_terminal=args.lambda_terminal,
    lambda_dense=args.lambda_dense, lambda_boundary=args.lambda_boundary,
    lambda_fidelity=args.lambda_fidelity,
    lambda_action_boundary=args.lambda_action_boundary,
    occupancy=occupancy,
    action_occupancy=action_occupancy,
    detach_segments=False,  # Smooth FM has stable Jacobians, no need to detach
).to(DEVICE)

optimizer = torch.optim.AdamW(iwm.parameters(), lr=args.lr, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)


# ─── Live evaluation function (d4rl v1) ───────────────────────────────

def run_eval_episodes(iwm_model, fwd_model, n_episodes, seed, max_steps,
                      goal_threshold, K_p, K_d):
    """
    Run n_episodes in d4rl v1 gym env.
    Post-horizon controller: hold position at last actually-reached state.

    Key differences from v3:
      - gym.make (not gymnasium.make)
      - obs is flat (4,), not Dict
      - goal from env._target (2,)
      - step returns 4-tuple (obs, reward, done, info)
    """
    env = make_umaze()
    iwm_model.eval()
    results = []

    for i in range(n_episodes):
        env.seed(seed + i)
        obs = env.reset()
        env.unwrapped.set_target()
        goal = env.unwrapped._target.copy()
        state = obs.astype(np.float32)
        goal_state = np.array([goal[0], goal[1], 0.0, 0.0], dtype=np.float32)

        # Plan
        result = iwm_model.plan(state, goal_state, forward_model=fwd_model)
        actions_plan = result['actions']
        predicted_traj = result['predicted_trajectory']

        # Execute
        executed_positions = [state[:2].copy()]
        success = False
        ep_return = 0.0
        hold_target = None

        for step in range(max_steps):
            if step < len(actions_plan):
                action = actions_plan[step]
            else:
                # Hold position at where we reached after planned actions
                action = (hold_target[:2] - state[:2]) + (0.0 - state[2:])

            action = np.clip(action, -1.0, 1.0)
            obs, reward, done, info = env.step(action.astype(np.float32))
            state = obs.astype(np.float32)
            ep_return += reward
            executed_positions.append(state[:2].copy())

            if step == len(actions_plan) - 1:
                hold_target = state.copy()

            if np.linalg.norm(state[:2] - goal) < goal_threshold and not success:
                success = True

            if done:
                break

        final_dist = np.linalg.norm(state[:2] - goal)
        results.append({
            'start': executed_positions[0].tolist(),
            'goal': goal.tolist(),
            'planned_xy': predicted_traj[:, :2].tolist() if predicted_traj is not None else None,
            'executed_xy': np.array(executed_positions).tolist(),
            'success': bool(success),
            'final_dist': float(final_dist),
            'return': float(ep_return),
        })

    env.close()
    return results


# ─── D4RL score references ────────────────────────────────────────────
D4RL_REFS = {
    'umaze':  {'random': 23.85,  'expert': 161.86},
    'medium': {'random': 13.13,  'expert': 277.39},
    'large':  {'random': 6.7,    'expert': 273.99},
}

d4rl_ref = D4RL_REFS[args.maze]


# ─── Training loop ────────────────────────────────────────────────────

print(f'\nTraining {args.maze} for {args.epochs} epochs on GPU {args.gpu}...', flush=True)
print(f'  λ_terminal={args.lambda_terminal}, λ_dense={args.lambda_dense}, '
      f'λ_boundary={args.lambda_boundary}, λ_fidelity={args.lambda_fidelity}', flush=True)
print(f'  Arch: d_model={args.d_model}, n_heads={args.n_heads}, n_layers={args.n_layers}, d_ff={args.d_ff}', flush=True)
print(f'  Live eval: {args.n_eval_episodes} episodes per epoch', flush=True)

progress_file = OUT_DIR / 'progress.json'
history_file = OUT_DIR / 'history.json'
eval_file = OUT_DIR / 'eval_episodes.json'

# ── Z-score sigma trackers ──
sigma_trackers = None
if args.zscore_loss:
    sigma_trackers = {
        'dense':    RunningStd(args.zscore_momentum, args.zscore_warmup),
        'terminal': RunningStd(args.zscore_momentum, args.zscore_warmup),
        'boundary': RunningStd(args.zscore_momentum, args.zscore_warmup),
        'fidelity': RunningStd(args.zscore_momentum, args.zscore_warmup),
    }
    print(f'  Z-score loss: momentum={args.zscore_momentum}, warmup={args.zscore_warmup}', flush=True)

best_val_loss = float('inf')
best_epoch = 0
train_history = []
val_history = []
eval_sr_history = []
d4rl_history = []
best_d4rl_score = float('-inf')
best_d4rl_epoch = 0
best_d4rl_sr = 0.0
start_epoch = 0

# ── Resume from checkpoint if exists ──
resume_ckpt = OUT_DIR / 'checkpoint.pt'
if resume_ckpt.exists():
    print(f'  Resuming from {resume_ckpt}...', flush=True)
    ckpt = torch.load(resume_ckpt, map_location=DEVICE, weights_only=False)
    iwm.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    start_epoch = ckpt['epoch'] + 1
    best_val_loss = ckpt['best_val_loss']
    best_epoch = ckpt['best_epoch']
    train_history = ckpt['train_history']
    val_history = ckpt['val_history']
    eval_sr_history = ckpt['eval_sr_history']
    d4rl_history = ckpt.get('d4rl_history', [])
    best_d4rl_score = ckpt.get('best_d4rl_score', float('-inf'))
    best_d4rl_epoch = ckpt.get('best_d4rl_epoch', 0)
    best_d4rl_sr = ckpt.get('best_d4rl_sr', 0.0)
    if sigma_trackers is not None and 'sigma_trackers' in ckpt:
        for k, sd in ckpt['sigma_trackers'].items():
            if k in sigma_trackers:
                sigma_trackers[k].load_state_dict(sd)
    print(f'  Resumed at epoch {start_epoch}/{args.epochs}, best_val={best_val_loss:.4f} @{best_epoch+1}', flush=True)

t_start = time.time()

for epoch in range(start_epoch, args.epochs):
    # ── Train ──
    iwm.train()
    epoch_losses = []

    for batch_idx, batch in enumerate(train_loader):
        if need_traj:
            s0, goal, gt_traj = batch
            gt_traj = gt_traj.to(DEVICE, non_blocking=True)
        else:
            s0, goal = batch
            gt_traj = None
        s0 = s0.to(DEVICE, non_blocking=True)
        goal = goal.to(DEVICE, non_blocking=True)

        # ── Z-score: adjust lambdas before compute_loss ──
        if sigma_trackers is not None and batch_idx > 0:
            # Use previous-batch sigmas to set effective lambdas = λ / σ
            shooting.lambda_dense    = args.lambda_dense    / sigma_trackers['dense'].std
            shooting.lambda_terminal = args.lambda_terminal / sigma_trackers['terminal'].std
            shooting.lambda_boundary = args.lambda_boundary / sigma_trackers['boundary'].std
            shooting.lambda_fidelity = args.lambda_fidelity / sigma_trackers['fidelity'].std

        loss, loss_dict = shooting.compute_loss(s0, goal, gt_trajectory=gt_traj)

        # ── Z-score: update running σ trackers with raw (unweighted) losses ──
        if sigma_trackers is not None:
            sigma_trackers['dense'].update(loss_dict['dense'])
            sigma_trackers['terminal'].update(loss_dict['terminal'])
            sigma_trackers['boundary'].update(loss_dict['boundary'])
            sigma_trackers['fidelity'].update(loss_dict['fidelity'])

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(iwm.parameters(), 1.0)
        optimizer.step()

        epoch_losses.append(loss_dict)

    scheduler.step()
    avg_train = {k: float(np.mean([d[k] for d in epoch_losses])) for k in epoch_losses[0]}
    train_history.append(avg_train)

    # ── Validate (always use original lambdas for comparable val loss) ──
    if sigma_trackers is not None:
        shooting.lambda_dense    = args.lambda_dense
        shooting.lambda_terminal = args.lambda_terminal
        shooting.lambda_boundary = args.lambda_boundary
        shooting.lambda_fidelity = args.lambda_fidelity

    iwm.eval()
    val_losses = []
    with torch.no_grad():
        for batch in val_loader:
            if need_traj:
                s0, goal, gt_traj = batch
                gt_traj = gt_traj.to(DEVICE, non_blocking=True)
            else:
                s0, goal = batch
                gt_traj = None
            s0 = s0.to(DEVICE, non_blocking=True)
            goal = goal.to(DEVICE, non_blocking=True)
            _, loss_dict = shooting.compute_loss(s0, goal, gt_trajectory=gt_traj)
            val_losses.append(loss_dict)

    avg_val = {k: float(np.mean([d[k] for d in val_losses])) for k in val_losses[0]}
    val_history.append(avg_val)

    # ── Live evaluation (d4rl v1 env) ──
    eval_results = run_eval_episodes(
        iwm, fwd_model, args.n_eval_episodes,
        seed=args.seed + epoch * 100,
        max_steps=EVAL_MAX_STEPS,
        goal_threshold=args.goal_threshold,
        K_p=args.K_p, K_d=args.K_d,
    )
    n_success = sum(e['success'] for e in eval_results)
    eval_sr = n_success / len(eval_results)
    eval_sr_history.append(eval_sr)
    mean_return = np.mean([e['return'] for e in eval_results])
    d4rl_score = 100.0 * (mean_return - d4rl_ref['random']) / (d4rl_ref['expert'] - d4rl_ref['random'])
    d4rl_history.append(d4rl_score)

    # Save eval episodes for monitor
    eval_file.write_text(json.dumps(eval_results, indent=2))

    # ── CORL-style best D4RL tracking ──
    if d4rl_score > best_d4rl_score:
        best_d4rl_score = d4rl_score
        best_d4rl_epoch = epoch
        best_d4rl_sr = eval_sr
        torch.save({
            'model_state_dict': iwm.state_dict(),
            'epoch': epoch,
            'val_loss': avg_val['total'],
            'eval_sr': eval_sr,
            'd4rl_score': d4rl_score,
            'selection_method': 'best_d4rl',
            'args': vars(args),
            'maze': args.maze,
            'horizon': HORIZON,
        }, OUT_DIR / 'best_d4rl_model.pt')

    # ── Checkpointing (val-loss based) ──
    if avg_val['total'] < best_val_loss:
        best_val_loss = avg_val['total']
        best_epoch = epoch
        torch.save({
            'model_state_dict': iwm.state_dict(),
            'epoch': epoch,
            'val_loss': best_val_loss,
            'eval_sr': eval_sr,
            'args': vars(args),
            'maze': args.maze,
            'horizon': HORIZON,
        }, OUT_DIR / 'best_model.pt')

    # Save resumable checkpoint every epoch
    ckpt_data = {
        'model_state_dict': iwm.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch,
        'train_history': train_history,
        'val_history': val_history,
        'eval_sr_history': eval_sr_history,
        'd4rl_history': d4rl_history,
        'best_d4rl_score': best_d4rl_score,
        'best_d4rl_epoch': best_d4rl_epoch,
        'best_d4rl_sr': best_d4rl_sr,
    }
    if sigma_trackers is not None:
        ckpt_data['sigma_trackers'] = {k: v.state_dict() for k, v in sigma_trackers.items()}
    torch.save(ckpt_data, OUT_DIR / 'checkpoint.pt')

    # Periodic checkpoint saving (for convergence analysis)
    if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
        ckpt_dir = OUT_DIR / args.checkpoint_subdir if args.checkpoint_subdir else OUT_DIR
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': iwm.state_dict(),
            'epoch': epoch,
            'val_loss': avg_val['total'],
            'eval_sr': eval_sr,
            'args': vars(args),
            'maze': args.maze,
            'horizon': HORIZON,
        }, ckpt_dir / f'checkpoint_epoch{epoch+1}.pt')

    # ── Logging ──
    elapsed = time.time() - t_start
    epochs_done = epoch - start_epoch + 1
    eta = elapsed / epochs_done * (args.epochs - epoch - 1) if epochs_done > 0 else 0

    progress = {
        'maze': args.maze,
        'epoch': epoch + 1,
        'total_epochs': args.epochs,
        'train': avg_train,
        'val': avg_val,
        'eval_sr': eval_sr,
        'eval_d4rl': d4rl_score,
        'eval_mean_return': mean_return,
        'eval_sr_history': eval_sr_history,
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch + 1,
        'best_d4rl_score': best_d4rl_score,
        'best_d4rl_epoch': best_d4rl_epoch + 1,
        'best_d4rl_sr': best_d4rl_sr,
        'n_evaluations': len(d4rl_history),
        'elapsed_s': round(elapsed, 1),
        'eta_s': round(eta, 1),
        'lr': scheduler.get_last_lr()[0],
    }
    if sigma_trackers is not None:
        progress['sigma'] = {k: round(v.std, 6) for k, v in sigma_trackers.items()}
    progress_file.write_text(json.dumps(progress, indent=2))
    history_file.write_text(json.dumps({
        'train': train_history,
        'val': val_history,
        'eval_sr': eval_sr_history,
        'd4rl': d4rl_history,
        'args': vars(args),
    }))

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(
            f'Epoch {epoch+1:3d}/{args.epochs} | '
            f'Train: {avg_train["total"]:.4f} | '
            f'Val: {avg_val["total"]:.4f} | '
            f'SR: {eval_sr:.0%} D4RL: {d4rl_score:.1f} | '
            f'Best: {best_val_loss:.4f} @{best_epoch+1} | '
            f'BestD4RL: {best_d4rl_score:.1f} @{best_d4rl_epoch+1} | '
            f'ETA: {eta/60:.0f}min',
            flush=True,
        )

# ─── Final ─────────────────────────────────────────────────────────────

total_time = time.time() - t_start

print(f'\nDone! {total_time/60:.1f} min', flush=True)
print(f'  === Model Selection by Val Loss (Our Method) ===', flush=True)
print(f'  Best val loss: {best_val_loss:.4f} @ epoch {best_epoch+1}', flush=True)
print(f'  Final SR: {eval_sr_history[-1]:.0%}, Best SR: {max(eval_sr_history):.0%}', flush=True)
print(f'  === Model Selection by D4RL Score (CORL Method) ===', flush=True)
print(f'  Best D4RL: {best_d4rl_score:.1f} @ epoch {best_d4rl_epoch+1}', flush=True)
print(f'  SR at best D4RL: {best_d4rl_sr:.0%}', flush=True)
print(f'  Lottery tickets used: {len(d4rl_history)}', flush=True)
if d4rl_history:
    print(f'  Gap (Best - Last): {best_d4rl_score - d4rl_history[-1]:+.1f}', flush=True)
print(f'  Saved to: {OUT_DIR}', flush=True)

# Final benchmark evaluation is explicit and separate from training diagnostics.
print('Run examples/maze2d_umaze/evaluate.py for the fixed-goal paper benchmark.', flush=True)
