"""Train the 16-step U-Maze forward model used in the paper.

Run from an editable installation of inverter. The default data and output
paths are resolved relative to this checkout. Model math, data sampling and
optimization follow the archived trainer. --seed controls NumPy window
sampling; the archived trainer did not seed PyTorch initialization or dropout.
"""

import sys, os, argparse, json, time

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from datetime import datetime
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ─── Args ───────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument('--maze', type=str, default='umaze', choices=['umaze'])
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--chunk_length', type=int, default=16)
parser.add_argument('--d_model', type=int, default=128)
parser.add_argument('--n_heads', type=int, default=4)
parser.add_argument('--n_layers', type=int, default=4)
parser.add_argument('--ff_mult', type=int, default=4)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--s0_noise', type=float, default=0.01,
                    help='Gaussian noise std added to s0 during training')
parser.add_argument('--n_train', type=int, default=500_000)
parser.add_argument('--n_val', type=int, default=50_000)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--name', type=str, default=None)
parser.add_argument('--out_dir', type=Path, default=PROJECT_ROOT / 'outputs/maze2d_umaze/forward')
parser.add_argument('--data_dir', type=Path, default=PROJECT_ROOT / 'data/maze2d_v1/umaze')
parser.add_argument('--save_every', type=int, default=0,
                    help='Save checkpoint every N epochs (0=disabled)')
parser.add_argument('--save_all', action='store_true',
                    help='Save every epoch (shortcut for --save_every 1, saves to checkpoints/ subdir)')
parser.add_argument('--resume', action='store_true',
                    help='Resume training from latest_checkpoint.pt in out_dir')
# Z-score loss args
parser.add_argument('--no_episode_aware', action='store_true',
                    help='Disable episode-boundary filtering; use flat random indexing')
parser.add_argument('--zscore_loss', action='store_true',
                    help='Enable per-group z-score loss normalisation (pos/vel)')
parser.add_argument('--zscore_momentum', type=float, default=0.99,
                    help='EMA momentum for running σ')
parser.add_argument('--zscore_warmup', type=int, default=100,
                    help='Number of batches before σ normalisation activates')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

from inverter.models.universal_transformer_fm import UniversalTransformerFM



# Auto-generate name if not provided
if args.name is None:
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    args.name = f'universal_smooth{args.s0_noise}_flat_{ts}'

if args.out_dir:
    OUT_DIR = Path(args.out_dir)
else:
    OUT_DIR = PROJECT_ROOT / 'maze2d' / args.maze / 'fm' / args.name
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ─── Z-score running σ tracker ─────────────────────────────────────────

class RunningStd:
    """EMA-based running standard deviation estimator.
    During the first `warmup` updates, returns 1.0 (no normalisation).
    """
    def __init__(self, momentum=0.99, warmup=100):
        self.momentum = momentum
        self.warmup = warmup
        self.n = 0
        self._mean = 0.0
        self._var = 0.0

    def update(self, x):
        self.n += 1
        alpha = min(1.0 - self.momentum, 1.0 / self.n)
        delta = x - self._mean
        self._mean += alpha * delta
        self._var = (1.0 - alpha) * (self._var + alpha * delta * delta)
        if self.n < self.warmup:
            return 1.0
        return max(self._var ** 0.5, 1e-8)

    @property
    def std(self):
        if self.n < self.warmup:
            return 1.0
        return max(self._var ** 0.5, 1e-8)

    def state_dict(self):
        return {'n': self.n, 'mean': self._mean, 'var': self._var,
                'momentum': self.momentum, 'warmup': self.warmup}

    def load_state_dict(self, d):
        self.n = d['n']
        self._mean = d['mean']
        self._var = d['var']
        self.momentum = d['momentum']
        self.warmup = d['warmup']


# ─── Dataset ───────────────────────────────────────────────────────────

class ChunkWindowDataset(Dataset):
    def __init__(self, states, actions, chunk_length, n_samples, valid_starts, seed=42):
        self.states = states
        self.actions = actions
        self.L = chunk_length
        self.n_samples = n_samples
        rng = np.random.RandomState(seed)
        self.indices = valid_starts[rng.randint(0, len(valid_starts), size=n_samples)]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        t = self.indices[idx]
        s0 = self.states[t]
        actions = self.actions[t:t + self.L]
        targets = self.states[t + 1:t + 1 + self.L]
        return (
            torch.from_numpy(s0.astype(np.float32)),
            torch.from_numpy(actions.astype(np.float32)),
            torch.from_numpy(targets.astype(np.float32)),
        )


# ─── Load data ─────────────────────────────────────────────────────────

print(f'\n{"="*70}')
print(f'Training UniversalTransformerFM for maze2d-{args.maze}-v1')
print(f'{"="*70}', flush=True)

data_dir = args.data_dir
all_states = np.load(data_dir / 'states.npy').astype(np.float32)
all_actions = np.load(data_dir / 'actions.npy').astype(np.float32)
print(f'  States: {all_states.shape}, Actions: {all_actions.shape}', flush=True)
print(f'  x=[{all_states[:,0].min():.3f}, {all_states[:,0].max():.3f}] '
      f'y=[{all_states[:,1].min():.3f}, {all_states[:,1].max():.3f}]', flush=True)

# Episode boundaries
L = args.chunk_length
max_start = len(all_actions) - L

if args.no_episode_aware:
    valid_starts = np.arange(max_start)
    print(f'  Episode-aware sampling DISABLED (--no_episode_aware)', flush=True)
    print(f'  Valid starts: {len(valid_starts):,} / {max_start:,}', flush=True)
else:
    diffs = np.linalg.norm(np.diff(all_states[:, :2], axis=0), axis=1)
    jump_threshold = {'umaze': 0.3, 'medium': 0.5, 'large': 0.8}[args.maze]
    jump_indices = set(np.where(diffs > jump_threshold)[0])

    valid_starts = []
    for t in range(max_start):
        window_bad = False
        for j in range(t, t + L):
            if j in jump_indices:
                window_bad = True
                break
        if not window_bad:
            valid_starts.append(t)
    valid_starts = np.array(valid_starts)
    print(f'  Valid starts: {len(valid_starts):,} / {max_start:,}', flush=True)

train_ds = ChunkWindowDataset(all_states, all_actions, L, args.n_train, valid_starts, seed=args.seed)
val_ds = ChunkWindowDataset(all_states, all_actions, L, args.n_val, valid_starts, seed=args.seed + 1)

train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=4, pin_memory=True, persistent_workers=True)
val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                        num_workers=4, pin_memory=True, persistent_workers=True)

# Fixed examples for monitoring
rng_ex = np.random.RandomState(999)
example_indices = valid_starts[rng_ex.randint(0, len(valid_starts), size=10)]
example_s0 = all_states[example_indices].astype(np.float32)
example_actions = np.stack([all_actions[t:t+L] for t in example_indices]).astype(np.float32)
example_targets = np.stack([all_states[t+1:t+1+L] for t in example_indices]).astype(np.float32)


# ─── Build model ──────────────────────────────────────────────────────

model = UniversalTransformerFM(
    state_dim=4,
    action_dim=2,
    chunk_length=L,
    d_model=args.d_model,
    n_heads=args.n_heads,
    n_layers=args.n_layers,
    ff_mult=args.ff_mult,
    dropout=args.dropout,
).to(DEVICE)

n_params = sum(p.numel() for p in model.parameters())
print(f'  Model: {model}', flush=True)
print(f'  Parameters: {n_params:,}', flush=True)
print(f'  s0_noise: {args.s0_noise}', flush=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
loss_fn = nn.MSELoss()

# Z-score loss trackers (pos=dims 0:2, vel=dims 2:4)
if args.zscore_loss:
    sigma_trackers = {
        'pos': RunningStd(momentum=args.zscore_momentum, warmup=args.zscore_warmup),
        'vel': RunningStd(momentum=args.zscore_momentum, warmup=args.zscore_warmup),
    }
    print(f'  Z-score loss: momentum={args.zscore_momentum}, warmup={args.zscore_warmup}', flush=True)

# Save config
config = {
    'experiment': {
        'name': args.name,
        'created': datetime.now().isoformat(timespec='seconds'),
        'script': 'examples/maze2d_umaze/train_forward.py',
        'gpu': args.gpu,
        'python': sys.executable,
    },
    'model': {
        'class': 'UniversalTransformerFM',
        'state_dim': 4,
        'action_dim': 2,
        'chunk_length': L,
        'd_model': args.d_model,
        'n_heads': args.n_heads,
        'n_layers': args.n_layers,
        'ff_mult': args.ff_mult,
        'dropout': args.dropout,
        'params': n_params,
    },
    'training': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
}
(OUT_DIR / 'config.json').write_text(json.dumps(config, indent=2))

np.savez(OUT_DIR / 'examples.npz',
         s0=example_s0, actions=example_actions,
         targets=example_targets[:, :, :2])


# ─── Training loop ────────────────────────────────────────────────────

print(f'\nTraining for {args.epochs} epochs on GPU {args.gpu}...', flush=True)

best_val_loss = float('inf')
best_epoch = 0
train_history = []
val_history = []
start_epoch = 0

# Resume from checkpoint if requested
if args.resume and (OUT_DIR / 'latest_checkpoint.pt').exists():
    print('\nResuming from latest_checkpoint.pt...', flush=True)
    ckpt = torch.load(OUT_DIR / 'latest_checkpoint.pt', map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    start_epoch = ckpt['epoch'] + 1
    best_val_loss = ckpt['best_val_loss']
    best_epoch = ckpt['best_epoch']
    train_history = ckpt.get('train_history', [])
    val_history = ckpt.get('val_history', [])
    if args.zscore_loss and 'sigma_trackers' in ckpt:
        for g, sd in ckpt['sigma_trackers'].items():
            if g in sigma_trackers:
                sigma_trackers[g].load_state_dict(sd)
    print(f'  Resuming from epoch {start_epoch} (best_val={best_val_loss:.6f}@{best_epoch+1})', flush=True)
    del ckpt

t_start = time.time()

for epoch in range(start_epoch, args.epochs):
    # ── Train ──
    model.train()
    epoch_losses = []

    for s0, actions, s_targets in train_loader:
        s0 = s0.to(DEVICE, non_blocking=True)
        actions = actions.to(DEVICE, non_blocking=True)
        s_targets = s_targets.to(DEVICE, non_blocking=True)

        # Add noise to s0 for robustness
        if args.s0_noise > 0:
            s0 = s0 + torch.randn_like(s0) * args.s0_noise

        s_pred = model(s0, actions)

        if args.zscore_loss:
            mse_pos = ((s_pred[:, :, :2] - s_targets[:, :, :2]) ** 2).mean()
            mse_vel = ((s_pred[:, :, 2:] - s_targets[:, :, 2:]) ** 2).mean()
            sig_pos = sigma_trackers['pos'].update(mse_pos.item())
            sig_vel = sigma_trackers['vel'].update(mse_vel.item())
            loss = mse_pos / sig_pos + mse_vel / sig_vel
        else:
            loss = loss_fn(s_pred, s_targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        epoch_losses.append(loss.item())

    scheduler.step()
    avg_train_loss = float(np.mean(epoch_losses))
    train_history.append(avg_train_loss)

    # ── Validate ──
    model.eval()
    val_losses = []
    per_step_pos = np.zeros(L)
    per_step_vel = np.zeros(L)
    per_step_count = 0

    with torch.no_grad():
        for s0, actions, s_targets in val_loader:
            s0 = s0.to(DEVICE, non_blocking=True)
            actions = actions.to(DEVICE, non_blocking=True)
            s_targets = s_targets.to(DEVICE, non_blocking=True)

            s_pred = model(s0, actions)
            loss = loss_fn(s_pred, s_targets)
            val_losses.append(loss.item())

            step_pos = ((s_pred[:, :, :2] - s_targets[:, :, :2]) ** 2).mean(dim=(0, 2))
            per_step_pos += step_pos.cpu().numpy()
            step_vel = ((s_pred[:, :, 2:] - s_targets[:, :, 2:]) ** 2).mean(dim=(0, 2))
            per_step_vel += step_vel.cpu().numpy()
            per_step_count += 1

    avg_val_loss = float(np.mean(val_losses))
    val_history.append(avg_val_loss)
    per_step_pos /= per_step_count
    per_step_vel /= per_step_count

    # ── Save predictions ──
    with torch.no_grad():
        ex_s0 = torch.from_numpy(example_s0).to(DEVICE)
        ex_act = torch.from_numpy(example_actions).to(DEVICE)
        ex_pred = model(ex_s0, ex_act)[:, :, :2].cpu().numpy()
    np.save(OUT_DIR / 'example_preds.npy', ex_pred)

    # ── Checkpoint ──
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_epoch = epoch
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_type': 'universal_transformer',
            'model_class': 'UniversalTransformerFM',
            'state_dim': 4,
            'action_dim': 2,
            'output_dim': 4,
            'chunk_length': L,
            'd_model': args.d_model,
            'n_heads': args.n_heads,
            'n_layers': args.n_layers,
            'ff_mult': args.ff_mult,
            'dropout': args.dropout,
            'hidden_dim': args.d_model,  # compat key for IWM loading
            's0_noise': args.s0_noise,
            'epoch': epoch,
            'val_loss': best_val_loss,
            'maze': args.maze,
            'pipeline': 'v1',
            'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        }, OUT_DIR / 'best_model.pt')

    # ── Log ──
    elapsed = time.time() - t_start
    eta = elapsed / (epoch + 1) * (args.epochs - epoch - 1)

    progress = {
        'maze': args.maze,
        'model': 'UniversalTransformerFM',
        'epoch': epoch + 1,
        'total_epochs': args.epochs,
        'train_loss': avg_train_loss,
        'val_loss': avg_val_loss,
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch + 1,
        'elapsed_s': round(elapsed, 1),
        'eta_s': round(eta, 1),
        'lr': scheduler.get_last_lr()[0],
        'per_step_pos_mse': per_step_pos.tolist(),
        'per_step_vel_mse': per_step_vel.tolist(),
    }
    if args.zscore_loss:
        progress['sigma_pos'] = sigma_trackers['pos'].std
        progress['sigma_vel'] = sigma_trackers['vel'].std
    (OUT_DIR / 'progress.json').write_text(json.dumps(progress, indent=2))
    (OUT_DIR / 'history.json').write_text(json.dumps({
        'train': train_history, 'val': val_history, 'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }))

    # Save latest checkpoint for resume
    ckpt_data = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch,
        'train_history': train_history,
        'val_history': val_history,
    }
    if args.zscore_loss:
        ckpt_data['sigma_trackers'] = {g: t.state_dict() for g, t in sigma_trackers.items()}
    torch.save(ckpt_data, OUT_DIR / 'latest_checkpoint.pt')

    # Periodic checkpoint saving
    save_interval = 1 if args.save_all else args.save_every
    if save_interval > 0 and (epoch + 1) % save_interval == 0:
        ckpt_dir = OUT_DIR / 'checkpoints' if args.save_all else OUT_DIR
        ckpt_dir.mkdir(exist_ok=True)
        ckpt_path = ckpt_dir / f'model_ep{epoch+1}.pt'
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_type': 'universal_transformer',
            'model_class': 'UniversalTransformerFM',
            'state_dim': 4,
            'action_dim': 2,
            'output_dim': 4,
            'chunk_length': L,
            'd_model': args.d_model,
            'n_heads': args.n_heads,
            'n_layers': args.n_layers,
            'ff_mult': args.ff_mult,
            'dropout': args.dropout,
            'hidden_dim': args.d_model,
            's0_noise': args.s0_noise,
            'epoch': epoch,
            'val_loss': avg_val_loss,
            'maze': args.maze,
            'pipeline': 'v1',
            'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        }, ckpt_path)

    if (epoch + 1) % 10 == 0 or epoch == 0:
        step_indices = [0, 3, 7, 11, 15]
        step_str = ' '.join([f's{i+1}={per_step_pos[i]:.5f}' for i in step_indices])
        print(
            f'Epoch {epoch+1:3d}/{args.epochs} | '
            f'Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f} | '
            f'Best: {best_val_loss:.6f} @{best_epoch+1} | '
            f'Steps: {step_str} | '
            f'ETA: {eta/60:.0f}min',
            flush=True,
        )

# ─── Done ──────────────────────────────────────────────────────────────

total_time = time.time() - t_start
print(f'\n{"="*70}')
print(f'Training Complete!', flush=True)
print(f'  Model: UniversalTransformerFM', flush=True)
print(f'  Total time: {total_time/60:.1f} min', flush=True)
print(f'  Best val MSE: {best_val_loss:.6f} (RMSE: {np.sqrt(best_val_loss):.4f})', flush=True)
print(f'  Best epoch: {best_epoch+1}/{args.epochs}', flush=True)
print(f'  Checkpoint: {OUT_DIR / "best_model.pt"}', flush=True)
print(f'{"="*70}\n')
