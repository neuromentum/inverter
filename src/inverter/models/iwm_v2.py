"""
IWM v2 — 1D CNN Encoder-Decoder with Multiple Shooting.

Maps (s_0, goal_state) → actions[128×2]

Key design:
- Full goal state [x, y, 0, 0], not just position
- CNN encoder-decoder outputs ONLY actions (not states)
- All position-based losses (dense, boundary) flow through frozen
  forward model that predicts all intermediate states per chunk
- No supervised action loss — trained via forward-model consistency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from typing import Dict, Optional
from scipy.ndimage import gaussian_filter


class IWM_V2_CNN(nn.Module):
    """
    1D CNN encoder-decoder inverse world model.

    Input:  [s_0 (4D), goal (4D)] = 8D condition
    Output: [128 × 2] = actions at each timestep (tanh-bounded)
    """

    def __init__(
        self,
        state_dim: int = 4,
        goal_dim: int = 4,      # full state including velocity
        action_dim: int = 2,
        horizon: int = 128,
        latent_dim: int = 256,
        decoder_channels: tuple = (256, 128, 64, 64, 32, 32, 16),
        output_dim: int = None,  # override output channels (e.g. 6 for v2 compat)
    ):
        super().__init__()

        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.latent_dim = latent_dim
        self._output_dim = output_dim or action_dim

        # Encoder: condition → latent
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + goal_dim, latent_dim),
            nn.GELU(),
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.LayerNorm(latent_dim),
        )

        # Decoder: latent (256×1) → (action_dim×128) via transposed 1D convolutions
        # Each ConvTranspose1d doubles the sequence length: 1→2→4→8→16→32→64→128
        decoder_layers = []
        in_ch = latent_dim
        for i, out_ch in enumerate(decoder_channels):
            decoder_layers.append(
                nn.ConvTranspose1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
            )
            decoder_layers.append(nn.GELU())
            decoder_layers.append(nn.BatchNorm1d(out_ch))
            in_ch = out_ch

        self.decoder = nn.Sequential(*decoder_layers)

        # Final projection — output_dim channels (action_dim for v3, state+action for v2)
        self.output_proj = nn.Conv1d(decoder_channels[-1], self._output_dim, kernel_size=3, padding=1)

        # Initialize output layer with small weights
        nn.init.uniform_(self.output_proj.weight, -0.01, 0.01)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            state: (B, state_dim) current state
            goal:  (B, goal_dim) goal state [x, y, vx, vy]

        Returns:
            actions: (B, horizon, action_dim) = (B, 128, 2), tanh-bounded
        """
        B = state.shape[0]

        # Encode condition
        cond = torch.cat([state, goal], dim=-1)  # (B, 8)
        z = self.encoder(cond)                    # (B, latent_dim)

        # Reshape for 1D conv: (B, latent_dim, 1)
        z = z.unsqueeze(-1)

        # Decode: (B, latent_dim, 1) → (B, decoder_channels[-1], 128)
        h = self.decoder(z)

        # Project to output channels: (B, output_dim, 128)
        out = self.output_proj(h)

        # Transpose to (B, 128, output_dim)
        out = out.transpose(1, 2)

        # Extract action channels (last action_dim channels for v2 compat, all for v3)
        if self._output_dim > self.action_dim:
            # V2 layout: first state_dim channels are states, last action_dim are actions
            actions = torch.tanh(out[:, :, self.state_dim:self.state_dim + self.action_dim])
        else:
            actions = torch.tanh(out)

        return actions  # (B, 128, action_dim)

    def plan(
        self,
        current_state: np.ndarray,
        goal_state: np.ndarray,
        forward_model: nn.Module = None,
    ) -> Dict:
        """
        Plan action sequence for evaluation.

        Args:
            current_state: (4,) current [x, y, vx, vy]
            goal_state: (4,) goal [x, y, 0, 0]
            forward_model: optional full-trajectory forward model for state prediction

        Returns:
            dict with 'actions', 'predicted_trajectory', 'planning_time'
        """
        t0 = time.time()
        device = next(self.parameters()).device

        s = torch.tensor(current_state, dtype=torch.float32, device=device).unsqueeze(0)
        g = torch.tensor(goal_state, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            actions = self.forward(s, g)  # (1, 128, 2)

            # If forward model available, get predicted trajectory
            if forward_model is not None:
                L = 16  # segment length
                K = actions.shape[1] // L
                all_states = []
                state = s
                for k in range(K):
                    seg_actions = actions[:, k*L:(k+1)*L, :]
                    seg_states = forward_model(state, seg_actions)  # (1, L, 4)
                    all_states.append(seg_states)
                    state = seg_states[:, -1, :]  # last state becomes start of next segment
                pred_states = torch.cat(all_states, dim=1)  # (1, 128, 4)
                states_np = pred_states[0].cpu().numpy()
            else:
                states_np = None

        actions_np = actions[0].cpu().numpy()

        return {
            'actions': np.clip(actions_np, -1.0, 1.0),
            'predicted_trajectory': states_np,
            'planning_time': time.time() - t0,
        }


def build_occupancy_grid(
    states_path: str = 'data/maze2d/umaze/states.npy',
    resolution: int = 64,
    sigma: float = 1.5,
    margin: float = 0.1,
    mode: str = 'binary',
) -> dict:
    """
    Build a smooth 2D occupancy grid from training data positions.

    Args:
        mode: 'binary' — any visited cell → 1.0 (original behaviour)
              'frequency' — log-scaled visit counts, so rarely-visited
                            cells get lower occupancy than frequently-visited
                            ones.  Discourages FM from visiting regions
                            where it has little training signal.

    Returns dict with:
        'grid':   (1, 1, H, W) float32 tensor, values in [0, 1]
        'x_min', 'x_max', 'y_min', 'y_max': coordinate bounds
        'mode':  'binary' or 'frequency'
    """
    states = np.load(states_path)
    pos = states[:, :2]  # (N, 2)

    x_min, x_max = pos[:, 0].min() - margin, pos[:, 0].max() + margin
    y_min, y_max = pos[:, 1].min() - margin, pos[:, 1].max() + margin

    grid = np.zeros((resolution, resolution), dtype=np.float32)
    bx = np.clip(((pos[:, 0] - x_min) / (x_max - x_min) * resolution).astype(int), 0, resolution - 1)
    by = np.clip(((pos[:, 1] - y_min) / (y_max - y_min) * resolution).astype(int), 0, resolution - 1)
    for i in range(len(pos)):
        grid[by[i], bx[i]] += 1

    if mode == 'binary':
        # Binary occupancy + Gaussian smoothing for soft gradients
        grid = (grid > 0).astype(np.float32)
    elif mode == 'frequency':
        # Log-scaled frequency: log(1 + count) preserves relative differences
        # while compressing the huge dynamic range of visit counts.
        # Unvisited cells stay at 0; cells with 1 visit get ~0.69;
        # cells with 1000 visits get ~6.9.  After normalisation → [0, 1].
        grid = np.log1p(grid)  # log(1 + count)
    else:
        raise ValueError(f"Unknown occupancy mode: '{mode}'. Expected 'binary' or 'frequency'.")

    grid = gaussian_filter(grid, sigma=sigma)
    if grid.max() > 0:
        grid = grid / grid.max()  # normalise to [0, 1]

    # Stats for logging
    nonzero = (grid > 0.01).sum()
    total = grid.size
    print(f'  Occupancy grid ({mode}): {nonzero}/{total} active cells '
          f'({nonzero/total*100:.1f}%), res={resolution}, sigma={sigma}', flush=True)
    if mode == 'frequency':
        print(f'    Grid value range: [{grid.min():.3f}, {grid.max():.3f}] '
              f'(mean of nonzero: {grid[grid>0.01].mean():.3f})', flush=True)

    # Shape for F.grid_sample: (1, 1, H, W)
    grid_tensor = torch.from_numpy(grid).float().unsqueeze(0).unsqueeze(0)

    return {
        'grid': grid_tensor,
        'x_min': float(x_min), 'x_max': float(x_max),
        'y_min': float(y_min), 'y_max': float(y_max),
        'mode': mode,
    }


def build_state_action_occupancy(
    states_path: str,
    actions_path: str,
    pos_resolution: int = 16,
    act_resolution: int = 16,
    sigma: float = 1.5,
    margin: float = 0.1,
) -> dict:
    """
    Build a 4D occupancy grid over (x, y, a0, a1) from training data.

    At each spatial cell, records which actions were observed.  Penalizes
    IWM actions that are out-of-distribution for the current FM-predicted
    position — stops the IWM from proposing action patterns that would
    drive the FM into OOD territory.

    Returns dict with:
        'grid':    (pos_res*pos_res, 1, act_res, act_res) float32 tensor
        + coordinate bounds and resolutions
    """
    states = np.load(states_path)
    actions = np.load(actions_path)
    pos = states[:, :2]

    x_min = pos[:, 0].min() - margin
    x_max = pos[:, 0].max() + margin
    y_min = pos[:, 1].min() - margin
    y_max = pos[:, 1].max() + margin

    PR, AR = pos_resolution, act_resolution

    # Bin positions
    bx = np.clip(((pos[:, 0] - x_min) / (x_max - x_min) * PR).astype(int), 0, PR - 1)
    by = np.clip(((pos[:, 1] - y_min) / (y_max - y_min) * PR).astype(int), 0, PR - 1)

    # Bin actions: [-1, 1] → [0, act_res)
    ba0 = np.clip(((actions[:, 0] + 1) / 2 * AR).astype(int), 0, AR - 1)
    ba1 = np.clip(((actions[:, 1] + 1) / 2 * AR).astype(int), 0, AR - 1)

    # Build 4D histogram
    grid = np.zeros((PR, PR, AR, AR), dtype=np.float32)
    for i in range(len(pos)):
        grid[bx[i], by[i], ba0[i], ba1[i]] += 1

    raw_occupied = (grid > 0).sum()
    raw_total = grid.size

    # Binary occupancy + Gaussian smoothing for soft gradients
    grid = (grid > 0).astype(np.float32)
    grid = gaussian_filter(grid, sigma=sigma)
    if grid.max() > 0:
        grid = grid / grid.max()  # normalise to [0, 1]

    # Reshape: (PR, PR, AR, AR) → (PR*PR, 1, AR, AR) for per-cell grid_sample
    grid_tensor = torch.from_numpy(
        grid.reshape(PR * PR, 1, AR, AR)
    ).float()

    print(f'  Action occupancy: {raw_occupied}/{raw_total} raw cells occupied '
          f'({raw_occupied / raw_total * 100:.1f}%), '
          f'pos_res={PR}, act_res={AR}, sigma={sigma}', flush=True)

    return {
        'grid': grid_tensor,
        'pos_res': PR,
        'act_res': AR,
        'x_min': float(x_min), 'x_max': float(x_max),
        'y_min': float(y_min), 'y_max': float(y_max),
    }


class MultipleShooting(nn.Module):
    """
    Multiple-shooting loss computation using full-trajectory chunk forward models.

    Breaks the 128-step horizon into K segments of L steps each.
    Each segment is evaluated by a chunk forward model that predicts ALL
    intermediate states from (start_state, L actions) in one shot — no
    sequential rollout, no compounding error, clean gradients.

    ALL position-based losses are computed on forward-model states.
    CNN outputs only actions — no junk state predictions.

    Losses:
    1. L_terminal:  final forward-model state must match goal [x, y, 0, 0]
    2. L_dense:     sum of exp(-dist_to_goal) along ALL forward-model states
    3. L_boundary:  penalise forward-model positions outside training data support
    4. L_fidelity:  MSE between FM-predicted trajectory and ground truth data
                    trajectory — punishes plans whose FM rollout diverges from
                    what would actually happen (linear gradient, no saturation)
    6. L_action_bnd: penalise (position, action) pairs outside training data
                     support.  Keeps IWM actions in-distribution for the FM so
                     it doesn't hallucinate under OOD inputs.
    """

    def __init__(
        self,
        iwm: IWM_V2_CNN,
        forward_model: nn.Module,
        segment_length: int = 16,
        lambda_terminal: float = 10.0,
        lambda_dense: float = 1.0,
        lambda_boundary: float = 5.0,
        lambda_fidelity: float = 0.0,
        lambda_action_boundary: float = 0.0,
        occupancy: Optional[dict] = None,
        action_occupancy: Optional[dict] = None,
        detach_segments: bool = False,
    ):
        super().__init__()
        self.iwm = iwm
        self.forward_model = forward_model
        self.segment_length = segment_length
        self.n_segments = iwm.horizon // segment_length
        self.lambda_terminal = lambda_terminal
        self.lambda_dense = lambda_dense
        self.lambda_boundary = lambda_boundary
        self.lambda_fidelity = lambda_fidelity
        self.lambda_action_boundary = lambda_action_boundary
        self.detach_segments = detach_segments

        # Register occupancy grid as buffer (moves with .to(device), not trained)
        if occupancy is not None:
            self.register_buffer('occ_grid', occupancy['grid'])  # (1, 1, H, W)
            self.occ_x_min = occupancy['x_min']
            self.occ_x_max = occupancy['x_max']
            self.occ_y_min = occupancy['y_min']
            self.occ_y_max = occupancy['y_max']
        else:
            self.occ_grid = None

        # Register state-action occupancy grid
        if action_occupancy is not None:
            self.register_buffer('act_occ_grid', action_occupancy['grid'])  # (PR*PR, 1, AR, AR)
            self.act_occ_pos_res = action_occupancy['pos_res']
            self.act_occ_act_res = action_occupancy['act_res']
            self.act_occ_x_min = action_occupancy['x_min']
            self.act_occ_x_max = action_occupancy['x_max']
            self.act_occ_y_min = action_occupancy['y_min']
            self.act_occ_y_max = action_occupancy['y_max']
        else:
            self.act_occ_grid = None

    def query_occupancy(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Look up occupancy values for a batch of 2D positions.

        Args:
            positions: (B, T, 2) or (B, 2) — x, y coordinates
        Returns:
            occupancy values in [0, 1], same leading shape.
            1 = corridor (training data), 0 = wall (no data)
        """
        assert self.occ_grid is not None, "No occupancy grid registered"
        squeeze = False
        if positions.dim() == 2:
            positions = positions.unsqueeze(1)  # (B, 1, 2)
            squeeze = True

        B, T, _ = positions.shape
        x = positions[:, :, 0]  # (B, T)
        y = positions[:, :, 1]  # (B, T)

        # Normalise to [-1, 1] for F.grid_sample
        gx = 2.0 * (x - self.occ_x_min) / (self.occ_x_max - self.occ_x_min) - 1.0
        gy = 2.0 * (y - self.occ_y_min) / (self.occ_y_max - self.occ_y_min) - 1.0

        # grid_sample expects (N, H_out, W_out, 2) in (x, y) order
        grid_coords = torch.stack([gx, gy], dim=-1)  # (B, T, 2)
        grid_coords = grid_coords.unsqueeze(1)        # (B, 1, T, 2)

        # Expand occupancy grid to batch size
        occ = self.occ_grid.expand(B, -1, -1, -1)     # (B, 1, H, W)

        # Bilinear interpolation — differentiable!
        sampled = F.grid_sample(
            occ, grid_coords,
            mode='bilinear', padding_mode='zeros', align_corners=True
        )  # (B, 1, 1, T)

        result = sampled[:, 0, 0, :]  # (B, T)
        if squeeze:
            result = result.squeeze(1)  # (B,)
        return result

    def query_action_occupancy(
        self, positions: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """
        Look up state-action occupancy: at this position, was this action
        observed in training data?

        Uses nearest-neighbour spatial lookup (non-differentiable w.r.t.
        position — we only need gradients w.r.t. actions) + bilinear
        action interpolation via grid_sample (differentiable).

        Args:
            positions: (B, T, 2) xy positions (pre-action states)
            actions:   (B, T, 2) actions in [-1, 1]
        Returns:
            (B, T) occupancy in [0, 1].  1 = in-distribution, 0 = OOD.
        """
        assert self.act_occ_grid is not None, "No action occupancy grid registered"
        B, T, _ = positions.shape
        PR = self.act_occ_pos_res

        # --- Map position → spatial cell index (nearest-neighbour) ---
        px = positions[:, :, 0]  # (B, T)
        py = positions[:, :, 1]
        ix = ((px - self.act_occ_x_min) / (self.act_occ_x_max - self.act_occ_x_min)
              * (PR - 1)).long().clamp(0, PR - 1)
        iy = ((py - self.act_occ_y_min) / (self.act_occ_y_max - self.act_occ_y_min)
              * (PR - 1)).long().clamp(0, PR - 1)
        cell_idx = ix * PR + iy  # (B, T)

        # Gather per-cell action histograms: (B*T, 1, AR, AR)
        flat_idx = cell_idx.reshape(-1)
        action_hists = self.act_occ_grid[flat_idx]  # (B*T, 1, AR, AR)

        # --- Map actions [-1, 1] → grid_sample coords [-1, 1] ---
        # actions are already in [-1, 1] from IWM tanh output
        # grid_sample (x, y): x indexes W (last dim = a1), y indexes H (= a0)
        a0 = actions[:, :, 0].reshape(-1)  # (B*T,)
        a1 = actions[:, :, 1].reshape(-1)
        grid_coords = torch.stack([a1, a0], dim=-1)          # (B*T, 2)
        grid_coords = grid_coords.unsqueeze(1).unsqueeze(1)  # (B*T, 1, 1, 2)

        sampled = F.grid_sample(
            action_hists, grid_coords,
            mode='bilinear', padding_mode='zeros', align_corners=True
        )  # (B*T, 1, 1, 1)

        return sampled[:, 0, 0, 0].reshape(B, T)

    def compute_loss(
        self,
        start_states: torch.Tensor,    # (B, 4)
        goal_states: torch.Tensor,     # (B, 4) — full state [x,y,0,0]
        gt_trajectory: Optional[torch.Tensor] = None,  # (B, H, 4) ground truth states
    ):
        """
        Compute loss with ALL position-based terms flowing through the frozen
        forward model. CNN only outputs actions.

        Architecture:
            CNN(s₀, goal) → actions[128×2]
            For each segment k=0..K-1:
                FM(state_k, actions[k*L:(k+1)*L]) → seg_states[L×4]
                state_{k+1} = seg_states[-1]
            Concatenate → fm_states[128×4]

        Gradients flow: loss → fm_states → actions → CNN weights.
        Forward model is frozen (no grad). Segments are chained sequentially
        so trajectory is continuous by construction.

        Returns:
            total_loss, loss_dict
        """
        B = start_states.shape[0]
        L = self.segment_length
        K = self.n_segments

        # Forward pass through IWM — actions only
        actions = self.iwm(start_states, goal_states)  # (B, 128, 2)

        goal_pos = goal_states[:, :2]  # (B, 2)

        # --- Sequential forward-model rollout ---
        all_fm_states = []
        state = start_states
        for k in range(K):
            seg_actions = actions[:, k*L:(k+1)*L, :]          # (B, L, 2)
            seg_states = self.forward_model(state, seg_actions)  # (B, L, 4)
            all_fm_states.append(seg_states)
            if self.detach_segments:
                # Detach state between segments to prevent gradient explosion
                # through long FM chains (spectral radius^K can explode).
                # Each segment still gets gradient from dense/boundary losses
                # on its own positions. Terminal loss affects only last segment.
                state = seg_states[:, -1, :].detach()
            else:
                state = seg_states[:, -1, :]  # endpoint → next segment start (WITH gradient)

        fm_states = torch.cat(all_fm_states, dim=1)  # (B, 128, 4)
        final_state = fm_states[:, -1, :]             # (B, 4)

        # --- Loss 1: Terminal (final FM state ≈ goal) ---
        terminal_loss = ((final_state - goal_states) ** 2).mean()

        # --- Loss 2: Dense reward along FM trajectory ---
        fm_pos = fm_states[:, :, :2]  # (B, 128, 2)
        dists = torch.norm(fm_pos - goal_pos.unsqueeze(1), dim=-1)  # (B, 128)
        dense_reward = torch.exp(-dists).mean()
        dense_loss = -dense_reward

        # --- Loss 3: Boundary (keep FM positions within training data support) ---
        if self.occ_grid is not None and self.lambda_boundary > 0:
            occ_vals = self.query_occupancy(fm_pos)  # (B, 128) in [0, 1]
            boundary_loss = (1.0 - occ_vals).mean()
        else:
            boundary_loss = torch.tensor(0.0, device=start_states.device)

        # --- Loss 4: Fidelity (FM trajectory ≈ data trajectory) ---
        # Provides a linear-in-distance gradient (unlike dense which saturates
        # via exp(-d)). Strong signal even when planned trajectory is far off.
        if gt_trajectory is not None and self.lambda_fidelity > 0:
            # Compare FM-predicted positions against ground truth data positions
            fidelity_loss = ((fm_pos - gt_trajectory[:, :, :2]) ** 2).mean()
        else:
            fidelity_loss = torch.tensor(0.0, device=start_states.device)

        # --- Loss 5: Action boundary (state-conditioned action occupancy) ---
        # Penalise IWM actions that are OOD for the current position.
        # Gradient: loss → grid_sample → action coords → IWM weights.
        # Position lookup is nearest-neighbour (non-differentiable) — only
        # used as context, not optimised.
        if self.act_occ_grid is not None and self.lambda_action_boundary > 0:
            # Pre-action positions: state BEFORE each action is applied
            pre_pos = torch.cat([
                start_states[:, :2].unsqueeze(1),   # (B, 1, 2) for t=0
                fm_states[:, :-1, :2].detach(),      # (B, H-1, 2) for t=1..H-1
            ], dim=1)  # (B, H, 2)
            act_occ_vals = self.query_action_occupancy(pre_pos, actions)  # (B, H)
            action_boundary_loss = (1.0 - act_occ_vals).mean()
        else:
            action_boundary_loss = torch.tensor(0.0, device=start_states.device)

        # --- Total ---
        total_loss = (
            self.lambda_dense * dense_loss
            + self.lambda_terminal * terminal_loss
            + self.lambda_boundary * boundary_loss
            + self.lambda_fidelity * fidelity_loss
            + self.lambda_action_boundary * action_boundary_loss
        )

        loss_dict = {
            'total': total_loss.item(),
            'dense': dense_loss.item(),
            'terminal': terminal_loss.item(),
            'boundary': boundary_loss.item() if torch.is_tensor(boundary_loss) else boundary_loss,
            'fidelity': fidelity_loss.item() if torch.is_tensor(fidelity_loss) else fidelity_loss,
            'action_bnd': action_boundary_loss.item() if torch.is_tensor(action_boundary_loss) else action_boundary_loss,
            'dense_reward': dense_reward.item(),
        }

        return total_loss, loss_dict