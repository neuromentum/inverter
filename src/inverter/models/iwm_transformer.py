"""
IWM Transformer — Causal Transformer Inverse World Model.

Maps (s_0, goal_state) → actions[128×2]

Design:
- Condition (s_0, goal) is projected to d_model and added to every position
- Learnable position embeddings for each of 128 timesteps
- Causal self-attention: each action token attends to all previous tokens
- Final linear projection → tanh → actions (B, 128, 2)

Same interface as IWM_V2_CNN so it plugs into MultipleShooting unchanged.
"""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional


class IWM_Transformer(nn.Module):
    """
    Causal Transformer inverse world model.

    Input:  [s_0 (4D), goal (4D)] = 8D condition
    Output: [128 × 2] = actions at each timestep (tanh-bounded)
    """

    def __init__(
        self,
        state_dim: int = 4,
        goal_dim: int = 4,
        action_dim: int = 2,
        horizon: int = 128,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.d_model = d_model

        # Condition encoder: (s_0, goal) → d_model
        self.cond_encoder = nn.Sequential(
            nn.Linear(state_dim + goal_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        # Learnable position embeddings for each timestep
        self.pos_embed = nn.Parameter(torch.randn(1, horizon, d_model) * 0.02)

        # Transformer decoder layers
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(
            decoder_layer,
            num_layers=n_layers,
        )

        # Output projection
        self.output_proj = nn.Linear(d_model, action_dim)

        # Small init for output layer
        nn.init.uniform_(self.output_proj.weight, -0.01, 0.01)
        nn.init.zeros_(self.output_proj.bias)

        # Causal mask (cached)
        self.register_buffer(
            'causal_mask',
            torch.triu(torch.ones(horizon, horizon, dtype=torch.bool), diagonal=1),
        )

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

        # Encode condition → (B, d_model)
        cond = torch.cat([state, goal], dim=-1)  # (B, 8)
        c = self.cond_encoder(cond)               # (B, d_model)

        # Broadcast condition to all positions + add positional embeddings
        # (B, 1, d_model) + (1, 128, d_model) → (B, 128, d_model)
        x = c.unsqueeze(1) + self.pos_embed

        # Causal self-attention
        x = self.transformer(x, mask=self.causal_mask)  # (B, 128, d_model)

        # Project to actions
        actions = torch.tanh(self.output_proj(x))  # (B, 128, 2)

        return actions

    def plan(
        self,
        current_state: np.ndarray,
        goal_state: np.ndarray,
        forward_model: nn.Module = None,
    ) -> Dict:
        """
        Plan action sequence for evaluation.
        Same interface as IWM_V2_CNN.plan().
        """
        t0 = time.time()
        device = next(self.parameters()).device

        s = torch.tensor(current_state, dtype=torch.float32, device=device).unsqueeze(0)
        g = torch.tensor(goal_state, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            actions = self.forward(s, g)  # (1, 128, 2)

            if forward_model is not None:
                L = 16
                K = actions.shape[1] // L
                all_states = []
                state = s
                for k in range(K):
                    seg_actions = actions[:, k*L:(k+1)*L, :]
                    seg_states = forward_model(state, seg_actions)
                    all_states.append(seg_states)
                    state = seg_states[:, -1, :]
                pred_states = torch.cat(all_states, dim=1)
                states_np = pred_states[0].cpu().numpy()
            else:
                states_np = None

        actions_np = actions[0].cpu().numpy()

        return {
            'actions': np.clip(actions_np, -1.0, 1.0),
            'predicted_trajectory': states_np,
            'planning_time': time.time() - t0,
        }
