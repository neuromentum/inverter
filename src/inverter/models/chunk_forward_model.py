"""
Chunk Forward Model — predicts state after L steps given (s₀, a₀, ..., a_{L-1}).

Unlike the single-step BlackBoxForwardModel which must be rolled out sequentially
(compounding errors), this model learns the L-step dynamics in one shot:

    F(s₀, [a₀, a₁, ..., a_{L-1}]) → s_L

Key advantages for IWM v2 training:
- Zero compounding error (learns ground-truth 16/32/64-step transitions)
- Clean gradients: single backward pass instead of L chained Jacobians
- Still uses residual prediction: s_L = s₀ + Δ

Architecture: 1D CNN over action sequence + state conditioning
- Encodes the action sequence with 1D convolutions (captures temporal patterns)
- Conditions on s₀ via FiLM (Feature-wise Linear Modulation)
- Predicts residual Δs = s_L - s₀
"""

import torch
import torch.nn as nn
import numpy as np


class ChunkForwardModel(nn.Module):
    """
    Chunk forward model: F(s₀, a_{0:L}) → s_L

    Uses 1D CNN over actions + state conditioning via FiLM.
    Predicts residual: Δs = F(s₀, actions), s_L = s₀ + Δs

    Args:
        state_dim: state dimensionality (default: 4 for [x,y,vx,vy])
        action_dim: action dimensionality (default: 2 for [ax,ay])
        chunk_length: number of steps L (e.g. 8, 16, 32, 64)
        hidden_dim: MLP hidden dimension
        n_layers: number of MLP layers after CNN features
    """

    def __init__(
        self,
        state_dim: int = 4,
        action_dim: int = 2,
        chunk_length: int = 16,
        hidden_dim: int = 256,
        n_layers: int = 3,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_length = chunk_length

        # ─── 1D CNN over action sequence ─────────────────────
        # Input: (B, action_dim, L) → features: (B, feat_dim)
        self.action_cnn = nn.Sequential(
            nn.Conv1d(action_dim, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),  # → (B, 64, 1)
        )
        cnn_feat_dim = 64

        # Also keep a flat path for the raw action sequence
        # (CNN captures patterns, flat captures exact positions)
        flat_action_dim = action_dim * chunk_length

        # ─── State conditioning (FiLM) ───────────────────────
        # s₀ modulates the action features
        self.state_to_film = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, cnn_feat_dim * 2),  # γ and β
        )

        # ─── Prediction MLP ─────────────────────────────────
        mlp_input_dim = cnn_feat_dim + flat_action_dim + state_dim
        layers = []
        in_dim = mlp_input_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.LayerNorm(hidden_dim))
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, state_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Predict state after chunk_length steps.

        Args:
            state:   (B, state_dim) — initial state s₀
            actions: (B, chunk_length, action_dim) — action sequence

        Returns:
            (B, state_dim) — predicted state s_L = s₀ + Δs
        """
        B = state.shape[0]

        # CNN features from action sequence
        # actions: (B, L, 2) → (B, 2, L) for Conv1d
        act_t = actions.transpose(1, 2)  # (B, action_dim, L)
        cnn_feat = self.action_cnn(act_t).squeeze(-1)  # (B, 64)

        # FiLM conditioning: state modulates action features
        film = self.state_to_film(state)  # (B, 128)
        gamma, beta = film.chunk(2, dim=-1)  # each (B, 64)
        cnn_feat = gamma * cnn_feat + beta

        # Flat action features
        flat_actions = actions.reshape(B, -1)  # (B, L*2)

        # Concatenate all features
        x = torch.cat([cnn_feat, flat_actions, state], dim=-1)

        # Predict residual
        delta = self.mlp(x)  # (B, state_dim)

        return state + delta  # residual prediction


class ChunkForwardModelFull(nn.Module):
    """
    Full-trajectory chunk forward model: predicts ALL intermediate states.

        F(s₀, [a₀, ..., a_{L-1}]) → [s₁, s₂, ..., s_L]

    Same CNN+FiLM architecture as ChunkForwardModel, but outputs L × state_dim
    instead of just state_dim. Each output is a residual from s₀.

    This gives us all 128 states (8 segments × 16 steps) with zero compounding
    error, enabling honest dense reward and boundary losses at every timestep.
    """

    def __init__(
        self,
        state_dim: int = 4,
        action_dim: int = 2,
        chunk_length: int = 16,
        hidden_dim: int = 512,
        n_layers: int = 4,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_length = chunk_length

        # ─── 1D CNN over action sequence ─────────────────────
        self.action_cnn = nn.Sequential(
            nn.Conv1d(action_dim, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),  # → (B, 64, 1)
        )
        cnn_feat_dim = 64

        flat_action_dim = action_dim * chunk_length

        # ─── State conditioning (FiLM) ───────────────────────
        self.state_to_film = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, cnn_feat_dim * 2),  # γ and β
        )

        # ─── Prediction MLP → L × state_dim outputs ─────────
        mlp_input_dim = cnn_feat_dim + flat_action_dim + state_dim
        output_dim = chunk_length * state_dim

        layers = []
        in_dim = mlp_input_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.LayerNorm(hidden_dim))
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Predict all intermediate and final states.

        Args:
            state:   (B, state_dim) — initial state s₀
            actions: (B, chunk_length, action_dim) — action sequence

        Returns:
            (B, chunk_length, state_dim) — predicted [s₁, s₂, ..., s_L]
            Each s_t = s₀ + Δs_t (residual from initial state)
        """
        B = state.shape[0]

        # CNN features
        act_t = actions.transpose(1, 2)  # (B, action_dim, L)
        cnn_feat = self.action_cnn(act_t).squeeze(-1)  # (B, 64)

        # FiLM conditioning
        film = self.state_to_film(state)
        gamma, beta = film.chunk(2, dim=-1)
        cnn_feat = gamma * cnn_feat + beta

        # Flat action features
        flat_actions = actions.reshape(B, -1)

        # Concatenate
        x = torch.cat([cnn_feat, flat_actions, state], dim=-1)

        # Predict L residuals
        deltas = self.mlp(x)  # (B, L * state_dim)
        deltas = deltas.reshape(B, self.chunk_length, self.state_dim)  # (B, L, 4)

        # Each output is s₀ + Δs_t
        return state.unsqueeze(1) + deltas  # (B, L, state_dim)


class ChunkForwardModelMLP(nn.Module):
    """
    Simpler variant: pure MLP with no CNN.
    Good baseline and possibly sufficient for small L.

    F(s₀, flatten(a_{0:L})) → s₀ + Δs
    """

    def __init__(
        self,
        state_dim: int = 4,
        action_dim: int = 2,
        chunk_length: int = 16,
        hidden_dim: int = 512,
        n_layers: int = 4,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_length = chunk_length

        input_dim = state_dim + action_dim * chunk_length

        layers = []
        in_dim = input_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.LayerNorm(hidden_dim))
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, state_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state:   (B, state_dim)
            actions: (B, chunk_length, action_dim)
        Returns:
            (B, state_dim) predicted state after chunk_length steps
        """
        flat = torch.cat([state, actions.reshape(state.shape[0], -1)], dim=-1)
        delta = self.net(flat)
        return state + delta
