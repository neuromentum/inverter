"""
Universal Transformer Forward Model — a single architecture for all environments.

    F(s₀, [a₀, ..., a_{L-1}]) → [ŝ₁, ŝ₂, ..., ŝ_L]

Unifies the separate TransformerForwardModel (maze2d), AntMazeXYTransformerFM,
AntMazeXYQWQZTransformerFM, and AntMazeFull29TransformerFM into one parameterized
class.  The architecture:

    1. Embed s₀ and each aₜ into d_model-dimensional tokens
    2. Prepend s₀ token to action tokens: [s₀, a₀, a₁, …, a_{L-1}]
    3. Add learned positional + type embeddings
    4. Run through a pre-norm causal Transformer encoder
    5. Read out at action token positions → residual Δ per step
    6. Output: s₀[output_dims] + Δ

Design choices (taken from the more mature antmaze implementations):
- 2-layer token embeddings (Linear→GELU→LN→Linear)
- Xavier uniform init, zero init on output head
- Pre-norm Transformer with final LayerNorm
- Dropout in output head
- Float -inf causal mask (computed once, cached)

To reproduce the specific variants:
    maze2d:         UniversalTransformerFM(state_dim=4, action_dim=2, chunk_length=16)
    antmaze-xy:     UniversalTransformerFM(state_dim=29, action_dim=8, output_dim=2,
                                            output_state_slice=slice(0,2), d_model=256)
    antmaze-xyqwqz: UniversalTransformerFM(state_dim=29, action_dim=8, output_dim=4,
                                            output_state_slice=[0,1,3,6], d_model=384,
                                            n_heads=6, n_layers=6)
    antmaze-full:   UniversalTransformerFM(state_dim=29, action_dim=8, d_model=384,
                                            n_heads=6, n_layers=6)
"""

import torch
import torch.nn as nn
from typing import List, Optional, Union


class UniversalTransformerFM(nn.Module):
    """
    General-purpose causal Transformer forward model.

    Predicts residuals: ŝₜ = s₀[output_dims] + Δₜ

    Args:
        state_dim:   Dimensionality of input state (4 for maze2d, 29 for antmaze).
        action_dim:  Dimensionality of actions (2 for maze2d, 8 for antmaze).
        chunk_length: Number of forward-prediction steps L.
        output_dim:  Dimensionality of predicted output per step.
                     Defaults to state_dim (predict all dims).
                     Set to 2 for xy-only prediction.
        output_state_slice: Which dims of s₀ to use as residual base.
                     Accepts a slice (e.g., slice(0,2) for xy) or a list of
                     int indices (e.g., [0,1,3,6] for non-contiguous xyqwqz).
                     Defaults to slice(None) (full state → output_dim must == state_dim).
        d_model:     Transformer hidden dimension.
        n_heads:     Number of attention heads.
        n_layers:    Number of Transformer layers.
        ff_mult:     FFN dimension multiplier (ff_dim = d_model * ff_mult).
        dropout:     Dropout rate.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_length: int = 16,
        output_dim: Optional[int] = None,
        output_state_slice: Optional[Union[slice, List[int]]] = None,
        d_model: int = 192,
        n_heads: int = 6,
        n_layers: int = 4,
        ff_mult: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_length = chunk_length
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers_tfm = n_layers
        self.ff_mult = ff_mult

        # Output configuration
        if output_dim is None:
            output_dim = state_dim
        self.output_dim = output_dim

        if output_state_slice is None:
            # Default: use full state as residual base (output_dim must match)
            if output_dim == state_dim:
                self._output_indices = None  # sentinel for slice(None)
            else:
                # Common case: predict first output_dim dims (e.g., xy)
                self._output_indices = list(range(output_dim))
        elif isinstance(output_state_slice, slice):
            # Convert slice → explicit index list for uniform handling
            # (slice(None) for full state is kept as None sentinel)
            if output_state_slice == slice(None):
                self._output_indices = None
            else:
                self._output_indices = list(range(*output_state_slice.indices(state_dim)))
        else:
            # List of int indices (e.g., [0, 1, 3, 6])
            self._output_indices = list(output_state_slice)

        # Validate
        if self._output_indices is not None:
            assert len(self._output_indices) == output_dim, (
                f"output_state_slice selects {len(self._output_indices)} dims "
                f"but output_dim={output_dim}"
            )

        # ─── Token embeddings ─────────────────────────────────
        # 2-layer projection: raw dim → d_model with nonlinearity + normalization
        self.state_embed = nn.Sequential(
            nn.Linear(state_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )
        self.action_embed = nn.Sequential(
            nn.Linear(action_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )

        # ─── Learned positional encoding (L+1 positions) ─────
        self.pos_embed = nn.Parameter(
            torch.randn(1, chunk_length + 1, d_model) * 0.02
        )

        # ─── Token type embed (state vs action) ──────────────
        self.type_embed = nn.Parameter(
            torch.randn(1, 2, d_model) * 0.02  # [0] = state, [1] = action
        )

        # ─── Pre-norm causal Transformer encoder ─────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model),
        )

        # ─── Output head ─────────────────────────────────────
        # d_model → output_dim (residual Δ per step)
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_dim),
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier uniform init for all linear layers, zeros for output bias."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Zero init on output layer → residual starts at Δ ≈ 0
        nn.init.zeros_(self.output_head[-1].weight)
        nn.init.zeros_(self.output_head[-1].bias)

    def apply_spectral_norm(self):
        """Apply spectral normalization to all Linear layers for Lipschitz bound."""
        from torch.nn.utils import spectral_norm
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                parts = name.rsplit('.', 1)
                if len(parts) == 2:
                    parent = dict(self.named_modules())[parts[0]]
                    attr = parts[1]
                else:
                    parent = self
                    attr = parts[0]
                setattr(parent, attr, spectral_norm(module))
        return self

    def forward(self, state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Predict states at each step via residual prediction.

        Args:
            state:   (B, state_dim) — initial state s₀
            actions: (B, L, action_dim) — action sequence [a₀, ..., a_{L-1}]

        Returns:
            (B, L, output_dim) — predicted states [ŝ₁, ..., ŝ_L]
            Each ŝₜ = s₀[output_dims] + Δₜ
        """
        B = state.shape[0]
        L = self.chunk_length

        # ─── Embed tokens ─────────────────────────────────────
        s_tok = self.state_embed(state).unsqueeze(1)   # (B, 1, d_model)
        a_tok = self.action_embed(actions)              # (B, L, d_model)

        # Sequence: [s₀, a₀, a₁, ..., a_{L-1}]
        tokens = torch.cat([s_tok, a_tok], dim=1)      # (B, L+1, d_model)

        # ─── Add positional + type embeddings ─────────────────
        tokens = tokens + self.pos_embed[:, :L + 1, :]
        tokens[:, 0:1, :] = tokens[:, 0:1, :] + self.type_embed[:, 0:1, :]
        tokens[:, 1:, :]  = tokens[:, 1:, :]  + self.type_embed[:, 1:2, :]

        # ─── Causal mask ──────────────────────────────────────
        # Position i can only attend to positions ≤ i
        causal_mask = torch.triu(
            torch.full((L + 1, L + 1), float('-inf'), device=tokens.device),
            diagonal=1,
        )

        # ─── Transformer forward ──────────────────────────────
        out = self.transformer(tokens, mask=causal_mask)  # (B, L+1, d_model)

        # ─── Readout at action positions → residuals ──────────
        out_actions = out[:, 1:, :]                       # (B, L, d_model)
        deltas = self.output_head(out_actions)             # (B, L, output_dim)

        # ─── Residual from s₀ ─────────────────────────────────
        if self._output_indices is None:
            s0_base = state                                    # (B, state_dim=output_dim)
        else:
            s0_base = state[:, self._output_indices]            # (B, output_dim)
        return s0_base.unsqueeze(1) + deltas               # (B, L, output_dim)

    def param_count(self) -> int:
        """Total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"state={self.state_dim}, action={self.action_dim}, "
            f"output={self.output_dim}, L={self.chunk_length}, "
            f"d={self.d_model}, heads={self.n_heads}, "
            f"layers={self.n_layers_tfm}, "
            f"params={self.param_count():,})"
        )
