"""
Transformer Forward Model — predicts ALL intermediate states using causal attention.

    F(s₀, [a₀, ..., a_{L-1}]) → [s₁, s₂, ..., s_L]

Unlike ChunkForwardModelFull (CNN+FiLM MLP) which collapses temporal structure
via AdaptiveAvgPool, this model uses causal self-attention so each output step t
attends to the initial state and all actions a₀..a_{t-1}. This naturally captures
how effects accumulate over time — particularly important for v1 dynamics where
cross-axis velocity coupling means the history of all past actions matters.

Key design:
- Same forward() signature as ChunkForwardModelFull: (B, 4), (B, L, 2) → (B, L, 4)
- Residual prediction: ŝ_t = s₀ + Δs_t (same as existing FM)
- NOT autoregressive — no predicted states fed back in → zero compounding error
- Causal mask: output at position t only depends on s₀ and a₀..a_{t-1}
"""

import torch
import torch.nn as nn
import math


class TransformerForwardModel(nn.Module):
    """
    Causal transformer forward model: F(s₀, a_{0:L}) → [s₁, ..., s_L]

    Sequence layout: [s₀_token, a₀_token, a₁_token, ..., a_{L-1}_token]
    Read out predictions at action token positions → Δs_t
    Final output: s₀ + Δs_t for each t

    Args:
        state_dim: state dimensionality (4 for [x,y,vx,vy])
        action_dim: action dimensionality (2 for [ax,ay])
        chunk_length: number of steps L (e.g. 16)
        d_model: transformer hidden dimension
        n_heads: number of attention heads
        n_layers: number of transformer layers
        d_ff: feedforward dimension (default: 4 * d_model)
        dropout: dropout rate
    """

    def __init__(
        self,
        state_dim: int = 4,
        action_dim: int = 2,
        chunk_length: int = 16,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_length = chunk_length
        self.d_model = d_model

        if d_ff == 0:
            d_ff = 4 * d_model

        # ─── Token embeddings ─────────────────────────────────
        # State token: s₀ (4D) → d_model
        self.state_embed = nn.Sequential(
            nn.Linear(state_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        # Action tokens: a_t (2D) → d_model
        self.action_embed = nn.Sequential(
            nn.Linear(action_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        # ─── Positional embeddings ────────────────────────────
        # Position 0 = state token, positions 1..L = action tokens
        seq_len = 1 + chunk_length
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        # Type embeddings: distinguish state vs action tokens
        self.type_embed = nn.Parameter(torch.randn(1, 2, d_model) * 0.02)

        # ─── Causal Transformer ───────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        # ─── Output projection ────────────────────────────────
        # Each action token position → predict state residual Δs_t
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, state_dim),
        )

        # Small init for final layer
        nn.init.uniform_(self.output_proj[-1].weight, -0.01, 0.01)
        nn.init.zeros_(self.output_proj[-1].bias)

        # ─── Causal mask ──────────────────────────────────────
        # Sequence: [s₀, a₀, a₁, ..., a_{L-1}]
        # s₀ (pos 0) sees only itself
        # a_t (pos t+1) sees s₀ and a₀..a_t (positions 0..t+1)
        # This is standard causal: position i sees positions 0..i
        self.register_buffer(
            'causal_mask',
            torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1),
        )

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
        L = self.chunk_length

        # Embed state token: (B, d_model)
        s_tok = self.state_embed(state)  # (B, d_model)

        # Embed action tokens: (B, L, d_model)
        a_tok = self.action_embed(actions)  # (B, L, d_model)

        # Build sequence: [s₀_token, a₀_token, ..., a_{L-1}_token]
        # (B, 1+L, d_model)
        seq = torch.cat([s_tok.unsqueeze(1), a_tok], dim=1)

        # Add positional embeddings
        seq = seq + self.pos_embed

        # Add type embeddings (0=state, 1=action)
        seq[:, 0:1, :] = seq[:, 0:1, :] + self.type_embed[:, 0:1, :]
        seq[:, 1:, :] = seq[:, 1:, :] + self.type_embed[:, 1:2, :]

        # Causal transformer
        seq = self.transformer(seq, mask=self.causal_mask)  # (B, 1+L, d_model)

        # Read out at action token positions (1..L) → predict Δs_t
        action_outputs = seq[:, 1:, :]  # (B, L, d_model)
        deltas = self.output_proj(action_outputs)  # (B, L, state_dim)

        # Residual from s₀
        return state.unsqueeze(1) + deltas  # (B, L, state_dim)


class TransformerForwardModelLarge(nn.Module):
    """
    Larger variant with separate encoder (for state) and decoder (for prediction).

    Uses cross-attention: action query tokens attend to state context.
    This is more expressive for modeling how state conditions action outcomes.

    Args: same as TransformerForwardModel but adds n_encoder_layers.
    """

    def __init__(
        self,
        state_dim: int = 4,
        action_dim: int = 2,
        chunk_length: int = 16,
        d_model: int = 192,
        n_heads: int = 6,
        n_encoder_layers: int = 2,
        n_decoder_layers: int = 4,
        d_ff: int = 0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_length = chunk_length
        self.d_model = d_model

        if d_ff == 0:
            d_ff = 4 * d_model

        # ─── State encoder → context tokens ───────────────────
        # Expand s₀ into multiple context tokens for richer conditioning
        n_context = 4  # 4 context tokens from s₀
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, d_model * n_context),
            nn.GELU(),
        )
        self.n_context = n_context
        self.ctx_pos_embed = nn.Parameter(
            torch.randn(1, n_context, d_model) * 0.02
        )

        # Optional encoder self-attention on context tokens
        if n_encoder_layers > 0:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads,
                dim_feedforward=d_ff, dropout=dropout,
                activation='gelu', batch_first=True, norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_encoder_layers)
        else:
            self.encoder = None

        # ─── Action query tokens ──────────────────────────────
        self.action_embed = nn.Sequential(
            nn.Linear(action_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.query_pos_embed = nn.Parameter(
            torch.randn(1, chunk_length, d_model) * 0.02
        )

        # ─── Decoder with cross-attention ─────────────────────
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_ff, dropout=dropout,
            activation='gelu', batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_decoder_layers)

        # ─── Output projection ────────────────────────────────
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, state_dim),
        )
        nn.init.uniform_(self.output_proj[-1].weight, -0.01, 0.01)
        nn.init.zeros_(self.output_proj[-1].bias)

        # Causal mask for decoder self-attention
        self.register_buffer(
            'causal_mask',
            torch.triu(torch.ones(chunk_length, chunk_length, dtype=torch.bool), diagonal=1),
        )

    def forward(self, state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Same interface as TransformerForwardModel."""
        B = state.shape[0]

        # Encode state → context tokens
        ctx = self.state_encoder(state)  # (B, n_context * d_model)
        ctx = ctx.reshape(B, self.n_context, self.d_model)  # (B, n_ctx, d_model)
        ctx = ctx + self.ctx_pos_embed

        if self.encoder is not None:
            ctx = self.encoder(ctx)  # (B, n_ctx, d_model)

        # Action query tokens
        queries = self.action_embed(actions) + self.query_pos_embed  # (B, L, d_model)

        # Decoder: queries attend to themselves (causal) + cross-attend to context
        out = self.decoder(
            queries, ctx,
            tgt_mask=self.causal_mask,
        )  # (B, L, d_model)

        deltas = self.output_proj(out)  # (B, L, state_dim)
        return state.unsqueeze(1) + deltas
