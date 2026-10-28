"""The trained PLSR extension to the upstream modulated cross-attention block."""

import torch
from torch import nn

from ..basic import sparse_cat
from ..attention.inference import FullSparseAttention
from ...norm import LayerNorm32
from .blocks import SparseFeedForwardNet


class PLSRTransformerBlock(nn.Module):
    """Time-modulated sparse block with LR injection and gated image attention.

    Apply target self-attention, add local LR features, then gate the image
    cross-attention residual before the feed-forward residual. The outer model
    supplies LR attention features and noise-level gate embeddings.
    """
    def __init__(self, channels, ctx_channels, num_heads, mlp_ratio, rope_freq,
                 qk_rms_norm, qk_rms_norm_cross):
        super().__init__()
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm3 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.self_attn = FullSparseAttention(
            channels, num_heads, type="self", use_rope=True,
            rope_freq=rope_freq, qk_rms_norm=qk_rms_norm,
        )
        self.cross_attn = FullSparseAttention(
            channels, num_heads, ctx_channels=ctx_channels, type="cross",
            qk_rms_norm=qk_rms_norm_cross,
        )
        self.mlp = SparseFeedForwardNet(channels, mlp_ratio)
        self.norm_lr = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        # A scalar gates LR injection; a feature-conditioned MLP gates the
        # image-attention residual independently of the timestep residual gates.
        self.lr_gate_weight = nn.Parameter(torch.zeros(1))
        self.cross_gate_mlp = SparseFeedForwardNet(channels * 4, mlp_ratio=0.125, out_channels=1)
        self.cross_gate_norm = LayerNorm32(channels * 4, elementwise_affine=True, eps=1e-6)
        self.modulation = nn.Parameter(torch.randn(6 * channels) / channels ** 0.5)

    def forward(self, x, mod, context, lr_x, lr_cross_gate_emb, coords_scaling):
        """Apply target self-attention, LR injection, gated image attention and the MLP."""

        # 1. Self-attention uses the shared timestep modulation plus per-block offsets.
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation + mod
        ).type(mod.dtype).chunk(6, dim=1)
        h = x.replace(self.norm1(x.feats))
        h = h * (1 + scale_msa) + shift_msa
        h = self.self_attn(h, coords_scaling=coords_scaling)
        x = x + h * gate_msa

        # 2. Inject aligned local LR features through a scalar gate.
        lr_h = lr_x.replace(self.norm_lr(lr_x.feats))
        lr_gate = torch.tanh(self.lr_gate_weight) / 2 + 0.5
        x = x.replace(x.feats + lr_gate.to(lr_h.dtype) * lr_h.feats)

        # 3. Gate image attention using target, LR, image and LR-noise-level features.
        h = self.cross_attn(x.replace(self.norm2(x.feats)), context)
        gate = sparse_cat([x, lr_x, h, lr_cross_gate_emb], dim=-1)
        gate = gate.replace(self.cross_gate_norm(gate.feats))
        gate = torch.tanh(self.cross_gate_mlp(gate).feats) / 2 + 0.5
        x = x + h.replace(h.feats * gate)

        # 4. Finish with the time-modulated feed-forward residual.
        h = x.replace(self.norm3(x.feats))
        h = h * (1 + scale_mlp) + shift_mlp
        h = self.mlp(h)
        return x + h * gate_mlp
