"""Inference architecture for the released Shape and PBR PLSR checkpoints."""

from functools import partial
import math
import warnings

import torch
from torch import nn
from torch.nn import functional as F

from .sparse_structure_flow import TimestepEmbedder
from ..modules import sparse as sp
from ..modules.norm import LayerNorm32
from ..modules.utils import convert_module_to, manual_cast, str_to_dtype
from ..modules.sparse.attention.inference import FullSparseAttention
from ..modules.sparse.attention.local_inference import LocalInferenceAttention
from ..modules.sparse.transformer.plsr import PLSRTransformerBlock


class PLSRFlowModel(nn.Module):
    """Shared modulation, convolutional LR features and gated local attention.

    State names match the trained architecture. ``load_pretrained_denoiser``
    loads the base weights followed by the LoRA/PLSR adapter.
    """

    @classmethod
    def from_config(cls, config):
        """Build the released inference architecture from its saved model arguments."""
        args = dict(config)
        # Select the checkpoint's supported architecture without mutating the
        # caller's configuration; initialization and training controls are unused.
        required = {
            "pe_mode": "rope", "share_mod": True, "lr_noise_level_mod": True,
            "lr_use_conv": True, "lr_injection": True, "gate_cross_attn": True,
            "fuse_mode": "cross",
        }
        for name, expected in required.items():
            if args.pop(name, None) != expected:
                raise ValueError(f"PLSR inference requires {name}={expected!r}")
        for name in ("initialization", "use_checkpoint", "lr_dropout", "lr_jitter", "lr_pool_distance_offset"):
            args.pop(name, None)
        if args.pop("lr_channel_limit", 0) != 0:
            raise ValueError("PLSR inference does not support LR channel truncation")
        return cls(**args)

    def __init__(self, resolution, in_channels, model_channels, cond_channels,
                 out_channels, num_blocks, lr_channels, lr_conv_ch_factor,
                 local_attn_configs, num_heads=None, num_head_channels=64,
                 mlp_ratio=4, rope_freq=(1.0, 10000.0), dtype="float32",
                 qk_rms_norm=False, qk_rms_norm_cross=False):
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.cond_channels = cond_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.num_heads = num_heads or model_channels // num_head_channels
        self.lr_channels = lr_channels
        local = dict(local_attn_configs)
        self.n_blocks_per_lr_cross = local.pop("n_blocks_per_lr_cross", 4)
        radius = local.pop("local_attn_radius", 1.7)
        neighbors = local.pop("local_attn_knn", 5)
        if local.pop("local_attn_falloff_mode", "cosine") != "cosine" or local:
            raise ValueError("PLSR inference requires cosine local attention with known settings")
        if self.n_blocks_per_lr_cross <= 0 or radius <= 0 or neighbors <= 0:
            raise ValueError("Local attention interval, radius and neighbor count must be positive")

        # Denoising backbone: time modulation, image-conditioned blocks and projection.
        self.t_embedder = TimestepEmbedder(model_channels)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(model_channels, 6 * model_channels))
        self.input_layer = sp.SparseLinear(in_channels, model_channels)
        self.blocks = nn.ModuleList([
            PLSRTransformerBlock(model_channels, cond_channels, self.num_heads,
                                 mlp_ratio, rope_freq, qk_rms_norm, qk_rms_norm_cross)
            for _ in range(num_blocks)
        ])
        self.out_layer = sp.SparseLinear(model_channels, out_channels)

        # LR conditioning: noise-level modulation and convolutions over decoder features.
        self.lr_t_embedder = TimestepEmbedder(model_channels)
        self.lr_adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(model_channels, 6 * model_channels))
        self.lr_cross_gate_modulation = nn.Sequential(nn.SiLU(), nn.Linear(model_channels, model_channels))
        hidden = int(model_channels * lr_conv_ch_factor)
        self.lr_in = sp.SparseLinear(lr_channels, hidden)
        self.lr_pre = sp.SparseConv3d(hidden, hidden, kernel_size=3, padding=None)
        self.lr_fuse_conv = sp.SparseConv3d(hidden, hidden, kernel_size=3, padding=None)
        self.lr_fuse_conv_activation = sp.SparseGELU()
        self.lr_fuse_conv_norm = LayerNorm32(hidden, elementwise_affine=False)
        self.lr_proj_layer = sp.SparseLinear(hidden, model_channels)
        self.lr_in_activation = sp.SparseGELU()
        self.lr_pre_activation = sp.SparseGELU()
        self.lr_in_norm = LayerNorm32(hidden, elementwise_affine=False)
        self.lr_pre_norm = LayerNorm32(hidden, elementwise_affine=False)
        # A local attention projection serves each group of Transformer blocks.
        self.local_attn_module = LocalInferenceAttention(radius, neighbors)
        self.local_mha = nn.ModuleList([
            FullSparseAttention(model_channels, self.num_heads, type="cross", qk_rms_norm=qk_rms_norm_cross)
            for _ in range(math.ceil(num_blocks / self.n_blocks_per_lr_cross))
        ])
        self.convert_to(str_to_dtype(dtype))

    @property
    def device(self):
        return next(self.parameters()).device

    def convert_to(self, dtype):
        # The requested dtype applies only to the Transformer torso and local
        # attention projections; LR convolutions and embedders remain FP32.
        """Set the Transformer and local-attention projection dtype; keep LR layers FP32."""
        self.dtype = dtype
        convert = partial(convert_module_to, dtype=dtype)
        self.blocks.apply(convert)
        self.local_mha.apply(convert)

    def clear_local_attn_cache(self):
        self.local_attn_module.empty_cache()

    def _forward(self, x, t, lr_x, cond, lr_noise_level, coords_scaling, local_cache_key):
        """Predict flow velocity using image tokens and local LR feature conditioning."""

        # 1. Embed the noisy target and convolve LR features in FP32.
        h = self.input_layer(x)
        lr_h = self.lr_in(lr_x.replace(lr_x.feats.float()))
        lr_h = self.lr_in_activation(lr_h.replace(self.lr_in_norm(lr_h.feats)))
        lr_h = self.lr_pre(lr_h)
        lr_h = self.lr_pre_activation(lr_h.replace(self.lr_pre_norm(lr_h.feats)))
        lr_h = self.lr_fuse_conv(lr_h)
        lr_h = self.lr_fuse_conv_activation(lr_h.replace(self.lr_fuse_conv_norm(lr_h.feats)))
        lr_h = self.lr_proj_layer(lr_h)

        # 2. Prepare backbone-dtype features, timestep modulation and gate embeddings.
        h, lr_h = manual_cast(h, self.dtype), manual_cast(lr_h, self.dtype)
        t_emb = manual_cast(self.adaLN_modulation(self.t_embedder(t)), self.dtype)
        lr_t_emb = self.lr_t_embedder(lr_noise_level)
        gate_emb = manual_cast(self.lr_cross_gate_modulation(lr_t_emb), self.dtype).unsqueeze(1)
        gate_emb = h.replace(torch.cat([
            emb.expand_as(h.feats[h.layout[i]]) for i, emb in enumerate(gate_emb)
        ], dim=0))
        t_emb = t_emb + manual_cast(self.lr_adaLN_modulation(lr_t_emb), self.dtype)
        cond = manual_cast(cond, self.dtype)

        # 3. Alternate local LR attention updates with image-conditioned block groups.
        for index, block in enumerate(self.blocks):
            # Reuse the LR attention feature for this entire block group. The
            # neighbor map is cached separately across denoising timesteps.
            if index % self.n_blocks_per_lr_cross == 0:
                mha = self.local_mha[index // self.n_blocks_per_lr_cross]
                lr_cross = self.local_attn_module(h, lr_h, mha, local_cache_key)
            h = block(h, t_emb, cond, lr_cross, gate_emb, coords_scaling)

        # 4. Restore the input dtype, then normalize and project the flow velocity.
        h = manual_cast(h, x.dtype)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = self.out_layer(h)
        if not torch.isfinite(h.feats).all():
            # Replace non-finite predictions before returning to the sampler.
            warnings.warn("Non-finite PLSR output replaced with finite values (legacy behavior)", RuntimeWarning)
            h = h.replace(torch.nan_to_num(h.feats, nan=0.0, posinf=1e5, neginf=-1e5))
        return h

    def forward(self, x, t, lr_x=None, cond=None, concat_cond=None,
                lr_noise_level=None, lr_concat_cond=None, coords_scaling=1.0,
                local_cache_key=None):
        """Predict one patch batch, adding Shape channels during the PBR pass.

        x and lr_x use patch-local coordinates. concat_cond supplies normalized
        refined Shape, while lr_concat_cond supplies LR Shape features for PBR.
        A supplied cache key must identify fixed query/context coordinates and batch
        layout. Without a key, neighborhood caching is limited to this call.
        """
        if self.training:
            raise RuntimeError("PLSRFlowModel is inference-only; load weights and call eval() first")
        if lr_x is None or lr_noise_level is None or cond is None:
            raise ValueError("PLSR requires LR features, LR noise level and image conditioning")
        # Append Shape channels for PBR; Shape generation uses neither concatenation.
        if concat_cond is not None:
            x = sp.sparse_cat([x, concat_cond], dim=-1)
        if lr_concat_cond is not None:
            lr_x = sp.sparse_cat([lr_x, lr_concat_cond], dim=-1)
        x.coords, lr_x.coords = x.coords.int(), lr_x.coords.int()
        if isinstance(cond, list):
            cond = sp.VarLenTensor.from_tensor_list(cond)

        # Reuse sampler-scoped geometry, or create a cache valid for this call only.
        temporary_cache = local_cache_key is None
        if temporary_cache:
            self.clear_local_attn_cache()
            local_cache_key = object()
        try:
            return self._forward(x, t, lr_x, cond, lr_noise_level, coords_scaling, local_cache_key)
        finally:
            if temporary_cache:
                self.clear_local_attn_cache()
