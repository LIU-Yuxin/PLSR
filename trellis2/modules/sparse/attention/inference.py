"""Full sparse attention with the upstream parameter layout."""

import torch

from .modules import SparseMultiHeadAttention
from .full_attn import sparse_scaled_dot_product_attention


class FullSparseAttention(SparseMultiHeadAttention):
    """Reuse the upstream projections, normalization and rotary embeddings."""

    def forward(self, x, context=None, coords_scaling=1.0):
        if self.attn_mode != "full":
            raise ValueError("Inference attention requires full attention")
        if self._type == "self":
            qkv = self._fused_pre(self._linear(self.to_qkv, x), num_fused=3)
            if self.qk_rms_norm or self.use_rope:
                q, k, v = qkv.unbind(dim=-3)
                if self.qk_rms_norm:
                    q, k = self.q_rms_norm(q), self.k_rms_norm(k)
                if self.use_rope:
                    q, k = self.rope(q, k, coords_scaling=coords_scaling)
                qkv = qkv.replace(torch.stack([q.feats, k.feats, v.feats], dim=1))
            h = sparse_scaled_dot_product_attention(qkv)
        else:
            q = self._reshape_chs(self._linear(self.to_q, x), (self.num_heads, -1))
            kv = self._fused_pre(self._linear(self.to_kv, context), num_fused=2)
            if self.qk_rms_norm:
                k, v = kv.unbind(dim=-3)
                h = sparse_scaled_dot_product_attention(self.q_rms_norm(q), self.k_rms_norm(k), v)
            else:
                h = sparse_scaled_dot_product_attention(q, kv)
        return self._linear(self.to_out, self._reshape_chs(h, (-1,)))
