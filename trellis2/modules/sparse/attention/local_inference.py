"""PLSR's radius-limited KNN attention for inference."""

import math

import torch
from torch import nn
from pytorch3d.ops import knn_points

from ...utils import manual_cast


@torch.no_grad()
def local_neighbor_indices(query, context, *, neighbors, radius):
    """Return global context row indices, with -1 outside the radius.

    When a batch has fewer than K context cells, retain PyTorch3D's padded
    indices in the neighbor set. The distance/radius mask is applied to those
    slots as well, so their multiplicity contributes to the attention weights.
    """
    indices = []
    for query_slice, context_slice in zip(query.layout, context.layout):
        q = query.coords[query_slice, 1:]
        k = context.coords[context_slice, 1:]
        if not len(q):
            continue
        if not len(k):
            index = torch.full((len(q), neighbors), -1, dtype=torch.int32, device=q.device)
        else:
            distance, index, _ = knn_points(q[None].float(), k[None].float(), K=neighbors)
            index = index[0].to(torch.int32) + context_slice.start
            index[distance[0] > radius ** 2] = -1
        indices.append(index)
    return torch.cat(indices).long()


class LocalInferenceAttention(nn.Module):
    """Cache geometry on CPU; recompute attention from current features.

    A cache key identifies fixed query/context coordinates and batch layout for
    one sampling pass. The sampler clears the cache on success and failure.
    """

    def __init__(self, radius, knn):
        super().__init__()
        self.radius = radius
        self.knn = knn
        self.falloff_multiplier = nn.Parameter(torch.tensor(0.5))
        self.local_cache = {}

    def empty_cache(self):
        self.local_cache.clear()
        torch.cuda.empty_cache()

    def forward(self, x, context, mha, local_cache_key):
        """Attend to nearby LR features and return rows aligned with the target tensor."""

        # 1. Reuse geometry indices and masks, never attention features or projections.
        if local_cache_key not in self.local_cache:
            indices = local_neighbor_indices(x, context, neighbors=self.knn, radius=self.radius)
            self.local_cache[local_cache_key] = (indices.cpu(), (indices >= 0).cpu())
        indices, valid = (value.to(x.device) for value in self.local_cache[local_cache_key])
        query_mask = valid.any(dim=-1)
        if not query_mask.any():
            raise ValueError("Local attention requires at least one query cell with an LR neighbor")
        indices = indices[query_mask].clamp(min=0)
        valid = valid[query_mask]
        query = x.replace(x.feats[query_mask], coords=x.coords[query_mask])

        # 2. Project current target queries and LR keys/values for the retained rows.
        q = mha._reshape_chs(mha._linear(mha.to_q, query), (mha.num_heads, -1))
        kv = mha._fused_pre(mha._linear(mha.to_kv, context), num_fused=2)
        k, v = kv.unbind(dim=-3)
        if mha.qk_rms_norm:
            q, k = mha.q_rms_norm(q), mha.k_rms_norm(k)

        # 3. Weight valid neighbors with cosine spatial falloff before softmax.
        relative = (q.coords[:, None, 1:] - context.coords[indices][:, :, 1:]).float()
        distance = torch.norm(relative, dim=-1)
        falloff = (0.5 + 0.5 * torch.cos(math.pi * distance / self.radius)) * self.falloff_multiplier
        falloff = manual_cast(falloff.clamp(min=1e-2, max=1.0), q.dtype)
        scores = torch.einsum('nhd,nkhd->nkh', q.feats * (1 / math.sqrt(mha.head_dim)), k.feats[indices])
        scores = scores + falloff.unsqueeze(-1).log().to(scores.dtype)
        scores = scores.masked_fill(~valid.unsqueeze(-1), float('-inf'))
        weights = torch.softmax(scores, dim=1)
        features = torch.einsum('nkh,nkhd->nhd', weights, v.feats[indices])
        features = mha.to_out(features.reshape(features.shape[0], -1))

        # 4. Scatter attended features back; targets without neighbors retain zeros.
        output = torch.zeros_like(x.feats)
        output[query_mask] = features
        return x.replace(output).to(x.dtype)
