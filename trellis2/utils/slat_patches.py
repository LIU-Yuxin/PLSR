"""Plan overlapping sparse-latent patches and map their rows to global tensors."""

from dataclasses import dataclass
from itertools import product

import torch

from ..modules.sparse import SparseTensor, sparse_cat


@dataclass(frozen=True)
class SLatPatch:
    """One instance's target rows and their patch-local representation.

    bbox stores global, half-open bounds (x0, y0, z0, x1, y1, z1). indices
    address rows of the full target tensor. coords reset the batch index to
    zero and subtract the bbox origin. weights contain one unnormalized
    overlap weight per selected row.
    """
    instance: int
    bbox: tuple[int, int, int, int, int, int]
    indices: torch.Tensor
    coords: torch.Tensor
    weights: torch.Tensor

    @property
    def key(self):
        """Identify this instance and bbox within a fixed sampling pass."""
        return f"{self.instance}/{'_'.join(str(v) for v in self.bbox)}"

    def extract(self, tensor):
        """Extract an aligned target tensor using the stored rows and local coordinates."""
        return SparseTensor(feats=tensor.feats[self.indices], coords=self.coords.clone())

    def crop_indices(self, tensor):
        """Find this instance's condition rows inside the half-open bbox."""
        origin = tensor.coords.new_tensor(self.bbox[:3])
        end = tensor.coords.new_tensor(self.bbox[3:])
        mask = (tensor.coords[:, 0] == self.instance) & (
            (tensor.coords[:, 1:] >= origin) & (tensor.coords[:, 1:] < end)
        ).all(dim=1)
        return mask.nonzero(as_tuple=True)[0]

    def crop(self, tensor, indices):
        """Crop condition rows and translate them into this patch's local frame."""
        coords = tensor.coords[indices].clone()
        coords[:, 0] = 0
        coords[:, 1:] -= coords.new_tensor(self.bbox[:3])
        return SparseTensor(feats=tensor.feats[indices], coords=coords)


def validate_slat(tensor, name):
    """Check the sparse storage and instance layout required for patch operations."""
    if not isinstance(tensor, SparseTensor) or tensor.feats.ndim != 2:
        raise ValueError(f"{name} must be a SparseTensor with [N, C] features")
    coords = tensor.coords
    if coords.ndim != 2 or coords.shape[1] != 4 or len(coords) == 0 or len(coords) != len(tensor.feats):
        raise ValueError(f"{name} must have nonempty [N, 4] coordinates aligned with features")
    if coords.dtype not in (torch.int32, torch.int64) or coords.device != tensor.feats.device:
        raise ValueError(f"{name} requires integer coordinates on the feature device")
    batches = coords[:, 0]
    unique = torch.unique_consecutive(batches)
    if not torch.equal(unique, torch.arange(len(unique), device=coords.device, dtype=coords.dtype)):
        raise ValueError(f"{name} requires contiguous, nonempty batches numbered from zero")
    if len(torch.unique(coords, dim=0)) != len(coords):
        raise ValueError(f"{name} contains duplicate coordinates")


def require_aligned(reference, condition, name):
    """Require an optional condition to share the reference coordinates and row order."""
    if condition is None:
        return
    validate_slat(condition, name)
    if condition.device != reference.device or not torch.equal(condition.coords, reference.coords):
        raise ValueError(f"{name} coordinates and row order must match its reference tensor")


def plan_slat_patches(tensor, patch_size=32, overlap=8):
    """Tile each instance with fixed-size overlapping cubes in x/y/z traversal order.

    Start at the occupied minimum and advance by patch_size - overlap along
    each axis. Bbox maxima are exclusive; boundary boxes may extend beyond
    occupied cells. Skip empty boxes and retain global row indices.

    Weights ramp linearly away from the nearest cube face and are clamped to
    1e-6. The sampler normalizes their sum when fusing overlapping predictions.
    """
    # 1. Check the sparse layout and patch geometry.
    validate_slat(tensor, "noise")
    if type(patch_size) is not int or patch_size < 2:
        raise ValueError("patch_size must be an integer >= 2")
    if type(overlap) is not int or not 0 <= overlap < patch_size:
        raise ValueError("overlap must be an integer in [0, patch_size)")
    # 2. Tile each instance independently, retaining only occupied boxes.
    result = []
    for instance, layout in enumerate(tensor.layout):
        xyz = tensor.coords[layout, 1:]
        low, high = xyz.amin(dim=0).tolist(), xyz.amax(dim=0).tolist()
        starts = [range(lo, hi + 1, patch_size - overlap) for lo, hi in zip(low, high)]
        for origin in product(*starts):
            local = xyz - xyz.new_tensor(origin)
            selected = ((local >= 0) & (local < patch_size)).all(dim=1).nonzero(as_tuple=True)[0]
            if len(selected) == 0:
                continue
            # Add neighboring context to underfilled windows without dropping their cells.
            if len(selected) <= 8:
                occupied = xyz[selected]
                lo, hi = occupied.amin(0), occupied.amax(0)
                shifted = xyz.new_tensor(origin)
                shifted = shifted + ((lo + hi + 1) // 2 - shifted - patch_size // 2).clamp(-8, 8)
                shifted = torch.minimum(torch.maximum(shifted, (hi + 1 - patch_size).clamp(min=0)), lo)
                shifted_local = xyz - shifted
                shifted_selected = ((shifted_local >= 0) & (shifted_local < patch_size)).all(dim=1).nonzero(as_tuple=True)[0]
                if len(shifted_selected) > len(selected):
                    origin = shifted.tolist()
                    local, selected = shifted_local, shifted_selected
            local = local[selected]
            # 3. Store local coordinates and unnormalized overlap weights.
            normalized = local / (patch_size - 1)
            weights = (torch.minimum(normalized, 1 - normalized) * 2).amin(dim=1, keepdim=True).clamp(min=1e-6)
            coords = torch.cat([torch.zeros_like(local[:, :1]), local], dim=1)
            bbox = (*origin, *(v + patch_size for v in origin))
            result.append(SLatPatch(instance, bbox, selected + layout.start, coords, weights))
    return result


def pack_sparse(tensors):
    """Pack local patches as separate instances in their supplied order."""
    return sparse_cat(tensors, dim=0)
