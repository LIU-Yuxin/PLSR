"""Spatial partitions for local VAE decoding, including overlapping surfaces."""

from dataclasses import dataclass
from itertools import product

import numpy as np
import torch

from ..modules.sparse import SparseTensor
from .slat_patches import validate_slat


@dataclass(frozen=True)
class DecodePatch:
    """Select latent rows and define their integer origin on the SLat grid."""
    indices: torch.Tensor
    origin: tuple[int, int, int]

    def extract(self, latent):
        """Gather rows and subtract the origin without changing feature values."""
        coords = latent.coords[self.indices].clone()
        coords[:, 1:] -= coords.new_tensor(self.origin)
        return SparseTensor(coords=coords, feats=latent.feats[self.indices])

    def local_to_canonical(self, resolution, local_resolution):
        """Map the decoded local cube to the asset's canonical unit cube.

        Four decoder subdivisions produce 16 voxels per SLat cell. A local
        vertex is (global_voxel - origin*16)/local_resolution - .5; this affine
        transform restores global_voxel/resolution - .5. Apply it after baking
        local voxel materials, so geometry and attribute sampling stay aligned.
        """
        scale = local_resolution / resolution
        transform = np.eye(4)
        transform[:3, :3] *= scale
        transform[:3, 3] = (scale - 1) / 2 + np.asarray(self.origin) * (16 / resolution)
        return transform


@dataclass(frozen=True)
class DecodedPart:
    """Mesh and voxel material remain in local space until texture baking ends."""
    mesh: object
    local_to_canonical: np.ndarray


def _mask(coords, bbox):
    """Select integer cells inside a half-open spatial box."""
    return ((coords >= coords.new_tensor(bbox[:3])) & (coords < coords.new_tensor(bbox[3:]))).all(dim=1)


def _recenter(coords, bbox, size, min_cells=2000):
    """Move an underfilled box toward occupied cells to provide more context.

    Keep its current cells while moving at most eight cells per axis, for at
    most ten iterations. Stop at min_cells; clamp negative bounds to zero.
    """
    for _ in range(10):
        occupied = coords[_mask(coords, bbox)]
        if len(occupied) == 0:
            raise ValueError("Recentering a decode patch produced an empty region")
        center = [int((lo + hi + 1) // 2) for lo, hi in zip(occupied.amin(0).tolist(), occupied.amax(0).tolist())]
        previous = [int((lo + hi) / 2) for lo, hi in zip(bbox[:3], bbox[3:])]
        center = [min(max(old - 8, new), old + 8) for old, new in zip(previous, center)]
        bbox = tuple(v - size // 2 for v in center) + tuple(v + size // 2 for v in center)
        if int(_mask(coords, bbox).sum()) >= min_cells:
            break
    return tuple(max(0, value) for value in bbox)


def _split(coords, bbox, size, overlap, cell_limit):
    """Split an oversized box without gaps, retaining the requested overlap.

    Try x/y/z proposals and choose the fewest parts, with stable axis tie order.
    Skip an axis if its children cannot shrink; a thin axis must not prevent
    splitting the other axes. Fail only when no proposal can satisfy the cap.
    """
    if int(_mask(coords, bbox).sum()) <= cell_limit:
        return [tuple(bbox)]
    proposals = []
    for axis in range(3):
        first, second = list(bbox), list(bbox)
        width = min(size, bbox[axis + 3] - bbox[axis])
        first[axis + 3] = bbox[axis] + (width + overlap + 1) // 2
        second[axis] = first[axis + 3] - overlap
        children = (tuple(first), tuple(second))
        if first[axis + 3] >= bbox[axis + 3] or second[axis] <= bbox[axis]:
            continue
        proposal = []
        for child in children:
            count = int(_mask(coords, child).sum())
            if count > cell_limit:
                try:
                    proposal.extend(_split(coords, child, size, overlap, cell_limit))
                except ValueError:
                    # This proposal cannot meet the cap; another axis may do so.
                    break
            elif count:
                proposal.append(child)
        else:
            if len(proposal) == 2:
                return proposal
            proposals.append(proposal)
    if not proposals:
        raise ValueError("Decode cell_limit is too small for this overlap; splitting cannot make progress")
    return min(proposals, key=len)


def plan_decode_patches(latent, *, local_resolution=1024, patch_size=64, overlap=2, cell_limit=25000):
    """Plan bounded local VAE inputs while retaining overlapping geometry.

    An asset already fitting the local grid and cell cap is decoded as one part.
    Otherwise tile the occupied region, recenter underfilled edge boxes and split
    dense boxes. Each part's minimum occupied coordinate defines its origin.
    Shape/PBR share the row selection. Parts cover all cells but do not assign
    unique ownership to overlap cells; these are separate from sampling patches.
    """
    validate_slat(latent, "decode latent")
    if latent.shape[0] != 1 or (latent.coords[:, 1:] < 0).any():
        raise ValueError("Local decoding requires one asset with nonnegative coordinates")
    if type(local_resolution) is not int or local_resolution <= 0 or local_resolution % 128:
        raise ValueError("local_resolution must be a positive multiple of 128")
    if type(patch_size) is not int or patch_size < 2 or patch_size % 2 or patch_size > local_resolution // 16:
        raise ValueError("patch_size must be positive, even and fit the local SLat grid")
    if type(overlap) is not int or not 0 <= overlap < patch_size:
        raise ValueError("overlap must be an integer in [0, patch_size)")
    if type(cell_limit) is not int or cell_limit <= 0:
        raise ValueError("cell_limit must be a positive integer")
    coords = latent.coords[:, 1:]
    if int(coords.max()) < local_resolution // 16 and len(coords) <= cell_limit:
        return [DecodePatch(torch.arange(len(coords), device=coords.device), (0, 0, 0))]
    low = [int(v) // 2 * 2 for v in coords.amin(0).tolist()]
    high = [(int(v) + 1) // 2 * 2 for v in coords.amax(0).tolist()]
    boxes = []
    covered = torch.zeros(len(coords), dtype=torch.bool, device=coords.device)
    # 1. Tile the occupied region, retaining overlaps in accepted boxes.
    for origin in product(*(range(lo, hi + 1, patch_size - overlap) for lo, hi in zip(low, high))):
        bbox = (*origin, *(v + patch_size for v in origin))
        mask = _mask(coords, bbox)
        # Skip boxes that add no cells. Accepted boxes retain their overlap;
        # coverage avoids redundant whole boxes, not duplicate cell ownership.
        if covered[mask].all():
            continue
        covered[mask] = True
        count = int(mask.sum())
        if count < 2000:
            recentered = _recenter(coords, bbox, patch_size)
            if int(_mask(coords, recentered).sum()) > cell_limit:
                # Prefer the original region when added context exceeds the cap.
                boxes.extend(_split(coords, bbox, patch_size, overlap, cell_limit))
            else:
                boxes.append(recentered)
        elif count > cell_limit:
            boxes.extend(_split(coords, bbox, patch_size, overlap, cell_limit))
        else:
            boxes.append(bbox)
    # 2. Gather each box and define the local grid origin.
    patches = []
    for bbox in boxes:
        indices = _mask(coords, bbox).nonzero(as_tuple=True)[0]
        if len(indices) == 0:
            continue
        origin = tuple(coords[indices].amin(0).tolist())
        patches.append(DecodePatch(indices, origin))
    if not patches:
        raise ValueError("No occupied decode patches were found")
    # 3. Check the cap and complete coverage before executing the VAE.
    selected = torch.zeros(len(coords), dtype=torch.bool, device=coords.device)
    for patch in patches:
        if len(patch.indices) > cell_limit:
            raise ValueError("Decode planning exceeded cell_limit")
        selected[patch.indices] = True
    if not selected.all():
        raise ValueError("Decode partition did not cover every input cell")
    return patches


def plan_coarse_decode_patches(latent):
    """Plan coarse parts on the global 64-cell grid at mesh resolution 1024.

    Use 48-cell boxes, four-cell padding/alignment/overlap, a 25,000-cell cap
    and recenter boxes below 200 cells. Keep global coordinates and consume
    every selected part. Unlike local SR decoding, no placement offset is needed.
    """
    validate_slat(latent, "coarse decode latent")
    patch_size, overlap, cell_limit = 48, 4, 25000
    coords = latent.coords[:, 1:]
    if latent.shape[0] != 1 or not ((coords >= 0) & (coords < 64)).all():
        raise ValueError("Original 1024 coarse decoding requires one asset on the 64-grid")
    if len(coords) <= cell_limit:
        return [DecodePatch(torch.arange(len(coords), device=coords.device), (0, 0, 0))]
    low = [int(v) // 4 * 4 - 4 for v in coords.amin(0).tolist()]
    high = [(int(v) + 4) // 4 * 4 + 4 for v in coords.amax(0).tolist()]
    boxes = []
    for origin in product(*(range(lo, hi + 1, patch_size - overlap) for lo, hi in zip(low, high))):
        bbox = (*origin, *(v + patch_size for v in origin))
        count = int(_mask(coords, bbox).sum())
        if count == 0:
            continue
        if count < 200:
            recentered = _recenter(coords, bbox, patch_size, min_cells=200)
            # Retain the original region if moving into dense context exceeds the cap.
            boxes.append(bbox if int(_mask(coords, recentered).sum()) > cell_limit else recentered)
        elif count > cell_limit:
            boxes.extend(_split(coords, bbox, patch_size, overlap, cell_limit))
        else:
            boxes.append(bbox)
    covered = torch.zeros(len(coords), dtype=torch.bool, device=coords.device)
    patches = []
    for bbox in boxes:
        mask = _mask(coords, bbox)
        if not mask.any():
            continue
        covered |= mask
        patches.append(DecodePatch(mask.nonzero(as_tuple=True)[0], (0, 0, 0)))
    if not covered.all():
        raise ValueError("Coarse decode partition did not cover every input cell")
    return patches
