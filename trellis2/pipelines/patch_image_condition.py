"""Local image conditions for one PLSR sampling pass."""

import torch

from .samplers.plsr_flow_euler import PatchCondition
from ..utils.slat_patches import validate_slat


class PatchImageConditioner:
    """Select visible views and cache their encoded patch crops for one pass.

    images: [B, V, 3, H, W] RGB tensors in [0, 1]. Intrinsics are normalized
    for those images and have zero skew. Extrinsics map canonical world
    coordinates to OpenCV camera axes (+z forward, +y down).
    support is the target SLat grid, whose resolution is voxel_resolution.
    encode_image takes [1, 3, 1024, 1024] and returns cond/neg_cond tensors.
    Image preprocessing and camera conversion belong to the calling pipeline.
    """

    def __init__(self, support, images, intrinsics, extrinsics, voxel_resolution, encode_image):
        """Bind one pass to fixed geometry, reference cameras and an image encoder."""
        validate_slat(support, "image conditioning support")
        if not isinstance(images, torch.Tensor) or images.ndim != 5 or images.shape[2] != 3:
            raise ValueError("images must have shape [B, V, 3, H, W]")
        batch, views = images.shape[:2]
        if batch != support.shape[0] or views < 1 or min(images.shape[-2:]) < 1:
            raise ValueError("images must provide at least one view per support instance")
        for name, tensor, shape in (("intrinsics", intrinsics, (batch, views, 3, 3)),
                                    ("extrinsics", extrinsics, (batch, views, 4, 4))):
            if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape:
                raise ValueError(f"{name} must have shape {shape}")
            if tensor.device != support.device or not torch.isfinite(tensor).all():
                raise ValueError(f"{name} must be finite and on the support device")
        if images.device != support.device or not images.is_floating_point():
            raise ValueError("images must be floating point tensors on the support device")
        if not torch.isfinite(images).all() or not ((images >= 0) & (images <= 1)).all():
            raise ValueError("images must be finite RGB values in [0, 1]")
        if type(voxel_resolution) is not int or voxel_resolution <= 0:
            raise ValueError("voxel_resolution must be a positive integer")
        self.support = support
        self.images = images
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics
        self.voxel_resolution = voxel_resolution
        self.encode_image = encode_image
        self._cache = {}

    def clear(self):
        """Release all CPU image tokens cached during this sampling pass."""
        self._cache.clear()

    def __enter__(self):
        """Start a sampling pass with an empty encoding cache."""
        self.clear()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Release cached tokens on completion or error without suppressing errors."""
        self.clear()

    @torch.no_grad()
    def __call__(self, patch, t):
        """Select a view afresh and reuse only its deterministic image encoding.

        t is accepted by the sampler callback interface; geometry and encoding
        do not depend on time. Returned tokens are on the reference image device.
        """
        from ..utils.patch_visibility import patch_view_crops, select_visible_view

        # 1. Map global latent rows to this instance's patch mask.
        layout = self.support.layout[patch.instance]
        coords = self.support.coords[layout, 1:]
        local_indices = patch.indices - layout.start
        expected_local = coords[local_indices] - coords.new_tensor(patch.bbox[:3])
        if not torch.equal(expected_local, patch.coords[:, 1:]):
            raise ValueError("Image condition patch does not match the supplied support")
        mask = torch.zeros(len(coords), dtype=torch.bool, device=coords.device)
        mask[local_indices] = True

        # 2. Project the patch into each view and crop around its visible support.
        crops, visible = patch_view_crops(
            coords, mask, self.voxel_resolution, self.images[patch.instance],
            self.intrinsics[patch.instance], self.extrinsics[patch.instance],
            crop_resolution=1024, dilate_mask=10,
        )

        # 3. Sample a view using visibility weights, even if its embedding is cached.
        view, image = select_visible_view(crops, visible)
        key = (patch.key, int(view))

        # 4. Cache only image encoding, keyed by patch and selected view.
        # Keep cached tokens on CPU rather than retaining every view on GPU.
        if key not in self._cache:
            encoded = self.encode_image(image)
            condition = PatchCondition(encoded["cond"], encoded["neg_cond"])
            self._cache[key] = PatchCondition(condition.cond.detach().cpu(), condition.neg_cond.detach().cpu())
            return condition
        cached = self._cache[key]
        return PatchCondition(cached.cond.to(image.device), cached.neg_cond.to(image.device))
