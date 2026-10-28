"""Align full reference images to the coarse mesh at the estimated cameras."""

import numpy as np
from PIL import Image
import torch

from ..utils.image_alignment import SilhouetteAlignment, match_silhouettes


def square_render_intrinsics(intrinsics, size):
    """Pad a rectangular camera canvas without changing projected pixel positions."""
    width, height = size
    side = max(size)
    left, top = (side - width) // 2, (side - height) // 2
    matrix = np.asarray(intrinsics).copy()
    matrix[0] *= width
    matrix[1] *= height
    matrix[0, 2] += left
    matrix[1, 2] += top
    matrix[:2] /= side
    return matrix, (left, top, left + width, top + height)


class RenderedReferenceAligner:
    """Use the PBR renderer's material alpha at the full reference resolution.

    Lighting and later depth layers do not affect its returned first-layer
    alpha. The renderer's alpha-only path avoids their cost at 4096 pixels.
    """

    def __init__(self, device="cuda"):
        """Select the CUDA device used for mesh alpha rasterization."""
        self.device = torch.device(device)

    @torch.no_grad()
    def render_masks(self, scene, sizes, intrinsics, extrinsics):
        """Render material-alpha masks on the corresponding reference canvases."""
        from ..renderers.pbr_mesh_renderer import PbrMeshRenderer
        from ..utils.pbr_scene import scene_to_pbr_mesh

        if not len(sizes) == len(intrinsics) == len(extrinsics):
            raise ValueError("Mask sizes and camera lists must have matching lengths")
        mesh = scene_to_pbr_mesh(scene, self.device)
        renderer = PbrMeshRenderer({"near": .01, "far": 100, "ssaa": 1}, device=self.device)
        masks = []
        for size, calibration, pose in zip(sizes, intrinsics, extrinsics):
            calibration, crop = square_render_intrinsics(calibration, size)
            renderer.rendering_options.resolution = max(size)
            alpha = renderer.render(
                mesh, torch.as_tensor(pose, dtype=torch.float32, device=self.device),
                torch.as_tensor(calibration, dtype=torch.float32, device=self.device),
                {}, alpha_only=True,
            ).alpha
            # Quantize alpha before fitting so foreground means at least one byte.
            masks.append(Image.fromarray((alpha.cpu().numpy() * 255).astype(np.uint8)).crop(crop))
        return masks

    @torch.no_grad()
    def __call__(self, scene, images, intrinsics, extrinsics):
        """Fit image scale and translation while keeping camera matrices fixed.

        Empty or degenerate masks remain unwarped so overlap diagnostics can
        report and reject that view without discarding other valid references.
        """
        targets = self.render_masks(scene, [image.size for image in images], intrinsics, extrinsics)
        results = []
        for image, target in zip(images, targets):
            if min(np.count_nonzero(np.asarray(image.getchannel("A"))),
                   np.count_nonzero(np.asarray(target))) < 2:
                results.append(SilhouetteAlignment(image.copy(), np.eye(2, 3, dtype=np.float32), target))
            else:
                results.append(match_silhouettes(image, target))
        return results
