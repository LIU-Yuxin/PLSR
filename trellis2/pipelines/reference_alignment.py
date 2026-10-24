"""Align full reference images to the coarse mesh at the estimated cameras."""

import numpy as np
from PIL import Image
import torch

from ..utils.image_alignment import match_silhouettes


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
    def __call__(self, scene, images, intrinsics, extrinsics):
        """Warp references to first-surface material alpha at their fixed cameras.

        Square-pad the render intrinsics for rectangular inputs, crop the mask
        back to the reference canvas, then fit image scale and translation.
        Returned affines act on image pixels; camera matrices are unchanged.
        """
        from ..renderers.pbr_mesh_renderer import PbrMeshRenderer
        from ..utils.pbr_scene import scene_to_pbr_mesh

        mesh = scene_to_pbr_mesh(scene, self.device)
        renderer = PbrMeshRenderer({"near": .01, "far": 100, "ssaa": 1}, device=self.device)
        results = []
        for image, calibration, pose in zip(images, intrinsics, extrinsics):
            calibration, crop = square_render_intrinsics(calibration, image.size)
            renderer.rendering_options.resolution = max(image.size)
            alpha = renderer.render(
                mesh, torch.as_tensor(pose, dtype=torch.float32, device=self.device),
                torch.as_tensor(calibration, dtype=torch.float32, device=self.device),
                {}, alpha_only=True,
            ).alpha
            # Quantize alpha before fitting so foreground means at least one byte.
            target = Image.fromarray((alpha.cpu().numpy() * 255).astype(np.uint8)).crop(crop)
            results.append(match_silhouettes(image, target))
        return results
