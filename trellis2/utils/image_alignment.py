"""Image-space silhouette matching used after reference camera estimation."""

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class SilhouetteAlignment:
    """Warped RGBA, source-to-target 2x3 pixel affine, and target alpha mask."""

    image: Image.Image
    matrix: np.ndarray
    target_alpha: Image.Image


def match_silhouettes(source: Image.Image, target_alpha: Image.Image) -> SilhouetteAlignment:
    """Fit uniform scale and translation from foreground centroid and spread.

    Use alpha >= 1 as foreground, one scalar coordinate standard deviation,
    a float32 affine matrix and nearest-neighbor RGBA resampling.
    The matrix moves the reference to the target camera's image plane; do not
    apply it to that camera's intrinsics a second time.
    """
    if source.mode != "RGBA" or target_alpha.mode != "L":
        raise ValueError("Silhouette matching requires an RGBA source and an L target alpha")
    moments = []
    for name, mask in (("Reference", np.asarray(source)[..., 3]),
                       ("Rendered coarse", np.asarray(target_alpha))):
        points = cv2.findNonZero((mask >= 1).astype(np.uint8))
        if points is None:
            raise ValueError(f"{name} silhouette is empty; check the image alpha and camera framing")
        points = points.squeeze(1)
        center = np.mean(points, axis=0)
        spread = np.std(points - center)
        if spread == 0:
            raise ValueError(f"{name} silhouette has no spatial extent")
        moments.append((center, spread))
    (source_center, source_spread), (target_center, target_spread) = moments
    scale = target_spread / (source_spread + 1e-9)
    matrix = np.eye(2, 3, dtype=np.float32)
    matrix[0, 0] = matrix[1, 1] = scale
    matrix[:, 2] = target_center - scale * source_center
    aligned = cv2.warpAffine(np.asarray(source), matrix, target_alpha.size, flags=cv2.INTER_NEAREST)
    return SilhouetteAlignment(Image.fromarray(aligned), matrix, target_alpha)
