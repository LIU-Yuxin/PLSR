"""Reference image preparation with corresponding normalized camera intrinsics."""

from dataclasses import dataclass

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from plsr.geometry import camera_to_canonical
from plsr.config import ReferenceImage
from plsr.inputs import has_useful_alpha
from ..utils.image_alignment import SilhouetteAlignment


@dataclass(frozen=True)
class PreparedViews:
    """Batched RGB conditions and cameras on one shared, padded canvas.

    Tensor shapes are [1,V,3,H,W], [1,V,3,3] and [1,V,4,4]. Cameras use
    normalized intrinsics and canonical world-to-camera extrinsics. Alignment
    records identify views whose images were moved onto the estimated canvas.
    """

    images: torch.Tensor
    intrinsics: torch.Tensor
    extrinsics: torch.Tensor
    alignments: tuple[tuple[int, SilhouetteAlignment], ...] = ()
    reference_alphas: tuple[Image.Image, ...] = ()
    camera_framing: tuple[tuple[int, dict], ...] = ()


def prepare_reference_images(views, input_to_canonical, remove_background=None, output_size=4096,
                             *, estimate_cameras=None, canonical_scene=None, align_images=None,
                             independent_camera_estimation=False):
    """Prepare images and cameras, estimating only independently missing fields.

    Resize each image by its long edge, composite onto black, and center-pad
    to a shared canvas. Apply the actual integer resize and padding to supplied
    intrinsics; transform supplied OpenCV poses into canonical mesh space.
    Existing alpha is retained, otherwise remove_background supplies it.

    The estimator fills only absent camera fields. Independent estimation sends
    each incomplete view to the estimator separately. Before image alignment,
    center these cameras by folding an image-plane transform into their intrinsics.
    The mesh and extrinsics remain unchanged; alignment then holds cameras fixed.
    The aligned RGBA intermediate stores black-composited RGB bytes;
    applying its alpha for the final condition retains two-stage soft-edge
    attenuation. Fully calibrated views composite once and are not warped.
    """
    if not views or type(output_size) is not int or output_size <= 0:
        raise ValueError("Reference views and a positive output_size are required")
    if type(independent_camera_estimation) is not bool:
        raise ValueError("independent_camera_estimation must be a boolean")
    images, cameras, intrinsics, rgba_images, alphas = [], [], [], [], []
    estimated_indices = [i for i, view in enumerate(views) if view.missing_camera_fields]
    missing = any(view.intrinsics is None or view.extrinsics is None for view in views)
    if missing and (estimate_cameras is None or canonical_scene is None):
        raise ValueError("Missing reference cameras require an estimator and a canonical scene")
    # 1. Resize/mask references and convert supplied camera fields.
    for view in views:
        with Image.open(view.path) as source:
            source.load()
            source = source.copy()
        width, height = source.size
        has_alpha = has_useful_alpha(source)
        source = source.convert("RGBA" if has_alpha else "RGB")
        scale = output_size / max(width, height)
        size = (max(1, int(width * scale)), max(1, int(height * scale)))
        image = source.resize(size, Image.Resampling.LANCZOS)
        if not has_alpha:
            if remove_background is None:
                raise ValueError(f"{view.path}: an RGB/opaque reference requires a background-removal model")
            image = remove_background(image.convert("RGB"))
            if image.mode != "RGBA" or image.size != size:
                raise ValueError("Background removal must return RGBA at the same image size")
        alphas.append(image.getchannel("A"))
        rgba = np.asarray(image).astype(np.float32) / 255
        images.append(torch.from_numpy((rgba[..., :3] * rgba[..., 3:4]).copy()).permute(2, 0, 1))
        # The alignment intermediate stores black-composited RGB and its alpha.
        # Quantize this stage before the final composite to retain soft-edge values.
        if align_images is not None and view.missing_camera_fields:
            rgba[..., :3] *= rgba[..., 3:4]
            rgba_images.append(Image.fromarray((rgba * 255).astype(np.uint8)))
        else:
            rgba_images.append(None)
        matrix = None
        if view.intrinsics is not None:
            matrix = np.asarray(view.intrinsics, dtype=np.float64).copy()
            if view.intrinsics_units == "normalized":
                matrix[0] *= width
                matrix[1] *= height
            matrix[0] *= size[0] / width
            matrix[1] *= size[1] / height
        intrinsics.append(matrix)
        cameras.append(None if view.extrinsics is None else camera_to_canonical(view.extrinsics, input_to_canonical))
    # 2. Pad to a common canvas and normalize pixel intrinsics on that canvas.
    height = max(image.shape[1] for image in images)
    width = max(image.shape[2] for image in images)
    for index, (image, matrix) in enumerate(zip(images, intrinsics)):
        h, w = image.shape[1:]
        left, top = (width - w) // 2, (height - h) // 2
        images[index] = F.pad(image, (left, width - w - left, top, height - h - top))
        if (w, h) != (width, height):
            padded_alpha = Image.new("L", (width, height))
            padded_alpha.paste(alphas[index], (left, top))
            alphas[index] = padded_alpha
        if rgba_images[index] is not None and (w, h) != (width, height):
            padded = Image.new("RGBA", (width, height))
            padded.paste(rgba_images[index], (left, top))
            rgba_images[index] = padded
        if matrix is not None:
            matrix[0, 2] += left
            matrix[1, 2] += top
            matrix[0] /= width
            matrix[1] /= height
    # 3. Estimate missing fields; retain the caller's supplied calibration/pose.
    if missing:
        pil_images = [Image.fromarray((image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)) for image in images]
        # Callbacks receive copies; supplied fields cannot be overwritten even
        # if an estimator returns predictions for every reference.
        predicted_intr, predicted_ext = list(intrinsics), list(cameras)
        groups = [[i] for i in estimated_indices] if independent_camera_estimation else [list(range(len(views)))]
        for indices in groups:
            group_intr, group_ext = estimate_cameras(
                canonical_scene, [pil_images[i] for i in indices],
                [None if intrinsics[i] is None else intrinsics[i].copy() for i in indices],
                [None if cameras[i] is None else cameras[i].copy() for i in indices],
            )
            if len(group_intr) != len(indices) or len(group_ext) != len(indices):
                raise ValueError("Camera estimation must return one calibration and pose per view")
            for i, calibration, extrinsic in zip(indices, group_intr, group_ext):
                predicted_intr[i], predicted_ext[i] = calibration, extrinsic
        intrinsics = [predicted_intr[i] if value is None else value for i, value in enumerate(intrinsics)]
        cameras = [predicted_ext[i] if value is None else value for i, value in enumerate(cameras)]
        for view, calibration, extrinsic in zip(views, intrinsics, cameras):
            if calibration is None or extrinsic is None:
                raise ValueError("Camera estimation left required fields missing")
            ReferenceImage(view.path, calibration, extrinsic)  # Validate the canonical camera contract.
    # 4. Warp only views with estimated fields, then build their RGB conditions.
    alignments, framing = [], []
    if estimated_indices and align_images is not None:
        from .camera_framing import center_camera_frames

        centered, records = center_camera_frames(
            canonical_scene, [intrinsics[i] for i in estimated_indices],
            [cameras[i] for i in estimated_indices],
        )
        for index, calibration, record in zip(estimated_indices, centered, records):
            intrinsics[index] = calibration
            framing.append((index, record))
        aligned = align_images(
            canonical_scene, [rgba_images[i] for i in estimated_indices],
            [np.asarray(intrinsics[i]).copy() for i in estimated_indices],
            [np.asarray(cameras[i]).copy() for i in estimated_indices],
        )
        if len(aligned) != len(estimated_indices):
            raise ValueError("Image alignment must return one result per estimated view")
        for index, result in zip(estimated_indices, aligned):
            if result.image.mode != "RGBA" or result.image.size != (width, height):
                raise ValueError("Aligned references must be RGBA on the prepared camera canvas")
            rgba = np.asarray(result.image).astype(np.float32) / 255
            images[index] = torch.from_numpy((rgba[..., :3] * rgba[..., 3:4]).copy()).permute(2, 0, 1)
            alphas[index] = result.image.getchannel("A")
            alignments.append((index, result))
    return PreparedViews(
        torch.stack(images)[None], torch.from_numpy(np.stack(intrinsics)).float()[None],
        torch.from_numpy(np.stack(cameras)).float()[None], tuple(alignments), tuple(alphas), tuple(framing),
    )
