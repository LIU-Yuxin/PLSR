"""Silhouette overlap previews and reference-view selection for inference."""

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


REFERENCE_COLOR = (38, 200, 235)
MESH_COLOR = (255, 157, 66)


def silhouette_overlap(reference_alpha, mesh_alpha):
    """Measure intersection over union using the alignment foreground rule."""
    if reference_alpha.mode != "L" or mesh_alpha.mode != "L" or reference_alpha.size != mesh_alpha.size:
        raise ValueError("Silhouette masks must be L images on the same canvas")
    reference = np.asarray(reference_alpha) >= 1
    mesh = np.asarray(mesh_alpha) >= 1
    reference_pixels, mesh_pixels = int(reference.sum()), int(mesh.sum())
    intersection = int(np.count_nonzero(reference & mesh))
    union = reference_pixels + mesh_pixels - intersection
    reason = None
    if reference_pixels < 2:
        reason = "Reference silhouette is empty or has no spatial extent"
    elif mesh_pixels < 2:
        reason = "Rendered mesh silhouette is empty or has no spatial extent"
    return {
        "iou": intersection / union if union else 0.0,
        "intersection_pixels": intersection, "union_pixels": union,
        "reference_pixels": reference_pixels, "mesh_pixels": mesh_pixels,
        "valid": reason is None, "reason": reason,
    }


def _save_overlay(path, image, reference_alpha, mesh_alpha, record, minimum):
    """Show both exclusive regions over the prepared reference image."""
    width, height = reference_alpha.size
    scale = min(1.0, 1024 / max(width, height))
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    rgb = image.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    rgb = Image.fromarray(np.round(rgb * 255).astype(np.uint8)).resize(size, Image.Resampling.LANCZOS)
    reference = np.asarray(reference_alpha.resize(size, Image.Resampling.NEAREST)) >= 1
    mesh = np.asarray(mesh_alpha.resize(size, Image.Resampling.NEAREST)) >= 1
    pixels = (np.asarray(rgb).astype(np.float32) * .35).astype(np.uint8)
    pixels[reference & mesh] = (np.asarray(rgb)[reference & mesh].astype(np.float32) * .35 + 150).clip(0, 255).astype(np.uint8)
    pixels[reference & ~mesh] = REFERENCE_COLOR
    pixels[mesh & ~reference] = MESH_COLOR
    canvas = Image.new("RGB", (max(size[0], 620), size[1] + 86), (23, 26, 30))
    canvas.paste(Image.fromarray(pixels), ((canvas.width - size[0]) // 2, 86))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default(size=18)
    except TypeError:
        font = ImageFont.load_default()
    decision = "KEPT" if record["accepted"] else "REJECTED"
    title = f'View {record["view"]} | IoU {record["iou"]:.1%} | minimum {minimum:.1%} | {decision}'
    draw.text((10, 8), title, fill="white", font=font)
    draw.text((10, 34), "Image only", fill=REFERENCE_COLOR, font=font)
    draw.text((155, 34), "Mesh only", fill=MESH_COLOR, font=font)
    draw.text((295, 34), "Overlap: light gray", fill=(220, 220, 220), font=font)
    if record["reason"]:
        draw.text((10, 60), record["reason"], fill=(255, 190, 170), font=font)
    canvas.save(path)


def diagnose_reference_views(views, scene, reference_images, output_dir, *, min_iou=0.6,
                             device="cuda", render_masks=None):
    """Save every view's post-alignment IoU and overlay before selecting views.

    Reuse target masks from estimated-camera alignment. Render calibrated
    views without warping their images or changing their supplied cameras.
    Empty and degenerate silhouettes are rejected even when min_iou is zero.
    """
    if isinstance(min_iou, bool) or not np.isfinite(min_iou) or not 0 <= min_iou <= 1:
        raise ValueError("min_iou must be a finite number in [0, 1]")
    count = len(reference_images)
    if (len(views.reference_alphas) != count or views.images.shape[1] != count
            or views.intrinsics.shape[1] != count or views.extrinsics.shape[1] != count):
        raise ValueError("Reference images, alpha masks and cameras must have matching lengths")
    targets = {index: alignment.target_alpha for index, alignment in views.alignments}
    missing = [index for index in range(count) if index not in targets]
    if missing:
        if render_masks is None:
            from .reference_alignment import RenderedReferenceAligner
            render_masks = RenderedReferenceAligner(device).render_masks
        masks = render_masks(
            scene, [views.reference_alphas[index].size for index in missing],
            [views.intrinsics[0, index].cpu().numpy() for index in missing],
            [views.extrinsics[0, index].cpu().numpy() for index in missing],
        )
        if len(masks) != len(missing):
            raise ValueError("Silhouette renderer did not return a mask for every view")
        targets.update(zip(missing, masks))
    output = Path(output_dir)
    previews = output / "view_diagnostics"
    previews.mkdir(parents=True, exist_ok=True)
    framing = dict(views.camera_framing)
    records = []
    for index, reference in enumerate(reference_images):
        record = {"view": index, "image": str(reference.path),
                  "camera_estimated_fields": list(reference.missing_camera_fields),
                  "image_aligned": any(index == i for i, _ in views.alignments),
                  **silhouette_overlap(views.reference_alphas[index], targets[index])}
        if index in framing and not framing[index]["applied"]:
            record.update(valid=False, reason=framing[index]["reason"])
        record["accepted"] = record["valid"] and record["iou"] >= min_iou
        if record["valid"] and not record["accepted"]:
            record["reason"] = "Silhouette IoU is below the configured minimum"
        record["overlay"] = f"view_diagnostics/view_{index:03d}.png"
        _save_overlay(output / record["overlay"], views.images[0, index],
                      views.reference_alphas[index], targets[index], record, min_iou)
        records.append(record)
    report = {
        "metric": "silhouette intersection over union after image alignment",
        "foreground": "alpha >= 1/255", "min_view_iou": float(min_iou),
        "kept_views": [r["view"] for r in records if r["accepted"]],
        "rejected_views": [r["view"] for r in records if not r["accepted"]],
        "views": records,
    }
    temporary = output / ".view_diagnostics.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output / "view_diagnostics.json")
    return report


def retain_reference_views(views, indices):
    """Select images, alpha and camera tensors together in original view order."""
    count = views.images.shape[1]
    if not indices or sorted(set(indices)) != list(indices) or any(i < 0 or i >= count for i in indices):
        raise ValueError("At least one valid reference-view index is required in input order")
    if list(indices) == list(range(count)):
        return views
    alignments = dict(views.alignments)
    framing = dict(views.camera_framing)
    return replace(
        views, images=views.images[:, indices], intrinsics=views.intrinsics[:, indices],
        extrinsics=views.extrinsics[:, indices],
        reference_alphas=tuple(views.reference_alphas[i] for i in indices),
        camera_framing=tuple((new, framing[old])
                             for new, old in enumerate(indices) if old in framing),
        alignments=tuple((new, alignments[old]) for new, old in enumerate(indices) if old in alignments),
    )
