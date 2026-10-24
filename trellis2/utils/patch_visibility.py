"""Project SLat patches and select reference images using voxel visibility."""

import math

import torch
from torch.nn import functional as F
from kornia.morphology import dilation


# diffoctreerast culls voxel centers at camera depth <= 0.2.
_NEAR_PLANE = 0.2


@torch.no_grad()
@torch.amp.autocast(device_type="cuda", enabled=False)
def render_patch_masks(coords, mask, voxel_resolution, intrinsics, extrinsics, resolution=128):
    """Render visible patch pixels in each camera as [V, 3, H, W].

    Cameras use OpenCV axes (+z forward, +y down) and zero-skew normalized
    intrinsics. coords contains one object's occupied grid cells. Selected cells are white;
    the remaining cells are opaque black occluders. Cell centers map to
    (coords + 0.5) / voxel_resolution - 0.5 in canonical world coordinates.
    Off-center principal points are mapped through a wider centered render
    because the rasterizer generates rays around the image center.
    The octree uses depth floor(log2(voxel_resolution)): non-power-of-two grids
    therefore have wider rendering voxels, e.g. 80/160 grids use widths 1/64
    and 1/128. This is an approximate visibility mask, not an exact surface test.
    """
    from ..representations.octree import DfsOctree
    from ..renderers.octree_renderer import OctreeRenderer

    renderer = OctreeRenderer()
    if renderer.unsupported:
        raise ImportError("Patch visibility requires diffoctreerast")
    renderer.rendering_options.resolution = resolution
    renderer.rendering_options.near = _NEAR_PLANE
    renderer.rendering_options.far = 100.0
    renderer.rendering_options.bg_color = (0, 0, 0)
    renderer.rendering_options.ssaa = 2
    representation = DfsOctree(
        depth=10, aabb=[-0.5, -0.5, -0.5, 1, 1, 1], device=coords.device,
        primitive="voxel", sh_degree=0, primitive_config={"solid": True},
    )
    representation.position = (coords.float() + 0.5) / voxel_resolution
    representation.depth = torch.full(
        (len(coords), 1), int(math.log2(voxel_resolution)), dtype=torch.uint8, device=coords.device,
    )
    colors = torch.ones_like(representation.position)
    colors[~mask] = 0
    def render_view(k, e):
        """Map centered rasterizer rays into the requested camera's image plane."""
        principal = torch.stack([k[0, 2], k[1, 2]])
        if torch.equal(principal, principal.new_full((2,), 0.5)):
            return renderer.render(representation, intrinsics=k, extrinsics=e, colors_overwrite=colors)["color"]

        # Cover both sides of the requested frustum before resampling; a simple
        # image translation would lose rays outside the centered field of view.
        extent = torch.maximum(principal, 1 - principal)
        centered = k.clone()
        centered[0, 0] /= 2 * extent[0]
        centered[1, 1] /= 2 * extent[1]
        centered[0, 2] = centered[1, 2] = 0.5
        visible = renderer.render(representation, intrinsics=centered, extrinsics=e, colors_overwrite=colors)["color"]
        axis = (torch.arange(resolution, device=visible.device, dtype=visible.dtype) + 0.5) / resolution
        y, x = torch.meshgrid(axis, axis, indexing="ij")
        grid = (torch.stack([x, y], dim=-1) - principal.to(visible)) / extent.to(visible)
        return F.grid_sample(visible[None], grid[None], mode="bilinear", padding_mode="zeros", align_corners=False)[0]

    return torch.stack([render_view(k, e) for k, e in zip(intrinsics, extrinsics)])


def projected_crop_boxes(bbox, intrinsics, extrinsics):
    """Project canonical-world bounds to one normalized crop per camera.

    Intrinsics use image coordinates in [0, 1]; extrinsics map world to camera.
    Expand the projected bounds to a square with a 1.2 margin and minimum side
    0.2, then clip to the image. Bounds behind the near plane yield an empty
    crop; bounds intersecting it use the full image to avoid singular projection.
    """
    x0, y0, z0, x1, y1, z1 = bbox
    corners = torch.tensor([
        [x0, y0, z0, 1], [x0, y0, z1, 1], [x0, y1, z0, 1], [x0, y1, z1, 1],
        [x1, y0, z0, 1], [x1, y0, z1, 1], [x1, y1, z0, 1], [x1, y1, z1, 1],
    ]).to(extrinsics)
    boxes = []
    for k, e in zip(intrinsics, extrinsics):
        depths = e[2] @ corners.T
        if depths.max() <= _NEAR_PLANE:
            boxes.append(extrinsics.new_zeros(4))
            continue
        if depths.min() <= _NEAR_PLANE:
            boxes.append(extrinsics.new_tensor([0, 0, 1, 1]))
            continue
        projected = k @ e[:3] @ corners.T
        projected = projected[:2] / projected[2:3]
        left, right = projected[0].min().item(), projected[0].max().item()
        top, bottom = projected[1].min().item(), projected[1].max().item()
        cx, cy = (left + right) / 2, (top + bottom) / 2
        half = max(max((right - left) / 2, (bottom - top) / 2) * 1.2, 0.2 / 2)
        boxes.append(torch.tensor([cx - half, cy - half, cx + half, cy + half]).to(extrinsics).clamp(0, 1))
    return boxes


def crop_and_resize(image, bbox, resolution):
    """Crop normalized bounds, pad to a square in pixels and resize bilinearly.

    An empty or offscreen crop returns a black image with the input dtype and
    device. Padding is symmetric, with an extra pixel on the right/bottom when
    the size difference is odd.
    """
    height, width = image.shape[-2:]
    left, top, right, bottom = (bbox * torch.tensor([width, height, width, height], device=image.device)).int()
    crop = image[:, :, top:bottom, left:right]
    height, width = crop.shape[-2:]
    if height == 0 or width == 0:
        return image.new_zeros((*image.shape[:2], resolution, resolution))
    size = max(height, width)
    pad_h, pad_w = (size - height) // 2, (size - width) // 2
    crop = F.pad(crop, (pad_w, size - width - pad_w, pad_h, size - height - pad_h))
    return F.interpolate(crop, size=(resolution, resolution), mode="bilinear", align_corners=False)


@torch.no_grad()
def patch_view_crops(coords, mask, voxel_resolution, images, intrinsics, extrinsics,
                     crop_resolution=1024, dilate_mask=10):
    """Return RGB and visibility crops [V, 3, crop_resolution, crop_resolution].

    Use occupied cell bounds, including the full final cell, for projection.
    Render visibility at 128 pixels, round and dilate there, then resize to the
    reference image dimensions. Apply identical crop/padding transforms to RGB
    and visibility; visibility controls view selection and does not mask RGB.
    """
    # 1. Map the patch's half-open cell bounds into canonical world coordinates.
    selected = coords[mask]
    if not len(selected):
        raise ValueError("Visibility projection requires a nonempty patch")
    bbox = torch.cat([selected.amin(dim=0), selected.amax(dim=0) + 1]).float() / voxel_resolution - 0.5
    boxes = projected_crop_boxes(bbox, intrinsics, extrinsics)

    # 2. Expand visible pixels at render resolution before mapping to the images.
    visible = render_patch_masks(coords, mask, voxel_resolution, intrinsics, extrinsics).to(images.device)
    if dilate_mask > 0:
        kernel = torch.ones((dilate_mask, dilate_mask), device=visible.device)
        visible = dilation(torch.round(visible), kernel)
    visible = F.interpolate(visible, size=images.shape[2:], mode="bilinear", align_corners=False)

    # 3. Keep each image crop and its visibility weights spatially aligned.
    def crop_views(values):
        """Apply the per-view boxes and the same square resize to a tensor batch."""
        return torch.cat([crop_and_resize(values[i:i + 1], box, crop_resolution) for i, box in enumerate(boxes)])

    return crop_views(images), crop_views(visible)


def select_visible_view(crops, visible):
    """Sample a view in proportion to mean visibility within its padded crop.

    Recompute the draw each sampling step. If every view has zero visibility,
    select view 0 deterministically without consuming random numbers.
    """
    ratios = visible.mean(dim=(1, 2, 3))
    total = ratios.sum()
    index = 0 if total == 0 else torch.multinomial(ratios / total, num_samples=1).item()
    return index, crops[index:index + 1]
