"""Bound object texture pixels independently of memory-limited decode parts."""

from dataclasses import dataclass
import math


def validate_texture_budget(texture_size, texture_budget):
    """Zero/None keeps fixed per-part textures; bounded sizes are powers of two."""
    if type(texture_size) is not int or texture_size <= 0:
        raise ValueError("texture_size must be a positive integer")
    if texture_budget is None:
        return
    if type(texture_budget) is not int or texture_budget < 0:
        raise ValueError("texture_budget must be zero or a positive power of two")
    if texture_budget:
        for name, size in (("texture_size", texture_size), ("texture_budget", texture_budget)):
            if size < 256 or size & (size - 1):
                raise ValueError(f"{name} must be a power of two >= 256 in budget mode")


@dataclass(frozen=True)
class TexturePlan:
    sizes: tuple[int, ...]
    budget: int | None

    @property
    def allocated_pixels(self):
        return sum(size ** 2 for size in self.sizes)

    def log(self, on_event):
        if on_event is not None:
            on_event("texture_plan", parts=len(self.sizes), bake_sizes=list(self.sizes),
                     texture_budget=self.budget, pixels_per_image_type=self.allocated_pixels,
                     material_images=2 * len(self.sizes),
                     weighting="selected_surface_latent_cells")


def plan_texture_budget(weights, *, texture_size=4096, texture_budget=8192):
    """Allocate per-part square maps by surface-support weights, without merging.

    Sparse surface cell counts approximate part area at a common voxel scale.
    Allocate area shares, then round edges to powers of two. Unused pixels are
    not forced onto small parts when a larger part's next upgrade cannot fit.
    The sum of size**2 is bounded by texture_budget**2 for each PBR image type.
    Both map types use the same size. Overlap cells count in every exported part
    but do not increase the total budget. A 256px minimum fails explicitly if
    the requested budget is too small, rather than silently exceeding it.
    """
    validate_texture_budget(texture_size, texture_budget)
    weights = tuple(float(value) for value in weights)
    if not weights or any(not math.isfinite(value) or value <= 0 for value in weights):
        raise ValueError("Texture allocation requires positive finite part weights")
    if not texture_budget:
        return TexturePlan((texture_size,) * len(weights), texture_budget)
    cap = min(texture_size, texture_budget)
    remaining = texture_budget ** 2 - len(weights) * 256 ** 2
    if remaining < 0:
        raise ValueError("Texture budget cannot fit 256px maps for every part; increase --texture-budget")
    targets = [float(256 ** 2)] * len(weights)
    active = list(range(len(weights)))
    while active and remaining > 0:
        weight_sum = sum(weights[index] for index in active)
        increments = {index: remaining * weights[index] / weight_sum for index in active}
        capped = [index for index in active if targets[index] + increments[index] >= cap ** 2]
        if capped:
            for index in capped:
                remaining -= cap ** 2 - targets[index]
                targets[index] = float(cap ** 2)
                active.remove(index)
        else:
            for index in active:
                targets[index] += increments[index]
            break
    sizes = [256] * len(weights)
    for index, target in enumerate(targets):
        while sizes[index] < cap and (2 * sizes[index]) ** 2 <= target + 1e-6:
            sizes[index] *= 2
    remaining = texture_budget ** 2 - sum(size ** 2 for size in sizes)
    while True:
        candidates = [index for index, size in enumerate(sizes)
                      if size < cap and targets[index] / size ** 2 >= 2
                      and 3 * size ** 2 <= remaining]
        if not candidates:
            break
        index = max(candidates, key=lambda index: (targets[index] / sizes[index] ** 2, -index))
        remaining -= 3 * sizes[index] ** 2
        sizes[index] *= 2
    return TexturePlan(tuple(sizes), texture_budget)
