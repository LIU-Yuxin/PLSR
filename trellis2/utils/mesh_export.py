"""Bake decoded voxel materials and assemble mesh parts in an explicit space."""

from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import warnings

import torch
import trimesh

from plsr.geometry import restore_glb_coordinates
from ..pipelines.slat_codec import AABB
from .slat_decode import DecodedPart
from .texture_budget import plan_texture_budget, validate_texture_budget


def _bake_mesh(data, *, target, texture_size, verbose):
    """Reconstruct the CPU payload on CUDA and bake its geometry with voxel PBR."""
    import o_voxel
    from ..representations import MeshWithVoxel

    torch.set_num_threads(4)
    mesh = MeshWithVoxel.from_dict(data, device="cuda")
    return o_voxel.postprocess.to_glb(
        vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs,
        coords=mesh.coords, attr_layout=mesh.layout, aabb=AABB, voxel_size=mesh.voxel_size,
        decimation_target=target, texture_size=texture_size,
        remesh=True, remesh_band=1, remesh_project=0, verbose=verbose,
    )


def _bake_coarse_mesh(data, *, target, texture_size, verbose):
    """Limit coarse face count for rasterization and bake with the supplied target."""
    import o_voxel
    from ..representations import MeshWithVoxel

    torch.set_num_threads(4)
    mesh = MeshWithVoxel.from_dict(data, device="cuda")
    mesh.simplify(16777216)  # The original nvdiffrast face-count limit.
    return o_voxel.postprocess.to_glb(
        vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs,
        coords=mesh.coords, attr_layout=mesh.layout, aabb=AABB, voxel_size=mesh.voxel_size,
        decimation_target=target, texture_size=texture_size,
        remesh=True, remesh_band=1, remesh_project=0, verbose=verbose,
    )


def _bake_part(part, *, target, texture_size, verbose, index, worker=_bake_mesh, on_event=None):
    """Bake CPU mesh data in an isolated CUDA process, retrying runtime failures.

    A native failure can invalidate a CUDA context. Each attempt uses a fresh
    process and the same inputs/settings; successful output contains no tensors.
    """
    data = part.to_dict()
    if torch.cuda.is_initialized():
        torch.cuda.empty_cache()
    for attempt in range(3):
        try:
            with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as executor:
                baked = executor.submit(worker, data, target=target, texture_size=texture_size, verbose=verbose).result()
            if on_event is not None:
                on_event("part_baked", part=index, bake_attempt=attempt + 1,
                         vertices=len(baked.vertices), faces=len(baked.faces))
            return baked
        except RuntimeError as exc:
            if on_event is not None:
                on_event("part_bake_retry" if attempt < 2 else "part_bake_failed",
                         part=index, bake_attempt=attempt + 1, error=f"{type(exc).__name__}: {exc}")
            if attempt == 2:
                raise RuntimeError(f"Mesh part {index} failed to bake after 3 attempts with unchanged settings") from exc
            warnings.warn(f"Mesh part {index} baking failed; retrying in a fresh process ({attempt + 2}/3): {exc}",
                          RuntimeWarning, stacklevel=2)
        except Exception as exc:
            if on_event is not None:
                on_event("part_bake_failed", part=index, bake_attempt=attempt + 1,
                         error=f"{type(exc).__name__}: {exc}")
            raise


def bake_coarse_mesh(mesh, *, texture_size=4096, index=0, verbose=False, on_event=None):
    """Bake a coarse mesh in the GLB (x,z,-y) frame.

    The mesh input path later applies +90 degrees about X before canonicalizing.
    Keep this frame here so generated and supplied GLBs share that input path.
    """
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError("The decoded coarse mesh is empty")
    return _bake_part(mesh.cpu(), target=1000000, texture_size=texture_size,
                      verbose=verbose, index=f"coarse_{index}", worker=_bake_coarse_mesh,
                      **({"on_event": on_event} if on_event is not None else {}))


def assemble_parts(parts, *, canonical_to_input=None, texture_size=4096, texture_budget=None,
                   part_weights=None, decimation_target=500000, verbose=False, on_event=None):
    """Bake each local part, restore its placement and assemble a scene.

    Undo the baker's (x,z,-y) axes, then apply local-to-canonical and optional
    canonical-to-input transforms, in that order. Parts remain separate scene
    meshes, retaining their overlapping surfaces.
    The per-part face target is min(decimation_target, max(1, faces//5));
    decimation_target=None uses the input face count. texture_size caps each
    map; texture_budget bounds total pixels per PBR image type across parts.
    Supply planned part_weights to allocate before consuming a streaming decode.
    Zero/None budget restores fixed texture_size maps on every part.
    Texture budgeting is opt-in; fixed-size maps remain the default.
    """
    if texture_size is None:
        texture_size = 4096
    validate_texture_budget(texture_size, texture_budget)
    if decimation_target is not None and (type(decimation_target) is not int or decimation_target <= 0):
        raise ValueError("decimation_target must be a positive integer or None")
    if part_weights is None and texture_budget:
        raise ValueError("Bounded streaming export requires planned part_weights")
    plan = None if part_weights is None else plan_texture_budget(
        part_weights, texture_size=texture_size, texture_budget=texture_budget)
    if plan is not None:
        plan.log(on_event)
    scene = trimesh.Scene()
    for index, part in enumerate(parts):
        if plan is not None and index >= len(plan.sizes):
            raise ValueError("Texture plan and decoded part count differ")
        part_texture_size = texture_size if plan is None else plan.sizes[index]
        transform = canonical_to_input
        if isinstance(part, DecodedPart):
            transform = part.local_to_canonical if transform is None else transform @ part.local_to_canonical
            part = part.mesh
        if len(part.vertices) == 0 or len(part.faces) == 0:
            if on_event is not None:
                on_event("part_empty", part=index, vertices=len(part.vertices), faces=len(part.faces))
            raise ValueError(f"Decoded mesh part {index} is empty")
        target = len(part.faces) if decimation_target is None else min(decimation_target, max(1, len(part.faces) // 5))
        if on_event is not None:
            on_event("part_bake_start", part=index, vertices=len(part.vertices), faces=len(part.faces),
                     material_voxels=len(part.coords), target_faces=target, texture_size=part_texture_size)
        textured = _bake_part(part, target=target, texture_size=part_texture_size, verbose=verbose, index=index,
                              **({"on_event": on_event} if on_event is not None else {}))
        # Geometry and material voxels share the local unit cube while baking.
        # Apply placement afterwards; postprocessing thresholds use local units.
        textured = restore_glb_coordinates(textured, transform)
        scene.add_geometry(textured, node_name=f"part_{index}", geom_name=f"part_{index}")
        if on_event is not None:
            on_event("part_merged", part=index, scene_parts=len(scene.geometry))
    if not scene.geometry:
        raise ValueError("At least one decoded mesh part is required")
    if plan is not None and len(scene.geometry) != len(plan.sizes):
        raise ValueError("Texture plan and decoded part count differ")
    if on_event is not None:
        on_event("merge_complete", scene_parts=len(scene.geometry))
    return scene
