"""Scene coordinates and cameras, independent of PyTorch and rendering."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh


def input_axis_rotation():
    """Rotate glTF input axes into model space: +90 degrees about X."""
    return trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])


@dataclass(frozen=True)
class CanonicalAsset:
    """A world-space scene in the model cube and its reversible transforms."""

    scene: trimesh.Scene
    input_to_canonical: np.ndarray
    canonical_to_input: np.ndarray


def flatten_scene(asset):
    """Bake every geometry instance's world transform, preserving its visual."""
    if isinstance(asset, (str, Path)):
        asset = trimesh.load_scene(asset, process=False)
    elif isinstance(asset, trimesh.Trimesh):
        asset = trimesh.Scene(asset)
    if not isinstance(asset, trimesh.Scene):
        raise TypeError("Expected a mesh, scene or mesh file path")
    flattened = trimesh.Scene()
    for node in asset.graph.nodes_geometry:
        transform, name = asset.graph[node]
        geometry = asset.geometry[name]
        if not isinstance(geometry, trimesh.Trimesh) or len(geometry.faces) == 0:
            raise ValueError(f"Scene node {node!r} must contain a nonempty triangle mesh")
        if not np.isfinite(transform).all() or not np.isfinite(geometry.vertices).all():
            raise ValueError(f"Scene node {node!r} has nonfinite geometry or transform")
        geometry = geometry.copy()
        geometry.apply_transform(transform)
        flattened.add_geometry(geometry, node_name=str(node), geom_name=str(node))
    if not flattened.geometry:
        raise ValueError("Scene contains no triangle meshes")
    return flattened


def canonicalize_mesh(asset):
    """Bake scene instances, rotate axes, then fit to a 0.99999-wide unit cube.

    Center the rotated world-space bounds and apply one positive uniform scale
    to every mesh. Return both transforms for camera conversion and export;
    leave the caller's scene unchanged.
    """
    scene = flatten_scene(asset)
    rotation = input_axis_rotation()
    # Rotate before measuring bounds; retain a small voxel-boundary margin.
    for geometry in scene.geometry.values():
        geometry.apply_transform(rotation)
    bounds = scene.bounds
    center = bounds.mean(axis=0)
    extent = np.ptp(bounds, axis=0).max()
    if not np.isfinite(extent) or extent <= 0:
        raise ValueError("Mesh must have finite, nonzero spatial extent")
    scale = 0.99999 / extent
    for geometry in scene.geometry.values():
        geometry.apply_translation(-center)
        geometry.apply_scale(scale)
    transform = rotation.copy()
    transform[:3, :3] *= scale
    transform[:3, 3] = -scale * center
    return CanonicalAsset(scene, transform, np.linalg.inv(transform))


def camera_to_canonical(extrinsics, input_to_canonical):
    """Transform OpenCV world-to-camera extrinsics, keeping a rigid rotation.

    The canonical scene's uniform scale also scales camera-space distances.
    Multiplying the top three rows by it preserves image projection while
    removing inverse scale from the camera rotation.
    """
    transform = np.asarray(input_to_canonical, dtype=np.float64)
    extrinsics = np.asarray(extrinsics, dtype=np.float64)
    if transform.shape != (4, 4) or extrinsics.shape != (4, 4):
        raise ValueError("Camera and canonical transforms must be 4x4 matrices")
    if not np.isfinite(transform).all() or not np.isfinite(extrinsics).all():
        raise ValueError("Camera and canonical transforms must be finite")
    scale = np.cbrt(np.linalg.det(transform[:3, :3]))
    if scale <= 0 or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3) * scale**2):
        raise ValueError("Canonical transform must contain a proper rotation and uniform positive scale")
    result = extrinsics @ np.linalg.inv(transform)
    result[:3] *= scale
    result[3] = [0, 0, 0, 1]
    return result


def prepare_pbr_scene(scene):
    """Split per-face PBR materials without changing vertices, UVs or textures.

    This stage supports glTF PBR materials, including constant materials without
    textures. Unsupported visual types fail explicitly instead of losing color.
    """
    result = trimesh.Scene()
    for name, geometry in flatten_scene(scene).geometry.items():
        visual = geometry.visual
        if not isinstance(visual, trimesh.visual.TextureVisuals):
            raise ValueError(f"{name}: expected a glTF PBR material; {type(visual).__name__} is not yet supported")
        material = visual.material
        if isinstance(material, trimesh.visual.material.MultiMaterial):
            face_materials = np.asarray(visual.face_materials)
            if face_materials.shape != (len(geometry.faces),) or face_materials.dtype.kind not in "iu":
                raise ValueError(f"{name}: invalid per-face material IDs")
            if np.any(face_materials < 0) or np.any(face_materials >= len(material.materials)):
                raise ValueError(f"{name}: per-face material ID is out of range")
            materials = enumerate(material.materials)
        else:
            face_materials = np.zeros(len(geometry.faces), dtype=np.int64)
            materials = [(0, material)]
        for index, material in materials:
            selected = face_materials == index
            if not selected.any():
                continue
            if not isinstance(material, trimesh.visual.material.PBRMaterial):
                raise ValueError(f"{name}: material {index} must be a glTF PBR material")
            if visual.uv is None and any(getattr(material, attr) is not None for attr in ("baseColorTexture", "metallicRoughnessTexture", "emissiveTexture", "normalTexture")):
                raise ValueError(f"{name}: textured PBR material requires UV coordinates")
            part = trimesh.Trimesh(
                vertices=geometry.vertices.copy(), faces=geometry.faces[selected].copy(),
                vertex_normals=geometry.vertex_normals.copy(), process=False,
                visual=trimesh.visual.TextureVisuals(
                    uv=None if visual.uv is None else visual.uv.copy(), material=material.copy(),
                ),
            )
            result.add_geometry(part, node_name=f"{name}/{index}", geom_name=f"{name}/{index}")
    return result


def restore_glb_coordinates(mesh, canonical_to_input=None):
    """Undo O-Voxel to_glb's axis conversion, then optionally restore input space."""
    transform = input_axis_rotation()
    if canonical_to_input is not None:
        transform = np.asarray(canonical_to_input) @ transform
    result = mesh.copy()
    result.apply_transform(transform)
    return result
