"""Adapt a transformed glTF PBR scene to the existing TRELLIS.2 renderer."""

import numpy as np
import torch

from plsr.geometry import prepare_pbr_scene
from ..representations.mesh import MeshWithPbrMaterial, PbrMaterial, Texture, AlphaMode


def scene_to_pbr_mesh(scene, device="cpu"):
    """Bake scene instances and preserve glTF material channels for rendering.

    Split per-face materials, flip V to the rasterizer's texture convention, and
    prepare compatible mipmapped textures. Constant PBR factors need no UVs.
    """
    scene = prepare_pbr_scene(scene)
    vertices, faces, material_ids, uv_coords, materials = [], [], [], [], []
    offset = 0
    for index, geometry in enumerate(scene.geometry.values()):
        material = geometry.visual.material
        rgba = None if material.baseColorTexture is None else np.array(material.baseColorTexture.convert("RGBA"))
        mr = None if material.metallicRoughnessTexture is None else np.array(material.metallicRoughnessTexture.convert("RGB"))

        def texture(array):
            """Convert an HWC byte image to a mipmap-compatible float texture."""
            if array is None:
                return None
            image = torch.from_numpy(np.ascontiguousarray(array)).float() / 255
            height, width = image.shape[:2]
            target = (1 << (height - 1).bit_length(), 1 << (width - 1).bit_length())
            if (height, width) != target:
                image = torch.nn.functional.interpolate(
                    image.permute(2, 0, 1)[None], target, mode="bilinear", align_corners=False,
                )[0].permute(1, 2, 0).contiguous()
            return Texture(image)

        factor = np.ones(4) if material.baseColorFactor is None else material.baseColorFactor / 255
        materials.append(PbrMaterial(
            base_color_texture=texture(None if rgba is None else rgba[..., :3]),
            base_color_factor=factor[:3].tolist(),
            metallic_texture=texture(None if mr is None else mr[..., 2:3]),
            metallic_factor=1.0 if material.metallicFactor is None else material.metallicFactor,
            roughness_texture=texture(None if mr is None else mr[..., 1:2]),
            roughness_factor=1.0 if material.roughnessFactor is None else material.roughnessFactor,
            alpha_texture=texture(None if rgba is None else rgba[..., 3:4]),
            # Constant alpha follows the same tensor path as sampled alpha.
            alpha_factor=torch.tensor(float(factor[3])),
            alpha_mode={"MASK": AlphaMode.MASK, "BLEND": AlphaMode.BLEND}.get(material.alphaMode, AlphaMode.OPAQUE),
            alpha_cutoff=0.5 if material.alphaCutoff is None else material.alphaCutoff,
        ))
        vertices.append(torch.tensor(geometry.vertices, dtype=torch.float32))
        faces.append(torch.tensor(geometry.faces, dtype=torch.int32) + offset)
        material_ids.append(torch.full((len(geometry.faces),), index, dtype=torch.int32))
        uv = np.zeros((len(geometry.faces), 3, 2)) if geometry.visual.uv is None else geometry.visual.uv[geometry.faces].copy()
        uv[..., 1] = 1 - uv[..., 1]
        uv_coords.append(torch.tensor(uv, dtype=torch.float32))
        offset += len(geometry.vertices)
    return MeshWithPbrMaterial(torch.cat(vertices), torch.cat(faces), torch.cat(material_ids),
                               torch.cat(uv_coords), materials).to(device)
