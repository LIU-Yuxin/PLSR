"""PBR mesh voxelization and reference VAE encoding/decoding."""

from contextlib import contextmanager

import torch
from torch.nn import functional as F

from .base import Pipeline
from ..modules.sparse import SparseTensor
from ..utils.slat_patches import require_aligned, validate_slat


PBR_LAYOUT = {"base_color": slice(0, 3), "metallic": slice(3, 4),
              "roughness": slice(4, 5), "alpha": slice(5, 6)}
AABB = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]


def _voxelize_prepared_scene(scene, resolution):
    """Voxelize a prepared PBR scene with the reference O-Voxel settings."""
    import o_voxel

    mesh = scene.to_mesh()
    if (mesh.bounds < -0.5).any() or (mesh.bounds > 0.5).any():
        raise ValueError("Voxelization requires a canonical mesh inside the unit cube")

    # 1. Voxelize surface geometry and sample PBR attributes on the same grid.
    coords, dual, intersected = o_voxel.convert.mesh_to_flexible_dual_grid(
        torch.from_numpy(mesh.vertices).float(), torch.from_numpy(mesh.faces).long(),
        voxel_size=1 / resolution, aabb=AABB,
        face_weight=1.0, boundary_weight=0.2, regularization_weight=1e-2,
    )
    material_coords, attributes = o_voxel.convert.textured_mesh_to_volumetric_attr(
        scene, voxel_size=1 / resolution, aabb=AABB,
    )

    # 2. Match geometry and material rows in Morton order before packing attributes.
    order = torch.argsort(o_voxel.serialize.encode_seq(coords))
    material_order = torch.argsort(o_voxel.serialize.encode_seq(material_coords))
    coords, dual, intersected = coords[order], dual[order], intersected[order]
    if len(coords) == 0 or not torch.equal(coords, material_coords[material_order]):
        raise ValueError("Geometry and material voxelization must cover exactly the same coordinates")
    attributes = {key: value[material_order] for key, value in attributes.items()}

    # 3. Store within-cell QEF offsets as bytes, matching VAE input quantization.
    # The QEF vertices are relative to the AABB origin; subtract each cell index.
    attributes["dual_vertices"] = ((dual * resolution - coords).clamp(0, 1) * 255).byte()
    attributes["intersected"] = (intersected[:, :1] + 2 * intersected[:, 1:2] + 4 * intersected[:, 2:3]).byte()
    return coords, attributes


def voxelize_pbr_scene(scene, resolution, *, independent_parts=False):
    """Pack O-Voxel geometry and PBR attributes in aligned Morton order.

    Coarse meshes exported from several overlapping decoder patches must be
    voxelized one part at a time. A joint QEF and material sample mixes their
    coincident surfaces and can turn a clean coarse mesh into broken geometry.
    For duplicate cells, keep the first part in scene order; all parts still
    contribute their non-overlapping cells. Ordinary supplied meshes retain the
    reference whole-scene behavior.
    """
    import o_voxel
    import trimesh
    from plsr.geometry import prepare_pbr_scene

    if type(resolution) is not int or resolution <= 0 or resolution % 128:
        raise ValueError("Encoding resolution must be a positive multiple of 128")
    scene = prepare_pbr_scene(scene)
    if not independent_parts or len(scene.geometry) == 1:
        return _voxelize_prepared_scene(scene, resolution)

    parts = [_voxelize_prepared_scene(trimesh.Scene(geometry), resolution)
             for geometry in scene.geometry.values()]
    coords = torch.cat([part[0] for part in parts])
    keys = o_voxel.serialize.encode_seq(coords)
    # Stable sorting makes the first scene part win when parts occupy the same cell.
    order = torch.argsort(keys, stable=True)
    sorted_keys = keys[order]
    first = torch.cat([torch.ones(1, dtype=torch.bool), sorted_keys[1:] != sorted_keys[:-1]])
    selected = order[first]
    attributes = {key: torch.cat([part[1][key] for part in parts])[selected]
                  for key in parts[0][1]}
    return coords[selected], attributes


class SLatCodec(Pipeline):
    """Run loaded Shape/PBR encoder pairs, decoder pairs, or both, stage by stage."""

    def __init__(self, models, device="cuda", low_vram=True):
        """Bind Shape/PBR model pairs and choose stage-wise device offloading."""
        encoders, decoders = {"shape_encoder", "pbr_encoder"}, {"shape_decoder", "pbr_decoder"}
        if set(models) not in (encoders, decoders, encoders | decoders) or any(m is None for m in models.values()):
            raise ValueError("Codec models must provide a Shape/PBR encoder pair, decoder pair, or both")
        super().__init__(models)
        self._device = torch.device(device)
        self.low_vram = low_vram

    def to(self, device):
        """Select the execution device, moving all models if offloading is disabled."""
        self._device = torch.device(device)
        if not self.low_vram:
            super().to(self.device)

    @contextmanager
    def _use_model(self, name):
        """Activate one model and offload after use or failure when enabled."""
        model = self.models[name]
        try:
            model.to(self.device)
            yield model
        finally:
            if self.low_vram:
                model.cpu()

    @torch.no_grad()
    def encode_voxels(self, coords, attributes):
        """Encode packed O-Voxel data into aligned, unnormalized 32-channel latents."""
        if "shape_encoder" not in self.models:
            raise ValueError("This codec was created without encoders")

        # 1. Unpack quantized voxel attributes and add the single-instance batch column.
        coords = coords.to(self.device, torch.int32)
        coords = torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=1)
        vertices = attributes["dual_vertices"].to(self.device).float() / 255.0
        bits = attributes["intersected"].to(self.device)
        intersected = torch.cat([bits % 2, bits // 2 % 2, bits // 4 % 2], dim=-1).bool()

        # 2. Form six-channel geometry and PBR inputs with their respective value ranges.
        geometry = SparseTensor(coords=coords, feats=torch.cat([vertices - 0.5, intersected.float() - 0.5], dim=1))
        material = geometry.replace(torch.cat([attributes[key].to(self.device) for key in PBR_LAYOUT], dim=-1) / 255.0 * 2 - 1)
        validate_slat(geometry, "geometry voxels")

        # 3. Encode each role separately while preserving a shared coordinate order.
        with self._use_model("shape_encoder") as encoder:
            shape = encoder(geometry)
        with self._use_model("pbr_encoder") as encoder:
            pbr = encoder(material)
        require_aligned(shape, pbr, "encoded PBR latent")
        return shape, pbr

    @torch.no_grad()
    def encode(self, canonical_scene, resolution, *, independent_parts=False):
        """Voxelize a canonical PBR scene and encode its Shape/PBR latent pair."""
        return self.encode_voxels(*voxelize_pbr_scene(
            canonical_scene, resolution, independent_parts=independent_parts))

    @torch.no_grad()
    def decode(self, shape, pbr, resolution, *, on_event=None):
        """Decode unnormalized latents into a surface and aligned voxel materials.

        resolution is the decoded voxel grid size, with 16 voxels per SLat cell.
        Shape predicts subdivision guides for PBR. Geometry is extracted and
        holes filled in the same unit cube used by the returned material volume.
        """
        if "shape_decoder" not in self.models:
            raise ValueError("This codec was created without decoders")
        from o_voxel.convert import flexible_dual_grid_to_mesh
        from ..representations import Mesh, MeshWithVoxel

        for name, slat in (("Shape", shape), ("PBR", pbr)):
            validate_slat(slat, name)
            if slat.shape[0] != 1 or slat.feats.shape[1] != 32:
                raise ValueError("Global decoding requires one asset with 32-channel VAE latents")
        require_aligned(shape, pbr, "PBR latent")
        if type(resolution) is not int or resolution <= 0 or resolution % 128:
            raise ValueError("Decode resolution must be a positive multiple of 128")

        # 1. Decode geometry and retain Shape's subdivision guides for PBR.
        with self._use_model("shape_decoder") as decoder:
            geometry, subs = decoder(shape.to(self.device))
            margin = decoder.voxel_margin
        if on_event is not None:
            on_event("shape_decoded", voxels=len(geometry.coords), guide_levels=len(subs))

        # 2. Convert geometry channels to dual-grid vertices, edge flags and a surface.
        vertices = (1 + 2 * margin) * F.sigmoid(geometry.feats[:, :3]) - margin
        intersected = geometry.feats[:, 3:6] > 0
        quad_lerp = F.softplus(geometry.feats[:, 6:7])
        mesh = Mesh(*flexible_dual_grid_to_mesh(
            geometry.coords[:, 1:], vertices, intersected, quad_lerp,
            aabb=AABB, grid_size=resolution, train=False,
        ))
        if on_event is not None:
            on_event("surface_extracted", vertices=len(mesh.vertices), faces=len(mesh.faces))

        # 3. Decode material attributes on Shape's support, then map them into [0, 1].
        with self._use_model("pbr_decoder") as decoder:
            material, _ = decoder(pbr.to(self.device), guide_subs=subs)
        if on_event is not None:
            on_event("material_decoded", voxels=len(material.coords))
        require_aligned(geometry, material, "decoded PBR voxels")
        material = material * 0.5 + 0.5

        # 4. Repair the surface and return it together with its volumetric PBR attributes.
        mesh.fill_holes()
        if not torch.isfinite(mesh.vertices).all() or (mesh.vertices.abs() > 1).any():
            raise RuntimeError("Decoded mesh contains invalid vertices after hole filling")
        return MeshWithVoxel(
            mesh.vertices, mesh.faces, origin=AABB[0], voxel_size=1 / resolution,
            coords=material.coords[:, 1:], attrs=material.feats,
            voxel_shape=torch.Size([*material.shape, *material.spatial_shape]), layout=PBR_LAYOUT,
        )

    def decode_parts(self, shape, pbr, resolution, *, local_resolution=1024,
                     patch_size=64, overlap=2, cell_limit=25000, on_event=None):
        """Yield CPU mesh parts in local space with explicit placement transforms.

        resolution describes the full asset; local_resolution describes each
        decoder cube. Keep overlapping geometry as separate parts. Shape and PBR
        use the same row selection and subdivision guides. The caller bakes each
        yielded part before placing it in the full scene; consume this iterator
        incrementally to bound GPU residency.
        """
        from ..utils.slat_decode import DecodedPart, plan_decode_patches

        if type(resolution) is not int or resolution <= 0 or resolution % 128:
            raise ValueError("Decode resolution must be a positive multiple of 128")
        require_aligned(shape, pbr, "PBR latent")

        # 1. Plan matching Shape/PBR patches in the decoder's local coordinate frame.
        patches = plan_decode_patches(shape, local_resolution=local_resolution,
                                      patch_size=patch_size, overlap=overlap, cell_limit=cell_limit)
        if on_event is not None:
            on_event("decode_plan", latent_cells=len(shape.coords), parts=len(patches),
                     selected_cells=sum(len(patch.indices) for patch in patches),
                     local_resolution=local_resolution, patch_size=patch_size,
                     overlap=overlap, cell_limit=cell_limit)

        # 2. Decode one patch at a time and move its result to CPU before yielding.
        for index, patch in enumerate(patches):
            if on_event is not None:
                on_event("part_extracted", part=index, latent_cells=len(patch.indices),
                         origin=list(patch.origin),
                         coordinate_min=shape.coords[patch.indices, 1:].amin(0).tolist(),
                         coordinate_max=shape.coords[patch.indices, 1:].amax(0).tolist())

            def part_event(stage, **details):
                """Attach the part index to a decoder progress event."""
                on_event(stage, part=index, **details)

            try:
                mesh = self.decode(patch.extract(shape), patch.extract(pbr), local_resolution,
                                   **({"on_event": part_event} if on_event is not None else {})).cpu()
            except Exception as exc:
                if on_event is not None:
                    on_event("part_decode_failed", part=index, error=f"{type(exc).__name__}: {exc}")
                raise
            if on_event is not None:
                on_event("part_decoded", part=index, vertices=len(mesh.vertices), faces=len(mesh.faces),
                         material_voxels=len(mesh.coords))
            # The placement transform restores this local cube after material baking.
            yield DecodedPart(mesh, patch.local_to_canonical(resolution, local_resolution))
            del mesh
