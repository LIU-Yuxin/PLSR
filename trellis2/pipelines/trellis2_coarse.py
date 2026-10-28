"""Original TRELLIS.2 weights and direct 1024 generation from a model repo."""

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import json

import numpy as np
from PIL import Image
import torch

from plsr.inputs import has_useful_alpha

from .base import Pipeline
from .samplers.flow_euler import FlowEulerGuidanceIntervalSampler
from .slat_codec import SLatCodec
from ..modules.sparse import SparseTensor


@dataclass(frozen=True)
class CoarseLatents:
    shape: SparseTensor
    pbr: SparseTensor
    condition_image: Image.Image
    resolution: int = 1024


def preprocess_coarse_image(image, remove_background=None):
    """Resize the long edge to 1024 and crop a square around the foreground.

    This crop is only the coarse generation condition. PLSR keeps its original
    full-resolution reference image, and warns before any resizing.
    """
    has_alpha = has_useful_alpha(image)
    image = image.convert("RGBA" if has_alpha else "RGB")
    scale = 1024 / max(image.size)
    size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    image = image.resize(size, Image.Resampling.LANCZOS)
    if not has_alpha:
        if remove_background is None:
            raise ValueError("RGB or opaque coarse inputs require a background-removal model")
        image = remove_background(image.convert("RGB"))
    if image.mode != "RGBA":
        raise ValueError("Background removal must return an RGBA image")
    foreground = np.argwhere(np.asarray(image)[..., 3] > .8 * 255)
    if len(foreground) == 0:
        raise ValueError("No foreground was found in the coarse input image")
    low, high = foreground.min(0), foreground.max(0)
    center_y, center_x = (low + high) / 2
    size = int((high - low).max())
    if size < 2:
        raise ValueError("The coarse foreground is too small to crop")
    image = image.crop((center_x - size // 2, center_y - size // 2,
                        center_x + size // 2, center_y + size // 2))
    pixels = np.asarray(image).astype(np.float32) / 255
    return Image.fromarray((pixels[..., :3] * pixels[..., 3:4] * 255).astype(np.uint8))


class Trellis2CoarsePipeline(Pipeline):
    """Run the needed upstream 1024 path, separately from PLSR's adapted weights.

    Sparse structure, Shape, PBR and image encoder are placed on the device by
    stage. The codec owns the two VAE decoders.
    """

    model_names = ("sparse_structure_flow_model", "sparse_structure_decoder",
                   "shape_slat_flow_model_1024", "tex_slat_flow_model_1024")

    def __init__(self, models, codec, image_encoder, config, *, remove_background=None, device="cuda", low_vram=True):
        if set(models) != set(self.model_names):
            raise ValueError("Coarse generation requires the original direct-1024 model set")
        super().__init__(models)
        self.codec = codec
        self.image_encoder = image_encoder
        self.remove_background = remove_background
        self.config = deepcopy(config)
        self.model_sources = {}
        self._device = torch.device(device)
        self.low_vram = low_vram
        self.configure_decode(25000)
        self.configure_export(4096, 0)
        self.samplers = {}
        for role in ("sparse_structure", "shape_slat", "tex_slat"):
            sampler = config[f"{role}_sampler"]
            if sampler["name"] != "FlowEulerGuidanceIntervalSampler":
                raise ValueError(f"Unsupported original coarse sampler: {sampler['name']}")
            self.samplers[role] = FlowEulerGuidanceIntervalSampler(**sampler["args"])

    @classmethod
    def from_pretrained(cls, path, *, image_encoder,
                        remove_background=None, device="cuda"):
        """Read upstream pipeline.json and each model's config/weights pair."""
        from plsr.pretrained import repo_file
        from .. import models as model_module
        from ..models.checkpoints import read_state_dict
        from ..models.slat_inference_flow import SLatInferenceFlowModel
        from ..models.sc_vaes.inference import create_slat_vae, SLatVaeDecoder

        config = json.loads(repo_file(path, "pipeline.json").read_text())["args"]
        requested = (*cls.model_names, "shape_slat_decoder", "tex_slat_decoder")
        loaded, sources = {}, {}
        for name in requested:
            prefix = config["models"][name]
            source = path
            if not prefix.startswith("ckpts/"):
                namespace, repo_name, prefix = prefix.split("/", 2)
                source = f"{namespace}/{repo_name}"
            model_config = json.loads(repo_file(source, f"{prefix}.json").read_text())
            weight_path = repo_file(source, f"{prefix}.safetensors")
            sources[name] = weight_path
            if name in ("shape_slat_flow_model_1024", "tex_slat_flow_model_1024"):
                if model_config["name"] != "SLatFlowModel":
                    raise ValueError("Coarse generation must load base SLatFlowModel weights")
                model = SLatInferenceFlowModel(**model_config["args"])
            elif name in ("shape_slat_decoder", "tex_slat_decoder"):
                model = create_slat_vae(model_config)
            else:
                model = getattr(model_module, model_config["name"])(**model_config["args"])
            state = read_state_dict(weight_path)
            if (name == "sparse_structure_flow_model"
                    and model_config["name"] == "SparseStructureFlowModel" and model.pe_mode == "rope"):
                # Upstream constructs this deterministic positional buffer from
                # the grid; its original checkpoint does not serialize it.
                # This is not a learned parameter or an adapter exemption.
                state.setdefault("rope_phases", model.rope_phases)
            model.load_state_dict(state, strict=True)
            del state
            # example.py applies this only to models that implement it; the
            # DiTs expose convert_to instead and retain their configured bf16.
            if hasattr(model, "convert_to_fp16"):
                model.convert_to_fp16()
            loaded[name] = model.eval().requires_grad_(False)
        codec = SLatCodec({"shape_decoder": SLatVaeDecoder(loaded.pop("shape_slat_decoder")),
                           "pbr_decoder": SLatVaeDecoder(loaded.pop("tex_slat_decoder"))}, device=device)
        pipeline = cls(loaded, codec, image_encoder, config, remove_background=remove_background, device=device)
        pipeline.model_sources = {name: str(prefix) for name, prefix in sources.items()}
        return pipeline

    @property
    def settings(self):
        """Record the executed coarse recipe separately from PLSR's settings."""
        models = {**self.models, **self.codec.models}
        return {
            "pipeline_type": "1024", "low_vram": self.low_vram,
            "model_sources": self.model_sources,
            "image_encoder": getattr(self.image_encoder, "source", None),
            "background_removal": getattr(self.remove_background, "source", None),
            "model_torso_dtypes": {name: str(model.dtype) for name, model in models.items() if hasattr(model, "dtype")},
            "samplers": {role: self.config[f"{role}_sampler"] for role in self.samplers},
            "remesh": True, "remesh_band": 1, "remesh_project": 0,
            "texture_size": self.texture_size, "texture_budget": self.texture_budget,
            "decimation_target": 1000000,
            "decode": {"patch_size": 48, "overlap": 4, "cell_limit": self.decode_cell_limit, "recenter_min_cells": 200},
        }

    def configure_decode(self, cell_limit):
        """Bound coarse VAE parts without changing its sampling recipe or grid."""
        if type(cell_limit) is not int or cell_limit <= 0:
            raise ValueError("cell_limit must be a positive integer")
        self.decode_cell_limit = cell_limit

    def configure_export(self, texture_size=4096, texture_budget=0):
        """Share SR's texture cap and object budget with generated coarse meshes."""
        from ..utils.texture_budget import validate_texture_budget
        validate_texture_budget(texture_size, texture_budget)
        self.texture_size, self.texture_budget = texture_size, texture_budget

    def to(self, device):
        self._device = torch.device(device)
        self.codec.to(device)
        if not self.low_vram:
            super().to(device)
            self.image_encoder.to(device)

    @contextmanager
    def _use(self, model):
        try:
            model.to(self.device)
            yield model
        finally:
            if self.low_vram:
                model.cpu()

    def get_cond(self, image, resolution):
        self.image_encoder.image_size = resolution
        with self._use(self.image_encoder) as encoder:
            condition = encoder([image])
        return {"cond": condition, "neg_cond": torch.zeros_like(condition)}

    def _sample(self, role, model, noise, condition, verbose, **kwargs):
        with self._use(model):
            return self.samplers[role].sample(
                model, noise, **condition, **self.config[f"{role}_sampler"]["params"],
                verbose=verbose, tqdm_desc=f"Coarse {role}", **kwargs,
            ).samples

    def _normalization(self, role, latent):
        value = self.config[f"{role}_slat_normalization"]
        return (torch.tensor(value["mean"])[None].to(latent.device),
                torch.tensor(value["std"])[None].to(latent.device))

    @torch.no_grad()
    def run(self, image, *, seed=42, verbose=True):
        image = preprocess_coarse_image(image, self.remove_background)
        torch.manual_seed(seed)
        cond_512 = self.get_cond(image, 512)
        cond_1024 = self.get_cond(image, 1024)
        flow = self.models["sparse_structure_flow_model"]
        noise = torch.randn(1, flow.in_channels, flow.resolution, flow.resolution, flow.resolution).to(self.device)
        structure = self._sample("sparse_structure", flow, noise, cond_512, verbose)
        with self._use(self.models["sparse_structure_decoder"]) as decoder:
            occupied = decoder(structure) > 0
        coords = torch.argwhere(occupied)[:, [0, 2, 3, 4]].int()
        if occupied.shape[2:] != (64, 64, 64) or len(coords) == 0:
            raise ValueError("Original 1024 coarse generation requires a nonempty 64-grid sparse structure")
        flow = self.models["shape_slat_flow_model_1024"]
        noise = SparseTensor(coords=coords, feats=torch.randn(len(coords), flow.in_channels).to(self.device))
        shape = self._sample("shape_slat", flow, noise, cond_1024, verbose)
        mean, std = self._normalization("shape", shape)
        shape = shape * std + mean
        # Preserve the upstream de-normalize/re-normalize sequence for PBR.
        shape_condition = (shape - mean) / std
        flow = self.models["tex_slat_flow_model_1024"]
        channels = flow.in_channels - shape_condition.feats.shape[1]
        noise = shape_condition.replace(torch.randn(len(coords), channels).to(self.device))
        pbr = self._sample("tex_slat", flow, noise, cond_1024, verbose, concat_cond=shape_condition)
        mean, std = self._normalization("tex", pbr)
        pbr = pbr * std + mean
        return CoarseLatents(shape, pbr, image)

    def decode_parts(self, latents, *, patches=None, on_event=None):
        """Decode all coarse parts sequentially, retaining global coordinates."""
        from ..utils.slat_decode import plan_coarse_decode_patches
        from ..utils.slat_patches import require_aligned

        if latents.resolution != 1024:
            raise ValueError("Original coarse decoding requires resolution 1024")
        require_aligned(latents.shape, latents.pbr, "coarse PBR latent")
        if patches is None:
            patches = plan_coarse_decode_patches(latents.shape, cell_limit=self.decode_cell_limit)
        if on_event is not None:
            on_event("decode_plan", latent_cells=len(latents.shape.coords), parts=len(patches),
                     selected_cells=sum(len(patch.indices) for patch in patches))
        for index, patch in enumerate(patches):
            if on_event is not None:
                on_event("part_extracted", part=index, latent_cells=len(patch.indices), origin=list(patch.origin))

            def part_event(stage, **details):
                on_event(stage, part=index, **details)

            try:
                mesh = self.codec.decode(patch.extract(latents.shape), patch.extract(latents.pbr), 1024,
                                         **({"on_event": part_event} if on_event is not None else {})).cpu()
            except Exception as exc:
                if on_event is not None:
                    on_event("part_decode_failed", part=index, error=f"{type(exc).__name__}: {exc}")
                raise
            if on_event is not None:
                on_event("part_decoded", part=index, vertices=len(mesh.vertices), faces=len(mesh.faces),
                         material_voxels=len(mesh.coords))
            yield mesh
            del mesh

    def export(self, latents, *, verbose=True, on_event=None):
        """Bake coarse mesh parts into a textured scene in GLB coordinates."""
        import trimesh
        from ..utils.mesh_export import bake_coarse_mesh
        from ..utils.slat_decode import plan_coarse_decode_patches
        from ..utils.texture_budget import plan_texture_budget

        scene = trimesh.Scene()
        patches = plan_coarse_decode_patches(latents.shape, cell_limit=self.decode_cell_limit)
        plan = plan_texture_budget([len(patch.indices) for patch in patches],
                                   texture_size=self.texture_size, texture_budget=self.texture_budget)
        plan.log(on_event)
        parts = self.decode_parts(latents, patches=patches, **({"on_event": on_event} if on_event is not None else {}))
        for index, mesh in enumerate(parts):
            texture_size = plan.sizes[index]
            if verbose:
                print(f"Baking coarse part {index + 1}", flush=True)
            if on_event is not None:
                on_event("part_bake_start", part=index, vertices=len(mesh.vertices),
                         faces=len(mesh.faces), material_voxels=len(mesh.coords),
                         target_faces=1000000, texture_size=texture_size)
            textured = bake_coarse_mesh(mesh, texture_size=texture_size, index=index, verbose=verbose,
                                        **({"on_event": on_event} if on_event is not None else {}))
            scene.add_geometry(textured, node_name=f"coarse_{index}", geom_name=f"coarse_{index}")
            if on_event is not None:
                on_event("part_merged", part=index, vertices=len(textured.vertices),
                         faces=len(textured.faces), scene_parts=len(scene.geometry))
        if not scene.geometry:
            raise ValueError("Coarse decoding produced no mesh parts")
        if on_event is not None:
            on_event("merge_complete", scene_parts=len(scene.geometry))
        return scene
