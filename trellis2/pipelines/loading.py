"""Assemble the TRELLIS.2 base components and the published PLSR adapters."""

import json

import torch

from plsr.pretrained import repo_file
from ..models.checkpoints import load_pretrained_denoiser, read_state_dict
from ..models.sc_vaes.inference import create_slat_vae, SLatVaeEncoder, SLatVaeDecoder
from ..models.sc_vaes.slat_feature_decoder import SLatFeatureDecoder
from ..modules.dinov3_image_encoder import DinoV3ImageEncoder
from .plsr_pipeline import PLSRPipeline
from .plsr_refinement import PLSRRefinementStep
from .slat_codec import SLatCodec


VAE_NAMES = {
    "shape_encoder": "shape_enc_next_dc_f16c32_fp16",
    "pbr_encoder": "tex_enc_next_dc_f16c32_fp16",
    "shape_decoder": "shape_dec_next_dc_f16c32_fp16",
    "pbr_decoder": "tex_dec_next_dc_f16c32_fp16",
}


def load_inference_pipeline(model_sources, *, pipeline_class=PLSRPipeline):
    """Preserve model precision, decoder sharing and stage-wise CPU offload."""
    plsr_repo_id, trellis2_repo_id = model_sources["plsr"], model_sources["trellis2"]

    # 1. Read component paths and inference defaults from the PLSR repository.
    config = json.loads(repo_file(plsr_repo_id, "pipeline.json").read_text())

    # 2. Load the TRELLIS.2 Shape/PBR VAE encoders and decoders.
    vaes = {}
    for role, name in VAE_NAMES.items():
        model_config = json.loads(repo_file(trellis2_repo_id, f"ckpts/{name}.json").read_text())
        model = create_slat_vae(model_config)
        model.load_state_dict(read_state_dict(repo_file(trellis2_repo_id, f"ckpts/{name}.safetensors")))
        wrapper = SLatVaeEncoder if role.endswith("encoder") else SLatVaeDecoder
        vaes[role] = wrapper(model.eval().requires_grad_(False))

    # 3. Construct each PLSR denoiser, then load its base and adapter weights.
    denoisers, configs = {}, {}
    for role in ("shape", "pbr"):
        component = config["models"][role]
        configs[role] = json.loads(repo_file(plsr_repo_id, component["config"]).read_text())
        denoisers[f"{role}_denoiser"] = load_pretrained_denoiser(
            configs[role], repo_file(trellis2_repo_id, component["base"]),
            repo_file(plsr_repo_id, component["adapter"]),
        )

    # 4. Share one DINOv3 encoder across all patch-image conditioning calls.
    image_encoder = DinoV3ImageEncoder.from_pretrained(model_sources["dino"], image_size=1024)

    def encode_image(image):
        """Encode one crop and release the encoder's GPU residency afterwards."""
        try:
            image_encoder.cuda()
            condition = image_encoder(image)
            return {"cond": condition, "neg_cond": torch.zeros_like(condition)}
        finally:
            image_encoder.cpu()

    # 5. Compose the codec and reusable refinement step. Feature decoders share
    # modules with the loaded VAEs rather than loading another copy of their weights.
    codec = SLatCodec(vaes)
    refinement = PLSRRefinementStep(
        {**denoisers, "shape_decoder": SLatFeatureDecoder(vaes["shape_decoder"]),
         "pbr_decoder": SLatFeatureDecoder(vaes["pbr_decoder"])}, encode_image,
        configs["shape"]["normalization"], configs["pbr"]["normalization"],
        shape_sigma_min=configs["shape"]["sigma_min"], pbr_sigma_min=configs["pbr"]["sigma_min"],
    )
    pipeline = pipeline_class(codec, refinement, inference_defaults=config["inference_defaults"])
    # Keep repository sources available for auxiliary stages loaded by later requests.
    pipeline.model_sources = dict(model_sources)
    pipeline._image_encoder = image_encoder
    pipeline.remove_background = None
    pipeline.pretrained_settings = {
        "inference_defaults": pipeline.inference_defaults,
        "loaded_stages": {"coarse": False, "camera_estimation": False,
                          "background_removal": False, "refinement": True},
        "low_vram": True, "reference_long_edge": 4096, "image_condition_size": 1024,
        "vae_torso_dtypes": {role: str(model.dtype) for role, model in vaes.items()},
        "denoiser_torso_dtypes": {role: str(model.dtype) for role, model in denoisers.items()},
    }
    return pipeline


def load_optional_stages(pipeline, config, inputs):
    """Reuse auxiliary models across requests; skip stages the input supplies."""
    sources = pipeline.model_sources
    # Use image alpha when available; otherwise obtain foreground masks with RMBG.
    if inputs.needs_background_removal and pipeline.remove_background is None:
        from .background_removal import BiRefNetBackgroundRemover

        pipeline.remove_background = BiRefNetBackgroundRemover.from_pretrained(sources["rmbg"])
    # Fully specified cameras bypass estimation and anchor rendering.
    if config.needs_camera_estimation and pipeline.camera_estimator is None:
        from mapanything.models import MapAnything
        from .camera_estimation import MapAnythingCameraEstimator

        model = MapAnything.from_pretrained(sources["mapanything"]).eval()
        pipeline.camera_estimator = MapAnythingCameraEstimator(model)
    # Single-image generation reuses the loaded image encoder for the coarse stage.
    if (config.mesh is None and pipeline.coarse_generator is None
            and not (config.output_dir / "coarse.glb").is_file()):
        from .trellis2_coarse import Trellis2CoarsePipeline

        pipeline.coarse_generator = Trellis2CoarsePipeline.from_pretrained(
            pipeline.model_sources["trellis2"], image_encoder=pipeline._image_encoder,
            remove_background=pipeline.remove_background,
        )
    if pipeline.coarse_generator is not None:
        pipeline.coarse_generator.remove_background = pipeline.remove_background
    pipeline.pretrained_settings["loaded_stages"].update(
        coarse=pipeline.coarse_generator is not None,
        camera_estimation=pipeline.camera_estimator is not None,
        background_removal=pipeline.remove_background is not None,
    )
