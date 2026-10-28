"""One PLSR resolution doubling, from VAE latents to refined VAE latents."""

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from time import perf_counter

import torch

from plsr.config import SamplingConfig
from .base import Pipeline
from .patch_image_condition import PatchImageConditioner
from .samplers.plsr_flow_euler import PLSRFlowEulerSampler
from ..modules.sparse import SparseTensor
from ..utils.slat_patches import require_aligned, validate_slat


@dataclass(frozen=True)
class RefinementResult:
    """Unnormalized VAE latents on the same grid, ready to save or decode."""
    shape: SparseTensor
    pbr: SparseTensor
    resolution: int


class PLSRRefinementStep(Pipeline):
    """Refine Shape and PBR VAE latents once at twice their input resolution.

    Decode LR features, sample Shape, then sample PBR conditioned on Shape.
    Reuse the supplied models, normalization and Euler samplers across calls;
    place each active model on the requested device and offload when enabled.

    Models are named shape_decoder, pbr_decoder, shape_denoiser and pbr_denoiser.
    The decoders are SLatFeatureDecoder instances. encode_image supplies image
    conditions for a [1,3,1024,1024] crop and owns image-encoder placement.
    Images and cameras must already be prepared together, with extrinsics in
    canonical mesh coordinates. Return an unnormalized VAE latent pair;
    mesh conversion, progressive scheduling and
    file output belong to PLSRPipeline.
    """

    def __init__(
        self, models, encode_image, shape_normalization, pbr_normalization, *,
        shape_sigma_min=1e-5, pbr_sigma_min=1e-5, device="cuda", low_vram=True,
    ):
        required = {"shape_decoder", "pbr_decoder", "shape_denoiser", "pbr_denoiser"}
        if set(models) != required or any(model is None for model in models.values()):
            raise ValueError(f"models must provide exactly {sorted(required)}")
        super().__init__(models)
        self._device = torch.device(device)
        self.low_vram = low_vram
        self.encode_image = encode_image
        self.normalization = {"shape": shape_normalization, "pbr": pbr_normalization}
        self.samplers = {
            "shape": PLSRFlowEulerSampler(shape_sigma_min),
            "pbr": PLSRFlowEulerSampler(pbr_sigma_min),
        }
        for role, normalization in self.normalization.items():
            mean = torch.tensor(normalization["mean"], dtype=torch.float32)
            std = torch.tensor(normalization["std"], dtype=torch.float32)
            if mean.shape != (32,) or std.shape != (32,):
                raise ValueError(f"{role} normalization must contain 32 channels")
            if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or not (std > 0).all():
                raise ValueError(f"{role} normalization requires finite means and positive finite stds")

    def to(self, device):
        """Select the execution device; move all models when offload is disabled."""
        self._device = torch.device(device)
        if not self.low_vram:
            super().to(self.device)

    @contextmanager
    def _use_model(self, name):
        """Place one active component on the target device; offload when enabled."""
        model = self.models[name]
        try:
            model.to(self.device)
            yield model
        finally:
            if self.low_vram:
                model.cpu()

    def _normalize(self, slat, role, inverse=False):
        # Match upstream's float32 constants, even if the features are half.
        """Convert between VAE latent values and the denoiser's normalized values."""
        parameters = self.normalization[role]
        mean = torch.tensor(parameters["mean"], dtype=torch.float32, device=slat.device)[None]
        std = torch.tensor(parameters["std"], dtype=torch.float32, device=slat.device)[None]
        return slat * std + mean if inverse else (slat - mean) / std

    @torch.no_grad()
    def prepare_lr_features(self, shape_slat, pbr_slat):
        """Decode unnormalized 32-channel latents to aligned 512-channel features."""
        for name, slat in (("shape_slat", shape_slat), ("pbr_slat", pbr_slat)):
            validate_slat(slat, name)
            if slat.feats.shape[1] != 32 or not torch.isfinite(slat.feats).all():
                raise ValueError(f"{name} must contain finite unnormalized 32-channel VAE latents")
        require_aligned(shape_slat, pbr_slat, "pbr_slat")
        shape_slat, pbr_slat = shape_slat.to(self.device), pbr_slat.to(self.device)
        # Shape predicts subdivision guides; PBR reuses them to keep both grids aligned.
        with self._use_model("shape_decoder") as decoder:
            shape, guide_subs = decoder(shape_slat)
        with self._use_model("pbr_decoder") as decoder:
            pbr, _ = decoder(pbr_slat, guide_subs=guide_subs)
        for role, features in (("shape", shape), ("pbr", pbr)):
            validate_slat(features, f"LR {role} features")
            if features.feats.shape[1] != 512 or not torch.isfinite(features.feats).all():
                raise ValueError(f"LR {role} decoder must return finite 512-channel features")
        require_aligned(shape, pbr, "LR PBR features")
        return shape, pbr

    @staticmethod
    def _diffuse_lr(features, t, noise, sigma_min):
        """Apply the flow-matching noise schedule to unnormalized LR features."""
        # Even at t=0, sigma_min leaves a small noise residual. Decoder features
        # are not SLat-normalized; sparse arithmetic broadcasts t per instance.
        t = t[:, None]
        return (1 - t) * features + (sigma_min + (1 - sigma_min) * t) * noise

    def _sample(self, role, lr_features, lr_t, conditioner, parameters, batch_size, *,
                patch_size=32, overlap=8, shape=None, lr_shape=None, verbose=True):
        """Draw target noise, sample one role and return unnormalized VAE latents."""
        model = self.models[f"{role}_denoiser"]
        # PBR receives normalized refined Shape as additional input channels.
        concat_cond = self._normalize(shape, "shape") if shape is not None else None
        channels = model.in_channels - (0 if concat_cond is None else concat_cond.feats.shape[1])
        # Draw target noise on CPU before transfer. LR noise is drawn on the
        # feature device in run(); both draw order and device affect seeded runs.
        noise = SparseTensor(
            feats=torch.randn(len(lr_features.coords), channels).to(self.device),
            coords=lr_features.coords.clone(),
        )
        with self._use_model(f"{role}_denoiser") as model, conditioner:
            slat = self.samplers[role].sample(
                model, noise, conditioner, lr_x=lr_features, lr_t=lr_t,
                concat_cond=concat_cond, lr_concat_cond=lr_shape,
                batch_size=batch_size, patch_size=patch_size, overlap=overlap,
                **asdict(parameters), verbose=verbose,
                tqdm_desc=f"Sampling {role} SLat",
            ).samples
        return self._normalize(slat, role, inverse=True)

    @torch.no_grad()
    def run(
        self, shape_slat, pbr_slat, images, intrinsics, extrinsics, *, resolution,
        shape_sampler_params, pbr_sampler_params, seed=42,
        shape_batch_size=12, pbr_batch_size=8, sampling_patch_size=32,
        sampling_patch_overlap=8, verbose=True, on_timing=None,
    ):
        """Refine once at twice the input resolution; output grid is resolution/16.

        Each image tensor uses [B,V,3,H,W], with normalized intrinsics [B,V,3,3]
        and canonical-world-to-camera extrinsics [B,V,4,4]. The output resolution
        is the mesh voxel resolution, not the SLat grid size. LR noise level is
        fixed at 0. Both Shape and PBR sampler configurations must be supplied.
        """

        # 1. Check that inputs belong to the preceding resolution's latent grid.
        if type(resolution) is not int or resolution <= 0 or resolution % 256:
            raise ValueError("A doubling's target resolution must be a positive multiple of 256")
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        if not all(isinstance(p, SamplingConfig) for p in (shape_sampler_params, pbr_sampler_params)):
            raise TypeError("Sampler parameters must be SamplingConfig instances")
        for size in (shape_batch_size, pbr_batch_size):
            if type(size) is not int or size <= 0:
                raise ValueError("Sampler batch sizes must be positive integers")
        if type(sampling_patch_size) is not int or sampling_patch_size < 2:
            raise ValueError("sampling_patch_size must be an integer >= 2")
        if (type(sampling_patch_overlap) is not int
                or not 0 <= sampling_patch_overlap < sampling_patch_size):
            raise ValueError("sampling_patch_overlap must be in [0, sampling_patch_size)")
        for name, slat in (("shape_slat", shape_slat), ("pbr_slat", pbr_slat)):
            validate_slat(slat, name)
            if not ((slat.coords[:, 1:] >= 0) & (slat.coords[:, 1:] < resolution // 32)).all():
                raise ValueError(f"{name} coordinates must belong to the preceding resolution's SLat grid")
        @contextmanager
        def timed(stage):
            """Report stage duration, synchronizing CUDA when a callback is supplied."""
            if on_timing is not None and torch.cuda.is_initialized():
                torch.cuda.synchronize()
            start = perf_counter()
            try:
                yield
            finally:
                if on_timing is not None:
                    if torch.cuda.is_initialized():
                        try:
                            torch.cuda.synchronize()
                        except RuntimeError:
                            pass
                    on_timing(stage, perf_counter() - start)

        # 2. Decode LR features onto the target grid, using shared subdivision guides.
        with timed("lr_feature_decode"):
            lr_shape, lr_pbr = self.prepare_lr_features(shape_slat, pbr_slat)

        # 3. Prepare local image conditioning and apply the LR noise schedule.
        with timed("conditioning_setup"):
            images, intrinsics, extrinsics = (tensor.to(self.device) for tensor in (images, intrinsics, extrinsics))
            conditioner = PatchImageConditioner(
                lr_shape, images, intrinsics, extrinsics, resolution // 16, self.encode_image,
            )
            torch.manual_seed(seed)
            lr_t = torch.zeros(lr_shape.shape[0], device=self.device)
            shape_noise = lr_shape.replace(torch.randn_like(lr_shape.feats))
            pbr_noise = lr_pbr.replace(torch.randn_like(lr_pbr.feats))
            lr_shape = self._diffuse_lr(lr_shape, lr_t, shape_noise, self.samplers["shape"].sigma_min)
            lr_pbr = self._diffuse_lr(lr_pbr, lr_t, pbr_noise, self.samplers["pbr"].sigma_min)

        # 4. Generate Shape latents first.
        with timed("shape_sampling"):
            shape = self._sample("shape", lr_shape, lr_t, conditioner, shape_sampler_params,
                                 shape_batch_size, patch_size=sampling_patch_size,
                                 overlap=sampling_patch_overlap, verbose=verbose)

        # 5. Generate PBR latents conditioned on refined Shape and LR Shape features.
        with timed("pbr_sampling"):
            pbr = self._sample("pbr", lr_pbr, lr_t, conditioner, pbr_sampler_params,
                               pbr_batch_size, patch_size=sampling_patch_size,
                               overlap=sampling_patch_overlap, shape=shape,
                               lr_shape=lr_shape, verbose=verbose)
        return RefinementResult(shape, pbr, resolution)
