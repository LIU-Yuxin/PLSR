"""PLSR patch predictions on top of the reference Euler/CFG implementation."""

from dataclasses import dataclass

import torch

from .flow_euler import FlowEulerSampler
from .classifier_free_guidance_mixin import ClassifierFreeGuidanceSamplerMixin
from .guidance_interval_mixin import GuidanceIntervalSamplerMixin
from ...modules.sparse import SparseTensor
from ...utils.slat_patches import plan_slat_patches, pack_sparse, require_aligned, validate_slat


@dataclass(frozen=True)
class PatchCondition:
    """Encoded image conditions for one patch, each shaped [1, tokens, channels]."""
    cond: torch.Tensor
    neg_cond: torch.Tensor


class _PLSRModelSampler(FlowEulerSampler):
    """Adapt the Euler model-call convention to PLSR's LR-conditioned signature."""
    def _inference_model(self, model, x_t, t, cond=None, **kwargs):
        # PLSR's third positional model argument is lr_x, unlike the base model.
        """Scale time to model units and pass image conditioning by keyword."""
        t = torch.full((x_t.shape[0],), 1000 * t, device=x_t.device, dtype=torch.float32)
        return model(x_t, t, cond=cond, **kwargs)


class PLSRFlowEulerSampler(GuidanceIntervalSamplerMixin, ClassifierFreeGuidanceSamplerMixin, _PLSRModelSampler):
    """Predict on local patches, fuse x0, then apply one global Euler update.

    Keep the fixed patch order for view selection and overlap accumulation.
    Stage image tokens on CPU and transfer only the current patch batch.
    Accumulate half/bfloat16 predictions and weights in FP32 before casting
    the fused prediction back to the latent dtype.
    """

    @torch.no_grad()
    def sample(
        self, model, noise, condition_fn, *, lr_x, lr_t, guidance_strength,
        concat_cond=None, lr_concat_cond=None, steps=12, rescale_t=3.0,
        guidance_interval=(0.6, 1.0), guidance_rescale=0.0,
        patch_size=32, overlap=8, batch_size=12, verbose=True, tqdm_desc="PLSR sampling",
    ):
        """Sample normalized SLat with already prepared LR decoder features.

        condition_fn(patch, t) returns PatchCondition; it owns image selection and
        encoding. LR coordinates use the same grid units as the target
        (scale factor 1). lr_t is a per-instance tensor in
        [0, 1]; LR features must already have the corresponding noise applied.
        """
        from plsr.config import SamplingConfig

        # 1. Fix the patch plan and locate each patch's LR conditioning cells.
        SamplingConfig(guidance_strength, steps, rescale_t, guidance_interval)
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if not 0 <= guidance_rescale <= 1:
            raise ValueError("guidance_rescale must be in [0, 1]")
        patches = plan_slat_patches(noise, patch_size, overlap)
        validate_slat(lr_x, "lr_x")
        if lr_x.device != noise.device or lr_x.shape[0] != noise.shape[0]:
            raise ValueError("lr_x must share the input device and instance count")
        require_aligned(noise, concat_cond, "concat_cond")
        require_aligned(lr_x, lr_concat_cond, "lr_concat_cond")
        if not isinstance(lr_t, torch.Tensor) or lr_t.shape != (noise.shape[0],) or lr_t.device != noise.device:
            raise ValueError("lr_t must be a per-instance tensor on the input device")
        if not torch.isfinite(lr_t).all() or not ((lr_t >= 0) & (lr_t <= 1)).all():
            raise ValueError("lr_t must be finite and in [0, 1]")
        lr_indices = [patch.crop_indices(lr_x) for patch in patches]
        if any(len(indices) == 0 for indices in lr_indices):
            raise ValueError("Every target patch must contain LR conditioning cells")

        # 2. Scope local-neighbor caches to this pass; reuse the Euler/CFG time loop.
        clear_cache = getattr(model, "clear_local_attn_cache", None)
        if clear_cache is not None:
            clear_cache()
        try:
            return super().sample(
                model, noise, steps=steps, rescale_t=rescale_t, verbose=verbose, tqdm_desc=tqdm_desc,
                patches=patches, lr_indices=lr_indices, condition_fn=condition_fn, batch_size=batch_size,
                lr_x=lr_x, lr_t=lr_t, concat_cond=concat_cond, lr_concat_cond=lr_concat_cond,
                guidance_strength=guidance_strength, guidance_interval=guidance_interval,
                guidance_rescale=guidance_rescale,
            )
        finally:
            if clear_cache is not None:
                clear_cache()

    def _get_model_prediction(
        self, model, x_t, t, cond=None, *, patches, lr_indices, condition_fn, batch_size,
        lr_x, lr_t, concat_cond, lr_concat_cond, **guidance,
    ):
        """Fuse all patch predictions for one denoising timestep.

        Every patch reads the same global x_t. Convert predicted velocities to x0,
        average overlapping predictions there, then convert back for the Euler update.
        """
        # Boundary weights can be as small as 1e-6. Accumulating their products
        # in half precision would distort even a constant overlapping prediction.
        accumulation_dtype = torch.float32 if x_t.dtype in (torch.float16, torch.bfloat16) else x_t.dtype
        full_pred = torch.zeros_like(x_t.feats, dtype=accumulation_dtype)
        full_weights = torch.zeros_like(x_t.feats[:, :1], dtype=accumulation_dtype)

        # 1. Select/encode every patch view before model calls. This order fixes
        # the random draws used for view selection at the current timestep.
        # Retain only CPU copies, so prefetching does not keep every patch's
        # image tokens resident on the GPU before the first model call.
        all_conditions = []
        for patch in patches:
            condition = condition_fn(patch, t)
            if not isinstance(condition, PatchCondition):
                raise TypeError("condition_fn must return PatchCondition")
            for features in (condition.cond, condition.neg_cond):
                if not isinstance(features, torch.Tensor) or features.ndim != 3 or features.shape[0] != 1:
                    raise ValueError("Patch image conditions must have shape [1, tokens, channels]")
                if features.device != x_t.device:
                    raise ValueError("Patch image conditions must be on the input device")
            all_conditions.append(PatchCondition(condition.cond.detach().cpu(), condition.neg_cond.detach().cpu()))
            del condition, features

        # 2. Batch target and LR crops in patch-local coordinates.
        for start in range(0, len(patches), batch_size):
            batch = patches[start:start + batch_size]
            indices = lr_indices[start:start + batch_size]
            conditions = all_conditions[start:start + batch_size]
            inputs = [patch.extract(x_t) for patch in batch]
            batch_x = pack_sparse(inputs)
            kwargs = {
                "lr_x": pack_sparse([patch.crop(lr_x, selected) for patch, selected in zip(batch, indices)]),
                "lr_noise_level": torch.cat([lr_t[p.instance:p.instance + 1] for p in batch]) * 1000,
                "local_cache_key": "/".join(p.key for p in batch),
            }
            if concat_cond is not None:
                kwargs["concat_cond"] = pack_sparse([patch.extract(concat_cond) for patch in batch])
            if lr_concat_cond is not None:
                kwargs["lr_concat_cond"] = pack_sparse([patch.crop(lr_concat_cond, selected) for patch, selected in zip(batch, indices)])

            # 3. Predict patch velocities, applying CFG within its configured interval.
            prediction = super()._inference_model(
                model, batch_x, t, cond=torch.cat([c.cond for c in conditions]).to(x_t.device),
                neg_cond=torch.cat([c.neg_cond for c in conditions]).to(x_t.device), **guidance, **kwargs,
            )
            if not isinstance(prediction, SparseTensor) or prediction.feats.shape != batch_x.feats.shape:
                raise ValueError("Model prediction must preserve the patch feature shape")
            if not torch.equal(prediction.coords, batch_x.coords):
                raise ValueError("Model prediction changed patch coordinates or row order")
            if not torch.isfinite(prediction.feats).all():
                raise ValueError("Model returned nonfinite patch predictions")

            # 4. Convert each patch velocity to x0 and accumulate on global rows.
            for index, (patch, patch_x) in enumerate(zip(batch, inputs)):
                pred_x0, _ = self._v_to_xstart_eps(patch_x, t, prediction[index:index + 1])
                # Preserve patch order while using the stable accumulation dtype.
                full_pred[patch.indices] += pred_x0.feats * patch.weights
                full_weights[patch.indices] += patch.weights
        if not (full_weights > 0).all():
            raise ValueError("Patch plan left target cells without a prediction")

        # 5. Normalize overlap weights; Euler advances the full state only after fusion.
        pred_x0 = x_t.replace((full_pred / full_weights.clamp(min=1e-6)).to(x_t.dtype))
        pred_v = self._xstart_to_pred(x_t, t, pred_x0)
        pred_eps = self._xstart_to_eps(x_t, t, pred_x0)
        return pred_x0, pred_eps, pred_v
