"""Bound MapAnything temporaries without removing views or global attention."""

import torch
from torch import nn
from types import MethodType
from dataclasses import fields, is_dataclass, replace


def _move_camera_tensors(value, device):
    """Move structured model features without changing their dtype or values."""
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if is_dataclass(value):
        return replace(value, **{field.name: _move_camera_tensors(getattr(value, field.name), device)
                                 for field in fields(value)})
    if isinstance(value, list):
        return [_move_camera_tensors(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_camera_tensors(item, device) for item in value)
    if isinstance(value, dict):
        return {key: _move_camera_tensors(item, device) for key, item in value.items()}
    return value


def _offload_multiview_features(module, args, output):
    if torch.is_grad_enabled() or module.training:
        raise RuntimeError("Camera feature offload is inference-only")
    print("Camera intermediate features offloaded to CPU", flush=True)
    return _move_camera_tensors(output, "cpu")


def _concatenate_camera_outputs(outputs, device):
    # The upstream downstream_head concatenates these same adaptor fields.
    # Keep completed batches on CPU until the complete output is needed.
    first = outputs[0]
    return first.__class__(**{
        key: torch.cat([getattr(output, key) for output in outputs], dim=0).to(device)
        for key in first.__dict__
    })


def _offloaded_downstream_head(model, dense_head_inputs, scale_head_inputs, img_shape,
                               memory_efficient_inference=False, minibatch_size=None):
    """Run the original heads per batch, staging completed outputs on CPU."""
    from uniception.models.prediction_heads.base import AdaptorInput, PredictionHeadInput, PredictionHeadTokenInput

    if torch.is_grad_enabled() or model.training:
        raise RuntimeError("Camera prediction offload is inference-only")
    if not memory_efficient_inference:
        raise ValueError("Camera feature offload requires memory-efficient inference")
    if model.pred_head_type not in ("linear", "dpt", "dpt+pose"):
        raise ValueError(f"Unsupported camera prediction head: {model.pred_head_type}")
    device = model.device
    count = (dense_head_inputs if model.pred_head_type == "linear" else dense_head_inputs[0]).shape[0]
    size = minibatch_size if minibatch_size is not None else model._compute_adaptive_minibatch_size()
    dense_outputs, pose_outputs = [], []
    for start in range(0, count, size):
        stop = min(start + size, count)
        batch = (dense_head_inputs[start:stop] if model.pred_head_type == "linear"
                 else [value[start:stop] for value in dense_head_inputs])
        batch = _move_camera_tensors(batch, device)
        dense = model.downstream_dense_head(batch, img_shape)
        dense_outputs.append(_move_camera_tensors(dense, "cpu"))
        if model.pred_head_type == "dpt+pose":
            pose = model.pose_head(PredictionHeadInput(last_feature=batch[-1]))
            pose = model.pose_adaptor(AdaptorInput(adaptor_feature=pose.decoded_channels, output_shape_hw=img_shape))
            pose_outputs.append(_move_camera_tensors(pose, "cpu"))
            del pose
        del batch, dense
        if stop % 10 == 0 or stop == count:
            print(f"Predicted camera views: {stop}/{count}", flush=True)
    dense = _concatenate_camera_outputs(dense_outputs, device)
    pose = _concatenate_camera_outputs(pose_outputs, device) if pose_outputs else None
    scale = model.scale_head(PredictionHeadTokenInput(last_feature=scale_head_inputs.to(device)))
    scale = model.scale_adaptor(AdaptorInput(adaptor_feature=scale.decoded_channels, output_shape_hw=img_shape))
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return dense, pose, scale.value.squeeze(-1)


def _encode_view_batches(model, views):
    """Use the original encoder per view group, then release its residency."""
    if torch.is_grad_enabled() or model.training:
        raise RuntimeError("Camera encoder batching and offload are inference-only")
    size = model._plsr_encoder_view_batch_size or len(views)
    features, registers = [], []
    has_registers = None
    try:
        for start in range(0, len(views), size):
            part_features, part_registers = model._plsr_original_encode_n_views(views[start:start + size])
            features.extend(part_features)
            if has_registers is None:
                has_registers = part_registers is not None
            if (part_registers is not None) != has_registers:
                raise ValueError("Camera encoder register layout changed between batches")
            if part_registers is not None:
                registers.extend(part_registers)
            completed = min(start + size, len(views))
            if completed % 10 == 0 or completed == len(views):
                print(f"Encoded camera views: {completed}/{len(views)}", flush=True)
        return tuple(features), tuple(registers) if has_registers else None
    finally:
        if model._plsr_offload_encoder:
            # Features remain on CUDA. The backbone is no longer used by the
            # geometric encoders, information sharing or prediction heads.
            model.encoder.cpu()
            torch.clear_autocast_cache()
            torch.cuda.empty_cache()
            print("Camera image backbone offloaded to CPU", flush=True)


class _ViewBatchedBlock(nn.Module):
    """Batch independent frame-attention blocks along their leading axis."""

    def __init__(self, module, batch_size):
        super().__init__()
        self.module, self.batch_size = module, batch_size

    def forward(self, x, xpos=None):
        if x.shape[0] <= self.batch_size:
            return self.module(x, xpos)
        if torch.is_grad_enabled() or self.training:
            raise RuntimeError("Camera view batching is inference-only")
        output = None
        for start in range(0, x.shape[0], self.batch_size):
            stop = start + self.batch_size
            # The upstream model uses a list of None values when positional
            # encoding is disabled. That placeholder is ignored by the block.
            positions = xpos
            if isinstance(xpos, torch.Tensor) and xpos.shape[0] != 1:
                positions = xpos[start:stop]
            part = self.module(x[start:stop], positions)
            if output is None:
                output = part.new_empty((x.shape[0], *part.shape[1:]))
            output[start:stop].copy_(part)
        return output


class _TokenBatchedMlp(nn.Module):
    """Bound pointwise MLP expansion while keeping every output token."""

    def __init__(self, module, batch_size):
        super().__init__()
        self.module, self.batch_size = module, batch_size

    def forward(self, x):
        flat = x.reshape(-1, x.shape[-1])
        if len(flat) <= self.batch_size:
            return self.module(x)
        if torch.is_grad_enabled() or self.training:
            raise RuntimeError("Camera token batching is inference-only")
        output = None
        for start in range(0, len(flat), self.batch_size):
            stop = start + self.batch_size
            part = self.module(flat[start:stop])
            if output is None:
                output = part.new_empty((len(flat), part.shape[-1]))
            output[start:stop].copy_(part)
        return output.reshape(*x.shape[:-1], output.shape[-1])


def configure_mapanything_batches(model, view_batch_size=None, token_batch_size=None, offload_encoder=False,
                                  offload_features=False):
    """Wrap only known independent operations after checkpoint loading."""
    from uniception.models.utils.transformer_blocks import Mlp, SwiGLUFFN

    for size in (view_batch_size, token_batch_size):
        if size is not None and (type(size) is not int or size <= 0):
            raise ValueError("Camera batch sizes must be positive integers or None")
    if type(offload_encoder) is not bool:
        raise ValueError("Camera encoder offload must be a boolean")
    if type(offload_features) is not bool:
        raise ValueError("Camera feature offload must be a boolean")
    active = view_batch_size is not None or token_batch_size is not None or offload_encoder or offload_features
    if active:
        try:
            blocks = model.info_sharing.self_attention_blocks
            for block in blocks:
                block = block.module if isinstance(block, _ViewBatchedBlock) else block
                mlp = block.mlp.module if isinstance(block.mlp, _TokenBatchedMlp) else block.mlp
                if token_batch_size is not None and not isinstance(mlp, (Mlp, SwiGLUFFN)):
                    raise TypeError(f"Unsupported camera MLP: {type(mlp).__name__}")
            if view_batch_size is not None or offload_encoder:
                if not callable(model._encode_n_views) or not isinstance(model.encoder, nn.Module):
                    raise TypeError("Unsupported camera image encoder")
            if offload_features:
                if model.pred_head_type not in ("linear", "dpt", "dpt+pose"):
                    raise TypeError(f"Unsupported camera prediction head: {model.pred_head_type}")
                for name in ("downstream_head", "downstream_dense_head", "scale_head", "scale_adaptor"):
                    if not callable(getattr(model, name)):
                        raise TypeError(f"Unsupported camera head component: {name}")
        except (AttributeError, TypeError) as exc:
            raise RuntimeError(
                "Camera batching requires the MapAnything/UniCeption versions selected by setup.sh or setup.ps1. "
                "Reinstall the camera dependency group before enabling these options."
            ) from exc
    if hasattr(model, "_plsr_feature_offload_hook"):
        model._plsr_feature_offload_hook.remove()
        del model._plsr_feature_offload_hook
    if offload_features:
        if not hasattr(model, "_plsr_original_downstream_head"):
            model._plsr_original_downstream_head = model.downstream_head
        model.downstream_head = MethodType(_offloaded_downstream_head, model)
        model._plsr_feature_offload_hook = model.info_sharing.register_forward_hook(_offload_multiview_features)
    elif hasattr(model, "_plsr_original_downstream_head"):
        model.downstream_head = model._plsr_original_downstream_head
    if view_batch_size is not None or offload_encoder:
        if not hasattr(model, "_plsr_original_encode_n_views"):
            model._plsr_original_encode_n_views = model._encode_n_views
        model._plsr_encoder_view_batch_size = view_batch_size
        model._plsr_offload_encoder = offload_encoder
        model._encode_n_views = MethodType(_encode_view_batches, model)
    elif hasattr(model, "_plsr_original_encode_n_views"):
        model._encode_n_views = model._plsr_original_encode_n_views
    blocks = model.info_sharing.self_attention_blocks
    for index in range(len(blocks)):
        block = blocks[index]
        if isinstance(block, _ViewBatchedBlock):
            block = block.module
        mlp = block.mlp
        if isinstance(mlp, _TokenBatchedMlp):
            mlp = mlp.module
        if token_batch_size is not None:
            if not isinstance(mlp, (Mlp, SwiGLUFFN)):
                raise ValueError(f"Unsupported camera pointwise MLP: {type(mlp).__name__}")
            block.mlp = _TokenBatchedMlp(mlp, token_batch_size).eval()
        else:
            block.mlp = mlp
        # UniCeption alternates global (even) and independent frame (odd)
        # blocks. Never split the token context of a global-attention block.
        blocks[index] = (_ViewBatchedBlock(block, view_batch_size).eval()
                         if view_batch_size is not None and index % 2 else block)
