"""Load TRELLIS.2 base weights followed by the published PLSR adapter."""

from copy import deepcopy
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model

from ..modules.sparse.lora import enable_sparse_lora


def read_state_dict(path) -> dict[str, torch.Tensor]:
    """Read flat tensor checkpoints with correct suffixes; never guess or retry formats."""
    path = Path(path)
    if path.suffix == ".pt":
        state = torch.load(path, map_location="cpu", weights_only=True)
    elif path.suffix == ".safetensors":
        from safetensors.torch import load_file
        state = load_file(str(path), device="cpu")
    else:
        raise ValueError(f"Checkpoint must use .pt or .safetensors: {path}")
    if not isinstance(state, dict) or not state or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()
    ):
        raise ValueError(f"Expected a nonempty flat tensor state_dict: {path}")
    return state


def _lora_config(model, config):
    # Training used substring exclusions, then enumerated Linear/Conv modules.
    # Keep that selection exactly; PEFT's own exclusion matching is different.
    config = deepcopy(config)
    if config.get("target_modules") != "all-linear" or not isinstance(config.get("exclude_modules"), list):
        raise ValueError("PLSR requires all-linear targets and an explicit substring exclusion list")
    exclusions = config["exclude_modules"]
    targets, excluded = [], []
    for name, module in model.named_modules():
        if not name:
            continue
        if any(part in name for part in exclusions):
            excluded.append(name)
        elif isinstance(module, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
            targets.append(name)
    if not targets:
        raise ValueError("LoRA selection matched no modules")
    config.update(target_modules=targets, exclude_modules=excluded)
    return LoraConfig(**config), len(targets)


def load_pretrained_denoiser(config, base_path, adapter_path):
    """Load the upstream base, then the trained LoRA and PLSR parameters.

    Keep the training-time target selection, SparseTensor adaptation, model
    precision and unmerged LoRA execution. Fixed gates/norms retain their
    constructor defaults; no separate checkpoint identity manifest is needed.
    """
    from .plsr_flow import PLSRFlowModel

    model = PLSRFlowModel.from_config(config["args"])
    model.load_state_dict(read_state_dict(base_path), strict=False)
    peft_config, _ = _lora_config(model, config["peft_cfg"])
    model = get_peft_model(model, peft_config)
    enable_sparse_lora(model)
    model.load_state_dict(read_state_dict(adapter_path), strict=False)
    return model.eval().requires_grad_(False)
