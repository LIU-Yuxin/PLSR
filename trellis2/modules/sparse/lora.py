"""Sparse input support for PEFT's LoRA branches, without patching PyTorch."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft.tuners.lora import Linear as LoraLinear

from .basic import VarLenTensor
from .linear import SparseLinear


class _FeatureLinear(nn.Linear):
    """Apply an ordinary Linear to features while retaining their sparse layout."""

    def forward(self, x):
        if isinstance(x, VarLenTensor):
            return x.replace(F.linear(x.feats, self.weight, self.bias))
        return F.linear(x, self.weight, self.bias)

    @classmethod
    def from_linear(cls, layer):
        # Reuse the Parameters: no initialization, dtype conversion or key change.
        with torch.device("meta"):
            result = cls(layer.in_features, layer.out_features, bias=layer.bias is not None)
        result.weight = layer.weight
        result.bias = layer.bias
        result.train(layer.training)
        return result


def enable_sparse_lora(model):
    """Adapt only A/B branches of LoRA layers whose base operation is SparseLinear."""
    for layer in model.modules():
        if not isinstance(layer, LoraLinear) or not isinstance(layer.get_base_layer(), SparseLinear):
            continue
        if layer.lora_variant or any(not isinstance(dropout, nn.Identity) for dropout in layer.lora_dropout.values()):
            raise ValueError("Sparse PLSR LoRA requires vanilla adapters with zero dropout")
        for branches in (layer.lora_A, layer.lora_B):
            for name, branch in branches.items():
                if not isinstance(branch, _FeatureLinear):
                    branches[name] = _FeatureLinear.from_linear(branch)
