"""Sparse attention exports, loaded only when their implementation is needed."""

import importlib
from pathlib import Path

__attributes = {
    "sparse_scaled_dot_product_attention": "full_attn",
    "sparse_windowed_scaled_dot_product_self_attention": "windowed_attn",
    "SparseMultiHeadRMSNorm": "modules",
    "SparseMultiHeadAttention": "modules",
    "SparseRotaryPositionEmbedder": "rope",
}

# Expose local attention when its optional implementation is available.
if (Path(__file__).parent / 'local_attn.py').is_file():
    __attributes.update({
        "SparseLocalAttention": "local_attn",
        "compute_knn_index": "local_attn",
    })

__all__ = list(__attributes)


def __getattr__(name):
    """Import the requested attention implementation and cache its export."""
    if name not in __attributes:
        raise AttributeError(f"module {__name__} has no attribute {name}")
    module = importlib.import_module(f".{__attributes[name]}", __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
