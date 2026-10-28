"""Sparse tensors, coordinate operations and neural layers with lazy exports."""

from . import config
import importlib
from pathlib import Path

__attributes = {
    'VarLenTensor': 'basic',
    'varlen_cat': 'basic',
    'varlen_unbind': 'basic',
    'SparseTensor': 'basic',
    'sparse_cat': 'basic',
    'sparse_unbind': 'basic',
    'SparseGroupNorm': 'norm',
    'SparseGroupNorm32': 'norm',
    'SparseReLU': 'nonlinearity',
    'SparseSiLU': 'nonlinearity',
    'SparseGELU': 'nonlinearity',
    'SparseActivation': 'nonlinearity',
    'SparseLinear': 'linear',
    'sparse_scaled_dot_product_attention': 'attention',
    'sparse_windowed_scaled_dot_product_self_attention': 'attention',
    'SparseRotaryPositionEmbedder':  'attention',
    'SparseMultiHeadAttention': 'attention',
    'SparseConv3d': 'conv',
    'SparseInverseConv3d': 'conv',
    'SparseDownsample': 'spatial',
    'SparseUpsample': 'spatial',
    'SparseSpatial2Channel': 'spatial',
    'SparseChannel2Spatial': 'spatial',
}

if (Path(__file__).parent / 'attention/local_attn.py').is_file():
    __attributes['SparseLocalAttention'] = 'attention'
if (Path(__file__).parent / 'serialize.py').is_file():
    __attributes.update({'encode_seq': 'serialize', 'decode_seq': 'serialize'})

__submodules = ['transformer', 'conv']

__all__ = list(__attributes.keys()) + __submodules

def __getattr__(name):
    """Import a registered sparse operation or submodule and cache its name."""
    if name not in globals():
        if name in __attributes:
            module_name = __attributes[name]
            module = importlib.import_module(f".{module_name}", __name__)
            globals()[name] = getattr(module, name)
        elif name in __submodules:
            module = importlib.import_module(f".{name}", __name__)
            globals()[name] = module
        else:
            raise AttributeError(f"module {__name__} has no attribute {name}")
    return globals()[name]


# For Pylance
if __name__ == '__main__':
    from .basic import *
    from .norm import *
    from .nonlinearity import *
    from .linear import *
    from .attention import *
    from .conv import *
    from .spatial import *
    from .serialize import *
    import transformer
    import conv
