"""Lazy access to the sparse-structure models used by inference."""

import importlib

__attributes = {
    "SparseStructureEncoder": "sparse_structure_vae",
    "SparseStructureDecoder": "sparse_structure_vae",
    "SparseStructureFlowModel": "sparse_structure_flow",
}

__all__ = list(__attributes)


def __getattr__(name):
    """Import a registered model class and cache its public name."""
    if name not in __attributes:
        raise AttributeError(f"module {__name__} has no attribute {name}")
    module = importlib.import_module(f".{__attributes[name]}", __name__)
    model_class = getattr(module, name)
    globals()[name] = model_class
    return model_class
