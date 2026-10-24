"""Lazy access to the PLSR pipeline, refinement step and TRELLIS.2 coarse stage.

Load PLSR repositories with ``plsr.PLSRPipeline.from_pretrained``.
"""

import importlib

__attributes = {
    "Trellis2CoarsePipeline": "trellis2_coarse",
    "PLSRPipeline": "plsr_pipeline",
    "PLSRRefinementStep": "plsr_refinement",
}

__all__ = [*__attributes, "samplers"]


def __getattr__(name):
    """Import a pipeline class or sampler package and cache its public name."""
    if name in __attributes:
        module = importlib.import_module(f".{__attributes[name]}", __name__)
        value = getattr(module, name)
    elif name == "samplers":
        value = importlib.import_module(".samplers", __name__)
    else:
        raise AttributeError(f"module {__name__} has no attribute {name}")
    globals()[name] = value
    return value
