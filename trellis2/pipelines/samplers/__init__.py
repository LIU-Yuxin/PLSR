"""Euler flow samplers and PLSR patch sampling, imported on demand."""

from .base import Sampler
import importlib

__all__ = ["Sampler", "FlowEulerSampler", "FlowEulerCfgSampler",
           "FlowEulerGuidanceIntervalSampler", "PLSRFlowEulerSampler"]


def __getattr__(name):
    """Load and cache an Euler sampler or the PLSR patch sampler by name."""
    if name not in __all__:
        raise AttributeError(f"module {__name__} has no attribute {name}")
    module = ".plsr_flow_euler" if name == "PLSRFlowEulerSampler" else ".flow_euler"
    value = getattr(importlib.import_module(module, __name__), name)
    globals()[name] = value
    return value
