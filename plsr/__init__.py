"""Public configuration and inference API for PLSR.

PLSRPipeline is resolved on first access.
Configuration and command-line help can be used without importing Torch.
"""

from .config import ReferenceImage, RefinementConfig, SamplingConfig
from .inference import RunOptions, run_inference

__all__ = ["ReferenceImage", "RefinementConfig", "SamplingConfig", "PLSRPipeline", "RunOptions", "run_inference"]


def __getattr__(name):
    """Resolve the public pipeline class when it is first requested."""
    if name == "PLSRPipeline":
        from trellis2.pipelines.plsr_pipeline import PLSRPipeline

        return PLSRPipeline
    raise AttributeError(name)
