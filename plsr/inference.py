"""Public repo-based inference entry point with automatic output reuse."""

from dataclasses import asdict, dataclass
import os
from time import perf_counter
import warnings

from .config import RefinementConfig
from .inputs import LowResolutionImageWarning
from .pretrained import load_inference_defaults, resolve_model_sources


RUNTIME_ENVIRONMENT = {"OPENCV_IO_ENABLE_OPENEXR": "1"}
if os.name != "nt":
    RUNTIME_ENVIRONMENT["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


@dataclass(frozen=True)
class RunOptions:
    """Per-run controls for patch batches, local decoding and textured mesh export."""
    texture_size: int | None = 4096
    decimation_target: int | None = 500000
    shape_batch_size: int = 12
    pbr_batch_size: int = 8
    camera_batch_size: int | None = None
    camera_token_batch_size: int | None = None
    camera_offload_encoder: bool = False
    camera_offload_features: bool = False
    sampling_patch_size: int = 32
    sampling_patch_overlap: int = 8
    decode_local_resolution: int = 1024
    decode_patch_size: int = 64
    decode_overlap: int = 2
    decode_cell_limit: int = 25000
    cpu_threads: int = 4

    @classmethod
    def for_low_memory(cls, **overrides):
        """Batch independent work and offload camera features; keep quality settings."""
        return cls(**{
            "shape_batch_size": 4, "pbr_batch_size": 4,
            "camera_batch_size": 1, "camera_token_batch_size": 2048,
            "camera_offload_encoder": True, "camera_offload_features": True,
            **overrides,
        })

    def __post_init__(self):
        """Validate execution sizes, patch overlap and optional mesh export controls."""
        if self.texture_size is not None and (type(self.texture_size) is not int or self.texture_size <= 0):
            raise ValueError("texture_size must be a positive integer or None")
        for name in ("shape_batch_size", "pbr_batch_size", "cpu_threads", "decode_cell_limit"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("camera_batch_size", "camera_token_batch_size"):
            size = getattr(self, name)
            if size is not None and (type(size) is not int or size <= 0):
                raise ValueError(f"{name} must be a positive integer or None")
        if type(self.camera_offload_encoder) is not bool:
            raise ValueError("camera_offload_encoder must be a boolean")
        if type(self.camera_offload_features) is not bool:
            raise ValueError("camera_offload_features must be a boolean")
        if type(self.sampling_patch_size) is not int or self.sampling_patch_size < 2:
            raise ValueError("sampling_patch_size must be an integer >= 2")
        if (type(self.sampling_patch_overlap) is not int
                or not 0 <= self.sampling_patch_overlap < self.sampling_patch_size):
            raise ValueError("sampling_patch_overlap must be in [0, sampling_patch_size)")
        if (type(self.decode_local_resolution) is not int or self.decode_local_resolution <= 0
                or self.decode_local_resolution % 128):
            raise ValueError("decode_local_resolution must be a positive multiple of 128")
        if (type(self.decode_patch_size) is not int or self.decode_patch_size < 2
                or self.decode_patch_size % 2 or self.decode_patch_size > self.decode_local_resolution // 16):
            raise ValueError("decode_patch_size must be even and fit the local SLat grid")
        if type(self.decode_overlap) is not int or not 0 <= self.decode_overlap < self.decode_patch_size:
            raise ValueError("decode_overlap must be in [0, decode_patch_size)")
        if self.decimation_target is not None and (type(self.decimation_target) is not int or self.decimation_target <= 0):
            raise ValueError("decimation_target must be a positive integer or None")


def run_inference(config, plsr_repo_id=None, *, trellis2_repo_id=None, dino_repo_id=None,
                  mapanything_repo_id=None, rmbg_repo_id=None, options=None, verbose=True):
    """Validate, assemble the required models, and run one progressive request.

    Use a fresh process so the established CUDA environment takes effect before
    importing Torch. Supplied environment values are retained and recorded.
    The caller must use a __main__ guard because GLB baking spawns workers.
    """
    model_sources = resolve_model_sources(
        plsr_repo_id, trellis2_repo_id, dino_repo_id=dino_repo_id,
        mapanything_repo_id=mapanything_repo_id, rmbg_repo_id=rmbg_repo_id,
    )
    if isinstance(config, dict):
        config = RefinementConfig.from_dict(config, defaults=load_inference_defaults(model_sources["plsr"]))
    if not isinstance(config, RefinementConfig):
        raise TypeError("run_inference requires input arguments as a dict or a RefinementConfig")
    options = RunOptions() if options is None else options
    if not isinstance(options, RunOptions):
        raise TypeError("options must be RunOptions")
    with warnings.catch_warnings():
        warnings.simplefilter("once", LowResolutionImageWarning)
        for name, value in RUNTIME_ENVIRONMENT.items():
            os.environ.setdefault(name, value)

        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("PLSR inference requires a CUDA GPU and the installed CUDA extensions")
        torch.set_num_threads(options.cpu_threads)
        from trellis2.pipelines.plsr_pipeline import PLSRPipeline
        if verbose:
            print("Loading pretrained PLSR inference models", flush=True)
        load_start = perf_counter()
        pipeline = PLSRPipeline.from_pretrained(
            model_sources["plsr"], trellis2_repo_id=model_sources["trellis2"], dino_repo_id=model_sources["dino"],
            mapanything_repo_id=model_sources["mapanything"], rmbg_repo_id=model_sources["rmbg"],
        )
        model_load_seconds = perf_counter() - load_start
        settings = dict(pipeline.pretrained_settings)
        settings.update(
            model_sources=model_sources, run_options=asdict(options),
            environment={name: os.environ[name] for name in RUNTIME_ENVIRONMENT},
            torch_version=torch.__version__, cuda_version=torch.version.cuda,
            gpu=torch.cuda.get_device_name(),
        )
        return pipeline.run(
            config, texture_size=options.texture_size,
            decimation_target=options.decimation_target, shape_batch_size=options.shape_batch_size,
            pbr_batch_size=options.pbr_batch_size,
            camera_batch_size=options.camera_batch_size,
            camera_token_batch_size=options.camera_token_batch_size,
            camera_offload_encoder=options.camera_offload_encoder,
            camera_offload_features=options.camera_offload_features,
            sampling_patch_size=options.sampling_patch_size, sampling_patch_overlap=options.sampling_patch_overlap,
            decode_local_resolution=options.decode_local_resolution, decode_patch_size=options.decode_patch_size,
            decode_overlap=options.decode_overlap, decode_cell_limit=options.decode_cell_limit,
            verbose=verbose, runtime_info=settings,
            model_load_seconds=model_load_seconds,
        )
