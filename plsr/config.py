"""Inference configuration, independent of model loading and CUDA."""

from dataclasses import asdict, dataclass, field
import json
import math
from numbers import Real
from pathlib import Path
from typing import Literal

import numpy as np


def _finite_number(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number.")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be a finite number.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number.")
    return number


def _matrix(value, size, name):
    try:
        matrix = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a {size}x{size} numeric matrix.") from exc
    if matrix.shape != (size, size) or matrix.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a {size}x{size} numeric matrix.")
    matrix = matrix.astype(np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite numbers.")
    return matrix


@dataclass(frozen=True)
class ReferenceImage:
    """One reference view, with independently optional camera matrices.

    Extrinsics are 4x4 input-world-to-camera transforms in the OpenCV camera
    convention (x right, y down, z forward). Intrinsics describe the original
    image, either in pixels or normalized by width for row 0 and height for row 1.
    Validation preserves supplied values; coordinate conversion happens later.
    """

    path: Path
    intrinsics: tuple[tuple[float, ...], ...] | None = None
    extrinsics: tuple[tuple[float, ...], ...] | None = None
    intrinsics_units: Literal["normalized", "pixels"] = "normalized"

    def __post_init__(self):
        object.__setattr__(self, "path", Path(self.path))
        if self.intrinsics_units not in ("normalized", "pixels"):
            raise ValueError("intrinsics_units must be 'normalized' or 'pixels'.")
        if self.intrinsics is not None:
            matrix = _matrix(self.intrinsics, 3, "intrinsics")
            if not np.allclose(matrix[2], [0, 0, 1], atol=1e-6, rtol=0):
                raise ValueError("intrinsics must have bottom row [0, 0, 1].")
            if matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
                raise ValueError("intrinsics focal lengths must be positive.")
            if matrix[0, 1] != 0 or matrix[1, 0] != 0:
                raise ValueError("Nonzero camera skew is not supported by the renderer.")
            object.__setattr__(self, "intrinsics", tuple(map(tuple, matrix.tolist())))
        if self.extrinsics is not None:
            matrix = _matrix(self.extrinsics, 4, "extrinsics")
            if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6, rtol=0):
                raise ValueError("extrinsics must have bottom row [0, 0, 0, 1].")
            rotation = matrix[:3, :3]
            if not (
                np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4, rtol=0)
                and math.isclose(np.linalg.det(rotation), 1.0, abs_tol=1e-4)
            ):
                raise ValueError("extrinsics must contain a proper rotation, without scale or reflection.")
            object.__setattr__(self, "extrinsics", tuple(map(tuple, matrix.tolist())))

    @property
    def missing_camera_fields(self):
        return tuple(name for name in ("intrinsics", "extrinsics") if getattr(self, name) is None)


@dataclass(frozen=True)
class SamplingConfig:
    """Parameters passed to the upstream guidance-interval sampler."""

    guidance_strength: float
    steps: int = 12
    rescale_t: float = 3.0
    guidance_interval: tuple[float, float] = (0.6, 1.0)

    def __post_init__(self):
        strength = _finite_number(self.guidance_strength, "guidance_strength")
        rescale = _finite_number(self.rescale_t, "rescale_t")
        if strength < 0:
            raise ValueError("guidance_strength must be nonnegative.")
        if rescale <= 0:
            raise ValueError("rescale_t must be positive.")
        if type(self.steps) is not int or self.steps <= 0:
            raise ValueError("steps must be a positive integer.")
        if not isinstance(self.guidance_interval, (list, tuple)) or len(self.guidance_interval) != 2:
            raise ValueError("guidance_interval must contain two values in [0, 1].")
        start, end = (_finite_number(t, "guidance_interval") for t in self.guidance_interval)
        if not 0 <= start <= end <= 1:
            raise ValueError("guidance_interval must satisfy 0 <= start <= end <= 1.")
        object.__setattr__(self, "guidance_strength", strength)
        object.__setattr__(self, "rescale_t", rescale)
        object.__setattr__(self, "guidance_interval", (start, end))


@dataclass(frozen=True)
class RefinementConfig:
    """A validated inference request; construction does not open input files.

    A missing mesh selects single-image coarse generation. Multiple references
    require a supplied mesh until a coarse-generation view policy is defined.
    """

    images: tuple[ReferenceImage, ...]
    output_dir: Path
    shape_sampler: SamplingConfig
    mesh: Path | None = None
    resolution_sequence: tuple[int, ...] = (640, 1280, 2560)
    pbr_sampler: SamplingConfig = field(default_factory=lambda: SamplingConfig(guidance_strength=2.5))
    seed: int = 42
    decode_intermediate: bool = False
    output_space: Literal["input", "canonical"] = "input"

    def __post_init__(self):
        sequence = self.resolution_sequence
        if not isinstance(sequence, (list, tuple)) or len(sequence) < 2:
            raise ValueError("resolution_sequence must contain a base resolution and at least one SR level.")
        if any(type(r) is not int or r <= 0 for r in sequence):
            raise ValueError("Resolutions must be positive integers.")
        if sequence[0] % 128:
            raise ValueError("The base resolution must be a multiple of 128.")
        if any(current * 2 != following for current, following in zip(sequence, sequence[1:])):
            raise ValueError("Each resolution must be exactly twice the preceding resolution.")
        if not isinstance(self.images, (list, tuple)) or not self.images:
            raise ValueError("At least one reference image is required.")
        if any(not isinstance(view, ReferenceImage) for view in self.images):
            raise TypeError("images must contain ReferenceImage instances.")
        if self.mesh is None and len(self.images) != 1:
            raise ValueError("Multiple reference images require a mesh; single-image generation accepts one image.")
        if not isinstance(self.shape_sampler, SamplingConfig) or not isinstance(self.pbr_sampler, SamplingConfig):
            raise TypeError("shape_sampler and pbr_sampler must be SamplingConfig instances.")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32).")
        if type(self.decode_intermediate) is not bool:
            raise ValueError("decode_intermediate must be a boolean.")
        if self.output_space not in ("input", "canonical"):
            raise ValueError("output_space must be 'input' or 'canonical'.")
        object.__setattr__(self, "images", tuple(self.images))
        object.__setattr__(self, "resolution_sequence", tuple(sequence))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.mesh is not None:
            object.__setattr__(self, "mesh", Path(self.mesh))

    @property
    def input_mode(self):
        return "mesh" if self.mesh is not None else "image"

    @property
    def needs_camera_estimation(self):
        return any(view.missing_camera_fields for view in self.images)

    @classmethod
    def from_dict(cls, data, base_dir=None, *, defaults=None):
        """Merge repo defaults with a request; paths resolve against base_dir."""
        if not isinstance(data, dict):
            raise ValueError("Inference configuration must be a JSON object.")
        defaults = {} if defaults is None else defaults
        options = {**defaults, **data}
        for name in ("shape_sampler", "pbr_sampler"):
            if name in data and isinstance(data[name], dict):
                options[name] = {**defaults.get(name, {}), **data[name]}
        if "shape_sampler" not in options:
            raise ValueError("shape_sampler with an explicit guidance_strength is required when no repo defaults are supplied.")
        base_dir = Path.cwd() if base_dir is None else Path(base_dir)

        def resolve_path(value):
            if not isinstance(value, str) or not value.strip():
                raise ValueError("Configuration paths must be nonempty strings.")
            path = Path(value).expanduser()
            return (base_dir / path).resolve()

        for name in ("shape_sampler", "pbr_sampler"):
            if name in options and not isinstance(options[name], SamplingConfig):
                options[name] = SamplingConfig(**options[name])
        views = options.get("images", [])
        if not isinstance(views, list):
            raise ValueError("images must be a list of objects containing a path.")
        parsed_views = []
        for view in views:
            if not isinstance(view, dict) or "path" not in view:
                raise ValueError("Each image must be an object containing a path.")
            parsed_views.append(ReferenceImage(**{**view, "path": resolve_path(view["path"])}))
        options["images"] = tuple(parsed_views)
        if "output_dir" not in options:
            raise ValueError("output_dir is required.")
        options["output_dir"] = resolve_path(options["output_dir"])
        if options.get("mesh") is not None:
            options["mesh"] = resolve_path(options["mesh"])
        return cls(**options)

    @classmethod
    def from_json(cls, path, *, defaults=None):
        path = Path(path).expanduser().resolve()
        with path.open(encoding="utf-8") as stream:
            return cls.from_dict(json.load(stream), base_dir=path.parent, defaults=defaults)

    def to_dict(self):
        data = asdict(self)
        data["output_dir"] = str(self.output_dir)
        data["mesh"] = str(self.mesh) if self.mesh is not None else None
        data["images"] = list(data["images"])
        data["resolution_sequence"] = list(self.resolution_sequence)
        for name in ("shape_sampler", "pbr_sampler"):
            data[name]["guidance_interval"] = list(data[name]["guidance_interval"])
        for view in data["images"]:
            view["path"] = str(view["path"])
            for name in ("intrinsics", "extrinsics"):
                if view[name] is not None:
                    view[name] = [list(row) for row in view[name]]
        return data
