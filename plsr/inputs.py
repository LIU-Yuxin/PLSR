"""Read-only input checks shared by CLI and pipeline entry points."""

from dataclasses import dataclass
from pathlib import Path
import warnings

from PIL import Image

from .config import RefinementConfig


MIN_REFERENCE_SHORT_EDGE = 1024


class LowResolutionImageWarning(UserWarning):
    """A reference image has too few source pixels for effective refinement."""


@dataclass(frozen=True)
class ImageInfo:
    path: Path
    width: int
    height: int
    mode: str
    missing_camera_fields: tuple[str, ...]
    has_alpha: bool = False

    @property
    def low_resolution(self):
        return min(self.width, self.height) < MIN_REFERENCE_SHORT_EDGE


@dataclass(frozen=True)
class InputReport:
    input_mode: str
    images: tuple[ImageInfo, ...]
    needs_camera_estimation: bool

    @property
    def needs_background_removal(self):
        return any(not image.has_alpha for image in self.images)


def has_useful_alpha(image):
    """Return whether decoded alpha contains any non-opaque pixels.

    Accept explicit alpha bands and palette/RGB transparency metadata. Images
    with fully opaque alpha still need foreground segmentation.
    """
    if "A" in image.getbands():
        return image.getchannel("A").getextrema()[0] < 255
    if "transparency" in image.info:
        return image.convert("RGBA").getchannel("A").getextrema()[0] < 255
    return False


def validate_inputs(config: RefinementConfig) -> InputReport:
    """Check paths and decode original images without creating any outputs.

    Mesh geometry and model weights are checked by their respective loaders.
    No resizing, camera estimation, model import, or GPU initialization occurs.
    """
    if config.output_dir.exists() and not config.output_dir.is_dir():
        raise ValueError(f"Output directory is an existing file: {config.output_dir}")
    if config.mesh is not None:
        if not config.mesh.is_file():
            raise FileNotFoundError(f"Mesh file does not exist: {config.mesh}")
        with config.mesh.open("rb") as stream:
            if not stream.read(1):
                raise ValueError(f"Mesh file is empty: {config.mesh}")

    images = []
    for view in config.images:
        if not view.path.is_file():
            raise FileNotFoundError(f"Reference image does not exist: {view.path}")
        try:
            with Image.open(view.path) as image:
                image.load()
                width, height = image.size
                mode = image.mode
                has_alpha = has_useful_alpha(image)
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            raise ValueError(f"Cannot decode reference image {view.path}: {exc}") from exc
        info = ImageInfo(view.path, width, height, mode, view.missing_camera_fields, has_alpha)
        if info.low_resolution:
            warnings.warn(
                f"Reference image '{view.path}' is {width}x{height}; its original short edge is below "
                f"{MIN_REFERENCE_SHORT_EDGE} pixels. It lacks sufficient detail for effective mesh refinement. "
                "Resizing cannot restore missing detail. Processing may continue; use a higher-resolution "
                "reference for better refinement.",
                LowResolutionImageWarning,
                stacklevel=2,
            )
        images.append(info)
    return InputReport(config.input_mode, tuple(images), config.needs_camera_estimation)
