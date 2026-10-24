"""Persist an aligned Shape/PBR latent pair without normalization or reordering."""

from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import torch

from ..modules.sparse import SparseTensor
from .slat_patches import require_aligned, validate_slat


def _validate_pair(shape, pbr, resolution):
    if type(resolution) is not int or resolution <= 0 or resolution % 128:
        raise ValueError("Latent resolution must be a positive multiple of 128")
    for name, latent in (("Shape", shape), ("PBR", pbr)):
        validate_slat(latent, name)
        if latent.shape[0] != 1 or latent.feats.shape[1] != 32:
            raise ValueError("Saved latents require one asset with 32 channels")
        if latent.dtype not in (torch.float16, torch.float32, torch.float64) or not torch.isfinite(latent.feats).all():
            raise ValueError("Saved latents require finite float16, float32 or float64 features")
        if not ((latent.coords[:, 1:] >= 0) & (latent.coords[:, 1:] < resolution // 16)).all():
            raise ValueError("Latent coordinates fall outside the declared resolution")
    require_aligned(shape, pbr, "PBR latent")


def save_latents(path, shape, pbr, resolution):
    """Atomically save both unnormalized VAE latents and their shared coordinates."""
    _validate_pair(shape, pbr, resolution)
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError("Latent files must use the .npz suffix")
    temporary = None
    try:
        with NamedTemporaryFile(dir=path.parent, prefix=f".{path.stem}-", suffix=".npz", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(
                stream, format_version=np.int64(1), resolution=np.int64(resolution),
                coords=shape.coords.detach().cpu().numpy(),
                shape=shape.feats.detach().cpu().numpy(), pbr=pbr.feats.detach().cpu().numpy(),
            )
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_latents(path, device="cpu"):
    """Return (Shape, PBR, resolution); never infer resolution from a filename."""
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError("Latent files must use the .npz suffix")
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {"format_version", "resolution", "coords", "shape", "pbr"}:
            raise ValueError("Expected a PLSR Shape/PBR latent-pair archive")
        for name in ("format_version", "resolution"):
            if data[name].shape != () or data[name].dtype.kind not in "iu":
                raise ValueError(f"Latent {name} must be an integer scalar")
        if int(data["format_version"]) != 1:
            raise ValueError("Unsupported latent archive version")
        resolution = int(data["resolution"])
        coords = torch.from_numpy(data["coords"]).to(device)
        shape = SparseTensor(coords=coords, feats=torch.from_numpy(data["shape"]).to(device))
        pbr = SparseTensor(coords=coords.clone(), feats=torch.from_numpy(data["pbr"]).to(device))
    _validate_pair(shape, pbr, resolution)
    return shape, pbr, resolution
