"""Inference-only execution of the reference VAE layers, sharing loaded weights."""

import torch
from torch import nn
from torch.nn import functional as F


def create_slat_vae(config):
    """Construct reference layers without resolving the research model registry.

    Keep construction, strict checkpoint loading and wrapping separate so the
    caller retains its original dtype-conversion order and CPU offload policy.
    """
    from .sparse_unet_vae import SparseUnetVaeEncoder, SparseUnetVaeDecoder

    name, args = config["name"], dict(config["args"])
    if name in ("FlexiDualGridVaeEncoder", "FlexiDualGridVaeDecoder"):
        from .fdg_vae import FlexiDualGridVaeEncoder, FlexiDualGridVaeDecoder

        model_class = {"FlexiDualGridVaeEncoder": FlexiDualGridVaeEncoder,
                       "FlexiDualGridVaeDecoder": FlexiDualGridVaeDecoder}[name]
        if name == "FlexiDualGridVaeEncoder":
            # Some encoder configs include this unused resolution argument.
            # Encoding is fully convolutional; spatial scale belongs to the codec.
            args.pop("resolution", None)
    else:
        classes = {"SparseUnetVaeEncoder": SparseUnetVaeEncoder,
                   "SparseUnetVaeDecoder": SparseUnetVaeDecoder}
        if name not in classes:
            raise ValueError(f"Unsupported inference SLat VAE: {name}")
        model_class = classes[name]
    return model_class(**args)


class SLatVaeEncoder(nn.Module):
    """Encode packed voxels to the posterior mean using shared VAE layers.

    The codec supplies centered geometry or normalized PBR channels. Residual
    stages use the configured torso dtype; projection and normalization restore
    the input dtype. Return the mean without sampling or latent normalization.
    """

    def __init__(self, encoder):
        """Share loaded encoder projections and stages in evaluation mode."""
        super().__init__()
        self.input_layer = encoder.input_layer
        self.blocks = encoder.blocks
        self.to_latent = encoder.to_latent
        self.dtype = encoder.dtype
        self.eval()

    @torch.no_grad()
    def forward(self, x):
        """Return an unnormalized mean latent on the encoder's reduced support."""
        # 1. Project packed voxel channels and downsample through encoder stages.
        h = self.input_layer(x).type(self.dtype)
        for stage in self.blocks:
            for block in stage:
                h = block(h)
        # 2. Normalize and project to posterior parameters; keep only the mean.
        h = h.type(x.dtype)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = self.to_latent(h)
        mean, _ = h.feats.chunk(2, dim=-1)
        return h.replace(mean)


class SLatVaeDecoder(nn.Module):
    """Reference full decoder, returning voxel attributes and subdivision guides.

    Mesh construction and any outer spatial partitioning belong to the codec.
    No implicit coordinate-dependent patching, training or feature-probe branch.
    """

    def __init__(self, decoder):
        """Share the loaded decoder layers and retain its dtype/subdivision policy."""
        super().__init__()
        self.from_latent = decoder.from_latent
        self.blocks = decoder.blocks
        self.output_layer = decoder.output_layer
        self.dtype = decoder.dtype
        self.pred_subdiv = decoder.pred_subdiv
        self.voxel_margin = getattr(decoder, "voxel_margin", None)
        self.eval()

    @torch.no_grad()
    def forward(self, x, guide_subs=None):
        """Decode voxel attributes, predicting Shape guides or consuming them for PBR.

        Input/output projections follow the latent dtype; residual stages use
        the configured torso dtype. Return raw output channels plus the guides
        predicted by Shape. Geometry activation and mesh extraction are separate.
        """
        # 1. Shape predicts subdivisions; PBR must reuse one guide per upsample.
        if self.pred_subdiv and guide_subs is not None:
            raise ValueError("Shape decoding predicts its own subdivisions")
        if not self.pred_subdiv and (guide_subs is None or len(guide_subs) != len(self.blocks) - 1):
            raise ValueError("PBR decoding requires Shape's subdivision guide at every stage")
        # 2. Execute every residual stage and subdivision on the same support.
        h = self.from_latent(x).type(self.dtype)
        subs = []
        for index, stage in enumerate(self.blocks):
            for block_index, block in enumerate(stage):
                if index < len(self.blocks) - 1 and block_index == len(stage) - 1:
                    if self.pred_subdiv:
                        h, sub = block(h)
                        subs.append(sub)
                    else:
                        h = block(h, subdiv=guide_subs[index])
                else:
                    h = block(h)
        # 3. Restore the projection dtype, normalize and predict voxel channels.
        h = h.type(x.dtype)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        return self.output_layer(h), subs
