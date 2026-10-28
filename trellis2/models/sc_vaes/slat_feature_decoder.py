"""The first two VAE stages used as PLSR's LR feature decoder."""

import torch
from torch import nn


class SLatFeatureDecoder(nn.Module):
    """Extract LR conditioning features after one VAE subdivision.

    Project VAE latents, run stage 0 including its subdivision, then run stage
    1's residual blocks. Stop before the second subdivision and the decoder's
    output normalization/projection. The released VAEs return 512 channels on
    a grid twice the input SLat resolution.

    Reuse the supplied decoder's modules and parameters. Shape predicts the
    subdivision guide; PBR consumes that same guide to produce aligned cells.
    Feature computation uses the supplied decoder's dtype.
    """

    def __init__(self, decoder):
        super().__init__()
        if len(decoder.blocks) < 3 or any(len(stage) < 2 for stage in decoder.blocks[:2]):
            raise ValueError("Feature decoding requires two VAE stages with subdivision blocks")
        self.from_latent = decoder.from_latent
        self.blocks = nn.ModuleList([
            nn.ModuleList(decoder.blocks[0]),
            nn.ModuleList(decoder.blocks[1][:-1]),
        ])
        self.dtype = decoder.dtype
        self.pred_subdiv = decoder.pred_subdiv
        self.eval()

    @torch.no_grad()
    def forward(self, x, guide_subs=None):
        """Return intermediate features and Shape subdivision guides on the doubled grid."""
        if self.pred_subdiv:
            if guide_subs is not None:
                raise ValueError("The Shape feature decoder predicts its own subdivisions")
        elif guide_subs is None or len(guide_subs) != 1:
            raise ValueError("The PBR feature decoder requires Shape's one subdivision guide")
        # 1. Project unnormalized VAE latents into decoder features.
        h = self.from_latent(x).type(self.dtype)

        # 2. Run stage 0, predicting or consuming the shared subdivision guide.
        for block in self.blocks[0][:-1]:
            h = block(h)
        if self.pred_subdiv:
            h, subdivision = self.blocks[0][-1](h)
            subdivisions = [subdivision]
        else:
            h = self.blocks[0][-1](h, subdiv=guide_subs[0])
            subdivisions = []

        # 3. Finish stage 1's residual blocks without its second subdivision.
        for block in self.blocks[1]:
            h = block(h)
        return h, subdivisions
