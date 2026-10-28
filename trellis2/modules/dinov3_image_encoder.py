"""Image conditioning tokens from a Hugging Face DINOv3 model."""

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import Normalize


class DinoV3ImageEncoder(nn.Module):
    def __init__(self, model, image_size=1024):
        super().__init__()
        self.model = model.eval()
        self.image_size = image_size
        self.normalize = Normalize([.485, .456, .406], [.229, .224, .225])
        self.eval()

    @classmethod
    def from_pretrained(cls, path, image_size=1024):
        from transformers import DINOv3ViTModel

        model = DINOv3ViTModel.from_pretrained(str(path), use_safetensors=True)
        encoder = cls(model, image_size)
        encoder.source = str(path)
        return encoder

    @torch.no_grad()
    def forward(self, images):
        device = next(self.model.parameters()).device
        if isinstance(images, list) and images and all(isinstance(image, Image.Image) for image in images):
            pixels = [np.asarray(image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS).convert("RGB"), dtype=np.float32) / 255
                      for image in images]
            images = torch.stack([torch.from_numpy(image).permute(2, 0, 1) for image in pixels])
        if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("DINOv3 requires RGB PIL images or a [B,3,H,W] tensor")
        if not torch.isfinite(images).all() or images.min() < 0 or images.max() > 1:
            raise ValueError("DINOv3 image tensors must be finite and in [0,1]")
        images = self.normalize(images.to(device))
        # Round inputs through bf16 before converting to the patch embedding's dtype.
        images = images.to(torch.bfloat16).to(self.model.embeddings.patch_embeddings.weight.dtype)
        features = self.model.embeddings(images, bool_masked_pos=None)
        positions = self.model.rope_embeddings(images)
        for layer in self.model.layer:
            features = layer(features, position_embeddings=positions)
        return F.layer_norm(features, features.shape[-1:])
