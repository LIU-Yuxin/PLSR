"""BiRefNet foreground masks with Hugging Face model loading."""

import torch
from torchvision import transforms


class BiRefNetBackgroundRemover:
    """Predict alpha from a 1024-pixel normalized RGB condition.

    Resize the predicted sigmoid mask back to the input canvas and attach it
    to the original RGB pixels. The caller retains camera/image resizing policy.
    """

    def __init__(self, model, *, device="cuda", low_vram=True):
        """Bind the segmentation model and its ImageNet-normalized transform."""
        self.model = model.eval()
        self.device = torch.device(device)
        self.low_vram = low_vram
        self.transform = transforms.Compose([
            transforms.Resize((1024, 1024)), transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        """Load BiRefNet from a model repo ID or matching local repo directory."""
        from transformers import AutoModelForImageSegmentation

        model = AutoModelForImageSegmentation.from_pretrained(
            str(path), use_safetensors=True, trust_remote_code=True,
        )
        remover = cls(model, **kwargs)
        remover.source = str(path)
        return remover

    @torch.no_grad()
    def __call__(self, image):
        """Return RGBA at the input size, offloading the model when enabled."""
        image = image.convert("RGB")
        try:
            self.model.to(self.device)
            prediction = self.model(self.transform(image)[None].to(self.device))[-1].sigmoid().cpu()
        finally:
            if self.low_vram:
                self.model.cpu()
        mask = transforms.ToPILImage()(prediction[0].squeeze()).resize(image.size)
        output = image.copy()
        output.putalpha(mask)
        return output
