from __future__ import annotations

import torch


def compute_batch_entropy(images: torch.Tensor, bins: int = 64) -> torch.Tensor:
    """Compute a compact per-image entropy signal from batched tensors."""
    if images.ndim != 4:
        raise ValueError("images must have shape [batch, channels, height, width]")

    grayscale = images.mean(dim=1)
    values = []
    for image in grayscale:
        image = image.detach()
        image = (image - image.min()) / (image.max() - image.min()).clamp_min(1e-6)
        hist = torch.histc(image.float(), bins=bins, min=0.0, max=1.0)
        prob = hist / hist.sum().clamp_min(1.0)
        entropy = -(prob * torch.log2(prob.clamp_min(1e-12))).sum()
        values.append(entropy)
    return torch.stack(values).to(device=images.device, dtype=images.dtype)

