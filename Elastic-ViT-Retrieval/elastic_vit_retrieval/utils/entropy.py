from __future__ import annotations

from typing import Dict

import torch


def compute_patch_entropy_vectorized(
    image: torch.Tensor,
    patch_size: int = 16,
    num_scales: int = 2,
    bins: int = 256,
    pad_value: float = 1.0e6,
) -> Dict[int, torch.Tensor]:
    if image.ndim == 3:
        if image.shape[0] == 3:
            image = 0.2989 * image[0] + 0.5870 * image[1] + 0.1140 * image[2]
        else:
            image = image[0]

    entropy_maps: Dict[int, torch.Tensor] = {}
    height, width = image.shape
    patch_sizes = [patch_size * (2 ** scale_idx) for scale_idx in range(num_scales)]

    for active_patch_size in patch_sizes:
        num_patches_h = (height + active_patch_size - 1) // active_patch_size
        num_patches_w = (width + active_patch_size - 1) // active_patch_size
        pad_h = num_patches_h * active_patch_size - height
        pad_w = num_patches_w * active_patch_size - width
        padded_image = torch.nn.functional.pad(image, (0, pad_w, 0, pad_h), mode="constant", value=0)
        patches = padded_image.unfold(0, active_patch_size, active_patch_size).unfold(1, active_patch_size, active_patch_size)
        patches = patches.reshape(num_patches_h * num_patches_w, -1)

        histograms = torch.stack(
            [torch.histc(patch, bins=bins, min=0, max=255) for patch in patches],
            dim=0,
        )
        probabilities = histograms / (active_patch_size * active_patch_size)
        entropy = -(probabilities * torch.log2(probabilities + 1.0e-10)).sum(dim=1)
        entropy_map = entropy.reshape(num_patches_h, num_patches_w)

        if pad_h > 0:
            entropy_map[-1, :] = pad_value
        if pad_w > 0:
            entropy_map[:, -1] = pad_value

        entropy_maps[active_patch_size] = entropy_map

    return entropy_maps
