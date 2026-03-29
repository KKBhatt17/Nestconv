import math

import torch
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD

from entropy import compute_patch_entropy_vectorized


def get_entropy_feature_dim(input_size, patch_size):
    patches_per_side = math.ceil(input_size / patch_size)
    return patches_per_side * patches_per_side


def build_entropy_router_input(images, patch_size):
    if images.dim() != 4:
        raise ValueError(f"Expected a 4D batch tensor, got shape {tuple(images.shape)}")

    mean = torch.tensor(IMAGENET_INCEPTION_MEAN, device=images.device, dtype=images.dtype).view(1, -1, 1, 1)
    std = torch.tensor(IMAGENET_INCEPTION_STD, device=images.device, dtype=images.dtype).view(1, -1, 1, 1)

    # Dataloader tensors are normalized. Restore them to the [0, 255] range expected by entropy.py.
    denormalized = (images.detach() * std + mean).clamp(0.0, 1.0) * 255.0

    features = []
    for image in denormalized:
        entropy_map = compute_patch_entropy_vectorized(
            image=image,
            patch_size=patch_size,
            num_scales=1,
        )[patch_size]
        features.append(entropy_map.reshape(-1))

    return torch.stack(features, dim=0)
