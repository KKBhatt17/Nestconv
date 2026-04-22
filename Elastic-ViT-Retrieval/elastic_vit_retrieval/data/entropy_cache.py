from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from elastic_vit_retrieval.utils.entropy import compute_patch_entropy_vectorized


def pil_to_entropy_tensor(image: Image.Image, image_size: int) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda tensor: tensor * 255.0),
        ]
    )
    return transform(image)


def compute_mean_entropy(image: Image.Image, image_size: int, patch_size: int, num_scales: int) -> float:
    image_tensor = pil_to_entropy_tensor(image, image_size)
    entropy_maps = compute_patch_entropy_vectorized(
        image_tensor,
        patch_size=patch_size,
        num_scales=num_scales,
    )
    final_key = patch_size * (2 ** (num_scales - 1))
    return float(entropy_maps[final_key].mean().item())


def _extract_image(sample):
    if isinstance(sample, dict):
        image = sample["image"]
    elif isinstance(sample, (list, tuple)):
        image = sample[0]
    else:
        raise ValueError("Unsupported dataset sample format for entropy caching.")
    if not isinstance(image, Image.Image):
        image = transforms.ToPILImage()(image)
    return image


def build_entropy_cache(
    dataset: Dataset,
    cache_path: str | Path,
    image_size: int,
    patch_size: int,
    num_scales: int,
) -> torch.Tensor:
    target = Path(cache_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return torch.load(target, map_location="cpu")

    entropy_values = []
    for index in range(len(dataset)):
        image = _extract_image(dataset[index])
        entropy_values.append(compute_mean_entropy(image, image_size, patch_size, num_scales))

    values = torch.tensor(entropy_values, dtype=torch.float32)
    torch.save(values, target)
    return values


def lookup_batch_entropy(entropy_values: torch.Tensor, indices: Sequence[int]) -> torch.Tensor:
    batch_indices = torch.as_tensor(indices, dtype=torch.long)
    return entropy_values[batch_indices]
