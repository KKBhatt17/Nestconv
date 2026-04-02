import os

import torch
from PIL import Image
from torchvision.transforms import functional as TF


PAD_VALUE = 1e6


def compute_patch_entropy_vectorized(image, patch_size=16, num_scales=2, bins=512, pad_value=PAD_VALUE):
    """
    Compute entropy maps for multiple patch sizes in the input image using vectorized operations.

    Args:
        image: torch.Tensor of shape (C, H, W) or (H, W) with values in range [0, 255]
        bins: number of bins for histogram (default: 512)
        pad_value: high entropy value to pad incomplete patches with (default: 1e6)

    Returns:
        entropy_maps: dict containing torch.Tensor entropy maps for each patch size
    """
    if len(image.shape) == 3:
        if image.shape[0] == 3:
            image = 0.2989 * image[0] + 0.5870 * image[1] + 0.1140 * image[2]
        else:
            image = image[0]

    entropy_maps = {}
    height, width = image.shape
    patch_sizes = [patch_size * (2 ** i) for i in range(num_scales)]

    for current_patch_size in patch_sizes:
        num_patches_h = (height + current_patch_size - 1) // current_patch_size
        num_patches_w = (width + current_patch_size - 1) // current_patch_size

        pad_h = num_patches_h * current_patch_size - height
        pad_w = num_patches_w * current_patch_size - width
        padded_image = torch.nn.functional.pad(image, (0, pad_w, 0, pad_h), mode='constant', value=0)

        patches = padded_image.unfold(0, current_patch_size, current_patch_size).unfold(1, current_patch_size, current_patch_size)
        patches = patches.reshape(num_patches_h * num_patches_w, -1)

        histograms = torch.stack([torch.histc(patch, bins=bins, min=0, max=255) for patch in patches])
        probabilities = histograms / (current_patch_size * current_patch_size)

        epsilon = 1e-10
        entropy = -torch.sum(probabilities * torch.log2(probabilities + epsilon), dim=1)
        entropy_map = entropy.reshape(num_patches_h, num_patches_w)

        if pad_h > 0:
            entropy_map[-1, :] = pad_value
        if pad_w > 0:
            entropy_map[:, -1] = pad_value

        entropy_maps[current_patch_size] = entropy_map

    return entropy_maps


def _load_image(image_source):
    if isinstance(image_source, (str, os.PathLike)):
        return Image.open(image_source).convert("RGB")
    if isinstance(image_source, Image.Image):
        return image_source.convert("RGB")
    if torch.is_tensor(image_source):
        return TF.to_pil_image(image_source).convert("RGB")
    raise TypeError(f"Unsupported image source type: {type(image_source)}")


def _resize_image(image, image_size, patch_size, no_resize=False):
    if no_resize:
        return image

    width, height = image.size
    if width < height:
        new_width = image_size
        new_height = int(height * (image_size / width))
    else:
        new_height = image_size
        new_width = int(width * (image_size / height))

    if width > height:
        new_width = max(patch_size, (new_width // (patch_size * 4)) * (patch_size * 4))
    else:
        new_height = max(patch_size, (new_height // (patch_size * 4)) * (patch_size * 4))

    return image.resize((new_width, new_height))


def process_image(image_source, image_size, patch_size, num_scales, no_resize=False):
    """Process a single image path/PIL image/tensor and return the entropy map for `patch_size`."""
    image = _load_image(image_source)
    image = _resize_image(image, image_size, patch_size, no_resize=no_resize)
    img_tensor = TF.to_tensor(image) * 255.0
    entropy_maps = compute_patch_entropy_vectorized(img_tensor, patch_size, num_scales)
    return entropy_maps[patch_size]


def compute_entropy_mean(image_source, image_size=366, patch_size=14, num_scales=3, no_resize=False):
    entropy_map = process_image(
        image_source,
        image_size=image_size,
        patch_size=patch_size,
        num_scales=num_scales,
        no_resize=no_resize,
    )
    valid_mask = entropy_map < (PAD_VALUE / 2)
    if torch.any(valid_mask):
        return float(entropy_map[valid_mask].mean().item())
    return float(entropy_map.mean().item())


# should be called as
# process_image(image_path, 366, 14, 3)
