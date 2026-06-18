import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Union
from torchvision.transforms import functional as TF
import math
import cv2
from PIL import Image

import math
import os

def compute_patch_entropy_vectorized(image, patch_size=16, num_scales=2, bins=512, pad_value=1e6):
    """
    Compute entropy maps for multiple patch sizes in the input image using vectorized operations.
    
    Args:
        image: torch.Tensor of shape (C, H, W) or (H, W) with values in range [0, 255]
        patch_sizes: list of patch sizes (default: [16, 32])
        bins: number of bins for histogram (default: 512)
        pad_value: high entropy value to pad incomplete patches with (default: 1e6)
    
    Returns:
        entropy_maps: dict containing torch.Tensor entropy maps for each patch size
    """
    if len(image.shape) == 3:
        # Convert to grayscale if image is RGB
        if image.shape[0] == 3:
            image = 0.2989 * image[0] + 0.5870 * image[1] + 0.1140 * image[2]
        else:
            image = image[0]
    
    entropy_maps = {}
    H, W = image.shape

    patch_sizes = [patch_size * (2**i) for i in range(num_scales)]

    for patch_size in patch_sizes:
        num_patches_h = (H + patch_size - 1) // patch_size  # Round up
        num_patches_w = (W + patch_size - 1) // patch_size  # Round up

        # Pad image to ensure it fits into patches cleanly
        pad_h = num_patches_h * patch_size - H
        pad_w = num_patches_w * patch_size - W
        padded_image = torch.nn.functional.pad(image, (0, pad_w, 0, pad_h), mode='constant', value=0)

        # Unfold the image into patches
        patches = padded_image.unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
        patches = patches.reshape(num_patches_h * num_patches_w, -1)

        # Compute histograms for all patches
        bin_edges = torch.linspace(0, 255, bins + 1, device=image.device)
        histograms = torch.stack([torch.histc(patch, bins=bins, min=0, max=255) for patch in patches])

        # Normalize histograms to get probabilities
        probabilities = histograms / (patch_size * patch_size)

        # Compute entropy: -sum(p * log2(p)), avoiding log(0)
        epsilon = 1e-10
        entropy = -torch.sum(probabilities * torch.log2(probabilities + epsilon), dim=1)

        # Reshape back to spatial dimensions
        entropy_map = entropy.reshape(num_patches_h, num_patches_w)

        # Assign a high value to padded regions
        if pad_h > 0:
            entropy_map[-1, :] = pad_value  # High entropy at bottom row
        if pad_w > 0:
            entropy_map[:, -1] = pad_value  # High entropy at right column

        entropy_maps[patch_size] = entropy_map

    return entropy_maps


def process_image(image_path, image_size, patch_size, num_scales, no_resize=False):
    """Process a single image and return the visualization based on the specified type and method."""
    print(f"Processing image: {image_path}")
    
    # Open the image
    image = Image.open(image_path)
    print(f"Original image size: {image.size}")
    
    if no_resize:
        # Use original image without resizing
        img = image
        print("Using original image size (no resize)")
    else:
        # Find the image size
        width, height = image.size

        # Calculate the target size for the shorter side
        # Make the shorter side always image_size
        if width < height:
            new_width = image_size
            new_height = int(height * (image_size / width))
        else:
            new_height = image_size
            new_width = int(width * (image_size / height))

        # Adjust the longer side to be a multiple of patch_size * 4
        if width > height:
            new_width = (new_width // (patch_size * 4)) * (patch_size * 4)
        else:
            new_height = (new_height // (patch_size * 4)) * (patch_size * 4)

        # Resize the image
        img = image.resize((new_width, new_height))
        print(f"Resized image size: {img.size}")
    
    img_tensor = TF.to_tensor(img) * 255.0
    entropy_maps = compute_patch_entropy_vectorized(img_tensor, patch_size, num_scales)

    return entropy_maps[14]

# should be called as
# process_image(image_path, 366, 14, 3)