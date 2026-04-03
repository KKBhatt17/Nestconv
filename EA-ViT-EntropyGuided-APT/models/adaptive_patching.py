import math
from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.layers import resample_abs_pos_embed

from entropy import compute_patch_entropy_vectorized


def select_patches_by_threshold(importance_maps: Dict[int, torch.Tensor], thresholds: Iterable[float]):
    patch_sizes = sorted(list(importance_maps.keys()))

    if len(patch_sizes) == 1:
        return {patch_sizes[0]: torch.ones_like(importance_maps[patch_sizes[0]])}

    thresholds = list(thresholds)
    if len(thresholds) != len(patch_sizes) - 1:
        raise ValueError(
            f"Number of thresholds ({len(thresholds)}) must be one less than number of patch sizes ({len(patch_sizes)})"
        )

    masks = {patch_sizes[0]: torch.ones_like(importance_maps[patch_sizes[0]])}
    for index in range(len(patch_sizes) - 1, 0, -1):
        current_size = patch_sizes[index]
        threshold = thresholds[index - 1]
        masks[current_size] = (importance_maps[current_size] < threshold).float()

    for index in range(len(patch_sizes) - 1, 0, -1):
        current_size = patch_sizes[index]
        for smaller_index in range(index):
            smaller_size = patch_sizes[smaller_index]
            scale_factor = current_size // smaller_size
            mask_upscaled = masks[current_size].repeat_interleave(scale_factor, dim=0).repeat_interleave(scale_factor, dim=1)
            small_height, small_width = importance_maps[smaller_size].shape
            mask_upscaled = mask_upscaled[:small_height, :small_width]
            masks[smaller_size] = masks[smaller_size] * (1 - mask_upscaled)

    return masks


class AdaptivePatchTokenizer:
    def __init__(
        self,
        base_patch_size: int,
        image_size: int,
        num_scales: int = 2,
        thresholds: Tuple[float, ...] = (5.5,),
        mean=IMAGENET_INCEPTION_MEAN,
        std=IMAGENET_INCEPTION_STD,
    ):
        self.base_patch_size = base_patch_size
        self.image_size = image_size
        self.num_scales = num_scales
        self.thresholds = tuple(thresholds)
        mean_tensor = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        std_tensor = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
        self.mean = mean_tensor
        self.std = std_tensor

    def _unnormalize(self, image: torch.Tensor):
        mean = self.mean.to(image.device, dtype=image.dtype)
        std = self.std.to(image.device, dtype=image.dtype)
        return torch.clamp((image * std + mean) * 255.0, 0, 255)

    def compute_importance_maps(self, image: torch.Tensor):
        unnormalized = self._unnormalize(image)[0]
        return compute_patch_entropy_vectorized(
            unnormalized,
            patch_size=self.base_patch_size,
            num_scales=self.num_scales,
        )

    def construct_masks(self, image: torch.Tensor):
        importance_maps = self.compute_importance_maps(image)
        return select_patches_by_threshold(importance_maps, self.thresholds)

    def construct_patch_groups(self, image: torch.Tensor, masks: Dict[int, torch.Tensor]):
        output_dict = {}
        _, channels, _, _ = image.shape
        base = self.base_patch_size

        base_grid = image.unfold(2, base, base).unfold(3, base, base)[0]
        base_grid = base_grid.permute(1, 2, 0, 3, 4).contiguous()
        base_grid_h, base_grid_w = base_grid.shape[:2]

        for scale_index in range(self.num_scales):
            current_patch_size = base * (2 ** scale_index)
            factor = current_patch_size // base
            current_mask = masks[current_patch_size].bool()

            scale_image = image
            if scale_index > 0:
                scale_image = F.interpolate(
                    scale_image,
                    scale_factor=1.0 / factor,
                    mode="bilinear",
                    align_corners=False,
                )

            resized_grid = scale_image.unfold(2, base, base).unfold(3, base, base)[0]
            resized_grid = resized_grid.permute(1, 2, 0, 3, 4).contiguous()
            output_dict[f"resized_patches_{current_patch_size}"] = resized_grid[current_mask]
            output_dict[f"pos_embed_mask_{current_patch_size}"] = current_mask.flatten()

            if scale_index == 0:
                continue

            current_grid_h = base_grid_h // factor
            current_grid_w = base_grid_w // factor
            grouped = base_grid.view(
                current_grid_h,
                factor,
                current_grid_w,
                factor,
                channels,
                base,
                base,
            )
            grouped = grouped.permute(0, 2, 1, 3, 4, 5, 6).contiguous()
            grouped = grouped.view(current_grid_h, current_grid_w, factor * factor, channels, base, base)
            output_dict[f"full_patches_{current_patch_size}"] = grouped[current_mask]

        return output_dict

    def tokenize(self, image: torch.Tensor):
        masks = self.construct_masks(image)
        return self.construct_patch_groups(image, masks)


class AdaptivePatchSequenceBuilder(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        embed_dim: int,
        num_scales: int = 2,
        thresholds: Tuple[float, ...] = (5.5,),
    ):
        super().__init__()
        self.image_size = image_size
        self.base_patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_scales = num_scales
        self.patch_sizes = [patch_size * (2 ** index) for index in range(num_scales)]
        self.tokenizer = AdaptivePatchTokenizer(
            base_patch_size=patch_size,
            image_size=image_size,
            num_scales=num_scales,
            thresholds=thresholds,
        )
        self.patch_attn = nn.Conv2d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=2,
            stride=2,
        )
        self.zero_conv = nn.Linear(embed_dim, embed_dim)
        nn.init.zeros_(self.zero_conv.weight)
        nn.init.zeros_(self.zero_conv.bias)

    def _embed_patches(self, patch_embed, patches: torch.Tensor):
        patch_tokens = patch_embed.proj(patches).flatten(2).transpose(1, 2).squeeze(1)
        patch_tokens = patch_embed.norm(patch_tokens)
        return patch_tokens

    def _select_pos_embed(self, base_pos_embed, mask, patch_size):
        if base_pos_embed is None:
            return None

        if base_pos_embed.shape[1] <= 1:
            raise ValueError("Adaptive patching requires learned positional embeddings with patch tokens.")

        prefix_tokens = 1
        base_grid_tokens = base_pos_embed.shape[1] - prefix_tokens
        base_grid_size = int(math.sqrt(base_grid_tokens))

        if patch_size == self.base_patch_size:
            pos_embed = base_pos_embed[:, prefix_tokens:]
        else:
            pos_embed = resample_abs_pos_embed(
                base_pos_embed,
                new_size=(self.image_size // patch_size, self.image_size // patch_size),
                old_size=(base_grid_size, base_grid_size),
                num_prefix_tokens=prefix_tokens,
            )[:, prefix_tokens:]

        flat_pos_embed = pos_embed.reshape(-1, pos_embed.shape[-1])
        return flat_pos_embed[mask]

    def forward(self, image, patch_embed, cls_token, base_pos_embed):
        patch_groups = self.tokenizer.tokenize(image)
        sequence_parts = []

        if cls_token is not None:
            cls_with_pos = cls_token.expand(1, -1, -1)
            if base_pos_embed is not None:
                cls_with_pos = cls_with_pos + base_pos_embed[:, :1]
            sequence_parts.append(cls_with_pos)

        selected_patch_count = 0
        for scale_index, current_patch_size in enumerate(self.patch_sizes):
            resized_patches = patch_groups[f"resized_patches_{current_patch_size}"]
            if resized_patches.numel() == 0:
                continue

            pos_mask = patch_groups[f"pos_embed_mask_{current_patch_size}"]
            pos_embed = self._select_pos_embed(base_pos_embed, pos_mask, current_patch_size)

            scale_tokens = self._embed_patches(patch_embed, resized_patches)
            if pos_embed is not None:
                scale_tokens = scale_tokens + pos_embed

            if scale_index > 0:
                factor = current_patch_size // self.base_patch_size
                full_patches = patch_groups[f"full_patches_{current_patch_size}"]
                full_tokens = self._embed_patches(
                    patch_embed,
                    full_patches.view(-1, full_patches.shape[-3], full_patches.shape[-2], full_patches.shape[-1]),
                )
                full_tokens = full_tokens.view(-1, factor, factor, self.embed_dim).permute(0, 3, 1, 2).contiguous()
                for _ in range(scale_index):
                    full_tokens = self.patch_attn(full_tokens)
                attn_scale = full_tokens.squeeze(-1).squeeze(-1)
                scale_tokens = scale_tokens + self.zero_conv(attn_scale)

            sequence_parts.append(scale_tokens.unsqueeze(0))
            selected_patch_count += scale_tokens.shape[0]

        sequence = torch.cat(sequence_parts, dim=1)
        return sequence, selected_patch_count
