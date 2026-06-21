"""Analytic GMACs for an elastic ViT sub-network.

The estimate is analytic on purpose: a runtime profiler (e.g. fvcore) would still
count masked-out matmuls at full width, whereas this reports the cost of the
*realized* sub-network — which is what gets deployed.

The core formula is identical to ``Elastic-ViT/elastic_vit/utils/flops.py`` so the
numbers line up with stage-2. With the full config at 197 tokens it yields
17.447 GMACs (the ViT-B/16@224 reference of ~17.44).
"""

from __future__ import annotations

from typing import Tuple

from elastic_vit_od.models.common import SubnetworkConfig


def grid_from_input(input_h: int, input_w: int, patch_size: int = 16) -> Tuple[int, int]:
    return input_h // patch_size, input_w // patch_size


def estimate_backbone_macs(
    config: SubnetworkConfig,
    embed_dim: int = 768,
    tokens: int = 197,
    max_heads: int = 12,
    patch_size: int = 16,
    in_chans: int = 3,
    include_patch_embed: bool = False,
) -> int:
    """Return MACs for the transformer blocks of the given sub-network.

    ``tokens`` is the sequence length including prefix/cls tokens
    (``grid_h * grid_w + num_prefix``).
    """
    head_dim = embed_dim // max_heads
    total = 0
    for mlp_width, num_heads in zip(config.mlp_widths, config.num_heads):
        attn_dim = num_heads * head_dim
        qkv = 3 * tokens * embed_dim * attn_dim
        attn = 2 * num_heads * tokens * tokens * head_dim
        proj = tokens * attn_dim * embed_dim
        mlp = 2 * tokens * embed_dim * mlp_width
        total += qkv + attn + proj + mlp
    if include_patch_embed:
        num_patches = tokens - 1
        total += num_patches * embed_dim * (in_chans * patch_size * patch_size)
    return int(total)


def estimate_backbone_gmacs(
    config: SubnetworkConfig,
    embed_dim: int = 768,
    grid_h: int = 14,
    grid_w: int = 14,
    num_prefix: int = 1,
    max_heads: int = 12,
    patch_size: int = 16,
    include_patch_embed: bool = False,
) -> float:
    tokens = grid_h * grid_w + num_prefix
    macs = estimate_backbone_macs(
        config,
        embed_dim=embed_dim,
        tokens=tokens,
        max_heads=max_heads,
        patch_size=patch_size,
        include_patch_embed=include_patch_embed,
    )
    return macs / 1e9
