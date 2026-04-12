from __future__ import annotations

from elastic_vit.models.common import SubnetworkConfig


def estimate_vit_macs(embed_dim: int, config: SubnetworkConfig, tokens: int = 197) -> int:
    head_dim = embed_dim // 12
    total = 0
    for mlp_width, num_heads in zip(config.mlp_widths, config.num_heads):
        attn_dim = num_heads * head_dim
        qkv = 3 * tokens * embed_dim * attn_dim
        attn = 2 * num_heads * tokens * tokens * head_dim
        proj = tokens * attn_dim * embed_dim
        mlp = 2 * tokens * embed_dim * mlp_width
        total += qkv + attn + proj + mlp
    return int(total)
