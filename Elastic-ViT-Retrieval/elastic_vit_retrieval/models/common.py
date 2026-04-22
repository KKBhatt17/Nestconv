from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence


DEFAULT_MLP_CHOICES = [768, 1536, 2304, 3072]
DEFAULT_HEAD_CHOICES = [3, 6, 9, 12]


@dataclass(frozen=True)
class SubnetworkConfig:
    mlp_widths: tuple[int, ...]
    num_heads: tuple[int, ...]

    @property
    def num_layers(self) -> int:
        return len(self.mlp_widths)

    def as_key(self) -> str:
        mlp = "-".join(str(value) for value in self.mlp_widths)
        heads = "-".join(str(value) for value in self.num_heads)
        return f"mlp:{mlp}|heads:{heads}"


def broadcast_layer_values(values: Sequence[int] | int, num_layers: int) -> List[int]:
    if isinstance(values, int):
        return [values] * num_layers
    if len(values) != num_layers:
        raise ValueError(f"Expected {num_layers} values, got {len(values)}")
    return list(values)


def make_subnetwork_config(
    mlp_widths: Sequence[int] | int,
    num_heads: Sequence[int] | int,
    num_layers: int,
) -> SubnetworkConfig:
    return SubnetworkConfig(
        mlp_widths=tuple(broadcast_layer_values(mlp_widths, num_layers)),
        num_heads=tuple(broadcast_layer_values(num_heads, num_layers)),
    )


def sorted_global_subnetworks(
    num_layers: int,
    mlp_choices: Iterable[int] = DEFAULT_MLP_CHOICES,
    head_choices: Iterable[int] = DEFAULT_HEAD_CHOICES,
) -> List[SubnetworkConfig]:
    configs = [
        make_subnetwork_config(mlp, heads, num_layers)
        for mlp in mlp_choices
        for heads in head_choices
    ]
    return sorted(configs, key=estimate_subnetwork_cost, reverse=True)


def estimate_block_params(embed_dim: int, mlp_width: int, num_heads: int) -> int:
    head_dim = embed_dim // max(DEFAULT_HEAD_CHOICES)
    attn_dim = num_heads * head_dim
    attn_params = 4 * embed_dim * attn_dim + 4 * attn_dim
    mlp_params = embed_dim * mlp_width + mlp_width + mlp_width * embed_dim + embed_dim
    return attn_params + mlp_params


def estimate_subnetwork_cost(config: SubnetworkConfig, embed_dim: int = 768) -> int:
    return sum(
        estimate_block_params(embed_dim, mlp_width, heads)
        for mlp_width, heads in zip(config.mlp_widths, config.num_heads)
    )
