"""Sub-network description + elastic masking helpers.

Self-contained port of the relevant pieces of
``Elastic-ViT/elastic_vit/models/common.py`` so this codebase has no runtime
dependency on the ``elastic_vit`` package.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Sequence

import torch


DEFAULT_MLP_CHOICES = [768, 1536, 2304, 3072]
DEFAULT_HEAD_CHOICES = [3, 6, 9, 12]


@dataclass(frozen=True)
class SubnetworkConfig:
    """A per-layer choice of MLP width and number of attention heads."""

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
    """Build a config from per-layer sequences or scalars broadcast to all layers."""
    return SubnetworkConfig(
        mlp_widths=tuple(broadcast_layer_values(mlp_widths, num_layers)),
        num_heads=tuple(broadcast_layer_values(num_heads, num_layers)),
    )


def largest_config(mlp_choices: Sequence[int], head_choices: Sequence[int], num_layers: int) -> SubnetworkConfig:
    return make_subnetwork_config(max(mlp_choices), max(head_choices), num_layers)


def build_hard_mask(active_width: int, max_width: int, device, dtype) -> torch.Tensor:
    """Nested prefix mask: 1 for the first ``active_width`` units, 0 elsewhere."""
    mask = torch.zeros(max_width, device=device, dtype=dtype)
    mask[:active_width] = 1.0
    return mask


def get_unlocked_choice_count(progress: float, unlock_fractions: Sequence[float]) -> int:
    """Large-to-small curriculum: how many of the (descending) choices are unlocked.

    ``progress`` is training completion in [0, 1]. Ported from
    ``Elastic-ViT/elastic_vit/engine/train_elastic.py``.
    """
    stages = sum(progress >= fraction for fraction in unlock_fractions)
    return max(1, min(stages, len(unlock_fractions)))


def sample_layerwise_subnetwork(
    num_layers: int,
    mlp_choices: Sequence[int],
    head_choices: Sequence[int],
    unlocked_count: int,
    rng: random.Random,
) -> SubnetworkConfig:
    """Sample one per-layer config from the currently unlocked (largest) choices.

    Mirrors ``sample_layerwise_subnetwork`` from Elastic-ViT but takes an explicit
    ``random.Random`` instance so sampling can be made deterministic per training
    step (identical across DDP ranks).
    """
    unlocked_mlp = sorted(mlp_choices, reverse=True)[:unlocked_count]
    unlocked_heads = sorted(head_choices, reverse=True)[:unlocked_count]
    return SubnetworkConfig(
        mlp_widths=tuple(rng.choice(unlocked_mlp) for _ in range(num_layers)),
        num_heads=tuple(rng.choice(unlocked_heads) for _ in range(num_layers)),
    )
