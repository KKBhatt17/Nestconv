from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


DEFAULT_MLP_CHOICES = [768, 1536, 2304, 3072]
DEFAULT_HEAD_CHOICES = [3, 6, 9, 12]


@dataclass
class RouterSoftConfig:
    mlp_probabilities: torch.Tensor
    head_probabilities: torch.Tensor


def build_nested_mask(choice_values: Sequence[int], probabilities: torch.Tensor, max_width: int) -> torch.Tensor:
    if probabilities.ndim == 1:
        mask = torch.zeros(max_width, device=probabilities.device, dtype=probabilities.dtype)
        for probability, width in zip(probabilities, choice_values):
            mask[:width] += probability
        return mask.clamp_(0.0, 1.0)
    if probabilities.ndim == 2:
        mask = torch.zeros(probabilities.shape[0], max_width, device=probabilities.device, dtype=probabilities.dtype)
        for choice_idx, width in enumerate(choice_values):
            mask[:, :width] += probabilities[:, choice_idx].unsqueeze(-1)
        return mask.clamp_(0.0, 1.0)
    raise ValueError("probabilities must have shape [choices] or [batch, choices]")

