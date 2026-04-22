from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from elastic_vit_retrieval.models.common import DEFAULT_HEAD_CHOICES, DEFAULT_MLP_CHOICES, SubnetworkConfig
from elastic_vit_retrieval.models.elastic_clip import RouterSoftConfig


class EntropyRouter(nn.Module):
    def __init__(
        self,
        num_layers: int,
        hidden_dim: int = 64,
        mlp_choices: list[int] | None = None,
        head_choices: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.mlp_choices = mlp_choices or list(DEFAULT_MLP_CHOICES)
        self.head_choices = head_choices or list(DEFAULT_HEAD_CHOICES)
        self.backbone = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.mlp_head = nn.Linear(hidden_dim, num_layers * len(self.mlp_choices))
        self.attn_head = nn.Linear(hidden_dim, num_layers * len(self.head_choices))

    def forward(self, mean_entropy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if mean_entropy.ndim == 0:
            mean_entropy = mean_entropy.unsqueeze(0)
        if mean_entropy.ndim == 1:
            mean_entropy = mean_entropy.unsqueeze(-1)
        features = self.backbone(mean_entropy)
        mlp_logits = self.mlp_head(features).view(-1, self.num_layers, len(self.mlp_choices))
        head_logits = self.attn_head(features).view(-1, self.num_layers, len(self.head_choices))
        return mlp_logits, head_logits

    def sample_relaxed_config(
        self,
        mean_entropy: torch.Tensor,
        temperature: float,
        hard: bool = False,
    ) -> Tuple[RouterSoftConfig, Dict[str, torch.Tensor]]:
        mlp_logits, head_logits = self(mean_entropy)
        mlp_probs = F.gumbel_softmax(mlp_logits, tau=temperature, hard=hard, dim=-1)
        head_probs = F.gumbel_softmax(head_logits, tau=temperature, hard=hard, dim=-1)
        soft_config = RouterSoftConfig(
            mlp_probabilities=mlp_probs.mean(dim=0),
            head_probabilities=head_probs.mean(dim=0),
        )
        return soft_config, {
            "mlp_logits": mlp_logits,
            "head_logits": head_logits,
            "mlp_probs": mlp_probs,
            "head_probs": head_probs,
        }

    def predict_hard_config(self, mean_entropy: torch.Tensor) -> SubnetworkConfig:
        mlp_logits, head_logits = self(mean_entropy)
        mlp_index = mlp_logits.argmax(dim=-1).squeeze(0)
        head_index = head_logits.argmax(dim=-1).squeeze(0)
        mlp_widths = tuple(self.mlp_choices[index.item()] for index in mlp_index)
        num_heads = tuple(self.head_choices[index.item()] for index in head_index)
        return SubnetworkConfig(mlp_widths=mlp_widths, num_heads=num_heads)


def expected_router_param_cost(
    mlp_probabilities: torch.Tensor,
    head_probabilities: torch.Tensor,
    embed_dim: int,
    mlp_choices: list[int],
    head_choices: list[int],
) -> torch.Tensor:
    head_dim = embed_dim // max(head_choices)
    mlp_costs = torch.tensor(
        [embed_dim * width + width + width * embed_dim + embed_dim for width in mlp_choices],
        device=mlp_probabilities.device,
        dtype=mlp_probabilities.dtype,
    )
    head_costs = torch.tensor(
        [4 * embed_dim * (heads * head_dim) + 4 * (heads * head_dim) for heads in head_choices],
        device=head_probabilities.device,
        dtype=head_probabilities.dtype,
    )
    mlp_expected = (mlp_probabilities * mlp_costs.view(1, -1)).sum(dim=-1)
    head_expected = (head_probabilities * head_costs.view(1, -1)).sum(dim=-1)
    return (mlp_expected + head_expected).sum()
