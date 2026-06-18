from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from elastic_vit.models.common import DEFAULT_HEAD_CHOICES, DEFAULT_MLP_CHOICES, SubnetworkConfig
from elastic_vit.models.elastic_vit import RouterSoftConfig


class LayerRouter(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_mlp_choices: int, num_head_choices: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.mlp_head = nn.Linear(hidden_dim, num_mlp_choices)
        self.attn_head = nn.Linear(hidden_dim, num_head_choices)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(inputs)
        return self.mlp_head(features), self.attn_head(features)


class InputAdaptiveRouter(nn.Module):
    def __init__(
        self,
        num_layers: int,
        embed_dim: int = 768,
        hidden_dim: int = 64,
        mlp_choices: list[int] | None = None,
        head_choices: list[int] | None = None,
        input_mode: str = "both",
    ) -> None:
        super().__init__()
        if input_mode not in {"entropy", "hidden", "both"}:
            raise ValueError("input_mode must be one of: entropy, hidden, both")
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.input_mode = input_mode
        self.mlp_choices = mlp_choices or list(DEFAULT_MLP_CHOICES)
        self.head_choices = head_choices or list(DEFAULT_HEAD_CHOICES)

        input_dim = 0
        if input_mode in {"entropy", "both"}:
            input_dim += 1
        if input_mode in {"hidden", "both"}:
            input_dim += embed_dim
        self.layer_routers = nn.ModuleList(
            [
                LayerRouter(input_dim, hidden_dim, len(self.mlp_choices), len(self.head_choices))
                for _ in range(num_layers)
            ]
        )

    def _prepare_inputs(self, entropy: torch.Tensor | None, hidden_state: torch.Tensor | None) -> torch.Tensor:
        features = []
        batch_size = None

        if hidden_state is not None:
            if hidden_state.ndim == 3:
                hidden_state = hidden_state[:, 0]
            if hidden_state.ndim != 2:
                raise ValueError("hidden_state must have shape [batch, embed_dim] or [batch, tokens, embed_dim]")
            batch_size = hidden_state.shape[0]

        if self.input_mode in {"entropy", "both"}:
            if entropy is None:
                raise ValueError(f"entropy input is required for router input_mode={self.input_mode}")
            if entropy.ndim == 0:
                entropy = entropy.view(1, 1)
            elif entropy.ndim == 1:
                entropy = entropy.unsqueeze(-1)
            elif entropy.ndim != 2 or entropy.shape[-1] != 1:
                raise ValueError("entropy must have shape [batch] or [batch, 1]")
            if hidden_state is not None:
                entropy = entropy.to(device=hidden_state.device, dtype=hidden_state.dtype)
            if batch_size is not None and entropy.shape[0] == 1 and batch_size > 1:
                entropy = entropy.expand(batch_size, -1)
            features.append(entropy)
            batch_size = entropy.shape[0]

        if self.input_mode in {"hidden", "both"}:
            if hidden_state is None:
                raise ValueError(f"hidden_state input is required for router input_mode={self.input_mode}")
            features.append(hidden_state)
            batch_size = hidden_state.shape[0]

        if batch_size is None:
            raise ValueError("router received no usable inputs")
        return torch.cat(features, dim=-1)

    def forward_layer(
        self,
        layer_idx: int,
        entropy: torch.Tensor | None = None,
        hidden_state: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs = self._prepare_inputs(entropy, hidden_state)
        return self.layer_routers[layer_idx](inputs)

    def forward(
        self,
        entropy: torch.Tensor | None = None,
        hidden_state: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs = self._prepare_inputs(entropy, hidden_state)
        mlp_logits = []
        head_logits = []
        for layer_router in self.layer_routers:
            layer_mlp_logits, layer_head_logits = layer_router(inputs)
            mlp_logits.append(layer_mlp_logits)
            head_logits.append(layer_head_logits)
        mlp_logits = torch.stack(mlp_logits, dim=1)
        head_logits = torch.stack(head_logits, dim=1)
        return mlp_logits, head_logits

    def sample_layer_config(
        self,
        layer_idx: int,
        entropy: torch.Tensor | None,
        hidden_state: torch.Tensor | None,
        temperature: float,
        hard: bool = False,
        deterministic: bool = False,
    ) -> Tuple[RouterSoftConfig, Dict[str, torch.Tensor]]:
        mlp_logits, head_logits = self.forward_layer(layer_idx, entropy=entropy, hidden_state=hidden_state)
        if deterministic:
            mlp_probs = F.one_hot(mlp_logits.argmax(dim=-1), num_classes=len(self.mlp_choices)).to(mlp_logits.dtype)
            head_probs = F.one_hot(head_logits.argmax(dim=-1), num_classes=len(self.head_choices)).to(head_logits.dtype)
        else:
            mlp_probs = F.gumbel_softmax(mlp_logits, tau=temperature, hard=hard, dim=-1)
            head_probs = F.gumbel_softmax(head_logits, tau=temperature, hard=hard, dim=-1)
        soft_config = RouterSoftConfig(
            mlp_probabilities=mlp_probs,
            head_probabilities=head_probs,
        )
        return soft_config, {
            "mlp_logits": mlp_logits,
            "head_logits": head_logits,
            "mlp_probs": mlp_probs,
            "head_probs": head_probs,
        }

    def sample_relaxed_config(
        self,
        entropy: torch.Tensor | None,
        temperature: float,
        hard: bool = False,
        hidden_state: torch.Tensor | None = None,
    ) -> Tuple[RouterSoftConfig, Dict[str, torch.Tensor]]:
        mlp_logits, head_logits = self(entropy=entropy, hidden_state=hidden_state)
        mlp_probs = F.gumbel_softmax(mlp_logits, tau=temperature, hard=hard, dim=-1)
        head_probs = F.gumbel_softmax(head_logits, tau=temperature, hard=hard, dim=-1)
        soft_config = RouterSoftConfig(mlp_probabilities=mlp_probs, head_probabilities=head_probs)
        return soft_config, {
            "mlp_logits": mlp_logits,
            "head_logits": head_logits,
            "mlp_probs": mlp_probs,
            "head_probs": head_probs,
        }

    def predict_hard_config(
        self,
        entropy: torch.Tensor | None = None,
        hidden_state: torch.Tensor | None = None,
    ) -> SubnetworkConfig:
        if self.input_mode != "entropy":
            raise ValueError("Full hard configs can be predicted before the ViT pass only in entropy-only mode")
        mlp_logits, head_logits = self(entropy=entropy, hidden_state=hidden_state)
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
    head_dim = embed_dim // max(DEFAULT_HEAD_CHOICES)
    mlp_costs = torch.tensor(
        [embed_dim * width + width + width * embed_dim + embed_dim for width in mlp_choices],
        device=mlp_probabilities.device,
        dtype=mlp_probabilities.dtype,
    )
    head_costs = torch.tensor(
        [
            3 * (heads * head_dim) * embed_dim
            + 3 * (heads * head_dim)
            + embed_dim * (heads * head_dim)
            + embed_dim
            for heads in head_choices
        ],
        device=head_probabilities.device,
        dtype=head_probabilities.dtype,
    )
    if mlp_probabilities.ndim == 2:
        mlp_expected = (mlp_probabilities * mlp_costs.view(1, -1)).sum(dim=-1)
        head_expected = (head_probabilities * head_costs.view(1, -1)).sum(dim=-1)
        return (mlp_expected + head_expected).sum()
    if mlp_probabilities.ndim == 3:
        mlp_expected = (mlp_probabilities * mlp_costs.view(1, 1, -1)).sum(dim=-1)
        head_expected = (head_probabilities * head_costs.view(1, 1, -1)).sum(dim=-1)
        return (mlp_expected + head_expected).sum(dim=-1).mean()
    raise ValueError("router probabilities must have shape [layers, choices] or [batch, layers, choices]")


EntropyRouter = InputAdaptiveRouter
