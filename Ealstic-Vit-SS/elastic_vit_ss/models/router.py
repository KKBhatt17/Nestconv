from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import DEFAULT_HEAD_CHOICES, DEFAULT_MLP_CHOICES, RouterSoftConfig


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

        if self.input_mode in {"hidden", "both"}:
            if hidden_state is None:
                raise ValueError(f"hidden_state input is required for router input_mode={self.input_mode}")
            features.append(hidden_state)

        if not features:
            raise ValueError("router received no usable inputs")
        return torch.cat(features, dim=-1)

    def sample_layer_config(
        self,
        layer_idx: int,
        entropy: torch.Tensor | None,
        hidden_state: torch.Tensor | None,
        temperature: float,
        hard: bool = False,
        deterministic: bool = False,
    ) -> tuple[RouterSoftConfig, Dict[str, torch.Tensor]]:
        inputs = self._prepare_inputs(entropy, hidden_state)
        mlp_logits, head_logits = self.layer_routers[layer_idx](inputs)
        if deterministic:
            mlp_probs = F.one_hot(mlp_logits.argmax(dim=-1), num_classes=len(self.mlp_choices)).to(mlp_logits.dtype)
            head_probs = F.one_hot(head_logits.argmax(dim=-1), num_classes=len(self.head_choices)).to(head_logits.dtype)
        else:
            mlp_probs = F.gumbel_softmax(mlp_logits, tau=temperature, hard=hard, dim=-1)
            head_probs = F.gumbel_softmax(head_logits, tau=temperature, hard=hard, dim=-1)
        return RouterSoftConfig(mlp_probs, head_probs), {
            "mlp_logits": mlp_logits,
            "head_logits": head_logits,
            "mlp_probs": mlp_probs,
            "head_probs": head_probs,
        }

