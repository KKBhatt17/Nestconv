"""Width-elastic ViT vision tower.

Ported from the Elastic-ViT classification codebase, trimmed for VQA use:

* Subnetworks are realized by **masking** (zeroing inactive heads / MLP neurons),
  not slicing, so the full weights are always present and a single module can be
  evaluated at any preset. This is what the curriculum trainer and the
  fixed-preset evaluator rely on.
* Forward accepts either a hard ``config: SubnetworkConfig`` (curriculum training
  and preset eval) or a soft ``RouterSoftConfig`` of per-choice probabilities.
  The soft path is retained for API compatibility with the original backbone; the
  VQA pipeline only uses the hard path.
* ``forward_tokens`` returns the **full normalized token sequence**
  ``[B, num_tokens, embed_dim]`` for cross-modal fusion. (The original backbone
  pooled to a single vector for classification — VQA fusion needs every token.)

The deployment-only ``StaticVisionTransformer`` / ``export_subnetwork`` path from
the classification codebase is intentionally omitted; VQA eval uses masking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import timm
import torch
import torch.nn as nn

from elastic_vqa.models.common import (
    DEFAULT_HEAD_CHOICES,
    DEFAULT_MLP_CHOICES,
    SubnetworkConfig,
    make_subnetwork_config,
)


def _build_nested_mask(choice_values: Sequence[int], probabilities: torch.Tensor, max_width: int) -> torch.Tensor:
    device = probabilities.device
    mask = torch.zeros(max_width, device=device, dtype=probabilities.dtype)
    for probability, width in zip(probabilities, choice_values):
        mask[:width] += probability
    return mask.clamp_(0.0, 1.0)


@dataclass
class RouterSoftConfig:
    mlp_probabilities: torch.Tensor
    head_probabilities: torch.Tensor


class ElasticAttention(nn.Module):
    def __init__(self, base_attention: nn.Module, embed_dim: int = 768, max_heads: int = 12) -> None:
        super().__init__()
        self.num_heads = max_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // max_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = base_attention.qkv
        self.attn_drop = base_attention.attn_drop
        self.proj = base_attention.proj
        self.proj_drop = base_attention.proj_drop

    def forward(
        self,
        x: torch.Tensor,
        active_heads: Optional[int] = None,
        head_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, num_tokens, _ = x.shape
        qkv = self.qkv(x).reshape(batch_size, num_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if head_mask is None and active_heads is not None:
            head_mask = torch.zeros(self.num_heads, device=x.device, dtype=x.dtype)
            head_mask[:active_heads] = 1.0
        if head_mask is not None:
            shaped_mask = head_mask.view(1, self.num_heads, 1, 1)
            q = q * shaped_mask
            k = k * shaped_mask
            v = v * shaped_mask

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(batch_size, num_tokens, self.embed_dim)

        if head_mask is not None:
            channel_mask = head_mask.view(self.num_heads, 1).expand(self.num_heads, self.head_dim).reshape(self.embed_dim)
            x = x * channel_mask.view(1, 1, self.embed_dim)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class ElasticMlp(nn.Module):
    def __init__(self, base_mlp: nn.Module, max_mlp_width: int = 3072) -> None:
        super().__init__()
        self.max_mlp_width = max_mlp_width
        self.fc1 = base_mlp.fc1
        self.act = base_mlp.act
        self.drop1 = base_mlp.drop1 if hasattr(base_mlp, "drop1") else nn.Identity()
        self.norm = base_mlp.norm if hasattr(base_mlp, "norm") else nn.Identity()
        self.fc2 = base_mlp.fc2
        self.drop2 = base_mlp.drop2 if hasattr(base_mlp, "drop2") else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        active_width: Optional[int] = None,
        mlp_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.drop1(self.act(self.fc1(x)))

        if mlp_mask is None and active_width is not None:
            mlp_mask = torch.zeros(self.max_mlp_width, device=x.device, dtype=x.dtype)
            mlp_mask[:active_width] = 1.0
        if mlp_mask is not None:
            x = x * mlp_mask.view(1, 1, self.max_mlp_width)

        x = self.fc2(self.norm(x))
        x = self.drop2(x)
        return x


class ElasticBlock(nn.Module):
    def __init__(self, base_block: nn.Module, embed_dim: int = 768) -> None:
        super().__init__()
        self.norm1 = base_block.norm1
        self.attn = ElasticAttention(base_block.attn, embed_dim=embed_dim)
        self.ls1 = base_block.ls1 if hasattr(base_block, "ls1") else nn.Identity()
        self.drop_path1 = base_block.drop_path1 if hasattr(base_block, "drop_path1") else nn.Identity()
        self.norm2 = base_block.norm2
        self.mlp = ElasticMlp(base_block.mlp)
        self.ls2 = base_block.ls2 if hasattr(base_block, "ls2") else nn.Identity()
        self.drop_path2 = base_block.drop_path2 if hasattr(base_block, "drop_path2") else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        active_heads: Optional[int] = None,
        active_mlp_width: Optional[int] = None,
        head_mask: Optional[torch.Tensor] = None,
        mlp_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), active_heads=active_heads, head_mask=head_mask)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x), active_width=active_mlp_width, mlp_mask=mlp_mask)))
        return x


class ElasticVisionTransformer(nn.Module):
    """Elastic ViT-B/16 vision tower producing token-level features for fusion."""

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = False,
        checkpoint_path: Optional[str] = None,
        mlp_choices: Optional[Sequence[int]] = None,
        head_choices: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        self.mlp_choices = list(mlp_choices or DEFAULT_MLP_CHOICES)
        self.head_choices = list(head_choices or DEFAULT_HEAD_CHOICES)
        # num_classes=0 makes timm drop the classification head; we only need features.
        base_model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        if checkpoint_path:
            self._load_filtered(base_model, checkpoint_path)

        self.patch_embed = base_model.patch_embed
        self.cls_token = base_model.cls_token
        self.pos_embed = base_model.pos_embed
        self.pos_drop = base_model.pos_drop
        self.blocks = nn.ModuleList([ElasticBlock(block, embed_dim=base_model.embed_dim) for block in base_model.blocks])
        self.norm = base_model.norm
        self.num_prefix_tokens = getattr(base_model, "num_prefix_tokens", 1)
        self.embed_dim = base_model.embed_dim
        self.num_layers = len(self.blocks)

    @staticmethod
    def _load_filtered(base_model: nn.Module, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("model", checkpoint)
        model_state = base_model.state_dict()
        filtered = {
            key: value
            for key, value in state_dict.items()
            if key in model_state and model_state[key].shape == value.shape
        }
        base_model.load_state_dict(filtered, strict=False)

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        return self.pos_drop(x + self.pos_embed)

    def forward_tokens(
        self,
        x: torch.Tensor,
        config: Optional[SubnetworkConfig] = None,
        soft_config: Optional[RouterSoftConfig] = None,
    ) -> torch.Tensor:
        """Return the full normalized token sequence ``[B, num_tokens, embed_dim]``."""
        x = self._embed(x)

        if config is None and soft_config is None:
            config = make_subnetwork_config(max(self.mlp_choices), max(self.head_choices), self.num_layers)

        for layer_idx, block in enumerate(self.blocks):
            if soft_config is not None:
                mlp_mask = _build_nested_mask(self.mlp_choices, soft_config.mlp_probabilities[layer_idx], max(self.mlp_choices))
                head_mask = _build_nested_mask(self.head_choices, soft_config.head_probabilities[layer_idx], max(self.head_choices))
                x = block(x, head_mask=head_mask, mlp_mask=mlp_mask)
            else:
                assert config is not None
                x = block(
                    x,
                    active_heads=config.num_heads[layer_idx],
                    active_mlp_width=config.mlp_widths[layer_idx],
                )

        return self.norm(x)

    def forward(
        self,
        x: torch.Tensor,
        config: Optional[SubnetworkConfig] = None,
        soft_config: Optional[RouterSoftConfig] = None,
    ) -> torch.Tensor:
        return self.forward_tokens(x, config=config, soft_config=soft_config)

    def freeze(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False
