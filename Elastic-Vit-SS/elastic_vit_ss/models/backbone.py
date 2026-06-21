"""Elastic ViT backbone for MMSegmentation decode heads.

A weight-shared elastic ``vit_base_patch16_224`` (heads/MLP neurons importance
ordered by Elastic-ViT stage-1, fine-tuned elastically in stage-2). Any prefix
sub-network is realized by masking. The active sub-network is selected explicitly
through :meth:`set_active_config` (no router) — during training the
``ElasticConfigHook`` sets a sampled config each iteration, during evaluation the
``eval_configs`` tool sets each fixed config in turn.

Outputs four single-stride feature maps resampled to strides {4, 8, 16, 32} as the
4-level pyramid expected by UPerHead.
"""

from __future__ import annotations

from typing import Sequence

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmseg.registry import MODELS

from .common import (
    DEFAULT_HEAD_CHOICES,
    DEFAULT_MLP_CHOICES,
    SubnetworkConfig,
    build_hard_mask,
    largest_config,
    make_subnetwork_config,
)


class ElasticAttention(nn.Module):
    """timm ViT attention with a per-head prefix mask."""

    def __init__(self, base_attention: nn.Module, embed_dim: int = 768, max_heads: int = 12) -> None:
        super().__init__()
        self.num_heads = max_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // max_heads
        self.scale = self.head_dim**-0.5
        self.qkv = base_attention.qkv
        self.attn_drop = base_attention.attn_drop
        self.proj = base_attention.proj
        self.proj_drop = base_attention.proj_drop

    def forward(self, x: torch.Tensor, head_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, num_tokens, _ = x.shape
        qkv = self.qkv(x).reshape(batch_size, num_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

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

        return self.proj_drop(self.proj(x))


class ElasticMlp(nn.Module):
    """timm ViT MLP with a prefix mask on the hidden width."""

    def __init__(self, base_mlp: nn.Module, max_mlp_width: int = 3072) -> None:
        super().__init__()
        self.max_mlp_width = max_mlp_width
        self.fc1 = base_mlp.fc1
        self.act = base_mlp.act
        self.drop1 = base_mlp.drop1 if hasattr(base_mlp, "drop1") else nn.Identity()
        self.norm = base_mlp.norm if hasattr(base_mlp, "norm") else nn.Identity()
        self.fc2 = base_mlp.fc2
        self.drop2 = base_mlp.drop2 if hasattr(base_mlp, "drop2") else nn.Identity()

    def forward(self, x: torch.Tensor, mlp_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.drop1(self.act(self.fc1(x)))
        if mlp_mask is not None:
            x = x * mlp_mask.view(1, 1, self.max_mlp_width)
        return self.drop2(self.fc2(self.norm(x)))


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

    def forward(self, x: torch.Tensor, head_mask: torch.Tensor | None = None, mlp_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), head_mask=head_mask)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x), mlp_mask=mlp_mask)))
        return x


@MODELS.register_module()
class ElasticViTBackbone(BaseModule):
    """Weight-shared elastic ViT backbone producing a 4-level feature pyramid."""

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = False,
        checkpoint_path: str | None = None,
        mlp_choices: Sequence[int] = DEFAULT_MLP_CHOICES,
        head_choices: Sequence[int] = DEFAULT_HEAD_CHOICES,
        out_indices: Sequence[int] = (2, 5, 8, 11),
        pyramid_scales: Sequence[float] = (4.0, 2.0, 1.0, 0.5),
        frozen_stages: int = -1,
        freeze_vit: bool = False,
        init_cfg: dict | None = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.mlp_choices = list(mlp_choices)
        self.head_choices = list(head_choices)
        self.out_indices = tuple(out_indices)
        self.pyramid_scales = tuple(pyramid_scales)
        self.frozen_stages = frozen_stages
        self.freeze_vit = freeze_vit

        base_model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.patch_embed = base_model.patch_embed
        self.cls_token = base_model.cls_token
        self.pos_embed = base_model.pos_embed
        self.pos_drop = base_model.pos_drop
        self.blocks = nn.ModuleList([ElasticBlock(block, embed_dim=base_model.embed_dim) for block in base_model.blocks])
        self.norm = base_model.norm
        self.embed_dim = base_model.embed_dim
        self.num_layers = len(self.blocks)
        self.num_prefix_tokens = getattr(base_model, "num_prefix_tokens", 1)

        # Active sub-network; defaults to the largest (full) model until set.
        self.active_config: SubnetworkConfig = largest_config(self.mlp_choices, self.head_choices, self.num_layers)

        if checkpoint_path:
            self._load_filtered_checkpoint(checkpoint_path)
        self._apply_freezing()

    # ---- sub-network control -------------------------------------------------
    def set_active_config(self, mlp_widths, num_heads) -> None:
        """Select the sub-network. Accepts per-layer sequences or broadcastable scalars."""
        self.active_config = make_subnetwork_config(mlp_widths, num_heads, self.num_layers)

    # ---- checkpoint ----------------------------------------------------------
    def _load_filtered_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("model", checkpoint)
        own_state = self.state_dict()
        filtered = {
            key: value
            for key, value in state_dict.items()
            if key in own_state and own_state[key].shape == value.shape
        }
        missing, unexpected = self.load_state_dict(filtered, strict=False)
        print(
            f"[ElasticViTBackbone] loaded {len(filtered)}/{len(own_state)} tensors from "
            f"{checkpoint_path} (missing={len(missing)}, unexpected={len(unexpected)})"
        )

    # ---- freezing ------------------------------------------------------------
    @staticmethod
    def _freeze_module(module: nn.Module) -> None:
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad = False

    def _apply_freezing(self) -> None:
        if self.freeze_vit:
            for module in [self.patch_embed, self.pos_drop, self.blocks, self.norm]:
                self._freeze_module(module)
            self.cls_token.requires_grad = False
            self.pos_embed.requires_grad = False
            return
        if self.frozen_stages < 0:
            return
        for module in [self.patch_embed, self.pos_drop]:
            self._freeze_module(module)
        self.cls_token.requires_grad = False
        self.pos_embed.requires_grad = False
        for block in self.blocks[: self.frozen_stages]:
            self._freeze_module(block)

    def train(self, mode: bool = True) -> "ElasticViTBackbone":
        super().train(mode)
        self._apply_freezing()
        return self

    # ---- helpers -------------------------------------------------------------
    def _tokens_to_map(self, x: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
        patch_tokens = x[:, self.num_prefix_tokens :]
        return patch_tokens.transpose(1, 2).reshape(x.shape[0], self.embed_dim, grid_h, grid_w).contiguous()

    def _resize_pos_embed(self, grid_h: int, grid_w: int) -> torch.Tensor:
        prefix = self.pos_embed[:, : self.num_prefix_tokens]
        pos_tokens = self.pos_embed[:, self.num_prefix_tokens :]
        old_size = int(round(pos_tokens.shape[1] ** 0.5))
        if old_size * old_size == pos_tokens.shape[1] and (old_size, old_size) == (grid_h, grid_w):
            return self.pos_embed
        pos_tokens = pos_tokens.reshape(1, old_size, old_size, self.embed_dim).permute(0, 3, 1, 2)
        pos_tokens = F.interpolate(pos_tokens, size=(grid_h, grid_w), mode="bicubic", align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(1, grid_h * grid_w, self.embed_dim)
        return torch.cat([prefix, pos_tokens], dim=1)

    # ---- forward -------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        batch_size, _, input_h, input_w = x.shape
        patch_size = self.patch_embed.patch_size
        patch_h, patch_w = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        grid_h, grid_w = input_h // patch_h, input_w // patch_w

        x = self.patch_embed(x)
        if x.ndim == 4:  # timm may return (B, C, H, W)
            x = x.flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x + self._resize_pos_embed(grid_h, grid_w).to(dtype=x.dtype, device=x.device))

        config = self.active_config
        max_mlp = max(self.mlp_choices)
        max_heads = max(self.head_choices)

        outputs = []
        for layer_idx, block in enumerate(self.blocks):
            mlp_width = config.mlp_widths[layer_idx]
            num_heads = config.num_heads[layer_idx]
            mlp_mask = build_hard_mask(mlp_width, max_mlp, x.device, x.dtype) if mlp_width < max_mlp else None
            head_mask = build_hard_mask(num_heads, max_heads, x.device, x.dtype) if num_heads < max_heads else None
            x = block(x, head_mask=head_mask, mlp_mask=mlp_mask)
            if layer_idx in self.out_indices:
                feature = self._tokens_to_map(self.norm(x), grid_h, grid_w)
                scale = self.pyramid_scales[len(outputs)]
                if scale != 1.0:
                    feature = F.interpolate(feature, scale_factor=scale, mode="bilinear", align_corners=False)
                outputs.append(feature)

        return tuple(outputs)
