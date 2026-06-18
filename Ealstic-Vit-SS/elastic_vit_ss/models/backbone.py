from __future__ import annotations

from pathlib import Path
from typing import Sequence

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmseg.registry import MODELS

from elastic_vit_ss.utils import compute_batch_entropy

from .common import DEFAULT_HEAD_CHOICES, DEFAULT_MLP_CHOICES, build_nested_mask
from .router import InputAdaptiveRouter


class ElasticAttention(nn.Module):
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
            if head_mask.ndim == 1:
                shaped_mask = head_mask.view(1, self.num_heads, 1, 1)
            elif head_mask.ndim == 2:
                shaped_mask = head_mask.view(batch_size, self.num_heads, 1, 1)
            else:
                raise ValueError("head_mask must have shape [heads] or [batch, heads]")
            q = q * shaped_mask
            k = k * shaped_mask
            v = v * shaped_mask

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(batch_size, num_tokens, self.embed_dim)

        if head_mask is not None:
            if head_mask.ndim == 1:
                channel_mask = head_mask.view(self.num_heads, 1).expand(self.num_heads, self.head_dim).reshape(self.embed_dim)
                x = x * channel_mask.view(1, 1, self.embed_dim)
            else:
                channel_mask = head_mask.view(batch_size, self.num_heads, 1)
                channel_mask = channel_mask.expand(batch_size, self.num_heads, self.head_dim).reshape(batch_size, self.embed_dim)
                x = x * channel_mask.view(batch_size, 1, self.embed_dim)

        return self.proj_drop(self.proj(x))


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

    def forward(self, x: torch.Tensor, mlp_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.drop1(self.act(self.fc1(x)))
        if mlp_mask is not None:
            if mlp_mask.ndim == 1:
                x = x * mlp_mask.view(1, 1, self.max_mlp_width)
            elif mlp_mask.ndim == 2:
                x = x * mlp_mask.view(x.shape[0], 1, self.max_mlp_width)
            else:
                raise ValueError("mlp_mask must have shape [width] or [batch, width]")
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
    """Input-adaptive elastic ViT backbone for MMSegmentation decode heads."""

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = False,
        checkpoint_path: str | None = None,
        router_checkpoint: str | None = None,
        router_input_mode: str = "both",
        router_hidden_dim: int = 64,
        mlp_choices: Sequence[int] = DEFAULT_MLP_CHOICES,
        head_choices: Sequence[int] = DEFAULT_HEAD_CHOICES,
        out_indices: Sequence[int] = (2, 5, 8, 11),
        pyramid_scales: Sequence[float] = (4.0, 2.0, 1.0, 0.5),
        router_temperature: float = 1.0,
        frozen_stages: int = -1,
        freeze_vit: bool = False,
        freeze_router: bool = False,
        init_cfg: dict | None = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.mlp_choices = list(mlp_choices)
        self.head_choices = list(head_choices)
        self.out_indices = tuple(out_indices)
        self.pyramid_scales = tuple(pyramid_scales)
        self.router_temperature = router_temperature
        self.frozen_stages = frozen_stages
        self.freeze_vit = freeze_vit
        self.freeze_router = freeze_router

        base_model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.patch_embed = base_model.patch_embed
        self.cls_token = base_model.cls_token
        self.pos_embed = base_model.pos_embed
        self.pos_drop = base_model.pos_drop
        self.blocks = nn.ModuleList([ElasticBlock(block, embed_dim=base_model.embed_dim) for block in base_model.blocks])
        self.norm = base_model.norm
        self.embed_dim = base_model.embed_dim
        self.num_prefix_tokens = getattr(base_model, "num_prefix_tokens", 1)

        self.router = InputAdaptiveRouter(
            num_layers=len(self.blocks),
            embed_dim=self.embed_dim,
            hidden_dim=router_hidden_dim,
            mlp_choices=self.mlp_choices,
            head_choices=self.head_choices,
            input_mode=router_input_mode,
        )
        if checkpoint_path:
            self._load_filtered_checkpoint(checkpoint_path)
        if router_checkpoint:
            self._load_router_checkpoint(router_checkpoint)
        self._apply_freezing()

    def _load_filtered_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("model", checkpoint)
        own_state = self.state_dict()
        filtered = {key: value for key, value in state_dict.items() if key in own_state and own_state[key].shape == value.shape}
        self.load_state_dict(filtered, strict=False)

    def _load_router_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("model", checkpoint)
        self.router.load_state_dict(state_dict, strict=False)

    @staticmethod
    def _freeze_module(module: nn.Module) -> None:
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad = False

    def _freeze_vit_parameters(self) -> None:
        for module in [self.patch_embed, self.pos_drop, self.blocks, self.norm]:
            self._freeze_module(module)
        self.cls_token.requires_grad = False
        self.pos_embed.requires_grad = False

    def _apply_freezing(self) -> None:
        if self.freeze_vit:
            self._freeze_vit_parameters()
        if self.freeze_router:
            self._freeze_module(self.router)
        if self.frozen_stages < 0:
            return
        for module in [self.patch_embed, self.pos_drop]:
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad = False
        self.cls_token.requires_grad = False
        self.pos_embed.requires_grad = False
        for block in self.blocks[: self.frozen_stages]:
            block.eval()
            for parameter in block.parameters():
                parameter.requires_grad = False

    def train(self, mode: bool = True) -> "ElasticViTBackbone":
        super().train(mode)
        self._apply_freezing()
        return self

    def _tokens_to_map(self, x: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
        patch_tokens = x[:, self.num_prefix_tokens :]
        return patch_tokens.transpose(1, 2).reshape(x.shape[0], self.embed_dim, grid_h, grid_w).contiguous()

    def _resize_pos_embed(self, grid_h: int, grid_w: int) -> torch.Tensor:
        prefix = self.pos_embed[:, : self.num_prefix_tokens]
        pos_tokens = self.pos_embed[:, self.num_prefix_tokens :]
        old_size = int(pos_tokens.shape[1] ** 0.5)
        pos_tokens = pos_tokens.reshape(1, old_size, old_size, self.embed_dim).permute(0, 3, 1, 2)
        pos_tokens = F.interpolate(pos_tokens, size=(grid_h, grid_w), mode="bicubic", align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(1, grid_h * grid_w, self.embed_dim)
        return torch.cat([prefix, pos_tokens], dim=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        input_images = x
        batch_size, _, input_h, input_w = x.shape
        patch_size = self.patch_embed.patch_size
        if isinstance(patch_size, int):
            patch_h = patch_w = patch_size
        else:
            patch_h, patch_w = patch_size
        grid_h, grid_w = input_h // patch_h, input_w // patch_w

        entropy = compute_batch_entropy(input_images) if self.router.input_mode in {"entropy", "both"} else None
        x = self.patch_embed(x)
        if x.ndim == 4:
            x = x.flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x + self._resize_pos_embed(grid_h, grid_w).to(dtype=x.dtype, device=x.device))

        outputs = []
        for layer_idx, block in enumerate(self.blocks):
            soft_config, _ = self.router.sample_layer_config(
                layer_idx,
                entropy=entropy,
                hidden_state=x[:, 0],
                temperature=self.router_temperature,
                hard=False,
                deterministic=not self.training,
            )
            mlp_mask = build_nested_mask(self.mlp_choices, soft_config.mlp_probabilities, max(self.mlp_choices))
            head_mask = build_nested_mask(self.head_choices, soft_config.head_probabilities, max(self.head_choices))
            x = block(x, head_mask=head_mask, mlp_mask=mlp_mask)
            if layer_idx in self.out_indices:
                feature = self._tokens_to_map(self.norm(x), grid_h, grid_w)
                scale = self.pyramid_scales[len(outputs)]
                if scale != 1.0:
                    feature = F.interpolate(feature, scale_factor=scale, mode="bilinear", align_corners=False)
                outputs.append(feature)

        return tuple(outputs)
