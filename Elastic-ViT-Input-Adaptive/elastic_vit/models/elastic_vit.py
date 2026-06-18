from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import timm
import torch
import torch.nn as nn

from elastic_vit.models.common import (
    DEFAULT_HEAD_CHOICES,
    DEFAULT_MLP_CHOICES,
    SubnetworkConfig,
    make_subnetwork_config,
)


def _build_nested_mask(choice_values: Sequence[int], probabilities: torch.Tensor, max_width: int) -> torch.Tensor:
    device = probabilities.device
    if probabilities.ndim == 1:
        mask = torch.zeros(max_width, device=device, dtype=probabilities.dtype)
        for probability, width in zip(probabilities, choice_values):
            mask[:width] += probability
        return mask.clamp_(0.0, 1.0)
    if probabilities.ndim == 2:
        mask = torch.zeros(probabilities.shape[0], max_width, device=device, dtype=probabilities.dtype)
        for choice_idx, width in enumerate(choice_values):
            mask[:, :width] += probabilities[:, choice_idx].unsqueeze(-1)
        return mask.clamp_(0.0, 1.0)
    raise ValueError("probabilities must have shape [choices] or [batch, choices]")


def _layer_probabilities(probabilities: torch.Tensor, layer_idx: int) -> torch.Tensor:
    if probabilities.ndim == 2:
        return probabilities[layer_idx]
    if probabilities.ndim == 3:
        return probabilities[:, layer_idx]
    raise ValueError("router probabilities must have shape [layers, choices] or [batch, layers, choices]")


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
            if mlp_mask.ndim == 1:
                x = x * mlp_mask.view(1, 1, self.max_mlp_width)
            elif mlp_mask.ndim == 2:
                x = x * mlp_mask.view(x.shape[0], 1, self.max_mlp_width)
            else:
                raise ValueError("mlp_mask must have shape [width] or [batch, width]")

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


class StaticAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // max(DEFAULT_HEAD_CHOICES)
        self.active_dim = self.num_heads * self.head_dim
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, self.active_dim * 3)
        self.proj = nn.Linear(self.active_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, _ = x.shape
        qkv = self.qkv(x).reshape(batch_size, num_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(batch_size, num_tokens, self.active_dim)
        return self.proj(x)


class StaticMlp(nn.Module):
    def __init__(self, embed_dim: int, mlp_width: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, mlp_width)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mlp_width, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class StaticBlock(nn.Module):
    def __init__(self, source_block: ElasticBlock, embed_dim: int, num_heads: int, mlp_width: int) -> None:
        super().__init__()
        self.norm1 = source_block.norm1
        self.attn = StaticAttention(embed_dim, num_heads)
        self.norm2 = source_block.norm2
        self.mlp = StaticMlp(embed_dim, mlp_width)
        self.drop_path1 = source_block.drop_path1
        self.drop_path2 = source_block.drop_path2
        self.ls1 = source_block.ls1
        self.ls2 = source_block.ls2
        self._copy_from_elastic(source_block, num_heads, mlp_width)

    def _copy_from_elastic(self, source_block: ElasticBlock, num_heads: int, mlp_width: int) -> None:
        head_dim = source_block.attn.head_dim
        active_dim = num_heads * head_dim
        with torch.no_grad():
            self.attn.qkv.weight.copy_(source_block.attn.qkv.weight[: 3 * active_dim])
            self.attn.qkv.bias.copy_(source_block.attn.qkv.bias[: 3 * active_dim])
            self.attn.proj.weight.copy_(source_block.attn.proj.weight[:, :active_dim])
            self.attn.proj.bias.copy_(source_block.attn.proj.bias)
            self.mlp.fc1.weight.copy_(source_block.mlp.fc1.weight[:mlp_width])
            self.mlp.fc1.bias.copy_(source_block.mlp.fc1.bias[:mlp_width])
            self.mlp.fc2.weight.copy_(source_block.mlp.fc2.weight[:, :mlp_width])
            self.mlp.fc2.bias.copy_(source_block.mlp.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class StaticVisionTransformer(nn.Module):
    def __init__(self, source_model: "ElasticVisionTransformer", config: SubnetworkConfig) -> None:
        super().__init__()
        self.patch_embed = source_model.patch_embed
        self.cls_token = source_model.cls_token
        self.pos_embed = source_model.pos_embed
        self.pos_drop = source_model.pos_drop
        self.norm = source_model.norm
        self.fc_norm = source_model.fc_norm
        self.head_drop = source_model.head_drop
        self.head = source_model.head
        self.global_pool = source_model.global_pool
        self.embed_dim = source_model.embed_dim
        self.num_prefix_tokens = source_model.num_prefix_tokens
        self.blocks = nn.ModuleList(
            [
                StaticBlock(block, self.embed_dim, num_heads, mlp_width)
                for block, num_heads, mlp_width in zip(source_model.blocks, config.num_heads, config.mlp_widths)
            ]
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        if self.global_pool == "avg":
            x = x[:, self.num_prefix_tokens :].mean(dim=1)
            return self.fc_norm(x)
        return x[:, 0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.head_drop(self.forward_features(x))
        return self.head(x)


class ElasticVisionTransformer(nn.Module):
    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = False,
        num_classes: int = 1000,
        checkpoint_path: Optional[str] = None,
        mlp_choices: Optional[Sequence[int]] = None,
        head_choices: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        self.mlp_choices = list(mlp_choices or DEFAULT_MLP_CHOICES)
        self.head_choices = list(head_choices or DEFAULT_HEAD_CHOICES)
        base_model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            state_dict = checkpoint.get("model", checkpoint)
            filtered_state_dict = {}
            model_state = base_model.state_dict()
            for key, value in state_dict.items():
                if key in model_state and model_state[key].shape == value.shape:
                    filtered_state_dict[key] = value
            base_model.load_state_dict(filtered_state_dict, strict=False)

        self.patch_embed = base_model.patch_embed
        self.cls_token = base_model.cls_token
        self.pos_embed = base_model.pos_embed
        self.pos_drop = base_model.pos_drop
        self.blocks = nn.ModuleList([ElasticBlock(block, embed_dim=base_model.embed_dim) for block in base_model.blocks])
        self.norm = base_model.norm
        self.fc_norm = getattr(base_model, "fc_norm", nn.Identity())
        self.head_drop = getattr(base_model, "head_drop", nn.Identity())
        self.head = base_model.head
        self.global_pool = getattr(base_model, "global_pool", "token")
        self.num_prefix_tokens = getattr(base_model, "num_prefix_tokens", 1)
        self.embed_dim = base_model.embed_dim
        self.num_layers = len(self.blocks)
        self.export_cache: Dict[str, StaticVisionTransformer] = {}

    def forward_features(
        self,
        x: torch.Tensor,
        config: Optional[SubnetworkConfig] = None,
        soft_config: Optional[RouterSoftConfig] = None,
        adaptive_router: Optional[nn.Module] = None,
        entropy: Optional[torch.Tensor] = None,
        router_temperature: float = 1.0,
        router_hard: bool = False,
        deterministic_routing: bool = False,
        return_routing: bool = False,
    ) -> torch.Tensor:
        if adaptive_router is not None and (config is not None or soft_config is not None):
            raise ValueError("adaptive_router cannot be combined with config or soft_config")

        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        if config is None and soft_config is None and adaptive_router is None:
            config = make_subnetwork_config(max(self.mlp_choices), max(self.head_choices), self.num_layers)

        routing = {"mlp_probabilities": [], "head_probabilities": []}
        for layer_idx, block in enumerate(self.blocks):
            if adaptive_router is not None:
                layer_soft_config, _ = adaptive_router.sample_layer_config(
                    layer_idx,
                    entropy=entropy,
                    hidden_state=x[:, 0],
                    temperature=router_temperature,
                    hard=router_hard,
                    deterministic=deterministic_routing,
                )
                routing["mlp_probabilities"].append(layer_soft_config.mlp_probabilities)
                routing["head_probabilities"].append(layer_soft_config.head_probabilities)
                mlp_mask = _build_nested_mask(
                    self.mlp_choices,
                    layer_soft_config.mlp_probabilities,
                    max(self.mlp_choices),
                )
                head_mask = _build_nested_mask(
                    self.head_choices,
                    layer_soft_config.head_probabilities,
                    max(self.head_choices),
                )
                x = block(x, head_mask=head_mask, mlp_mask=mlp_mask)
            elif soft_config is not None:
                mlp_mask = _build_nested_mask(
                    self.mlp_choices,
                    _layer_probabilities(soft_config.mlp_probabilities, layer_idx),
                    max(self.mlp_choices),
                )
                head_mask = _build_nested_mask(
                    self.head_choices,
                    _layer_probabilities(soft_config.head_probabilities, layer_idx),
                    max(self.head_choices),
                )
                x = block(x, head_mask=head_mask, mlp_mask=mlp_mask)
            else:
                assert config is not None
                x = block(
                    x,
                    active_heads=config.num_heads[layer_idx],
                    active_mlp_width=config.mlp_widths[layer_idx],
                )

        x = self.norm(x)
        if self.global_pool == "avg":
            x = x[:, self.num_prefix_tokens :].mean(dim=1)
            x = self.fc_norm(x)
        else:
            x = x[:, 0]

        if return_routing:
            if adaptive_router is None:
                return x, None
            return x, {
                "mlp_probabilities": torch.stack(routing["mlp_probabilities"], dim=1),
                "head_probabilities": torch.stack(routing["head_probabilities"], dim=1),
            }
        return x

    def forward(
        self,
        x: torch.Tensor,
        config: Optional[SubnetworkConfig] = None,
        soft_config: Optional[RouterSoftConfig] = None,
    ) -> torch.Tensor:
        x = self.head_drop(self.forward_features(x, config=config, soft_config=soft_config))
        return self.head(x)

    def forward_adaptive(
        self,
        x: torch.Tensor,
        router: nn.Module,
        entropy: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        hard: bool = False,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        features, routing = self.forward_features(
            x,
            adaptive_router=router,
            entropy=entropy,
            router_temperature=temperature,
            router_hard=hard,
            deterministic_routing=deterministic,
            return_routing=True,
        )
        logits = self.head(self.head_drop(features))
        assert routing is not None
        return logits, routing

    def export_subnetwork(self, config: SubnetworkConfig) -> StaticVisionTransformer:
        key = config.as_key()
        if key not in self.export_cache:
            self.export_cache[key] = StaticVisionTransformer(self, config)
        return self.export_cache[key]

    def freeze_backbone(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False
