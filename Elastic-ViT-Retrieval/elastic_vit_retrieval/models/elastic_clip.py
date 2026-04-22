from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel

from elastic_vit_retrieval.models.checkpoint_ops import load_state_dict
from elastic_vit_retrieval.models.common import (
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
    def __init__(self, base_attention: nn.Module, embed_dim: int, max_heads: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = max_heads
        self.head_dim = embed_dim // max_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = base_attention.q_proj
        self.k_proj = base_attention.k_proj
        self.v_proj = base_attention.v_proj
        self.out_proj = base_attention.out_proj
        self.dropout = float(getattr(base_attention, "dropout", 0.0))

    def forward(
        self,
        hidden_states: torch.Tensor,
        active_heads: Optional[int] = None,
        head_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, num_tokens, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(hidden_states).view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(hidden_states).view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        if head_mask is None and active_heads is not None:
            head_mask = torch.zeros(self.num_heads, device=hidden_states.device, dtype=hidden_states.dtype)
            head_mask[:active_heads] = 1.0
        if head_mask is not None:
            shaped_mask = head_mask.view(1, self.num_heads, 1, 1)
            query = query * shaped_mask
            key = key * shaped_mask
            value = value * shaped_mask

        attn_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        attn_probs = F.softmax(attn_scores, dim=-1)
        if self.dropout > 0.0:
            attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)

        attn_output = torch.matmul(attn_probs, value).transpose(1, 2).reshape(batch_size, num_tokens, self.embed_dim)
        if head_mask is not None:
            channel_mask = head_mask.view(self.num_heads, 1).expand(self.num_heads, self.head_dim).reshape(self.embed_dim)
            attn_output = attn_output * channel_mask.view(1, 1, self.embed_dim)
        return self.out_proj(attn_output)


class ElasticMlp(nn.Module):
    def __init__(self, base_mlp: nn.Module, max_mlp_width: int) -> None:
        super().__init__()
        self.max_mlp_width = max_mlp_width
        self.fc1 = base_mlp.fc1
        self.activation_fn = base_mlp.activation_fn
        self.fc2 = base_mlp.fc2

    def forward(
        self,
        hidden_states: torch.Tensor,
        active_width: Optional[int] = None,
        mlp_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        if mlp_mask is None and active_width is not None:
            mlp_mask = torch.zeros(self.max_mlp_width, device=hidden_states.device, dtype=hidden_states.dtype)
            mlp_mask[:active_width] = 1.0
        if mlp_mask is not None:
            hidden_states = hidden_states * mlp_mask.view(1, 1, self.max_mlp_width)
        return self.fc2(hidden_states)


class ElasticBlock(nn.Module):
    def __init__(self, base_block: nn.Module, embed_dim: int, max_heads: int, max_mlp_width: int) -> None:
        super().__init__()
        self.layer_norm1 = base_block.layer_norm1
        self.self_attn = ElasticAttention(base_block.self_attn, embed_dim=embed_dim, max_heads=max_heads)
        self.layer_norm2 = base_block.layer_norm2
        self.mlp = ElasticMlp(base_block.mlp, max_mlp_width=max_mlp_width)

    def forward(
        self,
        hidden_states: torch.Tensor,
        active_heads: Optional[int] = None,
        active_mlp_width: Optional[int] = None,
        head_mask: Optional[torch.Tensor] = None,
        mlp_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states, active_heads=active_heads, head_mask=head_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states, active_width=active_mlp_width, mlp_mask=mlp_mask)
        hidden_states = residual + hidden_states
        return hidden_states


class ElasticCLIPRetrievalModel(nn.Module):
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch16",
        checkpoint_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
        mlp_choices: Optional[Sequence[int]] = None,
        head_choices: Optional[Sequence[int]] = None,
        freeze_text: bool = True,
    ) -> None:
        super().__init__()
        self.mlp_choices = list(mlp_choices or DEFAULT_MLP_CHOICES)
        self.head_choices = list(head_choices or DEFAULT_HEAD_CHOICES)

        base_model = CLIPModel.from_pretrained(model_name, cache_dir=cache_dir)
        if checkpoint_path:
            checkpoint_state = load_state_dict(checkpoint_path)
            model_state = base_model.state_dict()
            filtered_state = {
                key: value
                for key, value in checkpoint_state.items()
                if key in model_state and model_state[key].shape == value.shape
            }
            base_model.load_state_dict(filtered_state, strict=False)

        vision_model = base_model.vision_model
        self.embeddings = vision_model.embeddings
        self.pre_layrnorm = vision_model.pre_layrnorm
        self.blocks = nn.ModuleList(
            [
                ElasticBlock(
                    block,
                    embed_dim=vision_model.config.hidden_size,
                    max_heads=max(self.head_choices),
                    max_mlp_width=max(self.mlp_choices),
                )
                for block in vision_model.encoder.layers
            ]
        )
        self.post_layernorm = vision_model.post_layernorm
        self.visual_projection = base_model.visual_projection
        self.text_model = base_model.text_model
        self.text_projection = base_model.text_projection
        self.logit_scale = base_model.logit_scale

        self.embed_dim = int(vision_model.config.hidden_size)
        self.projection_dim = int(base_model.config.projection_dim)
        self.num_layers = len(self.blocks)
        self.export_cache: Dict[str, nn.Module] = {}

        if freeze_text:
            self.freeze_text_tower()

    def freeze_text_tower(self) -> None:
        for module in (self.text_model, self.text_projection):
            for parameter in module.parameters():
                parameter.requires_grad = False

    def freeze_backbone(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False

    def forward_vision_features(
        self,
        pixel_values: torch.Tensor,
        config: Optional[SubnetworkConfig] = None,
        soft_config: Optional[RouterSoftConfig] = None,
    ) -> torch.Tensor:
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.pre_layrnorm(hidden_states)

        if config is None and soft_config is None:
            config = make_subnetwork_config(
                mlp_widths=max(self.mlp_choices),
                num_heads=max(self.head_choices),
                num_layers=self.num_layers,
            )

        for layer_idx, block in enumerate(self.blocks):
            if soft_config is not None:
                mlp_mask = _build_nested_mask(
                    self.mlp_choices,
                    soft_config.mlp_probabilities[layer_idx],
                    max(self.mlp_choices),
                )
                head_mask = _build_nested_mask(
                    self.head_choices,
                    soft_config.head_probabilities[layer_idx],
                    max(self.head_choices),
                )
                hidden_states = block(hidden_states, head_mask=head_mask, mlp_mask=mlp_mask)
            else:
                assert config is not None
                hidden_states = block(
                    hidden_states,
                    active_heads=config.num_heads[layer_idx],
                    active_mlp_width=config.mlp_widths[layer_idx],
                )

        pooled_output = self.post_layernorm(hidden_states[:, 0, :])
        return pooled_output

    def encode_image(
        self,
        pixel_values: torch.Tensor,
        config: Optional[SubnetworkConfig] = None,
        soft_config: Optional[RouterSoftConfig] = None,
        normalize: bool = True,
    ) -> torch.Tensor:
        image_features = self.forward_vision_features(pixel_values, config=config, soft_config=soft_config)
        image_features = self.visual_projection(image_features)
        if normalize:
            image_features = F.normalize(image_features, dim=-1)
        return image_features

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = getattr(text_outputs, "pooler_output", None)
        if pooled_output is None:
            pooled_output = text_outputs[1]
        text_features = self.text_projection(pooled_output)
        if normalize:
            text_features = F.normalize(text_features, dim=-1)
        return text_features

    def compute_similarity_logits(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scale = self.logit_scale.exp()
        logits_per_image = scale * image_features @ text_features.t()
        return logits_per_image, logits_per_image.t()

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        config: Optional[SubnetworkConfig] = None,
        soft_config: Optional[RouterSoftConfig] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        image_features = self.encode_image(pixel_values, config=config, soft_config=soft_config)
        text_features = self.encode_text(input_ids, attention_mask)
        logits_per_image, logits_per_text = self.compute_similarity_logits(image_features, text_features)
        return image_features, text_features, logits_per_image, logits_per_text
