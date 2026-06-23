"""Convert a BLIP vision encoder into a timm ViT state dict.

This is the one real piece of glue between BLIP and the elastic backbone, and the
integration point flagged in the plan as *verify first*. The elastic backbone and
the ``rearrange`` stage are written against timm's ``vit_base_patch16_*`` key
layout; BLIP's HuggingFace vision encoder uses different names. We map one to the
other so a BLIP checkpoint can be (a) rearranged and (b) loaded into
``ElasticVisionTransformer``.

Geometry note: ``Salesforce/blip-vqa-base`` operates at **384px** with 16px
patches (577 tokens). Pair it with ``vit_base_patch16_384`` and ``image_size:
384`` in the config. Position embeddings are interpolated if the grid sizes
differ, so a 224px target also works (with a small accuracy cost).

Both BLIP and timm use a **fused** ``qkv`` linear of shape ``[3*embed, embed]``,
so attention maps over directly. Run :func:`audit_key_mapping` once against your
installed ``transformers`` version before trusting the mapping end to end.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F


def _interpolate_pos_embed(pos_embed: torch.Tensor, target_tokens: int, num_prefix: int = 1) -> torch.Tensor:
    """Resize ``[1, src_tokens, dim]`` position embeddings to ``target_tokens``."""
    src_tokens = pos_embed.shape[1]
    if src_tokens == target_tokens:
        return pos_embed
    dim = pos_embed.shape[2]
    prefix = pos_embed[:, :num_prefix]
    grid = pos_embed[:, num_prefix:]
    src_grid = int(math.sqrt(grid.shape[1]))
    tgt_grid = int(math.sqrt(target_tokens - num_prefix))
    grid = grid.reshape(1, src_grid, src_grid, dim).permute(0, 3, 1, 2)
    grid = F.interpolate(grid, size=(tgt_grid, tgt_grid), mode="bicubic", align_corners=False)
    grid = grid.permute(0, 2, 3, 1).reshape(1, tgt_grid * tgt_grid, dim)
    return torch.cat([prefix, grid], dim=1)


def blip_vision_to_timm(
    blip_vision_state: Dict[str, torch.Tensor],
    num_layers: int = 12,
    target_pos_tokens: int | None = None,
) -> Dict[str, torch.Tensor]:
    """Remap a BLIP ``vision_model`` state dict to timm ViT keys.

    ``blip_vision_state`` keys may or may not carry the ``vision_model.`` prefix;
    both are accepted.
    """
    src = {k[len("vision_model."):] if k.startswith("vision_model.") else k: v for k, v in blip_vision_state.items()}
    out: Dict[str, torch.Tensor] = {}

    # Patch / cls / position embeddings.
    out["patch_embed.proj.weight"] = src["embeddings.patch_embedding.weight"]
    if "embeddings.patch_embedding.bias" in src:
        out["patch_embed.proj.bias"] = src["embeddings.patch_embedding.bias"]
    out["cls_token"] = src["embeddings.class_embedding"].reshape(1, 1, -1)
    pos = src["embeddings.position_embedding"]
    if target_pos_tokens is not None:
        pos = _interpolate_pos_embed(pos, target_pos_tokens)
    out["pos_embed"] = pos

    # Transformer blocks.
    for i in range(num_layers):
        s = f"encoder.layers.{i}."
        d = f"blocks.{i}."
        out[d + "norm1.weight"] = src[s + "layer_norm1.weight"]
        out[d + "norm1.bias"] = src[s + "layer_norm1.bias"]
        out[d + "attn.qkv.weight"] = src[s + "self_attn.qkv.weight"]
        out[d + "attn.qkv.bias"] = src[s + "self_attn.qkv.bias"]
        out[d + "attn.proj.weight"] = src[s + "self_attn.projection.weight"]
        out[d + "attn.proj.bias"] = src[s + "self_attn.projection.bias"]
        out[d + "norm2.weight"] = src[s + "layer_norm2.weight"]
        out[d + "norm2.bias"] = src[s + "layer_norm2.bias"]
        out[d + "mlp.fc1.weight"] = src[s + "mlp.fc1.weight"]
        out[d + "mlp.fc1.bias"] = src[s + "mlp.fc1.bias"]
        out[d + "mlp.fc2.weight"] = src[s + "mlp.fc2.weight"]
        out[d + "mlp.fc2.bias"] = src[s + "mlp.fc2.bias"]

    # Final norm (BLIP: post_layernorm -> timm: norm).
    if "post_layernorm.weight" in src:
        out["norm.weight"] = src["post_layernorm.weight"]
        out["norm.bias"] = src["post_layernorm.bias"]

    return out


def export_blip_vision_checkpoint(
    blip_model_name: str,
    output_path: str | Path,
    num_layers: int = 12,
    target_pos_tokens: int | None = None,
) -> Path:
    """Download a BLIP model and write its vision tower as a timm-style checkpoint.

    The result is the ``stage1.input_checkpoint`` consumed by ``rearrange``.
    """
    from transformers import BlipForQuestionAnswering

    blip = BlipForQuestionAnswering.from_pretrained(blip_model_name)
    vision_state = blip.vision_model.state_dict()
    timm_state = blip_vision_to_timm(vision_state, num_layers=num_layers, target_pos_tokens=target_pos_tokens)

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": timm_state, "meta": {"source": blip_model_name}}, target)
    return target


def audit_key_mapping(blip_vision_state: Dict[str, torch.Tensor], timm_reference_keys: List[str]) -> Tuple[List[str], List[str]]:
    """Return (timm keys we failed to produce, BLIP keys we ignored).

    Use this once to confirm the mapping is complete for your installed
    ``transformers`` version. ``timm_reference_keys`` is
    ``ElasticVisionTransformer(...).state_dict().keys()`` filtered to the source
    backbone, or a fresh ``timm.create_model(name, num_classes=0).state_dict()``.
    """
    produced = blip_vision_to_timm(blip_vision_state)
    missing = [k for k in timm_reference_keys if k not in produced]
    src = {k[len("vision_model."):] if k.startswith("vision_model.") else k: v for k, v in blip_vision_state.items()}
    mapped_src_prefixes = ("embeddings.", "encoder.layers.", "post_layernorm.")
    ignored = [k for k in src if not k.startswith(mapped_src_prefixes)]
    return missing, ignored
