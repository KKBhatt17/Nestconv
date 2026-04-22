from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import torch
from transformers import CLIPModel


def _load_checkpoint(path: str | Path) -> Dict[str, torch.Tensor]:
    raw = torch.load(path, map_location="cpu")
    if isinstance(raw, dict) and "state_dict" in raw:
        return raw["state_dict"]
    if isinstance(raw, dict) and "model" in raw:
        return raw["model"]
    if isinstance(raw, dict):
        return raw
    raise ValueError(f"Unsupported checkpoint format at {path}")


def load_state_dict(path: str | Path) -> Dict[str, torch.Tensor]:
    state_dict = _load_checkpoint(path)
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in ("module.", "model."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
        cleaned[new_key] = value
    return cleaned


def save_state_dict(path: str | Path, state_dict: Dict[str, torch.Tensor], meta: Dict[str, object]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": state_dict, "meta": meta}, target)


def _attn_prefix(block_idx: int) -> str:
    return f"vision_model.encoder.layers.{block_idx}.self_attn"


def _mlp_prefix(block_idx: int) -> str:
    return f"vision_model.encoder.layers.{block_idx}.mlp"


def compute_attention_head_importance(
    state_dict: Dict[str, torch.Tensor],
    block_idx: int,
    embed_dim: int,
    num_heads: int,
) -> torch.Tensor:
    prefix = _attn_prefix(block_idx)
    head_dim = embed_dim // num_heads

    q_weight = state_dict[f"{prefix}.q_proj.weight"].view(num_heads, head_dim, embed_dim)
    k_weight = state_dict[f"{prefix}.k_proj.weight"].view(num_heads, head_dim, embed_dim)
    v_weight = state_dict[f"{prefix}.v_proj.weight"].view(num_heads, head_dim, embed_dim)
    q_bias = state_dict[f"{prefix}.q_proj.bias"].view(num_heads, head_dim)
    k_bias = state_dict[f"{prefix}.k_proj.bias"].view(num_heads, head_dim)
    v_bias = state_dict[f"{prefix}.v_proj.bias"].view(num_heads, head_dim)
    out_weight = state_dict[f"{prefix}.out_proj.weight"].view(embed_dim, num_heads, head_dim)

    return (
        q_weight.pow(2).sum(dim=(1, 2))
        + k_weight.pow(2).sum(dim=(1, 2))
        + v_weight.pow(2).sum(dim=(1, 2))
        + q_bias.pow(2).sum(dim=1)
        + k_bias.pow(2).sum(dim=1)
        + v_bias.pow(2).sum(dim=1)
        + out_weight.pow(2).sum(dim=(0, 2))
    )


def reorder_attention_heads_(
    state_dict: Dict[str, torch.Tensor],
    block_idx: int,
    embed_dim: int,
    num_heads: int,
) -> torch.Tensor:
    prefix = _attn_prefix(block_idx)
    order = torch.argsort(
        compute_attention_head_importance(state_dict, block_idx, embed_dim, num_heads),
        descending=True,
    )
    head_dim = embed_dim // num_heads

    for projection_name in ("q_proj", "k_proj", "v_proj"):
        weight_key = f"{prefix}.{projection_name}.weight"
        bias_key = f"{prefix}.{projection_name}.bias"
        weight = state_dict[weight_key].view(num_heads, head_dim, embed_dim)
        bias = state_dict[bias_key].view(num_heads, head_dim)
        state_dict[weight_key] = weight[order].reshape(embed_dim, embed_dim)
        state_dict[bias_key] = bias[order].reshape(embed_dim)

    out_key = f"{prefix}.out_proj.weight"
    out_weight = state_dict[out_key].view(embed_dim, num_heads, head_dim)
    state_dict[out_key] = out_weight[:, order].reshape(embed_dim, embed_dim)
    return order


def compute_mlp_neuron_importance(
    state_dict: Dict[str, torch.Tensor],
    block_idx: int,
) -> torch.Tensor:
    prefix = _mlp_prefix(block_idx)
    fc1_weight = state_dict[f"{prefix}.fc1.weight"]
    fc1_bias = state_dict[f"{prefix}.fc1.bias"]
    fc2_weight = state_dict[f"{prefix}.fc2.weight"]
    return fc1_weight.pow(2).sum(dim=1) + fc2_weight.pow(2).sum(dim=0) + fc1_bias.pow(2)


def reorder_mlp_neurons_(state_dict: Dict[str, torch.Tensor], block_idx: int) -> torch.Tensor:
    prefix = _mlp_prefix(block_idx)
    order = torch.argsort(compute_mlp_neuron_importance(state_dict, block_idx), descending=True)
    state_dict[f"{prefix}.fc1.weight"] = state_dict[f"{prefix}.fc1.weight"][order]
    state_dict[f"{prefix}.fc1.bias"] = state_dict[f"{prefix}.fc1.bias"][order]
    state_dict[f"{prefix}.fc2.weight"] = state_dict[f"{prefix}.fc2.weight"][:, order]
    return order


def rearrange_clip_checkpoint(
    model_name: str,
    output_path: str | Path,
    cache_dir: str | None = None,
    num_layers: int = 12,
    embed_dim: int = 768,
    num_heads: int = 12,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, object]]:
    state_dict = CLIPModel.from_pretrained(model_name, cache_dir=cache_dir).state_dict()
    attention_orders = {}
    mlp_orders = {}

    for block_idx in range(num_layers):
        attention_orders[block_idx] = reorder_attention_heads_(state_dict, block_idx, embed_dim, num_heads)
        mlp_orders[block_idx] = reorder_mlp_neurons_(state_dict, block_idx)

    meta = {
        "source_model_name": model_name,
        "attention_orders": {idx: order.tolist() for idx, order in attention_orders.items()},
        "mlp_orders": {idx: order.tolist() for idx, order in mlp_orders.items()},
    }
    save_state_dict(output_path, state_dict, meta)
    return state_dict, meta
