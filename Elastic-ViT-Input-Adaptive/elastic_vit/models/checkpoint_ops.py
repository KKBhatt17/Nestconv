from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import torch


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


def compute_attention_head_importance(
    state_dict: Dict[str, torch.Tensor],
    block_idx: int,
    embed_dim: int,
    num_heads: int,
) -> torch.Tensor:
    prefix = f"blocks.{block_idx}.attn"
    head_dim = embed_dim // num_heads

    qkv_weight = state_dict[f"{prefix}.qkv.weight"].view(3, num_heads, head_dim, embed_dim)
    qkv_bias = state_dict[f"{prefix}.qkv.bias"].view(3, num_heads, head_dim)
    proj_weight = state_dict[f"{prefix}.proj.weight"].view(embed_dim, num_heads, head_dim)

    qkv_score = qkv_weight.pow(2).sum(dim=(0, 2, 3))
    bias_score = qkv_bias.pow(2).sum(dim=(0, 2))
    proj_score = proj_weight.pow(2).sum(dim=(0, 2))
    return qkv_score + bias_score + proj_score


def reorder_attention_heads_(
    state_dict: Dict[str, torch.Tensor],
    block_idx: int,
    embed_dim: int,
    num_heads: int,
) -> torch.Tensor:
    prefix = f"blocks.{block_idx}.attn"
    order = torch.argsort(
        compute_attention_head_importance(state_dict, block_idx, embed_dim, num_heads),
        descending=True,
    )
    head_dim = embed_dim // num_heads

    qkv_weight = state_dict[f"{prefix}.qkv.weight"].view(3, num_heads, head_dim, embed_dim)
    qkv_bias = state_dict[f"{prefix}.qkv.bias"].view(3, num_heads, head_dim)
    proj_weight = state_dict[f"{prefix}.proj.weight"].view(embed_dim, num_heads, head_dim)

    state_dict[f"{prefix}.qkv.weight"] = qkv_weight[:, order].reshape(3 * embed_dim, embed_dim)
    state_dict[f"{prefix}.qkv.bias"] = qkv_bias[:, order].reshape(3 * embed_dim)
    state_dict[f"{prefix}.proj.weight"] = proj_weight[:, order].reshape(embed_dim, embed_dim)
    return order


def compute_mlp_neuron_importance(
    state_dict: Dict[str, torch.Tensor],
    block_idx: int,
) -> torch.Tensor:
    prefix = f"blocks.{block_idx}.mlp"
    fc1_weight = state_dict[f"{prefix}.fc1.weight"]
    fc1_bias = state_dict[f"{prefix}.fc1.bias"]
    fc2_weight = state_dict[f"{prefix}.fc2.weight"]

    fc1_score = fc1_weight.pow(2).sum(dim=1)
    fc2_score = fc2_weight.pow(2).sum(dim=0)
    bias_score = fc1_bias.pow(2)
    return fc1_score + fc2_score + bias_score


def reorder_mlp_neurons_(state_dict: Dict[str, torch.Tensor], block_idx: int) -> torch.Tensor:
    prefix = f"blocks.{block_idx}.mlp"
    order = torch.argsort(compute_mlp_neuron_importance(state_dict, block_idx), descending=True)

    fc1_weight = state_dict[f"{prefix}.fc1.weight"]
    fc1_bias = state_dict[f"{prefix}.fc1.bias"]
    fc2_weight = state_dict[f"{prefix}.fc2.weight"]

    state_dict[f"{prefix}.fc1.weight"] = fc1_weight[order]
    state_dict[f"{prefix}.fc1.bias"] = fc1_bias[order]
    state_dict[f"{prefix}.fc2.weight"] = fc2_weight[:, order]
    return order


def rearrange_vit_checkpoint(
    input_path: str | Path,
    output_path: str | Path,
    num_layers: int = 12,
    embed_dim: int = 768,
    num_heads: int = 12,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, object]]:
    state_dict = load_state_dict(input_path)
    attention_orders = {}
    mlp_orders = {}

    for block_idx in range(num_layers):
        attention_orders[block_idx] = reorder_attention_heads_(state_dict, block_idx, embed_dim, num_heads)
        mlp_orders[block_idx] = reorder_mlp_neurons_(state_dict, block_idx)

    meta = {
        "attention_orders": {idx: order.tolist() for idx, order in attention_orders.items()},
        "mlp_orders": {idx: order.tolist() for idx, order in mlp_orders.items()},
    }
    save_state_dict(output_path, state_dict, meta)
    return state_dict, meta
