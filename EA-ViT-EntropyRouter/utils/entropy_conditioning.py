import csv
import os
from typing import Dict, List, Sequence, Union

import torch


def default_entropy_lookup_path(nsga_path: str) -> str:
    root, ext = os.path.splitext(nsga_path)
    if not ext:
        ext = ".csv"
    return f"{root}_entropy_lookup{ext}"


def decode_encoding(encoding: Union[Sequence[int], str]) -> List[int]:
    if isinstance(encoding, str):
        return [int(digit) for digit in encoding.strip()]
    return [int(digit) for digit in encoding]


def build_router_input(entropy_vectors: torch.Tensor, device: Union[str, torch.device]) -> torch.Tensor:
    if entropy_vectors.dim() == 1:
        batch_router_input = entropy_vectors.float()
    else:
        batch_router_input = entropy_vectors.float().mean(dim=0)
    return batch_router_input.to(device).unsqueeze(0)


def entropy_score_from_vectors(entropy_vectors: torch.Tensor) -> float:
    if entropy_vectors.dim() == 1:
        return float(entropy_vectors.float().mean().item())
    return float(entropy_vectors.float().mean(dim=1).mean().item())


def load_entropy_lookup(file_name: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with open(file_name, mode="r", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            parsed_row = {
                "batch_index": int(row["BatchIndex"]),
                "entropy_mean": float(row["EntropyMean"]),
                "encoding": row["Encoding"],
            }
            if "Accuracy" in row and row["Accuracy"] not in (None, ""):
                parsed_row["accuracy"] = float(row["Accuracy"])
            if "MACs" in row and row["MACs"] not in (None, ""):
                parsed_row["macs"] = float(row["MACs"])
            rows.append(parsed_row)

    rows.sort(key=lambda item: item["entropy_mean"])
    return rows


def select_lookup_entry(lookup_rows: List[Dict[str, object]], entropy_mean: float) -> Dict[str, object]:
    if not lookup_rows:
        raise ValueError("Entropy lookup table is empty.")
    return min(lookup_rows, key=lambda row: abs(float(row["entropy_mean"]) - float(entropy_mean)))


def encoding_to_mask_tensors(encoding: Union[Sequence[int], str], device: Union[str, torch.device]):
    digits = decode_encoding(encoding)

    embed_sum = int(sum(digits[:12]))
    emb_mask = torch.cat(
        (torch.ones(embed_sum, device=device), torch.zeros(12 - embed_sum, device=device)),
        dim=0,
    )

    depth_attn_mask = torch.tensor([float(i) for i in digits[12:24]], device=device)
    depth_mlp_mask = torch.tensor([float(i) for i in digits[24:36]], device=device)

    mha_list = []
    mlp_list = []

    for i in range(12):
        attn_sum = int(sum(digits[36 + i * 12: 36 + (i + 1) * 12]))
        mha_list.append(
            torch.cat(
                (torch.ones(attn_sum, device=device), torch.zeros(12 - attn_sum, device=device)),
                dim=0,
            )
        )

    for i in range(12):
        mlp_sum = int(sum(digits[180 + i * 8: 180 + (i + 1) * 8]))
        mlp_list.append(
            torch.cat(
                (torch.ones(mlp_sum, device=device), torch.zeros(8 - mlp_sum, device=device)),
                dim=0,
            )
        )

    return (
        torch.stack(mlp_list),
        torch.stack(mha_list),
        emb_mask,
        depth_mlp_mask,
        depth_attn_mask,
    )
