from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn.functional as F


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    @property
    def avg(self) -> float:
        if self.count == 0:
            return 0.0
        return self.sum / self.count

    def update(self, value: float, n: int = 1) -> None:
        self.sum += value * n
        self.count += n


def batch_retrieval_accuracy(logits_per_image: torch.Tensor) -> float:
    targets = torch.arange(logits_per_image.size(0), device=logits_per_image.device)
    predictions = logits_per_image.argmax(dim=-1)
    return (predictions == targets).float().mean().item() * 100.0


@torch.no_grad()
def compute_retrieval_recalls(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    image_to_text_map: Sequence[Sequence[int]],
    text_to_image_map: Sequence[int],
    ks: Iterable[int] = (1, 5, 10),
    chunk_size: int = 256,
) -> dict[str, float]:
    ks = tuple(sorted(set(int(k) for k in ks)))
    max_k = ks[-1]

    image_features = F.normalize(image_features.float(), dim=-1)
    text_features = F.normalize(text_features.float(), dim=-1)

    image_hits = {k: 0 for k in ks}
    for start in range(0, image_features.size(0), chunk_size):
        end = min(start + chunk_size, image_features.size(0))
        similarity = image_features[start:end] @ text_features.t()
        topk_indices = similarity.topk(k=max_k, dim=-1).indices.cpu().tolist()
        for local_idx, row in enumerate(topk_indices):
            positives = set(int(index) for index in image_to_text_map[start + local_idx])
            for k in ks:
                if any(int(candidate) in positives for candidate in row[:k]):
                    image_hits[k] += 1

    text_hits = {k: 0 for k in ks}
    for start in range(0, text_features.size(0), chunk_size):
        end = min(start + chunk_size, text_features.size(0))
        similarity = text_features[start:end] @ image_features.t()
        topk_indices = similarity.topk(k=max_k, dim=-1).indices.cpu().tolist()
        for local_idx, row in enumerate(topk_indices):
            positive = int(text_to_image_map[start + local_idx])
            for k in ks:
                if positive in row[:k]:
                    text_hits[k] += 1

    metrics = {}
    for k in ks:
        metrics[f"i2t_recall@{k}"] = 100.0 * image_hits[k] / max(len(image_to_text_map), 1)
        metrics[f"t2i_recall@{k}"] = 100.0 * text_hits[k] / max(len(text_to_image_map), 1)
    metrics["mean_recall"] = sum(metrics.values()) / max(len(metrics), 1)
    return metrics
