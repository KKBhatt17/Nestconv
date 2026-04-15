from __future__ import annotations

import torch


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


def accuracy_top1(logits: torch.Tensor, targets: torch.Tensor) -> float:
    predictions = logits.argmax(dim=-1)
    return (predictions == targets).float().mean().item() * 100.0


def multilabel_precision_recall_f1(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    predictions = (torch.sigmoid(logits) >= threshold).to(dtype=torch.float32)
    targets = targets.to(dtype=torch.float32)

    true_positive = (predictions * targets).sum().item()
    false_positive = (predictions * (1.0 - targets)).sum().item()
    false_negative = ((1.0 - predictions) * targets).sum().item()

    precision = true_positive / max(true_positive + false_positive, 1.0)
    recall = true_positive / max(true_positive + false_negative, 1.0)
    f1_score = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "precision": precision * 100.0,
        "recall": recall * 100.0,
        "f1": f1_score * 100.0,
    }
