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
