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


def vqa_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Exact-match accuracy over the answer vocab, in percent.

    Out-of-vocab targets (negative index) are counted as incorrect, so accuracy is
    reported over the full set including the vocab-coverage cap. The denominator is
    the full batch size.
    """
    predictions = logits.argmax(dim=-1)
    correct = (predictions == targets).float().sum().item()
    return correct / max(targets.numel(), 1) * 100.0
