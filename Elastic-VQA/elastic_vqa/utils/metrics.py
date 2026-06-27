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


def vqa_soft_accuracy(logits: torch.Tensor, answer_targets: torch.Tensor) -> float:
    """Official VQA accuracy over multiple human answers, in percent.

    ``answer_targets`` has shape ``[B, K]`` of answer vocab indices (one column per
    human annotation; out-of-vocab answers are ``UNANSWERABLE = -1``). Per sample the
    score is ``min(#humans whose answer == prediction, 3) / 3``. Predictions are
    always in-vocab (>= 0), so the ``-1`` columns can never produce a false match.
    """
    predictions = logits.argmax(dim=-1)
    matches = (answer_targets == predictions.unsqueeze(1)).sum(dim=1).float()
    return torch.clamp(matches / 3.0, max=1.0).mean().item() * 100.0
