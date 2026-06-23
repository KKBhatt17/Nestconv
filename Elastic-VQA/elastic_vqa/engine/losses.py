from __future__ import annotations

import torch
import torch.nn.functional as F

from elastic_vqa.data.vocab import UNANSWERABLE


def vqa_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over the answer vocab.

    Targets equal to ``UNANSWERABLE`` (answer outside the vocab) are ignored, so
    they contribute no gradient. They are still counted as incorrect by the
    accuracy metric -- keeping the headline number over the full eval set.
    """
    return F.cross_entropy(logits, targets, ignore_index=UNANSWERABLE)
