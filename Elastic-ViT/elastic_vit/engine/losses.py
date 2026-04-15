from __future__ import annotations

import torch
import torch.nn.functional as F


def classification_loss(logits: torch.Tensor, targets: torch.Tensor, task_type: str) -> torch.Tensor:
    if task_type == "multilabel":
        return F.binary_cross_entropy_with_logits(logits, targets.float())
    return F.cross_entropy(logits, targets)


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    task_type: str,
    temperature: float = 1.0,
) -> torch.Tensor:
    if task_type == "multilabel":
        teacher_targets = torch.sigmoid(teacher_logits / temperature)
        return F.binary_cross_entropy_with_logits(student_logits / temperature, teacher_targets) * (temperature ** 2)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)
