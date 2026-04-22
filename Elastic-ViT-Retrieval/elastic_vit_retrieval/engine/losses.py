from __future__ import annotations

import torch
import torch.nn.functional as F


def clip_contrastive_loss(logits_per_image: torch.Tensor, logits_per_text: torch.Tensor) -> torch.Tensor:
    targets = torch.arange(logits_per_image.size(0), device=logits_per_image.device)
    image_loss = F.cross_entropy(logits_per_image, targets)
    text_loss = F.cross_entropy(logits_per_text, targets)
    return 0.5 * (image_loss + text_loss)


def cosine_distillation_loss(student_embeddings: torch.Tensor, teacher_embeddings: torch.Tensor) -> torch.Tensor:
    student_embeddings = F.normalize(student_embeddings, dim=-1)
    teacher_embeddings = F.normalize(teacher_embeddings, dim=-1)
    cosine_similarity = F.cosine_similarity(student_embeddings, teacher_embeddings, dim=-1)
    return (1.0 - cosine_similarity).mean()
