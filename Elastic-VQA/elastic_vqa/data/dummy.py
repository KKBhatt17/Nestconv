"""Synthetic VQA dataset for smoke-testing the pipeline without any real data.

Generates random RGB images and samples from a handful of fixed question/answer
pairs. Lets you exercise rearrange -> curriculum train -> preset eval end to end
(shapes, loss, metrics, checkpointing) before downloading GQA or CLEVR.
"""

from __future__ import annotations

import random
from typing import Iterator, List, Tuple

import torch
from torch.utils.data import Dataset

from elastic_vqa.data.vocab import AnswerVocab

_QA_PAIRS: List[Tuple[str, str]] = [
    ("what color is the object?", "red"),
    ("how many objects are there?", "two"),
    ("is the object large?", "yes"),
    ("is the object small?", "no"),
    ("what shape is it?", "cube"),
]


def iter_train_answers(root=None) -> Iterator[str]:
    for _, answer in _QA_PAIRS:
        yield answer


class DummyVqaDataset(Dataset):
    def __init__(self, split: str, vocab: AnswerVocab, image_size: int, num_samples: int = 256, seed: int = 0) -> None:
        self.vocab = vocab
        self.image_size = image_size
        self.num_samples = num_samples
        # Deterministic per-split so train/val are stable across runs.
        self.rng = random.Random(seed + (0 if split == "train" else 1))
        self.samples = [self.rng.randrange(len(_QA_PAIRS)) for _ in range(num_samples)]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str, int]:
        question, answer = _QA_PAIRS[self.samples[index]]
        generator = torch.Generator().manual_seed(index)
        image = torch.rand(3, self.image_size, self.image_size, generator=generator)
        return image, question, self.vocab.encode(answer)
