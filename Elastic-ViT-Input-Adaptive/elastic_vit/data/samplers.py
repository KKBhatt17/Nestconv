from __future__ import annotations

import random
from typing import Iterator, List

import torch
from torch.utils.data import Sampler


class EntropySortedBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        entropy_values: torch.Tensor,
        batch_size: int,
        shuffle_batches: bool = True,
        drop_last: bool = False,
    ) -> None:
        self.entropy_values = entropy_values
        self.batch_size = batch_size
        self.shuffle_batches = shuffle_batches
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[List[int]]:
        sorted_indices = torch.argsort(self.entropy_values).tolist()
        batches = [
            sorted_indices[start : start + self.batch_size]
            for start in range(0, len(sorted_indices), self.batch_size)
        ]
        if self.drop_last and batches and len(batches[-1]) < self.batch_size:
            batches = batches[:-1]
        if self.shuffle_batches:
            random.shuffle(batches)
        for batch in batches:
            yield batch

    def __len__(self) -> int:
        total = len(self.entropy_values) // self.batch_size
        if not self.drop_last and len(self.entropy_values) % self.batch_size:
            total += 1
        return total
