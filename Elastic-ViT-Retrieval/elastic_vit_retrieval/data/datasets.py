from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from elastic_vit_retrieval.data.coco import (
    CocoRetrievalImageDataset,
    CocoRetrievalPairDataset,
    CocoRetrievalTextDataset,
)
from elastic_vit_retrieval.data.samplers import EntropySortedBatchSampler


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


@dataclass
class RetrievalEvalBundle:
    image_loader: DataLoader
    text_loader: DataLoader
    image_to_text_map: list[list[int]]
    text_to_image_map: list[int]


def build_transforms(image_size: int, train: bool):
    normalize = transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD)
    if train:
        return transforms.Compose(
            [
                transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )


def _collate_pair_batch(tokenizer, max_length: int):
    def collate_fn(samples: Sequence[dict[str, object]]) -> dict[str, torch.Tensor]:
        images = torch.stack([sample["image"] for sample in samples])
        captions = [str(sample["caption"]) for sample in samples]
        indices = torch.tensor([int(sample["index"]) for sample in samples], dtype=torch.long)
        tokenized = tokenizer(
            captions,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return {
            "pixel_values": images,
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "indices": indices,
        }

    return collate_fn


def _collate_image_batch(samples: Sequence[dict[str, object]]) -> dict[str, torch.Tensor]:
    images = torch.stack([sample["image"] for sample in samples])
    indices = torch.tensor([int(sample["index"]) for sample in samples], dtype=torch.long)
    image_ids = torch.tensor([int(sample["image_id"]) for sample in samples], dtype=torch.long)
    return {
        "pixel_values": images,
        "indices": indices,
        "image_ids": image_ids,
    }


def _collate_text_batch(tokenizer, max_length: int):
    def collate_fn(samples: Sequence[dict[str, object]]) -> dict[str, torch.Tensor]:
        captions = [str(sample["caption"]) for sample in samples]
        image_indices = torch.tensor([int(sample["image_index"]) for sample in samples], dtype=torch.long)
        text_indices = torch.tensor([int(sample["text_index"]) for sample in samples], dtype=torch.long)
        tokenized = tokenizer(
            captions,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "image_indices": image_indices,
            "text_indices": text_indices,
        }

    return collate_fn


def build_train_dataloader(dataset_cfg: Dict, runtime_cfg: Dict, tokenizer) -> DataLoader:
    train_dataset = CocoRetrievalPairDataset(
        root=dataset_cfg["root"],
        split="train",
        transform=build_transforms(dataset_cfg["image_size"], train=True),
        sample_caption=True,
    )
    return DataLoader(
        train_dataset,
        batch_size=runtime_cfg["batch_size"],
        shuffle=True,
        num_workers=runtime_cfg["num_workers"],
        pin_memory=True,
        collate_fn=_collate_pair_batch(tokenizer, dataset_cfg["text_max_length"]),
    )


def build_eval_bundle(dataset_cfg: Dict, runtime_cfg: Dict, tokenizer, split: str = "val") -> RetrievalEvalBundle:
    image_dataset = CocoRetrievalImageDataset(
        root=dataset_cfg["root"],
        split=split,
        transform=build_transforms(dataset_cfg["image_size"], train=False),
    )
    text_dataset = CocoRetrievalTextDataset(image_dataset.entries)

    image_to_text_map: list[list[int]] = []
    text_to_image_map: list[int] = []
    running_index = 0
    for image_index, entry in enumerate(image_dataset.entries):
        caption_indices = list(range(running_index, running_index + len(entry.captions)))
        image_to_text_map.append(caption_indices)
        text_to_image_map.extend([image_index] * len(entry.captions))
        running_index += len(entry.captions)

    eval_batch_size = int(runtime_cfg.get("eval_batch_size", runtime_cfg["batch_size"]))
    image_loader = DataLoader(
        image_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=runtime_cfg["num_workers"],
        pin_memory=True,
        collate_fn=_collate_image_batch,
    )
    text_loader = DataLoader(
        text_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=runtime_cfg["num_workers"],
        pin_memory=True,
        collate_fn=_collate_text_batch(tokenizer, dataset_cfg["text_max_length"]),
    )
    return RetrievalEvalBundle(
        image_loader=image_loader,
        text_loader=text_loader,
        image_to_text_map=image_to_text_map,
        text_to_image_map=text_to_image_map,
    )


def build_entropy_source_dataset(name: str, root: str | Path, split: str):
    if name.lower() != "coco_retrieval":
        raise ValueError(f"Unsupported dataset for entropy cache: {name}")
    return CocoRetrievalImageDataset(root=root, split=split, transform=None)


def build_entropy_sorted_loader(
    dataset_cfg: Dict,
    runtime_cfg: Dict,
    entropy_values: torch.Tensor,
    tokenizer,
    split: str = "train",
) -> DataLoader:
    dataset = CocoRetrievalPairDataset(
        root=dataset_cfg["root"],
        split=split,
        transform=build_transforms(dataset_cfg["image_size"], train=split == "train"),
        sample_caption=split == "train",
    )
    batch_sampler = EntropySortedBatchSampler(
        entropy_values=entropy_values,
        batch_size=runtime_cfg["batch_size"],
        shuffle_batches=split == "train",
        drop_last=split == "train",
    )
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=runtime_cfg["num_workers"],
        pin_memory=True,
        collate_fn=_collate_pair_batch(tokenizer, dataset_cfg["text_max_length"]),
    )
