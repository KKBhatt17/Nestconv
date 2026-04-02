import os
from typing import List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision.transforms import functional as TF
from tqdm import tqdm

from entropy import compute_entropy_mean
from .image_datasets import build_image_dataset


def _fallback_extract_raw_image(dataset, index):
    attr_names = ["transform", "target_transform", "transforms"]
    saved = {name: getattr(dataset, name) for name in attr_names if hasattr(dataset, name)}
    try:
        for name in saved:
            setattr(dataset, name, None)
        image, _ = dataset[index]
    finally:
        for name, value in saved.items():
            setattr(dataset, name, value)

    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if torch.is_tensor(image):
        return TF.to_pil_image(image).convert("RGB")
    raise TypeError(f"Unable to extract raw image for dataset type {type(dataset)}")


def extract_raw_image(dataset, index):
    if hasattr(dataset, "_samples"):
        image_path, _ = dataset._samples[index]
        return Image.open(image_path).convert("RGB")

    if hasattr(dataset, "_image_files"):
        image_path = dataset._image_files[index]
        return Image.open(image_path).convert("RGB")

    if hasattr(dataset, "_images"):
        image_path = dataset._images[index]
        return Image.open(image_path).convert("RGB")

    if hasattr(dataset, "data"):
        data = dataset.data[index]
        if torch.is_tensor(data):
            data = data.cpu().numpy()
        data = np.asarray(data)
        if data.ndim == 3 and data.shape[0] in (1, 3):
            data = np.transpose(data, (1, 2, 0))
        if data.ndim == 3 and data.shape[2] == 1:
            data = data.squeeze(2)
        return Image.fromarray(data.astype(np.uint8)).convert("RGB")

    return _fallback_extract_raw_image(dataset, index)


def _entropy_cache_path(args, split):
    cache_dir = os.path.abspath(args.entropy_cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    file_name = (
        f"{args.dataset}_{split}_img{args.entropy_image_size}_patch{args.entropy_patch_size}_"
        f"scales{args.entropy_num_scales}_{'noresize' if args.entropy_no_resize else 'resize'}.pt"
    )
    return os.path.join(cache_dir, file_name)


def load_or_compute_entropy_means(dataset, args, split):
    cache_path = _entropy_cache_path(args, split)
    if os.path.exists(cache_path):
        entropy_means = torch.load(cache_path)
        if len(entropy_means) == len(dataset):
            return entropy_means.float()

    entropy_means: List[float] = []
    for index in tqdm(range(len(dataset)), desc=f"precompute entropy {split}", mininterval=0.3):
        raw_image = extract_raw_image(dataset, index)
        entropy_means.append(
            compute_entropy_mean(
                raw_image,
                image_size=args.entropy_image_size,
                patch_size=args.entropy_patch_size,
                num_scales=args.entropy_num_scales,
                no_resize=args.entropy_no_resize,
            )
        )

    entropy_tensor = torch.tensor(entropy_means, dtype=torch.float32)
    torch.save(entropy_tensor, cache_path)
    return entropy_tensor


class EntropyAnnotatedDataset(Dataset):
    def __init__(self, dataset, entropy_means):
        self.dataset = dataset
        self.entropy_means = entropy_means.float()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        return image, label, self.entropy_means[index]


class SortedEntropyBatchSampler(Sampler[List[int]]):
    def __init__(self, sorted_indices, batch_size, shuffle_batches=True, drop_last=False):
        self.sorted_indices = list(sorted_indices)
        self.batch_size = batch_size
        self.shuffle_batches = shuffle_batches
        self.drop_last = drop_last

    def __len__(self):
        num_batches = len(self.sorted_indices) // self.batch_size
        if not self.drop_last and len(self.sorted_indices) % self.batch_size != 0:
            num_batches += 1
        return num_batches

    def __iter__(self):
        batches = []
        for start in range(0, len(self.sorted_indices), self.batch_size):
            batch = self.sorted_indices[start:start + self.batch_size]
            if len(batch) < self.batch_size and self.drop_last:
                continue
            batches.append(batch)

        if self.shuffle_batches and len(batches) > 1:
            permutation = torch.randperm(len(batches)).tolist()
            batches = [batches[index] for index in permutation]

        for batch in batches:
            yield batch


def build_entropy_guided_dataloaders(args):
    dataset_train, dataset_val, nb_classes = build_image_dataset(args)

    train_entropy_means = load_or_compute_entropy_means(dataset_train, args, split="train")
    val_entropy_means = load_or_compute_entropy_means(dataset_val, args, split="val")

    train_dataset = EntropyAnnotatedDataset(dataset_train, train_entropy_means)
    val_dataset = EntropyAnnotatedDataset(dataset_val, val_entropy_means)

    sorted_train_indices = torch.argsort(train_entropy_means).tolist()
    train_batch_sampler = SortedEntropyBatchSampler(
        sorted_indices=sorted_train_indices,
        batch_size=args.batch_size,
        shuffle_batches=True,
        drop_last=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
    )

    return train_loader, val_loader, nb_classes


def build_entropy_guided_eval_loader(args):
    _, dataset_val, nb_classes = build_image_dataset(args)
    val_entropy_means = load_or_compute_entropy_means(dataset_val, args, split="val")
    val_dataset = EntropyAnnotatedDataset(dataset_val, val_entropy_means)
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
    )
    return val_loader, nb_classes
