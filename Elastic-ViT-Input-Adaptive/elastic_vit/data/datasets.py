from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from elastic_vit.data.coco import COCO_CLASS_NAMES, CocoMultilabelDataset
from elastic_vit.data.cub import CUB200Dataset
from elastic_vit.data.entropy_cache import IndexedDataset
from elastic_vit.data.samplers import EntropySortedBatchSampler


DATASET_METADATA = {
    "imagenet1k": {"num_classes": 1000, "task_type": "multiclass"},
    "cifar10": {"num_classes": 10, "task_type": "multiclass"},
    "cifar100": {"num_classes": 100, "task_type": "multiclass"},
    "fgvc_aircraft": {"num_classes": 100, "task_type": "multiclass"},
    "stanford_cars": {"num_classes": 196, "task_type": "multiclass"},
    "oxford_iiit_pets": {"num_classes": 37, "task_type": "multiclass"},
    "cub200": {"num_classes": 200, "task_type": "multiclass"},
    "coco": {"num_classes": len(COCO_CLASS_NAMES), "task_type": "multilabel"},
}


def build_transforms(image_size: int, train: bool):
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    if train:
        return transforms.Compose(
            [
                transforms.Resize(image_size + 32),
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )


def _build_raw_dataset(name: str, root: str | Path, split: str, transform):
    dataset_name = name.lower()
    root = str(root)
    if dataset_name == "imagenet1k":
        return datasets.ImageFolder(Path(root) / split, transform=transform)
    if dataset_name == "cifar10":
        return datasets.CIFAR10(root=root, train=split == "train", transform=transform, download=False)
    if dataset_name == "cifar100":
        return datasets.CIFAR100(root=root, train=split == "train", transform=transform, download=False)
    if dataset_name == "fgvc_aircraft":
        target_split = "train" if split == "train" else "val"
        return datasets.FGVCAircraft(
            root=root,
            split=target_split,
            annotation_level="variant",
            transform=transform,
            download=False,
        )
    if dataset_name == "stanford_cars":
        target_split = "train" if split == "train" else "test"
        return datasets.StanfordCars(root=root, split=target_split, transform=transform, download=False)
    if dataset_name == "oxford_iiit_pets":
        target_split = "trainval" if split == "train" else "test"
        return datasets.OxfordIIITPet(
            root=root,
            split=target_split,
            target_types="category",
            transform=transform,
            download=False,
        )
    if dataset_name == "cub200":
        target_split = "train" if split == "train" else "test"
        return CUB200Dataset(root=root, split=target_split, transform=transform)
    if dataset_name == "coco":
        target_split = "train" if split == "train" else "val"
        return CocoMultilabelDataset(root=root, split=target_split, transform=transform)
    raise ValueError(f"Unsupported dataset: {name}")


def get_dataset_metadata(name: str) -> Dict[str, object]:
    dataset_name = name.lower()
    if dataset_name not in DATASET_METADATA:
        raise ValueError(f"Unsupported dataset metadata lookup: {name}")
    return DATASET_METADATA[dataset_name]


def build_dataset(name: str, root: str | Path, split: str, image_size: int):
    return _build_raw_dataset(name, root, split, build_transforms(image_size, train=split == "train"))


def build_entropy_source_dataset(name: str, root: str | Path, split: str):
    return _build_raw_dataset(name, root, split, transform=None)


def build_indexed_dataset(name: str, root: str | Path, split: str, image_size: int) -> IndexedDataset:
    dataset = build_dataset(name, root, split, image_size=image_size)
    return IndexedDataset(dataset)


def build_standard_dataloaders(dataset_cfg: Dict, runtime_cfg: Dict) -> Tuple[DataLoader, DataLoader]:
    train_dataset = build_dataset(
        dataset_cfg["name"],
        dataset_cfg["root"],
        "train",
        image_size=dataset_cfg["image_size"],
    )
    val_dataset = build_dataset(
        dataset_cfg["name"],
        dataset_cfg["root"],
        "val",
        image_size=dataset_cfg["image_size"],
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=runtime_cfg["batch_size"],
        shuffle=True,
        num_workers=runtime_cfg["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=runtime_cfg["batch_size"],
        shuffle=False,
        num_workers=runtime_cfg["num_workers"],
        pin_memory=True,
    )
    return train_loader, val_loader


def build_entropy_sorted_loader(dataset_cfg: Dict, runtime_cfg: Dict, entropy_values, split: str = "train") -> DataLoader:
    dataset = build_indexed_dataset(
        dataset_cfg["name"],
        dataset_cfg["root"],
        split,
        image_size=dataset_cfg["image_size"],
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
    )
