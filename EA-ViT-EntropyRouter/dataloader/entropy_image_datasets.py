import os
from typing import Optional

import torch
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from torch.utils.data import BatchSampler, DataLoader, Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from entropy import compute_patch_entropy_vectorized

from .crop import RandomResizedCrop


class EntropyChunkBatchSampler(BatchSampler):
    def __init__(self, dataset_size: int, batch_size: int, drop_last: bool = False, shuffle_batches: bool = False):
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle_batches = shuffle_batches

        self._batches = []
        start = 0
        while start < self.dataset_size:
            stop = start + self.batch_size
            if stop > self.dataset_size and self.drop_last:
                break
            self._batches.append(list(range(start, min(stop, self.dataset_size))))
            start = stop

    def __iter__(self):
        if self.shuffle_batches:
            order = torch.randperm(len(self._batches)).tolist()
        else:
            order = list(range(len(self._batches)))
        for batch_index in order:
            yield self._batches[batch_index]

    def __len__(self):
        return len(self._batches)


class EntropyAwareImageDataset(Dataset):
    def __init__(
        self,
        base_dataset: Dataset,
        image_transform,
        entropy_transform,
        cache_path: Optional[str],
        patch_size: int = 14,
    ):
        self.base_dataset = base_dataset
        self.image_transform = image_transform
        self.entropy_transform = entropy_transform
        self.patch_size = patch_size
        self.cache_path = cache_path

        self.entropy_vectors, self.entropy_means = self._load_or_compute_cache()
        self.sorted_indices = sorted(
            range(len(self.base_dataset)),
            key=lambda idx: float(self.entropy_means[idx]),
        )

    def _compute_entropy_vector(self, image) -> torch.Tensor:
        entropy_input = self.entropy_transform(image)
        entropy_map = compute_patch_entropy_vectorized(
            entropy_input * 255.0,
            patch_size=self.patch_size,
            num_scales=1,
        )[self.patch_size]
        return entropy_map.flatten().float()

    def _load_or_compute_cache(self):
        if self.cache_path and os.path.isfile(self.cache_path):
            cache = torch.load(self.cache_path, map_location="cpu")
            if len(cache["entropy_vectors"]) == len(self.base_dataset):
                return cache["entropy_vectors"], cache["entropy_means"]

        entropy_vectors = []
        entropy_means = []

        for index in range(len(self.base_dataset)):
            image, _ = self.base_dataset[index]
            entropy_vector = self._compute_entropy_vector(image)
            entropy_vectors.append(entropy_vector)
            entropy_means.append(entropy_vector.mean())

        entropy_vectors = torch.stack(entropy_vectors)
        entropy_means = torch.stack(entropy_means)

        if self.cache_path:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            torch.save(
                {
                    "entropy_vectors": entropy_vectors,
                    "entropy_means": entropy_means,
                },
                self.cache_path,
            )

        return entropy_vectors, entropy_means

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        sorted_index = self.sorted_indices[index]
        image, label = self.base_dataset[sorted_index]
        image = self.image_transform(image)
        entropy_vector = self.entropy_vectors[sorted_index]
        entropy_mean = self.entropy_means[sorted_index]
        return image, label, entropy_vector, entropy_mean


def build_entropy_image_dataset(args):
    mean = IMAGENET_INCEPTION_MEAN
    std = IMAGENET_INCEPTION_STD

    transform_train = transforms.Compose(
        [
            RandomResizedCrop(args.input_size, interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    transform_val = transforms.Compose(
        [
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(args.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    entropy_transform = transforms.Compose(
        [
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(args.input_size),
            transforms.ToTensor(),
        ]
    )

    dataset_train, dataset_val, nb_classes = _build_raw_image_dataset(args)

    cache_root = os.path.join(args.data_path, "entropy_cache")
    train_cache_path = os.path.join(
        cache_root,
        f"{args.dataset}_train_size{args.input_size}_patch{args.entropy_patch_size}.pt",
    )
    val_cache_path = os.path.join(
        cache_root,
        f"{args.dataset}_val_size{args.input_size}_patch{args.entropy_patch_size}.pt",
    )

    dataset_train = EntropyAwareImageDataset(
        base_dataset=dataset_train,
        image_transform=transform_train,
        entropy_transform=entropy_transform,
        cache_path=train_cache_path,
        patch_size=args.entropy_patch_size,
    )
    dataset_val = EntropyAwareImageDataset(
        base_dataset=dataset_val,
        image_transform=transform_val,
        entropy_transform=entropy_transform,
        cache_path=val_cache_path,
        patch_size=args.entropy_patch_size,
    )

    return dataset_train, dataset_val, nb_classes


def create_entropy_dataloader(args, dataset, shuffle_batches: bool, single_image: bool = False):
    if single_image:
        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
        )

    batch_sampler = EntropyChunkBatchSampler(
        dataset_size=len(dataset),
        batch_size=args.batch_size,
        drop_last=False,
        shuffle_batches=shuffle_batches,
    )
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
    )


def _build_raw_image_dataset(args):
    if args.dataset == "cifar100_full":
        dataset_train = datasets.CIFAR100(os.path.join(args.data_path, "cifar100"), train=True, download=True)
        dataset_val = datasets.CIFAR100(os.path.join(args.data_path, "cifar100"), train=False, download=True)
        nb_classes = 100

    elif args.dataset == "cifar10_full":
        dataset_train = datasets.CIFAR10(os.path.join(args.data_path, "cifar10"), train=True, download=True)
        dataset_val = datasets.CIFAR10(os.path.join(args.data_path, "cifar10"), train=False, download=True)
        nb_classes = 10

    elif args.dataset == "flowers102_full":
        from .flowers102 import Flowers102

        dataset_train = Flowers102(os.path.join(args.data_path, "flowers102"), split="train", download=True)
        dataset_val = Flowers102(os.path.join(args.data_path, "flowers102"), split="test", download=True)
        nb_classes = 102

    elif args.dataset == "svhn_full":
        from torchvision.datasets import SVHN

        dataset_train = SVHN(os.path.join(args.data_path, "svhn"), split="train", download=True)
        dataset_val = SVHN(os.path.join(args.data_path, "svhn"), split="test", download=True)
        nb_classes = 10

    elif args.dataset == "food101_full":
        from .food101 import Food101

        dataset_train = Food101(os.path.join(args.data_path, "food101"), split="train", download=True)
        dataset_val = Food101(os.path.join(args.data_path, "food101"), split="test", download=True)
        nb_classes = 101

    elif args.dataset == "fgvc_aircraft_full":
        from .fgvc_aircraft import FGVCAircraft

        dataset_train = FGVCAircraft(os.path.join(args.data_path, "fgvc_aircraft"), split="trainval", download=True)
        dataset_val = FGVCAircraft(os.path.join(args.data_path, "fgvc_aircraft"), split="test", download=True)
        nb_classes = 100

    elif args.dataset == "stanford_cars_full":
        from .stanford_cars import StanfordCars

        dataset_train = StanfordCars(os.path.join(args.data_path, "StanfordCars"), split="train", download=True)
        dataset_val = StanfordCars(os.path.join(args.data_path, "StanfordCars"), split="test", download=True)
        nb_classes = 196

    elif args.dataset == "dtd_full":
        from .dtd import DTD

        dataset_train = DTD(os.path.join(args.data_path, "dtd"), split="train", download=True)
        dataset_val = DTD(os.path.join(args.data_path, "dtd"), split="test", download=True)
        nb_classes = 47

    elif args.dataset == "oxford_iiit_pet_full":
        from .oxford_iiit_pet import OxfordIIITPet

        dataset_train = OxfordIIITPet(os.path.join(args.data_path, "oxford_iiit_pet"), split="trainval", download=True)
        dataset_val = OxfordIIITPet(os.path.join(args.data_path, "oxford_iiit_pet"), split="test", download=True)
        nb_classes = 37

    else:
        raise ValueError(args.dataset)

    return dataset_train, dataset_val, nb_classes
