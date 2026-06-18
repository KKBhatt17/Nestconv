from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Tuple

from PIL import Image
from torch.utils.data import Dataset


def _resolve_cub_root(root: str | Path) -> Path:
    root_path = Path(root)
    nested = root_path / "CUB_200_2011"
    return nested if nested.exists() else root_path


class CUB200Dataset(Dataset):
    def __init__(self, root: str | Path, split: str, transform: Callable | None = None) -> None:
        super().__init__()
        self.root = _resolve_cub_root(root)
        self.transform = transform
        self.split = split
        self.samples = self._load_samples()

    def _read_mapping(self, relative_path: str) -> Dict[int, str]:
        mapping: Dict[int, str] = {}
        file_path = self.root / relative_path
        with file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                key, value = line.strip().split(" ", 1)
                mapping[int(key)] = value
        return mapping

    def _read_int_mapping(self, relative_path: str) -> Dict[int, int]:
        mapping: Dict[int, int] = {}
        file_path = self.root / relative_path
        with file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                key, value = line.strip().split(" ", 1)
                mapping[int(key)] = int(value)
        return mapping

    def _load_samples(self) -> List[Tuple[Path, int]]:
        image_paths = self._read_mapping("images.txt")
        class_labels = self._read_int_mapping("image_class_labels.txt")
        split_flags = self._read_int_mapping("train_test_split.txt")
        want_train = self.split == "train"

        samples: List[Tuple[Path, int]] = []
        for image_id, relative_path in image_paths.items():
            is_train = split_flags[image_id] == 1
            if is_train != want_train:
                continue
            image_path = self.root / "images" / relative_path
            label = class_labels[image_id] - 1
            samples.append((image_path, label))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, target = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, target
