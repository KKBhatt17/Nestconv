from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Sequence

from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class CocoRetrievalEntry:
    image_id: int
    file_name: str
    captions: tuple[str, ...]


def _normalize_caption(caption: str) -> str:
    normalized = " ".join(caption.strip().split())
    return normalized if normalized else "an image"


def load_coco_retrieval_entries(
    root: str | Path,
    split: str,
    annotation_file: str | Path | None = None,
) -> List[CocoRetrievalEntry]:
    root_path = Path(root)
    split_name = "train2017" if split == "train" else "val2017"
    caption_path = Path(annotation_file) if annotation_file else root_path / "annotations" / f"captions_{split_name}.json"

    with caption_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    image_records = {int(record["id"]): record for record in raw["images"]}
    captions_by_image: dict[int, list[str]] = defaultdict(list)
    for annotation in raw["annotations"]:
        captions_by_image[int(annotation["image_id"])].append(_normalize_caption(annotation["caption"]))

    entries: List[CocoRetrievalEntry] = []
    for image_record in raw["images"]:
        image_id = int(image_record["id"])
        captions = tuple(captions_by_image.get(image_id, []))
        if not captions:
            continue
        entries.append(
            CocoRetrievalEntry(
                image_id=image_id,
                file_name=str(image_record["file_name"]),
                captions=captions,
            )
        )
    return entries


def load_image(root: str | Path, split: str, file_name: str) -> Image.Image:
    root_path = Path(root)
    image_dir = root_path / ("train2017" if split == "train" else "val2017")
    image = Image.open(image_dir / file_name)
    return image.convert("RGB")


class CocoRetrievalPairDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        transform: Callable | None = None,
        annotation_file: str | Path | None = None,
        sample_caption: bool = False,
    ) -> None:
        self.root = str(root)
        self.split = split
        self.transform = transform
        self.sample_caption = sample_caption
        self.entries = load_coco_retrieval_entries(root, split, annotation_file=annotation_file)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, object]:
        entry = self.entries[index]
        image = load_image(self.root, self.split, entry.file_name)
        if self.transform is not None:
            image = self.transform(image)
        caption = random.choice(entry.captions) if self.sample_caption else entry.captions[0]
        return {
            "image": image,
            "caption": caption,
            "index": index,
            "image_id": entry.image_id,
        }


class CocoRetrievalImageDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        transform: Callable | None = None,
        annotation_file: str | Path | None = None,
    ) -> None:
        self.root = str(root)
        self.split = split
        self.transform = transform
        self.entries = load_coco_retrieval_entries(root, split, annotation_file=annotation_file)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, object]:
        entry = self.entries[index]
        image = load_image(self.root, self.split, entry.file_name)
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "index": index,
            "image_id": entry.image_id,
        }


class CocoRetrievalTextDataset(Dataset):
    def __init__(self, entries: Sequence[CocoRetrievalEntry]) -> None:
        self.samples: list[dict[str, object]] = []
        for image_index, entry in enumerate(entries):
            for caption in entry.captions:
                self.samples.append(
                    {
                        "caption": caption,
                        "image_index": image_index,
                        "text_index": len(self.samples),
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        return self.samples[index]
