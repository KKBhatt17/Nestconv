"""CLEVR loader (VQA-as-classification).

Expected layout under ``dataset.root`` (the standard CLEVR v1.0 release)::

    <root>/
      questions/
        CLEVR_train_questions.json
        CLEVR_val_questions.json
      images/
        train/CLEVR_train_<id>.png
        val/CLEVR_val_<id>.png

CLEVR answers come from a tiny closed set (~28: yes/no, integers 0-10, colors,
shapes, sizes, materials), so the answer vocab fully covers the data -- making
CLEVR a clean fast smoke test for the whole rearrange -> curriculum -> preset-eval
pipeline before committing GPU time to GQA.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, List, Tuple

from PIL import Image
from torch.utils.data import Dataset

from elastic_vqa.data.vocab import AnswerVocab

_SPLIT_FILES = {
    "train": "CLEVR_train_questions.json",
    "val": "CLEVR_val_questions.json",
}
_IMAGE_SUBDIR = {"train": "train", "val": "val"}


def _questions_path(root: str | Path, split: str) -> Path:
    filename = _SPLIT_FILES.get(split, f"CLEVR_{split}_questions.json")
    return Path(root) / "questions" / filename


def _load_records(root: str | Path, split: str) -> List[dict]:
    with _questions_path(root, split).open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return [
        {
            "image_filename": entry["image_filename"],
            "question": entry["question"],
            "answer": str(entry["answer"]),
        }
        for entry in raw["questions"]
    ]


def iter_train_answers(root: str | Path) -> Iterator[str]:
    for record in _load_records(root, "train"):
        yield record["answer"]


class ClevrDataset(Dataset):
    def __init__(self, root: str | Path, split: str, vocab: AnswerVocab, transform=None) -> None:
        self.root = Path(root)
        self.split = split
        self.records = _load_records(root, split)
        self.vocab = vocab
        self.transform = transform
        self.images_dir = self.root / "images" / _IMAGE_SUBDIR.get(split, split)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Tuple[object, str, int]:
        record = self.records[index]
        image = Image.open(self.images_dir / record["image_filename"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = self.vocab.encode(record["answer"])
        return image, record["question"], label
