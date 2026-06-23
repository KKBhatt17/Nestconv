"""GQA loader (VQA-as-classification).

Expected layout under ``dataset.root``::

    <root>/
      questions/
        train_balanced_questions.json
        testdev_balanced_questions.json
      images/
        <imageId>.jpg

Each questions file maps ``questionId -> {imageId, question, answer, ...}``.
GQA has a single ground-truth ``answer`` per question, so plain top-1 over the
answer vocab is the standard accuracy metric.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset

from elastic_vqa.data.vocab import AnswerVocab

# Map our generic split names to GQA's balanced question files.
_SPLIT_FILES = {
    "train": "train_balanced_questions.json",
    "val": "testdev_balanced_questions.json",
}


def _questions_path(root: str | Path, split: str) -> Path:
    filename = _SPLIT_FILES.get(split, f"{split}_balanced_questions.json")
    return Path(root) / "questions" / filename


def _load_records(root: str | Path, split: str) -> List[dict]:
    with _questions_path(root, split).open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return [
        {"image_id": entry["imageId"], "question": entry["question"], "answer": entry["answer"]}
        for entry in raw.values()
    ]


def iter_train_answers(root: str | Path) -> Iterator[str]:
    for record in _load_records(root, "train"):
        yield record["answer"]


class GqaDataset(Dataset):
    def __init__(self, root: str | Path, split: str, vocab: AnswerVocab, transform=None) -> None:
        self.root = Path(root)
        self.records = _load_records(root, split)
        self.vocab = vocab
        self.transform = transform
        self.images_dir = self.root / "images"

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Tuple[object, str, int]:
        record = self.records[index]
        image = Image.open(self.images_dir / f"{record['image_id']}.jpg").convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = self.vocab.encode(record["answer"])
        return image, record["question"], label
