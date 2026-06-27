"""Dataset registry, tokenizer, and collation for VQA training.

All VQA datasets share ``task_type == "vqa"`` (classification over the answer
vocab). A dataset is described by its raw-answer iterator (for vocab building) and
its ``Dataset`` factory. Batches are ``(images, questions, labels)`` triples; the
collate fn stacks images, tokenizes the question strings with the BLIP tokenizer,
and stacks answer labels.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Callable, Dict, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from elastic_vqa.data import clevr, dummy, gqa, okvqa
from elastic_vqa.data.transforms import build_image_transform
from elastic_vqa.data.vocab import AnswerVocab, load_or_build_vocab

# task_type is kept for parity with the classification codebase's branching, but
# every entry here is "vqa"; it exists so losses/metrics can dispatch uniformly.
DATASET_METADATA: Dict[str, Dict[str, object]] = {
    "gqa": {"task_type": "vqa", "module": gqa, "dataset_cls": gqa.GqaDataset},
    "clevr": {"task_type": "vqa", "module": clevr, "dataset_cls": clevr.ClevrDataset},
    "okvqa": {"task_type": "vqa", "module": okvqa, "dataset_cls": okvqa.OkvqaDataset},
    "dummy": {"task_type": "vqa", "module": dummy, "dataset_cls": None},
}


def get_dataset_metadata(name: str) -> Dict[str, object]:
    key = name.lower()
    if key not in DATASET_METADATA:
        raise ValueError(f"Unsupported VQA dataset: {name}")
    return DATASET_METADATA[key]


def build_vocab(dataset_cfg: Dict) -> AnswerVocab:
    name = dataset_cfg["name"].lower()
    meta = get_dataset_metadata(name)
    module = meta["module"]
    root = dataset_cfg.get("root")
    return load_or_build_vocab(
        path=dataset_cfg["answer_vocab_path"],
        answers_provider=partial(module.iter_train_answers, root),
        top_k=int(dataset_cfg["answer_vocab_size"]),
    )


def build_dataset(dataset_cfg: Dict, split: str, vocab: AnswerVocab) -> Dataset:
    name = dataset_cfg["name"].lower()
    meta = get_dataset_metadata(name)
    image_size = int(dataset_cfg["image_size"])
    transform = build_image_transform(image_size, train=split == "train")

    if name == "dummy":
        return dummy.DummyVqaDataset(
            split=split,
            vocab=vocab,
            image_size=image_size,
            num_samples=int(dataset_cfg.get("num_samples", 256)),
        )

    if name == "okvqa":
        return okvqa.OkvqaDataset(
            root=dataset_cfg["root"],
            split=split,
            vocab=vocab,
            transform=transform,
            images_root=dataset_cfg.get("images_root"),
        )

    dataset_cls = meta["dataset_cls"]
    return dataset_cls(root=dataset_cfg["root"], split=split, vocab=vocab, transform=transform)


def build_tokenizer(blip_model_name: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(blip_model_name)


def make_collate_fn(tokenizer, max_length: int) -> Callable:
    def collate(batch):
        # Items are (image, question, label) or, for datasets with multiple human
        # answers (OK-VQA), (image, question, label, answer_idxs). The 4th field is
        # optional so single-answer datasets are unaffected.
        fields = list(zip(*batch))
        images, questions, labels = fields[0], fields[1], fields[2]
        pixel_values = torch.stack(list(images), dim=0)
        encoded = tokenizer(
            list(questions),
            padding="longest",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        out = {
            "pixel_values": pixel_values,
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        if len(fields) > 3:
            out["answer_targets"] = torch.tensor(fields[3], dtype=torch.long)
        return out

    return collate


def build_dataloaders(
    dataset_cfg: Dict,
    runtime_cfg: Dict,
    vocab: AnswerVocab,
    tokenizer,
) -> Tuple[DataLoader, DataLoader]:
    collate = make_collate_fn(tokenizer, max_length=int(dataset_cfg["max_question_length"]))
    train_dataset = build_dataset(dataset_cfg, "train", vocab)
    val_dataset = build_dataset(dataset_cfg, "val", vocab)
    train_loader = DataLoader(
        train_dataset,
        batch_size=runtime_cfg["batch_size"],
        shuffle=True,
        num_workers=runtime_cfg["num_workers"],
        pin_memory=True,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=runtime_cfg["batch_size"],
        shuffle=False,
        num_workers=runtime_cfg["num_workers"],
        pin_memory=True,
        collate_fn=collate,
    )
    return train_loader, val_loader
