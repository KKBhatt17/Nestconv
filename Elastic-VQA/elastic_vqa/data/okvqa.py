"""OK-VQA loader (VQA-as-classification, VQAv2/COCO-style annotations).

OK-VQA (https://okvqa.allenai.org/) is a knowledge-based VQA benchmark over COCO
images. Unlike GQA/CLEVR it uses VQAv2-style annotations: questions and answers
live in *separate* JSON files merged by ``question_id``, and each question carries
**10 human answers**.

Expected layout under ``dataset.root`` (official OK-VQA v1.1 + COCO images)::

    <root>/
      questions/
        OpenEnded_mscoco_train2014_questions.json   # {"questions": [{image_id, question, question_id}, ...]}
        OpenEnded_mscoco_val2014_questions.json
      annotations/
        mscoco_train2014_annotations.json           # {"annotations": [{question_id, image_id, answers:[{answer,...}x10]}, ...]}
        mscoco_val2014_annotations.json
      images/
        train2014/COCO_train2014_000000XXXXXX.jpg
        val2014/COCO_val2014_000000XXXXXX.jpg

``train`` -> ``train2014`` files/dir, ``val`` -> ``val2014``. OK-VQA's ``val``
split is the public evaluation set (answers included).

The single training label is the most frequent (officially-normalized) answer, so
the existing cross-entropy path is unchanged; evaluation uses the standard VQA
soft accuracy over all 10 answers (``answer_idxs`` returned per item, scored by
``utils.metrics.vqa_soft_accuracy``).
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterator, List, Tuple

from PIL import Image
from torch.utils.data import Dataset

from elastic_vqa.data.vocab import AnswerVocab

_QUESTION_FILES = {
    "train": "OpenEnded_mscoco_train2014_questions.json",
    "val": "OpenEnded_mscoco_val2014_questions.json",
}
_ANNOTATION_FILES = {
    "train": "mscoco_train2014_annotations.json",
    "val": "mscoco_val2014_annotations.json",
}
_IMAGE_SUBDIR = {"train": "train2014", "val": "val2014"}


# --- Official VQA answer normalization -------------------------------------
# Mirrors VQAEval.processPunctuation + processDigitArticle so soft accuracy
# matches the OK-VQA leaderboard convention.
_CONTRACTIONS = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
    "couldnt": "couldn't", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't",
    "hadnt": "hadn't", "hasnt": "hasn't", "havent": "haven't", "hed": "he'd",
    "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's",
    "Id": "I'd", "Im": "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd",
    "itll": "it'll", "lets": "let's", "maam": "ma'am", "mightve": "might've",
    "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "shant": "shan't",
    "shes": "she's", "shouldve": "should've", "shouldnt": "shouldn't", "thats": "that's",
    "theres": "there's", "theyd": "they'd", "theyll": "they'll", "theyre": "they're",
    "theyve": "they've", "wasnt": "wasn't", "wed": "we'd", "weve": "we've",
    "werent": "weren't", "whatll": "what'll", "whatre": "what're", "whats": "what's",
    "whatve": "what've", "whens": "when's", "whered": "where'd", "wheres": "where's",
    "wholl": "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll",
    "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve": "would've",
    "wouldnt": "wouldn't", "youd": "you'd", "youll": "you'll", "youre": "you're",
    "youve": "you've",
}
_MANUAL_MAP = {
    "none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}
_ARTICLES = {"a", "an", "the"}
_PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
_COMMA_STRIP = re.compile(r"(\d)(,)(\d)")
_PUNCT = [
    ";", "/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-", ">",
    "<", "@", "`", ",", "?", "!",
]


def _process_punctuation(text: str) -> str:
    out = text
    for punct in _PUNCT:
        if (punct + " " in text) or (" " + punct in text) or (_COMMA_STRIP.search(text) is not None):
            out = out.replace(punct, "")
        else:
            out = out.replace(punct, " ")
    out = _PERIOD_STRIP.sub("", out, re.UNICODE)
    return out


def _process_digit_article(text: str) -> str:
    words = []
    for word in text.lower().split():
        word = _MANUAL_MAP.get(word, word)
        if word in _ARTICLES:
            continue
        words.append(_CONTRACTIONS.get(word, word))
    return " ".join(words)


def _process_answer(answer: str) -> str:
    answer = answer.replace("\n", " ").replace("\t", " ").strip()
    answer = _process_punctuation(answer)
    answer = _process_digit_article(answer)
    return answer


# --- Loading ---------------------------------------------------------------
def _questions_path(root: str | Path, split: str) -> Path:
    filename = _QUESTION_FILES.get(split, f"OpenEnded_mscoco_{split}2014_questions.json")
    return Path(root) / "questions" / filename


def _annotations_path(root: str | Path, split: str) -> Path:
    filename = _ANNOTATION_FILES.get(split, f"mscoco_{split}2014_annotations.json")
    return Path(root) / "annotations" / filename


def _image_path(root: str | Path, split: str, image_id: int) -> Path:
    subdir = _IMAGE_SUBDIR.get(split, f"{split}2014")
    return Path(root) / "images" / subdir / f"COCO_{subdir}_{int(image_id):012d}.jpg"


def _load_records(root: str | Path, split: str) -> List[dict]:
    with _questions_path(root, split).open("r", encoding="utf-8") as handle:
        questions = json.load(handle)["questions"]
    with _annotations_path(root, split).open("r", encoding="utf-8") as handle:
        annotations = json.load(handle)["annotations"]

    answers_by_qid = {}
    for ann in annotations:
        answers_by_qid[ann["question_id"]] = [_process_answer(a["answer"]) for a in ann["answers"]]

    records = []
    for entry in questions:
        answers = answers_by_qid.get(entry["question_id"])
        if not answers:
            continue
        label = Counter(answers).most_common(1)[0][0]
        records.append(
            {
                "image_id": entry["image_id"],
                "question": entry["question"],
                "answers": answers,
                "label": label,
            }
        )
    return records


def iter_train_answers(root: str | Path) -> Iterator[str]:
    for record in _load_records(root, "train"):
        yield from record["answers"]


class OkvqaDataset(Dataset):
    def __init__(self, root: str | Path, split: str, vocab: AnswerVocab, transform=None) -> None:
        self.root = Path(root)
        self.split = split
        self.records = _load_records(root, split)
        self.vocab = vocab
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Tuple[object, str, int, List[int]]:
        record = self.records[index]
        image = Image.open(_image_path(self.root, self.split, record["image_id"])).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = self.vocab.encode(record["label"])
        answer_idxs = [self.vocab.encode(answer) for answer in record["answers"]]
        return image, record["question"], label, answer_idxs
