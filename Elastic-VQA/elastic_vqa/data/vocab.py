"""Answer vocabulary for VQA-as-classification.

The vocab is the set of answer strings the model can predict. It is built once
from the training annotations (top-K most frequent answers) and cached to JSON;
both train and eval reuse the same file. Answers outside the vocab map to
``UNANSWERABLE`` (index ``-1``), which the loss ignores and the accuracy metric
counts as incorrect -- i.e. headline accuracy is over the *full* eval set,
including the vocab-coverage cap (see README "Metrics").
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

UNANSWERABLE = -1


class AnswerVocab:
    def __init__(self, answers: List[str]) -> None:
        self.answers = answers
        self.answer_to_index: Dict[str, int] = {answer: idx for idx, answer in enumerate(answers)}

    def __len__(self) -> int:
        return len(self.answers)

    def encode(self, answer: str) -> int:
        return self.answer_to_index.get(normalize_answer(answer), UNANSWERABLE)

    def decode(self, index: int) -> str:
        if index == UNANSWERABLE or index >= len(self.answers):
            return "<unanswerable>"
        return self.answers[index]

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            json.dump(self.answers, handle, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "AnswerVocab":
        with Path(path).open("r", encoding="utf-8") as handle:
            answers = json.load(handle)
        return cls(answers)


def normalize_answer(answer: str) -> str:
    return answer.strip().lower()


def build_answer_vocab(answers: Iterable[str], top_k: int) -> AnswerVocab:
    counter = Counter(normalize_answer(answer) for answer in answers)
    most_common = [answer for answer, _ in counter.most_common(top_k)]
    return AnswerVocab(most_common)


def load_or_build_vocab(
    path: str | Path,
    answers_provider,
    top_k: int,
) -> AnswerVocab:
    """Load the cached vocab if present, otherwise build it via ``answers_provider``.

    ``answers_provider`` is a zero-arg callable returning an iterable of raw answer
    strings (deferred so we only read annotations when the cache is missing).
    """
    target = Path(path)
    if target.exists():
        return AnswerVocab.load(target)
    vocab = build_answer_vocab(answers_provider(), top_k)
    vocab.save(target)
    return vocab
