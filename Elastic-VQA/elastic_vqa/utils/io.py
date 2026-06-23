from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def save_checkpoint(path: str | Path, model, optimizer, epoch: int, extra: Dict[str, Any] | None = None) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, target)
