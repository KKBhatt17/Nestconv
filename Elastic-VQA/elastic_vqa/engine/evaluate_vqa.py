"""Fixed-preset evaluation for the elastic VQA model.

Evaluates the trained model at each preset in ``stage2.eval_presets`` (e.g.
largest / large / medium / small / smallest), reporting exact-match VQA accuracy,
CE loss, and vision-tower MACs per preset -- making the accuracy-vs-compute
trade-off legible.
"""

from __future__ import annotations

from typing import Dict, Iterable, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from elastic_vqa.engine.losses import vqa_loss
from elastic_vqa.models.common import SubnetworkConfig, make_subnetwork_config
from elastic_vqa.utils.flops import estimate_vit_macs
from elastic_vqa.utils.metrics import AverageMeter, vqa_accuracy, vqa_soft_accuracy


@torch.no_grad()
def evaluate_subnetwork(
    model,
    data_loader: DataLoader,
    device: torch.device,
    config: SubnetworkConfig,
) -> Dict[str, float]:
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for batch in tqdm(data_loader, desc="eval", leave=False):
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        logits = model(pixel_values, input_ids, attention_mask, config=config)
        loss = vqa_loss(logits, labels)

        batch_size = labels.size(0)
        loss_meter.update(loss.item(), batch_size)
        # Datasets with multiple human answers (OK-VQA) carry ``answer_targets`` and
        # are scored with the official VQA soft accuracy; others use exact-match.
        if "answer_targets" in batch:
            answer_targets = batch["answer_targets"].to(device, non_blocking=True)
            acc_meter.update(vqa_soft_accuracy(logits, answer_targets), batch_size)
        else:
            acc_meter.update(vqa_accuracy(logits, labels), batch_size)

    macs = estimate_vit_macs(model.embed_dim, config)
    return {"loss": loss_meter.avg, "accuracy": acc_meter.avg, "macs": float(macs)}


@torch.no_grad()
def evaluate_subnetwork_levels(
    model,
    data_loader: DataLoader,
    device: torch.device,
    presets: Iterable[dict],
    num_layers: int,
) -> List[Dict[str, float]]:
    results = []
    for preset in presets:
        config = make_subnetwork_config(
            mlp_widths=preset["mlp_width"],
            num_heads=preset["num_heads"],
            num_layers=num_layers,
        )
        metrics = evaluate_subnetwork(model, data_loader, device, config)
        metrics["name"] = preset["name"]
        results.append(metrics)
    return results
