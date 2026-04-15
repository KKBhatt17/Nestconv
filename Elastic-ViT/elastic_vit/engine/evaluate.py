from __future__ import annotations

from typing import Dict, Iterable, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from elastic_vit.engine.losses import classification_loss
from elastic_vit.models.common import SubnetworkConfig, make_subnetwork_config
from elastic_vit.utils.flops import estimate_vit_macs
from elastic_vit.utils.metrics import AverageMeter, accuracy_top1, multilabel_precision_recall_f1


@torch.no_grad()
def evaluate_subnetwork(
    model,
    data_loader: DataLoader,
    device: torch.device,
    config: SubnetworkConfig,
    task_type: str,
) -> Dict[str, float]:
    model.eval()
    loss_meter = AverageMeter()
    primary_meter = AverageMeter()
    precision_meter = AverageMeter()
    recall_meter = AverageMeter()
    f1_meter = AverageMeter()

    for images, targets in tqdm(data_loader, desc="eval", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images, config=config)
        loss = classification_loss(logits, targets, task_type=task_type)
        batch_size = images.size(0)
        loss_meter.update(loss.item(), batch_size)
        if task_type == "multilabel":
            metrics = multilabel_precision_recall_f1(logits, targets)
            primary_meter.update(metrics["f1"], batch_size)
            precision_meter.update(metrics["precision"], batch_size)
            recall_meter.update(metrics["recall"], batch_size)
            f1_meter.update(metrics["f1"], batch_size)
        else:
            primary_meter.update(accuracy_top1(logits, targets), batch_size)

    macs = estimate_vit_macs(model.embed_dim, config)
    result = {"loss": loss_meter.avg, "macs": float(macs)}
    if task_type == "multilabel":
        result.update(
            {
                "precision": precision_meter.avg,
                "recall": recall_meter.avg,
                "f1": f1_meter.avg,
            }
        )
    else:
        result["top1"] = primary_meter.avg
    return result


@torch.no_grad()
def evaluate_subnetwork_levels(
    model,
    data_loader: DataLoader,
    device: torch.device,
    presets: Iterable[dict],
    num_layers: int,
    task_type: str,
) -> List[Dict[str, float]]:
    results = []
    for preset in presets:
        config = make_subnetwork_config(
            mlp_widths=preset["mlp_width"],
            num_heads=preset["num_heads"],
            num_layers=num_layers,
        )
        metrics = evaluate_subnetwork(model, data_loader, device, config, task_type=task_type)
        metrics["name"] = preset["name"]
        results.append(metrics)
    return results
