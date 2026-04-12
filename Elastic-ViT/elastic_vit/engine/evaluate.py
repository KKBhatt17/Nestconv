from __future__ import annotations

from typing import Dict, Iterable, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from elastic_vit.engine.losses import classification_loss
from elastic_vit.models.common import SubnetworkConfig, make_subnetwork_config
from elastic_vit.utils.flops import estimate_vit_macs
from elastic_vit.utils.metrics import AverageMeter, accuracy_top1


@torch.no_grad()
def evaluate_subnetwork(model, data_loader: DataLoader, device: torch.device, config: SubnetworkConfig) -> Dict[str, float]:
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for images, targets in tqdm(data_loader, desc="eval", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images, config=config)
        loss = classification_loss(logits, targets)
        acc = accuracy_top1(logits, targets)
        batch_size = images.size(0)
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(acc, batch_size)

    macs = estimate_vit_macs(model.embed_dim, config)
    return {"loss": loss_meter.avg, "top1": acc_meter.avg, "macs": float(macs)}


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
