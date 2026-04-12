from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List

import torch
from torch.optim import AdamW
from tqdm import tqdm

from elastic_vit.data.datasets import DATASET_NUM_CLASSES, build_standard_dataloaders
from elastic_vit.engine.evaluate import evaluate_subnetwork_levels
from elastic_vit.engine.losses import classification_loss
from elastic_vit.models.common import SubnetworkConfig
from elastic_vit.models.elastic_vit import ElasticVisionTransformer
from elastic_vit.utils.io import ensure_dir, save_checkpoint
from elastic_vit.utils.metrics import AverageMeter, accuracy_top1


def get_unlocked_choice_count(epoch: int, epochs: int, unlock_fractions: List[float]) -> int:
    progress = (epoch + 1) / max(epochs, 1)
    stages = sum(progress >= fraction for fraction in unlock_fractions)
    return max(1, min(stages, len(unlock_fractions)))


def sample_layerwise_subnetwork(num_layers: int, mlp_choices: List[int], head_choices: List[int], unlocked_count: int) -> SubnetworkConfig:
    unlocked_mlp = sorted(mlp_choices, reverse=True)[:unlocked_count]
    unlocked_heads = sorted(head_choices, reverse=True)[:unlocked_count]
    return SubnetworkConfig(
        mlp_widths=tuple(random.choice(unlocked_mlp) for _ in range(num_layers)),
        num_heads=tuple(random.choice(unlocked_heads) for _ in range(num_layers)),
    )


def run_elastic_training(config: Dict) -> None:
    dataset_cfg = config["dataset"]
    runtime_cfg = config["runtime"]
    step_cfg = config["stage2"]

    device = torch.device(runtime_cfg["device"])
    num_classes = DATASET_NUM_CLASSES[dataset_cfg["name"].lower()]
    model = ElasticVisionTransformer(
        model_name=step_cfg.get("model_name", "vit_base_patch16_224"),
        pretrained=step_cfg.get("use_timm_pretrained", False),
        num_classes=num_classes,
        checkpoint_path=step_cfg.get("checkpoint_path"),
        mlp_choices=step_cfg["mlp_choices"],
        head_choices=step_cfg["head_choices"],
    ).to(device)

    train_loader, val_loader = build_standard_dataloaders(dataset_cfg, runtime_cfg)
    optimizer = AdamW(
        model.parameters(),
        lr=step_cfg["optimizer"]["lr"],
        weight_decay=step_cfg["optimizer"]["weight_decay"],
    )

    output_dir = Path(step_cfg["output_dir"])
    ensure_dir(output_dir)

    history = []
    for epoch in range(step_cfg["epochs"]):
        model.train()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        unlocked_choice_count = get_unlocked_choice_count(
            epoch,
            step_cfg["epochs"],
            step_cfg["curriculum"]["unlock_fractions"],
        )

        for images, targets in tqdm(train_loader, desc=f"elastic train {epoch + 1}/{step_cfg['epochs']}"):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            chosen_config = sample_layerwise_subnetwork(
                num_layers=model.num_layers,
                mlp_choices=step_cfg["mlp_choices"],
                head_choices=step_cfg["head_choices"],
                unlocked_count=unlocked_choice_count,
            )

            optimizer.zero_grad(set_to_none=True)
            logits = model(images, config=chosen_config)
            loss = classification_loss(logits, targets)
            loss.backward()
            optimizer.step()

            batch_size = images.size(0)
            loss_meter.update(loss.item(), batch_size)
            acc_meter.update(accuracy_top1(logits, targets), batch_size)

        eval_results = evaluate_subnetwork_levels(
            model,
            val_loader,
            device,
            step_cfg["eval_presets"],
            model.num_layers,
        )
        epoch_summary = {
            "epoch": epoch + 1,
            "train_loss": loss_meter.avg,
            "train_top1": acc_meter.avg,
            "unlocked_choice_count": unlocked_choice_count,
            "evaluations": eval_results,
        }
        history.append(epoch_summary)
        print(json.dumps(epoch_summary, indent=2))

        save_checkpoint(
            output_dir / f"elastic_epoch_{epoch + 1}.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch + 1,
            extra={"history": history},
        )
