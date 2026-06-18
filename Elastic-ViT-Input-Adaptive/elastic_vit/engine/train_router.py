from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import torch
from torch.optim import AdamW
from tqdm import tqdm

from elastic_vit.data.datasets import (
    build_entropy_sorted_loader,
    build_entropy_source_dataset,
    get_dataset_metadata,
)
from elastic_vit.data.entropy_cache import build_entropy_cache, lookup_batch_entropy
from elastic_vit.engine.losses import distillation_loss
from elastic_vit.models.common import SubnetworkConfig
from elastic_vit.models.elastic_vit import ElasticVisionTransformer
from elastic_vit.models.router import InputAdaptiveRouter, expected_router_param_cost
from elastic_vit.utils.flops import estimate_vit_macs
from elastic_vit.utils.io import ensure_dir, save_checkpoint
from elastic_vit.utils.metrics import AverageMeter, accuracy_top1, multilabel_precision_recall_f1


def routing_to_subnetwork_config(routing: Dict[str, torch.Tensor], step_cfg: Dict, sample_idx: int = 0) -> SubnetworkConfig:
    mlp_indices = routing["mlp_probabilities"][sample_idx].argmax(dim=-1)
    head_indices = routing["head_probabilities"][sample_idx].argmax(dim=-1)
    return SubnetworkConfig(
        mlp_widths=tuple(step_cfg["mlp_choices"][index.item()] for index in mlp_indices),
        num_heads=tuple(step_cfg["head_choices"][index.item()] for index in head_indices),
    )


def average_routed_macs(model, routing: Dict[str, torch.Tensor], step_cfg: Dict) -> float:
    batch_size = routing["mlp_probabilities"].shape[0]
    total = 0.0
    for sample_idx in range(batch_size):
        config = routing_to_subnetwork_config(routing, step_cfg, sample_idx=sample_idx)
        total += float(estimate_vit_macs(model.embed_dim, config))
    return total / max(batch_size, 1)


@torch.no_grad()
def evaluate_router(model, router, data_loader, entropy_values, device, step_cfg, task_type: str):
    model.eval()
    router.eval()
    loss_meter = AverageMeter()
    primary_meter = AverageMeter()
    precision_meter = AverageMeter()
    recall_meter = AverageMeter()
    f1_meter = AverageMeter()
    macs_meter = AverageMeter()

    for images, targets, indices in tqdm(data_loader, desc="router eval", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        batch_entropy = lookup_batch_entropy(entropy_values, indices).to(device)

        logits, routing = model.forward_adaptive(
            images,
            router=router,
            entropy=batch_entropy,
            deterministic=True,
        )
        full_logits = model(images)
        kd_loss = distillation_loss(
            logits,
            full_logits,
            task_type=task_type,
            temperature=step_cfg["distillation_temperature"],
        )
        efficiency_loss = expected_router_param_cost(
            routing["mlp_probabilities"],
            routing["head_probabilities"],
            embed_dim=model.embed_dim,
            mlp_choices=step_cfg["mlp_choices"],
            head_choices=step_cfg["head_choices"],
        )
        loss = (
            step_cfg["lambda_param"] * efficiency_loss
            + step_cfg["beta"] * torch.relu(kd_loss - step_cfg["kd_threshold"])
        )
        macs = average_routed_macs(model, routing, step_cfg)

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
        macs_meter.update(float(macs), batch_size)

    result = {"loss": loss_meter.avg, "macs": macs_meter.avg}
    if task_type == "multilabel":
        result["precision"] = precision_meter.avg
        result["recall"] = recall_meter.avg
        result["f1"] = f1_meter.avg
    else:
        result["top1"] = primary_meter.avg
    return result


def run_router_training(config: Dict) -> None:
    dataset_cfg = config["dataset"]
    runtime_cfg = config["runtime"]
    step_cfg = config["stage3"]
    device = torch.device(runtime_cfg["device"])
    dataset_meta = get_dataset_metadata(dataset_cfg["name"])
    num_classes = int(dataset_meta["num_classes"])
    task_type = str(dataset_meta["task_type"])

    model = ElasticVisionTransformer(
        model_name=step_cfg.get("model_name", "vit_base_patch16_224"),
        pretrained=False,
        num_classes=num_classes,
        checkpoint_path=step_cfg["elastic_checkpoint"],
        mlp_choices=step_cfg["mlp_choices"],
        head_choices=step_cfg["head_choices"],
    ).to(device)
    model.freeze_backbone()
    model.eval()

    router = InputAdaptiveRouter(
        num_layers=model.num_layers,
        embed_dim=model.embed_dim,
        hidden_dim=step_cfg["router_hidden_dim"],
        mlp_choices=step_cfg["mlp_choices"],
        head_choices=step_cfg["head_choices"],
        input_mode=step_cfg.get("router_input_mode", "both"),
    ).to(device)

    entropy_dataset = build_entropy_source_dataset(
        dataset_cfg["name"],
        dataset_cfg["root"],
        "train",
    )
    entropy_cache_path = Path(step_cfg["entropy_cache_dir"]) / f"{dataset_cfg['name']}_train_entropy.pt"
    entropy_values = build_entropy_cache(
        entropy_dataset,
        entropy_cache_path,
        image_size=dataset_cfg["image_size"],
        patch_size=dataset_cfg["patch_size"],
        num_scales=dataset_cfg["entropy_num_scales"],
    )

    val_entropy_dataset = build_entropy_source_dataset(
        dataset_cfg["name"],
        dataset_cfg["root"],
        "val",
    )
    val_entropy_cache_path = Path(step_cfg["entropy_cache_dir"]) / f"{dataset_cfg['name']}_val_entropy.pt"
    val_entropy_values = build_entropy_cache(
        val_entropy_dataset,
        val_entropy_cache_path,
        image_size=dataset_cfg["image_size"],
        patch_size=dataset_cfg["patch_size"],
        num_scales=dataset_cfg["entropy_num_scales"],
    )

    train_loader = build_entropy_sorted_loader(dataset_cfg, runtime_cfg, entropy_values, split="train")
    val_loader = build_entropy_sorted_loader(dataset_cfg, runtime_cfg, val_entropy_values, split="val")

    optimizer = AdamW(
        router.parameters(),
        lr=step_cfg["optimizer"]["lr"],
        weight_decay=step_cfg["optimizer"]["weight_decay"],
    )

    output_dir = Path(step_cfg["output_dir"])
    ensure_dir(output_dir)
    history = []

    for epoch in range(step_cfg["epochs"]):
        router.train()
        loss_meter = AverageMeter()
        primary_meter = AverageMeter()
        precision_meter = AverageMeter()
        recall_meter = AverageMeter()
        f1_meter = AverageMeter()
        macs_meter = AverageMeter()

        temperature = max(
            step_cfg["temperature"]["min"],
            step_cfg["temperature"]["start"] * (step_cfg["temperature"]["decay"] ** epoch),
        )

        for images, targets, indices in tqdm(train_loader, desc=f"router train {epoch + 1}/{step_cfg['epochs']}"):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            batch_entropy = lookup_batch_entropy(entropy_values, indices).to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                teacher_logits = model(images)
            student_logits, routing = model.forward_adaptive(
                images,
                router=router,
                entropy=batch_entropy,
                temperature=temperature,
                hard=False,
            )
            kd_loss = distillation_loss(
                student_logits,
                teacher_logits,
                task_type=task_type,
                temperature=step_cfg["distillation_temperature"],
            )
            efficiency_loss = expected_router_param_cost(
                routing["mlp_probabilities"],
                routing["head_probabilities"],
                embed_dim=model.embed_dim,
                mlp_choices=step_cfg["mlp_choices"],
                head_choices=step_cfg["head_choices"],
            )
            total_loss = (
                step_cfg["lambda_param"] * efficiency_loss
                + step_cfg["beta"] * torch.relu(kd_loss - step_cfg["kd_threshold"])
            )
            total_loss.backward()
            optimizer.step()

            with torch.no_grad():
                hard_logits, hard_routing = model.forward_adaptive(
                    images,
                    router=router,
                    entropy=batch_entropy,
                    deterministic=True,
                )
            batch_size = images.size(0)
            loss_meter.update(total_loss.item(), batch_size)
            if task_type == "multilabel":
                metrics = multilabel_precision_recall_f1(hard_logits, targets)
                primary_meter.update(metrics["f1"], batch_size)
                precision_meter.update(metrics["precision"], batch_size)
                recall_meter.update(metrics["recall"], batch_size)
                f1_meter.update(metrics["f1"], batch_size)
            else:
                primary_meter.update(accuracy_top1(hard_logits, targets), batch_size)
            macs_meter.update(average_routed_macs(model, hard_routing, step_cfg), batch_size)

        eval_metrics = evaluate_router(model, router, val_loader, val_entropy_values, device, step_cfg, task_type=task_type)
        epoch_summary = {
            "epoch": epoch + 1,
            "temperature": temperature,
            "train_loss": loss_meter.avg,
            "train_macs": macs_meter.avg,
            "eval": eval_metrics,
        }
        if task_type == "multilabel":
            epoch_summary["train_precision"] = precision_meter.avg
            epoch_summary["train_recall"] = recall_meter.avg
            epoch_summary["train_f1"] = f1_meter.avg
        else:
            epoch_summary["train_top1"] = primary_meter.avg
        history.append(epoch_summary)
        print(json.dumps(epoch_summary, indent=2))

        save_checkpoint(
            output_dir / f"router_epoch_{epoch + 1}.pt",
            model=router,
            optimizer=optimizer,
            epoch=epoch + 1,
            extra={"history": history, "router_input_mode": step_cfg.get("router_input_mode", "both")},
        )
