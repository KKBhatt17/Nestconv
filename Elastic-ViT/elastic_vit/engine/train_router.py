from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import torch
from torch.optim import AdamW
from tqdm import tqdm

from elastic_vit.data.datasets import (
    DATASET_NUM_CLASSES,
    build_entropy_sorted_loader,
    build_entropy_source_dataset,
)
from elastic_vit.data.entropy_cache import build_entropy_cache, lookup_batch_entropy
from elastic_vit.engine.losses import distillation_kl
from elastic_vit.models.elastic_vit import ElasticVisionTransformer
from elastic_vit.models.router import EntropyRouter, expected_router_param_cost
from elastic_vit.utils.flops import estimate_vit_macs
from elastic_vit.utils.io import ensure_dir, save_checkpoint
from elastic_vit.utils.metrics import AverageMeter, accuracy_top1


@torch.no_grad()
def evaluate_router(model, router, data_loader, entropy_values, device, step_cfg):
    model.eval()
    router.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    macs_meter = AverageMeter()

    for images, targets, indices in tqdm(data_loader, desc="router eval", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        batch_entropy = lookup_batch_entropy(entropy_values, indices).to(device)
        config = router.predict_hard_config(batch_entropy.mean().unsqueeze(0))

        logits = model(images, config=config)
        full_logits = model(images)
        kd_loss = distillation_kl(logits, full_logits, temperature=step_cfg["distillation_temperature"])
        one_hot_mlp = torch.zeros(model.num_layers, len(step_cfg["mlp_choices"]), device=device)
        one_hot_head = torch.zeros(model.num_layers, len(step_cfg["head_choices"]), device=device)
        for layer_idx, mlp_width in enumerate(config.mlp_widths):
            one_hot_mlp[layer_idx, step_cfg["mlp_choices"].index(mlp_width)] = 1.0
        for layer_idx, num_heads in enumerate(config.num_heads):
            one_hot_head[layer_idx, step_cfg["head_choices"].index(num_heads)] = 1.0
        efficiency_loss = expected_router_param_cost(
            one_hot_mlp,
            one_hot_head,
            embed_dim=model.embed_dim,
            mlp_choices=step_cfg["mlp_choices"],
            head_choices=step_cfg["head_choices"],
        )
        loss = (
            step_cfg["lambda_param"] * efficiency_loss
            + step_cfg["beta"] * torch.relu(kd_loss - step_cfg["kd_threshold"])
        )
        acc = accuracy_top1(logits, targets)
        macs = estimate_vit_macs(model.embed_dim, config)

        batch_size = images.size(0)
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(acc, batch_size)
        macs_meter.update(float(macs), batch_size)

    return {"loss": loss_meter.avg, "top1": acc_meter.avg, "macs": macs_meter.avg}


def run_router_training(config: Dict) -> None:
    dataset_cfg = config["dataset"]
    runtime_cfg = config["runtime"]
    step_cfg = config["stage3"]
    device = torch.device(runtime_cfg["device"])
    num_classes = DATASET_NUM_CLASSES[dataset_cfg["name"].lower()]

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

    router = EntropyRouter(
        num_layers=model.num_layers,
        hidden_dim=step_cfg["router_hidden_dim"],
        mlp_choices=step_cfg["mlp_choices"],
        head_choices=step_cfg["head_choices"],
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
        acc_meter = AverageMeter()
        macs_meter = AverageMeter()

        temperature = max(
            step_cfg["temperature"]["min"],
            step_cfg["temperature"]["start"] * (step_cfg["temperature"]["decay"] ** epoch),
        )

        for images, targets, indices in tqdm(train_loader, desc=f"router train {epoch + 1}/{step_cfg['epochs']}"):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            batch_entropy = lookup_batch_entropy(entropy_values, indices).to(device)
            mean_entropy = batch_entropy.mean().unsqueeze(0)

            optimizer.zero_grad(set_to_none=True)
            soft_config, _ = router.sample_relaxed_config(mean_entropy, temperature=temperature, hard=False)

            with torch.no_grad():
                teacher_logits = model(images)
            student_logits = model(images, soft_config=soft_config)
            kd_loss = distillation_kl(student_logits, teacher_logits, temperature=step_cfg["distillation_temperature"])
            efficiency_loss = expected_router_param_cost(
                soft_config.mlp_probabilities,
                soft_config.head_probabilities,
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

            hard_config = router.predict_hard_config(mean_entropy)
            with torch.no_grad():
                hard_logits = model(images, config=hard_config)
            batch_size = images.size(0)
            loss_meter.update(total_loss.item(), batch_size)
            acc_meter.update(accuracy_top1(hard_logits, targets), batch_size)
            macs_meter.update(float(estimate_vit_macs(model.embed_dim, hard_config)), batch_size)

        eval_metrics = evaluate_router(model, router, val_loader, val_entropy_values, device, step_cfg)
        epoch_summary = {
            "epoch": epoch + 1,
            "temperature": temperature,
            "train_loss": loss_meter.avg,
            "train_top1": acc_meter.avg,
            "train_macs": macs_meter.avg,
            "eval": eval_metrics,
        }
        history.append(epoch_summary)
        print(json.dumps(epoch_summary, indent=2))

        save_checkpoint(
            output_dir / f"router_epoch_{epoch + 1}.pt",
            model=router,
            optimizer=optimizer,
            epoch=epoch + 1,
            extra={"history": history},
        )
