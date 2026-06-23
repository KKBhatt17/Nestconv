"""Stage 2: curriculum (sandwich) elastic training for VQA.

Ported from the Elastic-ViT classifier's ``train_elastic`` with the router and
entropy-sorted loading removed. Each step samples a random per-layer subnetwork
from the currently *unlocked* width choices and trains the VQA model under it; a
large-to-small curriculum gradually unlocks smaller widths as training
progresses. At the end of each epoch the model is evaluated at the fixed
``eval_presets``.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List

import torch
from torch.optim import AdamW
from tqdm import tqdm

from elastic_vqa.data.datasets import build_dataloaders, build_tokenizer, build_vocab, get_dataset_metadata
from elastic_vqa.engine.evaluate_vqa import evaluate_subnetwork_levels
from elastic_vqa.engine.losses import vqa_loss
from elastic_vqa.models.common import SubnetworkConfig
from elastic_vqa.models.vqa_model import VqaElasticModel
from elastic_vqa.utils.io import ensure_dir, save_checkpoint
from elastic_vqa.utils.metrics import AverageMeter, vqa_accuracy


def get_unlocked_choice_count(epoch: int, epochs: int, unlock_fractions: List[float]) -> int:
    progress = (epoch + 1) / max(epochs, 1)
    stages = sum(progress >= fraction for fraction in unlock_fractions)
    return max(1, min(stages, len(unlock_fractions)))


def sample_layerwise_subnetwork(
    num_layers: int,
    mlp_choices: List[int],
    head_choices: List[int],
    unlocked_count: int,
) -> SubnetworkConfig:
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
    get_dataset_metadata(dataset_cfg["name"])  # validate dataset is supported

    vocab = build_vocab(dataset_cfg)
    tokenizer = build_tokenizer(step_cfg.get("blip_model_name", "Salesforce/blip-vqa-base"))

    model = VqaElasticModel(
        num_answers=len(vocab),
        vision_model_name=step_cfg.get("vision_model_name", "vit_base_patch16_224"),
        blip_model_name=step_cfg.get("blip_model_name", "Salesforce/blip-vqa-base"),
        vision_checkpoint=step_cfg.get("vision_checkpoint"),
        vision_pretrained=step_cfg.get("use_timm_pretrained", False),
        mlp_choices=step_cfg["mlp_choices"],
        head_choices=step_cfg["head_choices"],
        head_hidden_dim=step_cfg.get("head_hidden_dim", 0),
        freeze_text=step_cfg.get("freeze_text", True),
    ).to(device)

    train_loader, val_loader = build_dataloaders(dataset_cfg, runtime_cfg, vocab, tokenizer)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable,
        lr=step_cfg["optimizer"]["lr"],
        weight_decay=step_cfg["optimizer"]["weight_decay"],
    )

    output_dir = Path(step_cfg["output_dir"])
    ensure_dir(output_dir)
    vocab.save(output_dir / "answer_vocab.json")

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

        for batch in tqdm(train_loader, desc=f"elastic train {epoch + 1}/{step_cfg['epochs']}"):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            chosen_config = sample_layerwise_subnetwork(
                num_layers=model.num_layers,
                mlp_choices=step_cfg["mlp_choices"],
                head_choices=step_cfg["head_choices"],
                unlocked_count=unlocked_choice_count,
            )

            optimizer.zero_grad(set_to_none=True)
            logits = model(pixel_values, input_ids, attention_mask, config=chosen_config)
            loss = vqa_loss(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            loss_meter.update(loss.item(), batch_size)
            acc_meter.update(vqa_accuracy(logits, labels), batch_size)

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
            "train_accuracy": acc_meter.avg,
            "unlocked_choice_count": unlocked_choice_count,
            "evaluations": eval_results,
        }
        history.append(epoch_summary)
        print(json.dumps(epoch_summary, indent=2))

        save_checkpoint(
            output_dir / f"vqa_elastic_epoch_{epoch + 1}.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch + 1,
            extra={"history": history},
        )
