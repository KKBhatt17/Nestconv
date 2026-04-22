from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List

import torch
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoTokenizer

from elastic_vit_retrieval.data.datasets import build_eval_bundle, build_train_dataloader
from elastic_vit_retrieval.engine.evaluate import evaluate_subnetwork_levels
from elastic_vit_retrieval.engine.losses import clip_contrastive_loss, cosine_distillation_loss
from elastic_vit_retrieval.models.common import SubnetworkConfig
from elastic_vit_retrieval.models.elastic_clip import ElasticCLIPRetrievalModel
from elastic_vit_retrieval.utils.io import ensure_dir, save_checkpoint
from elastic_vit_retrieval.utils.metrics import AverageMeter, batch_retrieval_accuracy


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


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_elastic_training(config: Dict) -> None:
    dataset_cfg = config["dataset"]
    pretrained_cfg = config["pretrained"]
    runtime_cfg = config["runtime"]
    step_cfg = config["stage2"]

    _set_seed(int(runtime_cfg.get("seed", 42)))
    device = torch.device(runtime_cfg["device"])

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_cfg["model_name"],
        cache_dir=pretrained_cfg.get("cache_dir"),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = ElasticCLIPRetrievalModel(
        model_name=pretrained_cfg["model_name"],
        checkpoint_path=step_cfg.get("checkpoint_path"),
        cache_dir=pretrained_cfg.get("cache_dir"),
        mlp_choices=step_cfg["mlp_choices"],
        head_choices=step_cfg["head_choices"],
        freeze_text=True,
    ).to(device)

    train_loader = build_train_dataloader(dataset_cfg, runtime_cfg, tokenizer)
    eval_bundle = build_eval_bundle(dataset_cfg, runtime_cfg, tokenizer, split="val")
    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=step_cfg["optimizer"]["lr"],
        weight_decay=step_cfg["optimizer"]["weight_decay"],
    )

    output_dir = Path(step_cfg["output_dir"])
    ensure_dir(output_dir)
    history = []

    for epoch in range(step_cfg["epochs"]):
        model.train()
        model.text_model.eval()
        loss_meter = AverageMeter()
        contrastive_meter = AverageMeter()
        distillation_meter = AverageMeter()
        accuracy_meter = AverageMeter()

        unlocked_choice_count = get_unlocked_choice_count(
            epoch,
            step_cfg["epochs"],
            step_cfg["curriculum"]["unlock_fractions"],
        )

        for batch in tqdm(train_loader, desc=f"elastic train {epoch + 1}/{step_cfg['epochs']}"):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)

            chosen_config = sample_layerwise_subnetwork(
                num_layers=model.num_layers,
                mlp_choices=step_cfg["mlp_choices"],
                head_choices=step_cfg["head_choices"],
                unlocked_count=unlocked_choice_count,
            )

            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                teacher_image_features = model.encode_image(pixel_values)
                text_features = model.encode_text(input_ids, attention_mask)
            student_image_features = model.encode_image(pixel_values, config=chosen_config)
            logits_per_image, logits_per_text = model.compute_similarity_logits(student_image_features, text_features)

            contrastive_loss = clip_contrastive_loss(logits_per_image, logits_per_text)
            distillation_loss = cosine_distillation_loss(student_image_features, teacher_image_features)
            total_loss = (
                step_cfg["contrastive_weight"] * contrastive_loss
                + step_cfg["distillation_weight"] * distillation_loss
            )
            total_loss.backward()
            optimizer.step()

            batch_size = pixel_values.size(0)
            loss_meter.update(total_loss.item(), batch_size)
            contrastive_meter.update(contrastive_loss.item(), batch_size)
            distillation_meter.update(distillation_loss.item(), batch_size)
            accuracy_meter.update(batch_retrieval_accuracy(logits_per_image), batch_size)

        eval_results = evaluate_subnetwork_levels(
            model,
            eval_bundle,
            device,
            step_cfg["eval_presets"],
            model.num_layers,
        )
        epoch_summary = {
            "epoch": epoch + 1,
            "train_loss": loss_meter.avg,
            "train_contrastive_loss": contrastive_meter.avg,
            "train_distillation_loss": distillation_meter.avg,
            "train_batch_match": accuracy_meter.avg,
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
