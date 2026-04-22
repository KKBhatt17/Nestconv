from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict

import torch
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoTokenizer

from elastic_vit_retrieval.data.datasets import (
    build_entropy_sorted_loader,
    build_entropy_source_dataset,
    build_eval_bundle,
)
from elastic_vit_retrieval.data.entropy_cache import build_entropy_cache, lookup_batch_entropy
from elastic_vit_retrieval.engine.evaluate import evaluate_router_retrieval
from elastic_vit_retrieval.engine.losses import clip_contrastive_loss, cosine_distillation_loss
from elastic_vit_retrieval.models.elastic_clip import ElasticCLIPRetrievalModel
from elastic_vit_retrieval.models.router import EntropyRouter, expected_router_param_cost
from elastic_vit_retrieval.utils.flops import estimate_vit_macs
from elastic_vit_retrieval.utils.io import ensure_dir, save_checkpoint
from elastic_vit_retrieval.utils.metrics import AverageMeter, batch_retrieval_accuracy


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_router_training(config: Dict) -> None:
    dataset_cfg = config["dataset"]
    pretrained_cfg = config["pretrained"]
    runtime_cfg = config["runtime"]
    step_cfg = config["stage3"]

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
        checkpoint_path=step_cfg["elastic_checkpoint"],
        cache_dir=pretrained_cfg.get("cache_dir"),
        mlp_choices=step_cfg["mlp_choices"],
        head_choices=step_cfg["head_choices"],
        freeze_text=True,
    ).to(device)
    model.freeze_backbone()
    model.eval()

    router = EntropyRouter(
        num_layers=model.num_layers,
        hidden_dim=step_cfg["router_hidden_dim"],
        mlp_choices=step_cfg["mlp_choices"],
        head_choices=step_cfg["head_choices"],
    ).to(device)

    train_entropy_dataset = build_entropy_source_dataset(dataset_cfg["name"], dataset_cfg["root"], "train")
    train_entropy_cache_path = Path(step_cfg["entropy_cache_dir"]) / "coco_retrieval_train_entropy.pt"
    train_entropy_values = build_entropy_cache(
        train_entropy_dataset,
        train_entropy_cache_path,
        image_size=dataset_cfg["image_size"],
        patch_size=dataset_cfg["patch_size"],
        num_scales=dataset_cfg["entropy_num_scales"],
    )

    val_entropy_dataset = build_entropy_source_dataset(dataset_cfg["name"], dataset_cfg["root"], "val")
    val_entropy_cache_path = Path(step_cfg["entropy_cache_dir"]) / "coco_retrieval_val_entropy.pt"
    val_entropy_values = build_entropy_cache(
        val_entropy_dataset,
        val_entropy_cache_path,
        image_size=dataset_cfg["image_size"],
        patch_size=dataset_cfg["patch_size"],
        num_scales=dataset_cfg["entropy_num_scales"],
    )

    train_loader = build_entropy_sorted_loader(dataset_cfg, runtime_cfg, train_entropy_values, tokenizer, split="train")
    eval_bundle = build_eval_bundle(dataset_cfg, runtime_cfg, tokenizer, split="val")

    optimizer = AdamW(
        router.parameters(),
        lr=step_cfg["optimizer"]["lr"],
        weight_decay=step_cfg["optimizer"]["weight_decay"],
    )

    output_dir = Path(step_cfg["output_dir"])
    ensure_dir(output_dir)
    history = []

    for epoch in range(step_cfg["epochs"]):
        model.eval()
        router.train()
        loss_meter = AverageMeter()
        distillation_meter = AverageMeter()
        retrieval_meter = AverageMeter()
        accuracy_meter = AverageMeter()
        macs_meter = AverageMeter()

        temperature = max(
            step_cfg["temperature"]["min"],
            step_cfg["temperature"]["start"] * (step_cfg["temperature"]["decay"] ** epoch),
        )

        for batch in tqdm(train_loader, desc=f"router train {epoch + 1}/{step_cfg['epochs']}"):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            indices = batch["indices"]
            batch_entropy = lookup_batch_entropy(train_entropy_values, indices).to(device)
            mean_entropy = batch_entropy.mean().unsqueeze(0)

            optimizer.zero_grad(set_to_none=True)
            soft_config, _ = router.sample_relaxed_config(mean_entropy, temperature=temperature, hard=False)

            with torch.no_grad():
                teacher_image_features = model.encode_image(pixel_values)
                text_features = model.encode_text(input_ids, attention_mask)

            student_image_features = model.encode_image(pixel_values, soft_config=soft_config)
            student_logits_per_image, student_logits_per_text = model.compute_similarity_logits(
                student_image_features,
                text_features,
            )
            retrieval_loss = clip_contrastive_loss(student_logits_per_image, student_logits_per_text)
            distillation_loss = cosine_distillation_loss(student_image_features, teacher_image_features)
            efficiency_loss = expected_router_param_cost(
                soft_config.mlp_probabilities,
                soft_config.head_probabilities,
                embed_dim=model.embed_dim,
                mlp_choices=step_cfg["mlp_choices"],
                head_choices=step_cfg["head_choices"],
            )
            total_loss = (
                step_cfg["lambda_param"] * efficiency_loss
                + step_cfg["beta"] * torch.relu(distillation_loss - step_cfg["kd_threshold"])
                + step_cfg["gamma_retrieval"] * retrieval_loss
            )
            total_loss.backward()
            optimizer.step()

            hard_config = router.predict_hard_config(mean_entropy)
            with torch.no_grad():
                hard_image_features = model.encode_image(pixel_values, config=hard_config)
                hard_logits_per_image, _ = model.compute_similarity_logits(hard_image_features, text_features)

            batch_size = pixel_values.size(0)
            loss_meter.update(total_loss.item(), batch_size)
            distillation_meter.update(distillation_loss.item(), batch_size)
            retrieval_meter.update(retrieval_loss.item(), batch_size)
            accuracy_meter.update(batch_retrieval_accuracy(hard_logits_per_image), batch_size)
            macs_meter.update(float(estimate_vit_macs(model.embed_dim, hard_config)), batch_size)

        eval_metrics = evaluate_router_retrieval(
            model,
            router,
            eval_bundle,
            val_entropy_values,
            device,
        )
        epoch_summary = {
            "epoch": epoch + 1,
            "temperature": temperature,
            "train_loss": loss_meter.avg,
            "train_distillation_loss": distillation_meter.avg,
            "train_retrieval_loss": retrieval_meter.avg,
            "train_batch_match": accuracy_meter.avg,
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
