from __future__ import annotations

from typing import Dict, Iterable, Optional

import torch
from tqdm import tqdm

from elastic_vit_retrieval.data.datasets import RetrievalEvalBundle
from elastic_vit_retrieval.data.entropy_cache import lookup_batch_entropy
from elastic_vit_retrieval.models.common import SubnetworkConfig, make_subnetwork_config
from elastic_vit_retrieval.utils.flops import estimate_vit_macs
from elastic_vit_retrieval.utils.metrics import AverageMeter, compute_retrieval_recalls


def _reorder_features(features: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    ordered = torch.empty_like(features)
    ordered[indices.long()] = features
    return ordered


@torch.no_grad()
def encode_text_corpus(model, text_loader, device: torch.device) -> torch.Tensor:
    model.eval()
    text_features = []
    text_indices = []
    for batch in tqdm(text_loader, desc="encode text", leave=False):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        features = model.encode_text(input_ids, attention_mask)
        text_features.append(features.cpu())
        text_indices.append(batch["text_indices"].cpu())
    return _reorder_features(torch.cat(text_features, dim=0), torch.cat(text_indices, dim=0))


@torch.no_grad()
def encode_image_corpus(
    model,
    image_loader,
    device: torch.device,
    config: Optional[SubnetworkConfig] = None,
) -> torch.Tensor:
    model.eval()
    image_features = []
    image_indices = []
    for batch in tqdm(image_loader, desc="encode image", leave=False):
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        features = model.encode_image(pixel_values, config=config)
        image_features.append(features.cpu())
        image_indices.append(batch["indices"].cpu())
    return _reorder_features(torch.cat(image_features, dim=0), torch.cat(image_indices, dim=0))


@torch.no_grad()
def evaluate_subnetwork(
    model,
    eval_bundle: RetrievalEvalBundle,
    device: torch.device,
    config: SubnetworkConfig,
    text_features: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    if text_features is None:
        text_features = encode_text_corpus(model, eval_bundle.text_loader, device)
    image_features = encode_image_corpus(model, eval_bundle.image_loader, device, config=config)
    metrics = compute_retrieval_recalls(
        image_features=image_features,
        text_features=text_features,
        image_to_text_map=eval_bundle.image_to_text_map,
        text_to_image_map=eval_bundle.text_to_image_map,
    )
    metrics["macs"] = float(estimate_vit_macs(model.embed_dim, config))
    return metrics


@torch.no_grad()
def evaluate_subnetwork_levels(
    model,
    eval_bundle: RetrievalEvalBundle,
    device: torch.device,
    presets: Iterable[dict],
    num_layers: int,
) -> list[Dict[str, float]]:
    text_features = encode_text_corpus(model, eval_bundle.text_loader, device)
    results = []
    for preset in presets:
        config = make_subnetwork_config(
            mlp_widths=preset["mlp_width"],
            num_heads=preset["num_heads"],
            num_layers=num_layers,
        )
        metrics = evaluate_subnetwork(model, eval_bundle, device, config=config, text_features=text_features)
        metrics["name"] = preset["name"]
        results.append(metrics)
    return results


@torch.no_grad()
def evaluate_router_retrieval(
    model,
    router,
    eval_bundle: RetrievalEvalBundle,
    entropy_values: torch.Tensor,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    router.eval()
    text_features = encode_text_corpus(model, eval_bundle.text_loader, device)
    image_features = []
    image_indices = []
    macs_meter = AverageMeter()

    for batch in tqdm(eval_bundle.image_loader, desc="router eval", leave=False):
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        indices = batch["indices"]
        batch_entropy = lookup_batch_entropy(entropy_values, indices).to(device)
        config = router.predict_hard_config(batch_entropy.mean().unsqueeze(0))
        features = model.encode_image(pixel_values, config=config)
        image_features.append(features.cpu())
        image_indices.append(indices.cpu())
        macs_meter.update(float(estimate_vit_macs(model.embed_dim, config)), pixel_values.size(0))

    ordered_image_features = _reorder_features(torch.cat(image_features, dim=0), torch.cat(image_indices, dim=0))
    metrics = compute_retrieval_recalls(
        image_features=ordered_image_features,
        text_features=text_features,
        image_to_text_map=eval_bundle.image_to_text_map,
        text_to_image_map=eval_bundle.text_to_image_map,
    )
    metrics["macs"] = macs_meter.avg
    return metrics
