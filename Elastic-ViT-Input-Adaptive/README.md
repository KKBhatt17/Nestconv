# Elastic Nested ViT

This repository implements a three-stage pipeline for training a nested elastic ViT and an input-adaptive router:

1. Rearrange a pretrained `vit_base_patch16_224` checkpoint so important attention heads and MLP neurons come first.
2. Fine-tune the elastic ViT by sampling subnetworks in a large-to-small curriculum.
3. Freeze the elastic ViT and train per-layer routers with Gumbel-Softmax to choose the next layer's runtime subnetwork.

## Quick Start

```bash
pip install -r requirements.txt
python -m elastic_vit.cli rearrange --config configs/imagenet1k.yaml
python -m elastic_vit.cli train-elastic --config configs/imagenet1k.yaml
python -m elastic_vit.cli train-router --config configs/imagenet1k.yaml
```

Shell wrappers are available in `scripts/`.

## Input-Adaptive Routing

Stage 3 places a router before every ViT layer. Each router predicts only the MLP width and attention-head count for the next layer, then the selected elastic masks are applied before the following layer routes.

Set `stage3.router_input_mode` to choose the router input:

- `entropy`: image entropy only
- `hidden`: previous layer hidden state only
- `both`: concatenate image entropy and previous layer hidden state

## Supported Datasets

- ImageNet-1K
- CIFAR-10
- CIFAR-100
- FGVC Aircraft
- Stanford Cars
- Oxford IIIT Pets
- CUB-200-2011 via a custom loader for the official metadata layout
- COCO 2017 as an 80-way multilabel classification benchmark using image-level object-presence labels

## Task Notes

- Standard datasets use multiclass training with cross-entropy and top-1 accuracy.
- COCO uses multilabel training with BCE-with-logits and reports precision, recall, and F1.
