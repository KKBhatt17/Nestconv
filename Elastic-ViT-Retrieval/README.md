# Elastic-ViT-Retrieval

This repository adapts the original Elastic-ViT classification pipeline to image-text retrieval with `openai/clip-vit-base-patch16` on COCO captions.

The workflow keeps the same three stages:

1. Rearrange the pretrained CLIP vision tower so important attention heads and MLP neurons appear first.
2. Fine-tune an elastic CLIP image encoder with nested subnetworks while keeping the CLIP text tower frozen.
3. Train an entropy router that chooses a vision subnetwork at runtime from image entropy.

## Repository Layout

- `elastic_vit_retrieval/`: training, evaluation, models, and data code
- `configs/coco_retrieval.yaml`: COCO retrieval configuration
- `scripts/`: thin shell wrappers for the three stages

## Installation

```bash
pip install -r requirements.txt
```

## COCO Layout

Expected COCO directory layout:

```text
/path/to/coco
  train2017/
  val2017/
  annotations/
    captions_train2017.json
    captions_val2017.json
```

This implementation uses the official COCO caption annotations and evaluates retrieval on the validation split.

## Quick Start

```bash
python -m elastic_vit_retrieval.cli rearrange --config configs/coco_retrieval.yaml
python -m elastic_vit_retrieval.cli train-elastic --config configs/coco_retrieval.yaml
python -m elastic_vit_retrieval.cli train-router --config configs/coco_retrieval.yaml
```

## Notes

- Only the CLIP vision tower is made elastic; the text tower stays frozen.
- Retrieval evaluation reports image-to-text and text-to-image recall at `1/5/10`, plus mean recall.
- The router predicts one subnetwork per entropy-sorted batch, matching the batching strategy used in the original codebase.
