# Elastic-ViT-SS

MMSegmentation codebase for ADE20K semantic segmentation with UPerNet and an input-adaptive Elastic ViT backbone.

## Model

- `configs/upernet_elastic_vit_ade20k.py`: UPerNet decode head over Elastic ViT pyramid features.
- `configs/upernet_elastic_vit_ade20k_router_ft.py`: router-only fine-tuning variant.
- Validation metric: `mIoU` on ADE20K validation.

Set these fields in the config before training:

- `model.backbone.checkpoint_path`: initial Elastic ViT backbone checkpoint.
- `model.backbone.router_checkpoint`: optional trained input-adaptive router checkpoint.
- `model.backbone.router_input_mode`: `entropy`, `hidden`, or `both`.

The base config trains UPerNet and the input-adaptive router jointly. To fine-tune only the router for ADE20K, first train a dense-task checkpoint, then launch `upernet_elastic_vit_ade20k_router_ft.py`. It freezes the ViT backbone and sets zero LR for the decode heads, leaving `backbone.router` trainable.

## Commands

If MMSegmentation is installed with pip, run from this directory:

```bash
cd /path/to/Ealstic-Vit-SS

# Single GPU
python tools/train.py configs/upernet_elastic_vit_ade20k.py

# Multi GPU
torchrun --nproc_per_node=8 tools/train.py configs/upernet_elastic_vit_ade20k.py --launcher pytorch

# Router-only fine-tuning from a trained dense-task checkpoint
torchrun --nproc_per_node=8 tools/train.py configs/upernet_elastic_vit_ade20k_router_ft.py --launcher pytorch --cfg-options load_from=/path/to/dense_task_checkpoint.pth

# Validation
python tools/test.py configs/upernet_elastic_vit_ade20k.py /path/to/checkpoint.pth
```

## ADE20K Dataset Layout

Place ADE20K under `data/ADEChallengeData2016/` or update `data_root` in the config:

```text
data/ADEChallengeData2016/
  images/
    training/
      ADE_train_00000001.jpg
      ...
    validation/
      ADE_val_00000001.jpg
      ...
  annotations/
    training/
      ADE_train_00000001.png
      ...
    validation/
      ADE_val_00000001.png
      ...
```
