# Elastic-ViT-SS

MMSegmentation codebase for ADE20K semantic segmentation with UPerNet and an input-adaptive Elastic ViT backbone.

## Model

- `configs/upernet_elastic_vit_ade20k.py`: UPerNet decode head over Elastic ViT pyramid features.
- Validation metric: `mIoU` on ADE20K validation.

Set these fields in the config before training:

- `model.backbone.checkpoint_path`: elastic ViT checkpoint from stage 2.
- `model.backbone.router_checkpoint`: optional trained input-adaptive router checkpoint.
- `model.backbone.router_input_mode`: `entropy`, `hidden`, or `both`.

## Commands

Run from an MMSegmentation checkout or an environment where `tools/train.py`, `tools/test.py`, and `tools/dist_train.sh` are available:

```bash
PYTHONPATH=/path/to/Ealstic-Vit-SS:$PYTHONPATH bash tools/dist_train.sh /path/to/Ealstic-Vit-SS/configs/upernet_elastic_vit_ade20k.py 8
PYTHONPATH=/path/to/Ealstic-Vit-SS:$PYTHONPATH python tools/test.py /path/to/Ealstic-Vit-SS/configs/upernet_elastic_vit_ade20k.py /path/to/checkpoint.pth
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

