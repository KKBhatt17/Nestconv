# Elastic-ViT-OD

MMDetection codebase for COCO object detection / instance segmentation with an input-adaptive Elastic ViT backbone.

## Models

- `configs/cascade_mask_rcnn_elastic_vit_coco.py`: Cascade Mask R-CNN with bbox and mask heads. Use this option to report Box AP, Box AP50, Box AP75, Mask AP, Mask AP50, and Mask AP75.
- `configs/reppoints_v2_elastic_vit_coco.py`: RepPoints-style detector using the Elastic ViT backbone. This is bbox-only in stock MMDetection, so it reports Box AP, Box AP50, and Box AP75. Mask AP requires a RepPointsV2 instance-mask extension/head.

Set these fields in the config before training:

- `model.backbone.checkpoint_path`: elastic ViT checkpoint from stage 2.
- `model.backbone.router_checkpoint`: optional trained input-adaptive router checkpoint.
- `model.backbone.router_input_mode`: `entropy`, `hidden`, or `both`.

## Commands

Run from an MMDetection checkout or an environment where `tools/train.py`, `tools/test.py`, and `tools/dist_train.sh` are available:

```bash
PYTHONPATH=/path/to/Ealstic-Vit-OD:$PYTHONPATH bash tools/dist_train.sh /path/to/Ealstic-Vit-OD/configs/cascade_mask_rcnn_elastic_vit_coco.py 8
PYTHONPATH=/path/to/Ealstic-Vit-OD:$PYTHONPATH python tools/test.py /path/to/Ealstic-Vit-OD/configs/cascade_mask_rcnn_elastic_vit_coco.py /path/to/checkpoint.pth
```

For COCO `test-dev2017`, `tools/test.py` writes result JSON files under `work_dirs/.../test-dev.*.json`; submit those files to the COCO evaluation server for the 20k test-dev scores.

## COCO Dataset Layout

Place COCO under `data/coco/` or update `data_root` in the configs:

```text
data/coco/
  annotations/
    instances_train2017.json
    instances_val2017.json
    image_info_test-dev2017.json
  train2017/
    000000000009.jpg
    ...
  val2017/
    000000000139.jpg
    ...
  test2017/
    000000000001.jpg
    ...
```

