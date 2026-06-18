# Elastic-ViT-OD

MMDetection codebase for COCO object detection / instance segmentation with an input-adaptive Elastic ViT backbone.

## Models

- `configs/cascade_mask_rcnn_elastic_vit_coco.py`: Cascade Mask R-CNN with bbox and mask heads. Use this option to report Box AP, Box AP50, Box AP75, Mask AP, Mask AP50, and Mask AP75.
- `configs/reppoints_v2_elastic_vit_coco.py`: RepPoints-style detector using the Elastic ViT backbone. This is bbox-only in stock MMDetection, so it reports Box AP, Box AP50, and Box AP75. Mask AP requires a RepPointsV2 instance-mask extension/head.
- `configs/cascade_mask_rcnn_elastic_vit_coco_router_ft.py`: router-only fine-tuning variant for Cascade Mask R-CNN.
- `configs/reppoints_v2_elastic_vit_coco_router_ft.py`: router-only fine-tuning variant for RepPoints-style detection.

Set these fields in the config before training:

- `model.backbone.checkpoint_path`: initial Elastic ViT backbone checkpoint.
- `model.backbone.router_checkpoint`: optional trained input-adaptive router checkpoint.
- `model.backbone.router_input_mode`: `entropy`, `hidden`, or `both`.

The base configs train the detector and input-adaptive router jointly. To fine-tune only the router for the task, first train a dense-task checkpoint, then launch one of the `*_router_ft.py` configs. These freeze the ViT backbone and set zero LR for the neck/detector heads, leaving `backbone.router` trainable.

## Commands

If MMDetection is installed with pip, run from this directory:

```bash
cd /path/to/Ealstic-Vit-OD

# Single GPU
python tools/train.py configs/cascade_mask_rcnn_elastic_vit_coco.py

# Multi GPU
torchrun --nproc_per_node=8 tools/train.py configs/cascade_mask_rcnn_elastic_vit_coco.py --launcher pytorch

# Router-only fine-tuning from a trained dense-task checkpoint
torchrun --nproc_per_node=8 tools/train.py configs/cascade_mask_rcnn_elastic_vit_coco_router_ft.py --launcher pytorch --cfg-options load_from=/path/to/dense_task_checkpoint.pth

# Test / format COCO test-dev results
python tools/test.py configs/cascade_mask_rcnn_elastic_vit_coco.py /path/to/checkpoint.pth
```

For COCO `test-dev2017`, the local `tools/test.py` writes result JSON files under `work_dirs/.../test-dev.*.json`; submit those files to the COCO evaluation server for the 20k test-dev scores.

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
