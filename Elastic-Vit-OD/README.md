# Elastic-Vit-OD

COCO object detection / instance segmentation with the **weight-shared Elastic ViT**
backbone from [`../Elastic-ViT`](../Elastic-ViT), built on **MMDetection 3.x**.

The backbone is initialized from the Elastic-ViT **stage-2** checkpoint and trained
end-to-end with the detector. Training is *elastic / weight-shared*: an
`ElasticConfigHook` samples one per-layer sub-network per iteration (large-to-small
curriculum), so the single trained model can be evaluated at **any** sub-network.
There is **no router** — you choose the evaluation sub-networks yourself.

## Models

- `configs/mask_rcnn_elastic_vit_coco.py` — Mask R-CNN. Reports **Box AP / AP50 / AP75**
  and **Mask AP / AP50 / AP75**.
- `configs/reppoints_elastic_vit_coco.py` — stock RepPoints (v1), bbox-only. Reports
  **Box AP / AP50 / AP75**.

Both feed a standard `FPN` from 4 ViT taps (`out_indices=(2,5,8,11)`, resampled to
strides {4,8,16,32}).

## Install

```bash
pip install -r requirements.txt
# mmcv must match your torch/CUDA; see https://mmcv.readthedocs.io for the right wheel.
```

## Set the backbone checkpoint

In the chosen config set `backbone_checkpoint` to the stage-2 file, e.g.
`../Elastic-ViT/outputs/elastic/imagenet1k/elastic_epoch_30.pt`, or pass it on the CLI:

```bash
--cfg-options model.backbone.checkpoint_path=/abs/path/elastic_epoch_30.pt
```

## Train

```bash
# Single GPU
python tools/train.py configs/mask_rcnn_elastic_vit_coco.py

# Multi GPU (torch.run / torchrun)
torchrun --nproc_per_node=8 tools/train.py configs/mask_rcnn_elastic_vit_coco.py --launcher pytorch
# or
bash scripts/train_mask_rcnn.sh 8
```

DDP note: sub-network sampling is seeded by the global step, so every rank runs the
*same* sub-network each iteration. Masking keeps all parameters "used", so DDP needs
no `find_unused_parameters`.

## Evaluate N sub-networks with GMACs

Edit `configs/eval_presets.yaml` to set your N configurations, then:

```bash
# Single GPU
python tools/eval_configs.py configs/mask_rcnn_elastic_vit_coco.py work_dirs/.../epoch_12.pth \
    --configs-file configs/eval_presets.yaml --input-hw 800 1280

# Multi GPU
bash scripts/eval_configs.sh 8 configs/mask_rcnn_elastic_vit_coco.py work_dirs/.../epoch_12.pth
```

This prints, per sub-network, the COCO metrics (Box/Mask AP/AP50/AP75) plus the analytic
**backbone GMACs** at `--input-hw`, and writes `eval_configs.json` / `.csv` to the work dir.

GMACs are analytic (`elastic_vit_od/utils/flops.py`): they reflect the *realized*
sub-network rather than the masked full-width forward a profiler would count. The full
ViT-B/16 backbone at 224×224 (14×14+1 tokens) is ≈ **17.44 GMACs**, matching Elastic-ViT.

## COCO layout

```text
data/coco/
  annotations/instances_train2017.json
  annotations/instances_val2017.json
  train2017/ ...
  val2017/ ...
```
Override `data_root` in the configs if your data lives elsewhere.
