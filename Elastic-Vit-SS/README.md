# Elastic-Vit-SS

ADE20K semantic segmentation with the **weight-shared Elastic ViT** backbone from
[`../Elastic-ViT`](../Elastic-ViT), built on **MMSegmentation 1.x**.

The backbone is initialized from the Elastic-ViT **stage-2** checkpoint and trained
end-to-end with the decode head. Training is *elastic / weight-shared*: an
`ElasticConfigHook` samples one per-layer sub-network per iteration (large-to-small
curriculum), so the single trained model can be evaluated at **any** sub-network.
There is **no router** — you choose the evaluation sub-networks yourself.

## Model

- `configs/upernet_elastic_vit_ade20k.py` — UPerNet (`UPerHead` over 4 ViT taps
  `out_indices=(2,5,8,11)` resampled to strides {4,8,16,32}, plus an `FCNHead` aux).
  Reports **mIoU** (and aAcc/mAcc).

## Install

```bash
pip install -r requirements.txt
# mmcv must match your torch/CUDA; see https://mmcv.readthedocs.io for the right wheel.
```

## Set the backbone checkpoint

In the config set `backbone_checkpoint` to the stage-2 file, e.g.
`../Elastic-ViT/outputs/elastic/imagenet1k/elastic_epoch_30.pt`, or pass it on the CLI:

```bash
--cfg-options model.backbone.checkpoint_path=/abs/path/elastic_epoch_30.pt
```

## Train

```bash
# Single GPU
python tools/train.py configs/upernet_elastic_vit_ade20k.py

# Multi GPU (torch.run / torchrun)
torchrun --nproc_per_node=8 tools/train.py configs/upernet_elastic_vit_ade20k.py --launcher pytorch
# or
bash scripts/train_upernet.sh 8
```

DDP note: sub-network sampling is seeded by the global step, so every rank runs the
*same* sub-network each iteration. Masking keeps all parameters "used", so DDP needs
no `find_unused_parameters`.

## Evaluate N sub-networks with GMACs

Edit `configs/eval_presets.yaml` to set your N configurations, then:

```bash
# Single GPU
python tools/eval_configs.py configs/upernet_elastic_vit_ade20k.py work_dirs/.../iter_160000.pth \
    --configs-file configs/eval_presets.yaml --input-hw 512 512

# Multi GPU
bash scripts/eval_configs.sh 8 configs/upernet_elastic_vit_ade20k.py work_dirs/.../iter_160000.pth
```

This prints, per sub-network, the mIoU plus the analytic **backbone GMACs** at
`--input-hw`, and writes `eval_configs.json` / `.csv` to the work dir.

GMACs are analytic (`elastic_vit_ss/utils/flops.py`): they reflect the *realized*
sub-network rather than the masked full-width forward a profiler would count. The full
ViT-B/16 backbone at 224×224 (14×14+1 tokens) is ≈ **17.44 GMACs**, matching Elastic-ViT.

## ADE20K layout

```text
data/ADEChallengeData2016/
  images/training/ ...
  images/validation/ ...
  annotations/training/ ...
  annotations/validation/ ...
```
Override `data_root` in the config if your data lives elsewhere.
