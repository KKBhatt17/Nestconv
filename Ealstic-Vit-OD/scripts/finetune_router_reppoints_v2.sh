#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT=${1:?Usage: finetune_router_reppoints_v2.sh /path/to/dense_task_checkpoint.pth}
CONFIG=${CONFIG:-configs/reppoints_v2_elastic_vit_coco_router_ft.py}
GPUS=${GPUS:-8}

torchrun --nproc_per_node="${GPUS}" tools/train.py "${CONFIG}" --launcher pytorch --cfg-options load_from="${CHECKPOINT}"
