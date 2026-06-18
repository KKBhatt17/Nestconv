#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/reppoints_v2_elastic_vit_coco.py}
GPUS=${GPUS:-8}

torchrun --nproc_per_node="${GPUS}" tools/train.py "${CONFIG}" --launcher pytorch
