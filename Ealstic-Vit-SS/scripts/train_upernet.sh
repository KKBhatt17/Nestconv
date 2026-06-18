#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/upernet_elastic_vit_ade20k.py}
GPUS=${GPUS:-8}

torchrun --nproc_per_node="${GPUS}" tools/train.py "${CONFIG}" --launcher pytorch
