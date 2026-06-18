#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/cascade_mask_rcnn_elastic_vit_coco.py}
GPUS=${GPUS:-8}

bash tools/dist_train.sh "${CONFIG}" "${GPUS}"

