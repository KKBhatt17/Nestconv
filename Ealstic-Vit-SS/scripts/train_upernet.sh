#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/upernet_elastic_vit_ade20k.py}
GPUS=${GPUS:-8}

bash tools/dist_train.sh "${CONFIG}" "${GPUS}"

