#!/usr/bin/env bash
set -euo pipefail

NGPUS="${1:-8}"
CONFIG="configs/upernet_elastic_vit_ade20k.py"

torchrun --nproc_per_node="${NGPUS}" tools/train.py "${CONFIG}" --launcher pytorch "${@:2}"
