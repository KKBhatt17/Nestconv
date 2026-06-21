#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/eval_configs.sh <ngpus> <config.py> <checkpoint.pth> [extra args]
NGPUS="${1:?usage: eval_configs.sh <ngpus> <config> <checkpoint>}"
CONFIG="${2:?config required}"
CHECKPOINT="${3:?checkpoint required}"

torchrun --nproc_per_node="${NGPUS}" tools/eval_configs.py \
    "${CONFIG}" "${CHECKPOINT}" \
    --configs-file configs/eval_presets.yaml \
    --launcher pytorch "${@:4}"
