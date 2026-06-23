#!/usr/bin/env bash
# Run the full VQA pipeline for one config: rearrange -> curriculum train.
# Usage: bash scripts/run_all.sh configs/gqa.yaml
set -euo pipefail

CONFIG="${1:?usage: run_all.sh <config.yaml>}"

python -m elastic_vqa.cli rearrange     --config "$CONFIG"
python -m elastic_vqa.cli train-elastic --config "$CONFIG"
