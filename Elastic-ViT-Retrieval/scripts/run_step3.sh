#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:?usage: bash scripts/run_step3.sh <config>}"
python -m elastic_vit_retrieval.cli train-router --config "${CONFIG_PATH}"
