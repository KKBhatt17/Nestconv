#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:?usage: bash scripts/run_step2.sh <config>}"
python -m elastic_vit.cli train-elastic --config "${CONFIG_PATH}"
