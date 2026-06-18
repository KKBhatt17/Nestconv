#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:?usage: bash scripts/run_step1.sh <config>}"
python -m elastic_vit.cli rearrange --config "${CONFIG_PATH}"
