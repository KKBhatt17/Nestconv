#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:?usage: bash scripts/run_all.sh <config>}"
bash scripts/run_step1.sh "${CONFIG_PATH}"
bash scripts/run_step2.sh "${CONFIG_PATH}"
bash scripts/run_step3.sh "${CONFIG_PATH}"
