#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1}
CHECKPOINT=${2}

python tools/test.py "${CONFIG}" "${CHECKPOINT}"

