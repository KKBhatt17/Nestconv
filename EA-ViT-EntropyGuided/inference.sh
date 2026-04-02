#!/bin/bash

STAGE='inference entropy guided'
DATASET="cifar10_full"
DEVICE="cuda:0"
CHECKPOINT="your trained checkpoint path after stage2"

python inference.py \
  --stage $STAGE \
  --dataset $DATASET \
  --stage2_checkpoint_path $CHECKPOINT \
  --device $DEVICE \
