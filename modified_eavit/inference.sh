#!/bin/bash

STAGE='inference eval'
BATCH_SIZE=16
DATASET="cifar10_full"
DEVICE="cuda:0"
CHECKPOINT="your trained checkpoint path after stage2"
ENTROPY_PATCH_SIZE=16

python inference.py \
  --stage $STAGE \
  --batch_size $BATCH_SIZE \
  --dataset $DATASET \
  --stage2_checkpoint_path $CHECKPOINT \
  --entropy_patch_size $ENTROPY_PATCH_SIZE \
  --device $DEVICE \
