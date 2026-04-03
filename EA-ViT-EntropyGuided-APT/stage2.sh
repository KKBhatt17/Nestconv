#!/bin/bash

STAGE='stage2_entropy_guided'
BATCH_SIZE=256
DATASET="cifar10_full"
DEVICE="cuda:0"
LR=1e-5
MIN_LR=1e-7
ACCUM_ITER=1
CHECKPOINT="your trained checkpoint path after stage1"
NSGA_path="./NSGA/cifar10.csv"
CONSTRAINT_GUIDE_PATH="./constraint_guide.csv"
GEN_ID=300

python train_stage2.py \
  --stage $STAGE \
  --batch_size $BATCH_SIZE \
  --lr $LR \
  --max_lr $LR \
  --min_lr $MIN_LR \
  --accum_iter $ACCUM_ITER \
  --dataset $DATASET \
  --stage1_checkpoint_path $CHECKPOINT \
  --nsga_path $NSGA_path \
  --constraint_guide_path $CONSTRAINT_GUIDE_PATH \
  --gen_id $GEN_ID \
  --device $DEVICE \
