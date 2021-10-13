#!/bin/bash

DATASET_ROOT="/hdd/mliuzzolino/datasets"  # Specify location of datasets
EXPERIMENT_ROOT="/hdd/mliuzzolino/cascaded_nets"  # Specify experiment root
SPLIT_IDXS_ROOT="/hdd/mliuzzolino/split_idxs"  # Specify root of dataset split_idxs

MODEL="resnet18_small"  # resnet10, resnet18, resnet18_small, resnet34, resnet50, densenet_cifar
DATASET_NAME="CIFAR10"  # CIFAR10, CIFAR100, TinyImageNet, ImageNet2012
EXPERIMENT_NAME="${MODEL}_${DATASET_NAME}"

# Model params
TRAIN_MODE="cascaded"  # baseline, cascaded
CASCADED_SCHEME="parallel"  # serial, parallel
MULTIPLE_FCS=false

# LAMBDA_VALS=(1.0)  # To sweep, set as list. E.g., LAMBDA_VALS=(0.0 0.5 0.8 1.0)
LAMBDA_VALS=(0.5) # (0.83 0.9) (0.0 1.0)
TAU_WEIGHTED_LOSS=false
PRETRAINED_WEIGHTS=false
USE_ALL_ICS=false

DISTILLATION=true
DISTILLATION_ALPHA=0.5
DISTILLATION_TEMP=1.0
# TEACHER_DIR="/hdd/mliuzzolino/cascaded_nets/resnet18_cifar10/experiments/std,lr_0.1,wd_0.0005,seed_42"
# TEACHER_DIR="/hdd/mliuzzolino/cascaded_nets/resnet18_ImageNet2012/experiments/std,lr_0.01,wd_0.0005,seed_42"
TEACHER_DIR="teacher_ckpts/resnet18_CIFAR10"

# Optimizer / LR Scheduling
LR_MILESTONES=(30 60 90)
LR=0.01  # 0.01 for all cases, but for imagenet tdlambda=1.0, use 0.001
WEIGHT_DECAY=0.0005
MOMENTUM=0.9
NESTEROV=true

# General / Dataset / Train params
DEVICE=0
RANDOM_SEEDS=(42)  # To sweep, set as list. E.g., RANDOM_SEEDS=(42 542 1042)
EPOCHS=120
BATCH_SIZE=32  # 128
NUM_WORKERS=4
DEBUG=false

for RANDOM_SEED in "${RANDOM_SEEDS[@]}"
do
    for LAMBDA_VAL in "${LAMBDA_VALS[@]}"
    do
      if [[ "$DISTILLATION" = true ]]
      then
        echo "Distillation!"
      else
        echo "Not distillation!"
      fi
      exit
      cmd=( python train.py )   # create array with one element
      cmd+=( --device $DEVICE )
      cmd+=( --random_seed $RANDOM_SEED )
      cmd+=( --dataset_root $DATASET_ROOT )
      cmd+=( --dataset_name $DATASET_NAME )
      ${DISTILLATION} && cmd+=( --distillation )
      cmd+=( --distillation_alpha $DISTILLATION_ALPHA )
      cmd+=( --distillation_temperature $DISTILLATION_TEMP )
      cmd+=( --teacher_dir $TEACHER_DIR )
      cmd+=( --split_idxs_root $SPLIT_IDXS_ROOT )
      cmd+=( --experiment_root $EXPERIMENT_ROOT )
      cmd+=( --experiment_name $EXPERIMENT_NAME )
      cmd+=( --n_epochs $EPOCHS )
      cmd+=( --model_key $MODEL )
      cmd+=( --cascaded_scheme $CASCADED_SCHEME )
      cmd+=( --lambda_val $LAMBDA_VAL )
      cmd+=( --train_mode $TRAIN_MODE )
      cmd+=( --batch_size $BATCH_SIZE )
      cmd+=( --num_workers $NUM_WORKERS )
      cmd+=( --learning_rate $LR )
      cmd+=( --lr_milestones "${LR_MILESTONES[@]}" )
      cmd+=( --momentum $MOMENTUM )
      cmd+=( --weight_decay $WEIGHT_DECAY )
      ${NESTEROV} && cmd+=( --nesterov )
      ${TAU_WEIGHTED_LOSS} && cmd+=( --tau_weighted_loss )
      ${PRETRAINED_WEIGHTS} && cmd+=( --use_pretrained_weights )
      ${MULTIPLE_FCS} && cmd+=( --multiple_fcs )
      ${USE_ALL_ICS} && cmd+=( --use_all_ICs )
      ${DEBUG} && cmd+=( --debug ) && echo "DEBUG MODE ENABLED"

      # Run command
      "${cmd[@]}"
    done
done