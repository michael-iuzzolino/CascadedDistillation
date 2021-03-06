#!/bin/bash

DATASET_ROOT="/hdd/mliuzzolino/datasets"  # Specify location of datasets
EXPERIMENT_ROOT="/hdd/mliuzzolino/cascaded_distillation_nets"  # Specify experiment root
SPLIT_IDXS_ROOT="/hdd/mliuzzolino/split_idxs"  # Specify root of dataset split_idxs

MODEL="resnet18"  # resnet18, resnet18_small, resnet34, resnet50, densenet_cifar
DATASET_NAME="CIFAR100"  # CIFAR10, CIFAR100, TinyImageNet, ImageNet2012
EXPERIMENT_NAME="${MODEL}_${DATASET_NAME}"

# Model params
TRAIN_MODE="cascaded"  # baseline, cascaded
CASCADED_SCHEME="parallel"  # serial, parallel
MULTIPLE_FCS=false

# LAMBDA_VALS # To sweep, set as list. E.g., LAMBDA_VALS=(0.0 0.5 0.8 1.0)
LAMBDA_VALS=(0.5)
TAU_WEIGHTED_LOSS=false
PRETRAINED_WEIGHTS=false
USE_ALL_ICS=false

DISTILLATION=false
DISTILLATION_ALPHAS=(1.0)  #  0.5 0.75 1.0)
DISTILLATION_TEMP=1.0
DISTILLATION_LOSS_MODE="external" # external, internal
TRAINABLE_TEMP=true
TEACHER_ROOT="/hdd/mliuzzolino/cascaded_distillation_nets/resnet18_CIFAR100/experiments"
# TEACHER_EXP_DIR="td(0.0),parallel,lr_0.01,wd_0.0005,seed_42"
TEACHER_EXP_DIR="td(0.9),parallel,lr_0.01,wd_0.001,seed_42"
# TEACHER_EXP_DIR="td(0.5),parallel,lr_0.01,wd_0.001,seed_42"
# TEACHER_EXP_DIR="td(1.0),parallel,lr_0.01,wd_0.0005,seed_42"
TEACHER_DIR="$TEACHER_ROOT/$TEACHER_EXP_DIR"

# Optimizer / LR Scheduling
LR_MILESTONES=(30 60 90)
LR=0.01  # 0.01 for all cases, but for imagenet tdlambda=1.0, use 0.001
WEIGHT_DECAY=0.005  # 0.0005
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
        for DISTILLATION_ALPHA in "${DISTILLATION_ALPHAS[@]}"
        do
          cmd=( python train.py )   # create array with one element
          cmd+=( --device $DEVICE )
          cmd+=( --random_seed $RANDOM_SEED )
          cmd+=( --dataset_root $DATASET_ROOT )
          cmd+=( --dataset_name $DATASET_NAME )
          ${DISTILLATION} && cmd+=( --distillation )
          cmd+=( --distillation_loss_mode $DISTILLATION_LOSS_MODE )
          cmd+=( --distillation_alpha $DISTILLATION_ALPHA )
          cmd+=( --distillation_temperature $DISTILLATION_TEMP )
          ${TRAINABLE_TEMP} && cmd+=( --trainable_temp )
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
      else
        cmd=( python train.py )   # create array with one element
        cmd+=( --device $DEVICE )
        cmd+=( --random_seed $RANDOM_SEED )
        cmd+=( --dataset_root $DATASET_ROOT )
        cmd+=( --dataset_name $DATASET_NAME )
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
      fi
    done
done