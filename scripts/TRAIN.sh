#!/usr/bin/env bash
# Train FineAMT using experiment.py with DDP (multi-GPU).
# Consumes the pre-packed .npy/.npz arrays produced by step 8 of CACHE_DATASET.sh.

CURRENT_DIR=$(pwd)
MAESTRO_DIR=$CURRENT_DIR/../dataset/corpus/MAESTRO-V3
DATASET_DIR=$MAESTRO_DIR/dataset                  # contains train/, valid/, test/ subdirs

CHECKPOINT_DIR=$CURRENT_DIR/../checkpoints

export PYTHONPATH=$CURRENT_DIR/..

# Detect number of GPUs (default 1)
NGPUS=${NGPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}
NGPUS=${NGPUS:-1}

torchrun --nproc_per_node=$NGPUS \
    $CURRENT_DIR/../experiment/refine_experiment.py \
    --dataset_dir       $DATASET_DIR \
    --checkpoint_dir    $CHECKPOINT_DIR \
    --p_row             0.5 \
    --p_flip            0.03 \
    --lambda_correction 1.0 \
    --seed              0
