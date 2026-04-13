#!/usr/bin/env bash
# Train FineAMT using experiment.py with DDP (multi-GPU)
# Paths mirror the layout created by CACHE_DATASET.sh

CURRENT_DIR=$(pwd)
DATASET_DIR=$CURRENT_DIR/../dataset/corpus/MAESTRO-V3

LIST_DIR=$DATASET_DIR/list
FEATURE_DIR=$DATASET_DIR/feature
LABEL_DIR=$DATASET_DIR/label
MIDI_DIR=$DATASET_DIR/midi
NORM_DIR=$DATASET_DIR/norm

CHECKPOINT_DIR=$CURRENT_DIR/../checkpoints

export PYTHONPATH=$CURRENT_DIR/..

# Detect number of GPUs (default 2)
NGPUS=${NGPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}
NGPUS=${NGPUS:-1}

torchrun --nproc_per_node=$NGPUS \
    $CURRENT_DIR/../experiment/refine_experiment.py \
    --seed          0
