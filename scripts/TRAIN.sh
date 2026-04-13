#!/usr/bin/env bash
# Train FineAMT using experiment.py with DDP (multi-GPU)
# Paths mirror the layout created by CACHE_DATASET.sh

CURRENT_DIR=$(pwd)
DATASET_DIR=$CURRENT_DIR/../dataset/corpus/MAESTRO-V3

LIST_DIR=$DATASET_DIR/list
FEATURE_DIR=$DATASET_DIR/feature/spec
LABEL_DIR=$DATASET_DIR/label
MIDI_DIR=$DATASET_DIR/midi
NORM_DIR=$DATASET_DIR/midi/norm

CHECKPOINT_DIR=$CURRENT_DIR/../checkpoints

export PYTHONPATH=$CURRENT_DIR/..

# Detect number of GPUs (default 2)
NGPUS=${NGPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}
NGPUS=${NGPUS:-1}

torchrun --nproc_per_node=$NGPUS \
    $CURRENT_DIR/../experiment/refine_experiment.py \
    --list_dir       $LIST_DIR \
    --feature_dir    $FEATURE_DIR \
    --midi_dir       $MIDI_DIR \
    --midi_cache_dir $NORM_DIR \
    --checkpoint_dir $CHECKPOINT_DIR \
    --seed           0
