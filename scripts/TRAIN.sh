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
    $CURRENT_DIR/../experiment/experiment.py \
    -d_list      $LIST_DIR \
    -d_feature   $FEATURE_DIR \
    -d_label     $LABEL_DIR \
    -d_midi      $MIDI_DIR \
    -d_cache     $NORM_DIR \
    --epochs        20 \
    --batch_size    32 \
    --lr            3e-4 \
    --blocks        8 \
    --dim           512 \
    --feature_dim   512 \
    --scheduler     onecycle \
    --threshold     0.5 \
    --wandb_project fine-amt \
    --checkpoint_dir $CHECKPOINT_DIR \
    --metadata_workers 8 \
    --num_workers   4 \
    --prefetch_factor 4 \
    --grad_accum_steps 1 \
    --metric_interval 8 \
    --seed          0
