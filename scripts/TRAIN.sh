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

TRAIN_SCRIPT=$CURRENT_DIR/../experiment/refine_experiment.py
TRAIN_ARGS=(
    --dataset_dir       "$DATASET_DIR"
    --checkpoint_dir    "$CHECKPOINT_DIR"
    --n_mels            128
    --dt                0.016
    --p_row             0.5
    --p_flip            0.03
    --lambda_correction 1.0
    --seed              0
)

if [ "$NGPUS" = "1" ]; then
    # Skip torchrun for single-GPU so Python tracebacks aren't swallowed by the
    # elastic launcher (it reports `error_file: <N/A>` since main isn't @record-decorated).
    python "$TRAIN_SCRIPT" "${TRAIN_ARGS[@]}"
else
    export TORCHELASTIC_ERROR_FILE=/tmp/torchelastic_error.json
    torchrun --nproc_per_node=$NGPUS "$TRAIN_SCRIPT" "${TRAIN_ARGS[@]}"
    rc=$?
    if [ $rc -ne 0 ] && [ -f "$TORCHELASTIC_ERROR_FILE" ]; then
        echo "----- torchrun child failed; traceback from $TORCHELASTIC_ERROR_FILE -----"
        cat "$TORCHELASTIC_ERROR_FILE"
    fi
    exit $rc
fi
