#!/usr/bin/env bash
# Run the FineAMT ablation study: trains one model without label conditioning
# (no_midi) and one without label perturbation (no_perturb), back-to-back.
# Consumes the same pre-packed dataset as TRAIN.sh.
#
# Usage:
#   bash ABLATE.sh                # run both ablations
#   bash ABLATE.sh no_midi        # run only the "no_midi" variant
#   bash ABLATE.sh no_perturb     # run only the "no_perturb" variant

set -e

CURRENT_DIR=$(pwd)
MAESTRO_DIR=$CURRENT_DIR/../dataset/corpus/MAESTRO-V3
DATASET_DIR=$MAESTRO_DIR/dataset                   # contains train/, valid/, test/
CHECKPOINT_ROOT=$CURRENT_DIR/../checkpoints/ablation

export PYTHONPATH=$CURRENT_DIR/..

# Detect number of GPUs (default 1)
NGPUS=${NGPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}
NGPUS=${NGPUS:-1}

TRAIN_SCRIPT=$CURRENT_DIR/../experiment/refine_ablation.py

# Ablations to run (override via positional args)
if [ "$#" -gt 0 ]; then
    ABLATIONS=("$@")
else
    ABLATIONS=("no_midi" "no_perturb")
fi

run_one() {
    local ABL=$1
    local CKPT_DIR=$CHECKPOINT_ROOT/$ABL
    mkdir -p "$CKPT_DIR"

    local ARGS=(
        --ablation        "$ABL"
        --dataset_dir     "$DATASET_DIR"
        --checkpoint_dir  "$CKPT_DIR"
        --wandb_project   refine-amt-ablation
        --wandb_name      "ablation-$ABL"
        --n_mels          128
        --dt              0.016
        --lambda_correction 1.0
        --seed            0
        --wandb_log_checkpoints
        --wandb_ckpt_alias best
    )

    echo "============================================================"
    echo " Running ablation: $ABL"
    echo " Checkpoints     : $CKPT_DIR"
    echo " GPUs            : $NGPUS"
    echo "============================================================"

    if [ "$NGPUS" = "1" ]; then
        python "$TRAIN_SCRIPT" "${ARGS[@]}"
    else
        export TORCHELASTIC_ERROR_FILE=/tmp/torchelastic_error.json
        torchrun --nproc_per_node=$NGPUS "$TRAIN_SCRIPT" "${ARGS[@]}"
        local rc=$?
        if [ $rc -ne 0 ] && [ -f "$TORCHELASTIC_ERROR_FILE" ]; then
            echo "----- torchrun child failed; traceback from $TORCHELASTIC_ERROR_FILE -----"
            cat "$TORCHELASTIC_ERROR_FILE"
            exit $rc
        fi
    fi
}

for ABL in "${ABLATIONS[@]}"; do
    case "$ABL" in
        no_midi|no_perturb) run_one "$ABL" ;;
        *) echo "Unknown ablation: $ABL (expected: no_midi, no_perturb)"; exit 2 ;;
    esac
done

echo "All ablations complete."
