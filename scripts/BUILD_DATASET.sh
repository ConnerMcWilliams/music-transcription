#!/usr/bin/env bash
# Pack the cached spectrograms + beat-grid MIDI labels into the per-split
# .npy/.npz arrays consumed by RefineDataset. Assumes the upstream artifacts
# from CACHE_DATASET.sh (steps 1-7) already exist.

CURRENT_DIR=$(pwd)
DATASET_SCRIPTS=$CURRENT_DIR/../dataset
CORPUS_DIR=$CURRENT_DIR/../dataset/corpus
MAESTRO_DIR=$CORPUS_DIR/MAESTRO-V3
LIST_DIR=$MAESTRO_DIR/list
FEATURE_DIR=$MAESTRO_DIR/feature/spec
MIDI_NORM_DIR=$MAESTRO_DIR/midi/norm
DATASET_DIR=$MAESTRO_DIR/dataset
CONFIG_FILE=$DATASET_SCRIPTS/config.json

export PYTHONPATH=$CURRENT_DIR/..

mkdir -p $DATASET_DIR
python -m dataset.build_dataset \
    -d_dataset    $DATASET_DIR \
    -d_list       $LIST_DIR \
    -d_feature    $FEATURE_DIR \
    -d_midi_cache $MIDI_NORM_DIR \
    -d_config     $CONFIG_FILE
