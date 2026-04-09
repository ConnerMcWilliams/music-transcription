## MAESTRO v3.0.0
CURRENT_DIR=$(pwd)
# Scripts -> 'Music Transcription' -> dataset
DATASET_SCRIPTS=$CURRENT_DIR/../dataset
# Scripts -> 'Music Transcription' -> dataset -> corpus -> MAESTRO-V3
CORPUS_DIR=$CURRENT_DIR/../dataset/corpus
MAESTRO_DIR=$CORPUS_DIR/MAESTRO-V3
LIST_DIR=$MAESTRO_DIR/list
MIDI_DIR=$MAESTRO_DIR/midi
WAV_DIR=$MAESTRO_DIR/wav
FEATURE_DIR=$MAESTRO_DIR/feature
NOTE_DIR=$MAESTRO_DIR/note
LABEL_DIR=$MAESTRO_DIR/label
NORM_DIR=$MAESTRO_DIR/norm
REFERENCE_DIR=$MAESTRO_DIR/reference
DATASET_DIR=$MAESTRO_DIR/dataset
CONFIG_FILE=$DATASET_SCRIPTS/config.json

# 7. beat-normalize cached spectrograms and labels
mkdir -p $NORM_DIR
python $DATASET_SCRIPTS/cache_spec.py \
    -d_list $LIST_DIR \
    -d_feature $FEATURE_DIR \
    -d_label $LABEL_DIR \
    -d_midi $MIDI_DIR \
    -d_out $NORM_DIR