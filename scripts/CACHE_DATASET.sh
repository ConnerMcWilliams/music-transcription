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
CONFIG_FILE=$CURRENT_DIR/corpus/config.json

# 1. download MAESTRO v3.0.0 data and expand them
mkdir -p $MAESTRO_DIR

FILE=./maestro-v3.0.0.zip
if test -f "$FILE"; then
    echo "$FILE exists, proceed to unzip"
else 
    echo "$FILE does not exist. Downloading..."    
    wget https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip ./
fi

unzip maestro-v3.0.0.zip -d $CORPUS_DIR
# ($CORPUS_DIR/maestro-v3.0.0)

# 2. make lists that include train/valid/test split
mkdir -p $LIST_DIR
python $DATASET_SCRIPTS/make_list_maestro.py -i $MAESTRO_DIR/maestro-v3.0.0/maestro-v3.0.0.csv

# 3. rename the files
mkdir -p $MIDI_DIR
mkdir -p $WAV_DIR
python3 $DATASET_SCRIPTS/rename_maestro.py -d_i $MAESTRO_DIR/maestro-v3.0.0 -d_o $MAESTRO_DIR -d_list $LIST_DIR

# 4. convert wav to log-mel spectrogram
mkdir -p $FEATURE_DIR
python3 $DATASET_SCRIPTS/conv_wav2fe.py -d_list $LIST_DIR -d_wav $WAV_DIR -d_feature $FEATURE_DIR -config $CONFIG_FILE

# 5. convert midi to note
mkdir -p $NOTE_DIR
python3 $DATASET_SCRIPTS/conv_midi2note.py -d_list $LIST_DIR -d_midi $MIDI_DIR -d_note $NOTE_DIR -config $CONFIG_FILE

# 6. convert note to label
mkdir -p $LABEL_DIR
python3 $DATASET_SCRIPTS/conv_note2label.py -d_list $LIST_DIR -d_note $NOTE_DIR -d_label $LABEL_DIR -config $CONFIG_FILE

# 7. normalize and cache everything
mkdir -p $NORM_DIR/feature
mkdir -p $NORM_DIR/label
python $DATASET_SCRIPTS/cache_norm.py

# 8. convert txt to reference for evaluation
mkdir -p $REFERENCE_DIR
python3 $DATASET_SCRIPTS/conv_note2ref.py -f_list $LIST_DIR/valid.list -d_note $NOTE_DIR -d_ref $REFERENCE_DIR
python3 $DATASET_SCRIPTS/conv_note2ref.py -f_list $LIST_DIR/test.list -d_note $NOTE_DIR -d_ref $REFERENCE_DIR

# 9. make dataset
mkdir -p $DATASET_DIR
python3 $DATASET_SCRIPTS/make_dataset.py -f_config_in $CONFIG_FILE -f_config_out $DATASET_DIR/config.json -d_dataset $DATASET_DIR -d_list $LIST_DIR -d_feature $FEATURE_DIR -d_label $LABEL_DIR -n_div_train 4 -n_div_valid 1 -n_div_test 1