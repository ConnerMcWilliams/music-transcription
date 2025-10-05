from dataset.transforms import log_mel
from utils.display_midi import display_spectrogram, display_midi_from_roll, display_spectrogram_with_beat
from experiment import run_experiment
from config import (MODEL_VARIANTS, CSV_PATH, MAESTRO_ROOT, BEATS_PER_CLIP, SUBDIVISIONS_PER_BEAT, HOP_LENGTH)
from train import load_or_compute_pos_weight
from dataset.beat_dataset import MaestroDatasetWithWindowingInBeats
import pandas as pd
from models.onset_and_frames import OnsetAndFrames
from data import make_loader


def main() :
    """
    Return the splits from 2017 and 2018
    """
    metadata = pd.read_csv(CSV_PATH)
    
    train_split = metadata[
        ((metadata['year'] == 2018) | (metadata['year'] == 2017)) &
        (metadata['split'] == 'train')
    ]
    
    val_split = metadata[
        ((metadata['year'] == 2018) | (metadata['year'] == 2017)) &
        (metadata['split'] == 'validation')
    ]
    
    print("Loading Train")
    train_data = MaestroDatasetWithWindowingInBeats(train_split, MAESTRO_ROOT,
                                                    mel_tx=log_mel,
                                                    n_mels=229,
                                                    subdivisions=12,
                                                    beats_per_window=8,
    hop_beats=8,
    hop_length=HOP_LENGTH,
    sr_target=16000,
    waveform_tx=None,
    mel_aug_prewarp=None,
    mel_aug_postwarp=None,
    target_transform=None,
    return_time_labels=False)
    print("Loading Val")
    val_split = MaestroDatasetWithWindowingInBeats(val_split, MAESTRO_ROOT,
                                                   mel_tx=log_mel,
                                                    n_mels=229,
                                                    subdivisions=12,
                                                    beats_per_window=8,
    hop_beats=8,                 # non-overlapping; set <8 for overlap
    hop_length=HOP_LENGTH,
    sr_target=16000,
    waveform_tx=None,
    mel_aug_prewarp=None,
    mel_aug_postwarp=None,
    target_transform=None,
    return_time_labels=False)
    
    print(MODEL_VARIANTS[4])
    
    run_experiment(train_loader=make_loader(train_data), 
                   val_loader=make_loader(val_split), 
                   variant=MODEL_VARIANTS[4])
    
    

if __name__ == "__main__" :
    main()