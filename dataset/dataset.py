from torch.utils.data import Dataset
import torchaudio
from pretty_midi import PrettyMIDI
from config import FRAMES_PER_CLIP, SAMPLES_PER_CLIP, FRAME_RATE, SAMPLE_RATE
import os
import torch
import torch.nn.functional as F

class MaestroDataset(Dataset):
    def __init__(self, metadata, root_dir, transform=None, target_transform=None):
        self.metadata = metadata
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        
        # Paths
        midi_path = os.path.join(self.root_dir, row['midi_filename'])
        audio_path = os.path.join(self.root_dir, row['audio_filename'])
        
        # Load MIDI → piano roll
        midi = PrettyMIDI(midi_file=midi_path)
        piano_roll = torch.tensor(midi.get_piano_roll(fs=FRAME_RATE), dtype=torch.float32)

        # Load audio
        waveform, sr = torchaudio.load(audio_path)
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Calculate alignment
        samples_per_frame = sr // FRAME_RATE
        max_audio_len = waveform.shape[1]
        max_frame_len = piano_roll.shape[1]
        max_total_len = min(max_audio_len, max_frame_len * samples_per_frame)

        max_start_sample = max_total_len - SAMPLES_PER_CLIP
        start_sample = torch.randint(0, max(1, max_start_sample), (1,)).item()

        end_sample = start_sample + SAMPLES_PER_CLIP
        start_frame = start_sample // samples_per_frame
        end_frame = start_frame + FRAMES_PER_CLIP

        # Slice
        waveform = waveform[:, start_sample:end_sample]
        piano_roll = piano_roll[:, start_frame:end_frame]

        # Pad if shorter than required
        if waveform.shape[1] < SAMPLES_PER_CLIP:
            waveform = F.pad(waveform, (0, SAMPLES_PER_CLIP - waveform.shape[1]))
        if piano_roll.shape[1] < FRAMES_PER_CLIP:
            piano_roll = F.pad(piano_roll, (0, FRAMES_PER_CLIP - piano_roll.shape[1]))

        # Apply transforms
        if self.transform:
            waveform = self.transform(waveform)
        if self.target_transform:
            piano_roll = self.target_transform(piano_roll)

        return waveform, piano_roll

class MaestroDatasetWithWindowing(Dataset) :
    def __init__(self, metadata, root_dir, transform=None, target_transform=None):
        self.metadata = metadata
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.metadata)