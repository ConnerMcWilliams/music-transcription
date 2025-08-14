from torch.utils.data import Dataset
import torchaudio
from pretty_midi import PrettyMIDI
from config import FRAMES_PER_CLIP, SAMPLES_PER_CLIP, FRAME_RATE, SAMPLE_RATE
import os
import torch
import torch.nn.functional as F
import math
import tqdm

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
        
        waveform = waveform.squeeze(0)

        return waveform, piano_roll

class MaestroDatasetWithWindowing(Dataset) :
    def __init__(self, metadata, root_dir, hop_samples=None, transform=None, target_transform=None):
        """
        Build a dataset where each __getitem__ returns ONE window:
        (waveform_window [1, SAMPLES_PER_CLIP], piano_roll_window [128, FRAMES_PER_CLIP])

        Args:
          hop_samples: step between consecutive windows in samples.
                       If None, use SAMPLES_PER_CLIP (non-overlapping).
        """
        self.metadata = metadata.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform

        self.hop_samples = hop_samples or SAMPLES_PER_CLIP
        self.samples_per_frame = None  # set per file (depends on sr)
        self.index = []  # list of (row_idx, start_sample)

        # Precompute window start positions per track (without loading audio fully)
        for i, row in self.metadata.iterrows():
            audio_path = os.path.join(self.root_dir, row["audio_filename"])
            midi_path  = os.path.join(self.root_dir, row["midi_filename"])

            # Fast probe for audio length & sr
            info = torchaudio.info(audio_path)
            sr = info.sample_rate
            num_samples = info.num_frames
            samples_per_frame = sr // FRAME_RATE

            # Load MIDI once to know number of frames
            roll_frames = PrettyMIDI(midi_path).get_end_time() * FRAME_RATE
            max_frame_len = int(math.ceil(roll_frames))
            max_audio_len = num_samples
            max_total_len = min(max_audio_len, max_frame_len * samples_per_frame)

            if max_total_len <= 0:
                continue

            # How many windows?
            n = max(1, 1 + (max(0, max_total_len - SAMPLES_PER_CLIP)) // self.hop_samples)
            for w in range(n):
                start_sample = w * self.hop_samples
                if start_sample + SAMPLES_PER_CLIP > max_total_len:
                    # clamp last window to end
                    start_sample = max(0, max_total_len - SAMPLES_PER_CLIP)
                self.index.append((i, start_sample))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        row_idx, start_sample = self.index[idx]
        row = self.metadata.iloc[row_idx]

        audio_path = os.path.join(self.root_dir, row["audio_filename"])
        midi_path  = os.path.join(self.root_dir, row["midi_filename"])

        # Load audio
        waveform, sr = torchaudio.load(audio_path)      # [C, N]
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        samples_per_frame = sr // FRAME_RATE
        end_sample = start_sample + SAMPLES_PER_CLIP

        # Slice audio
        wav = waveform[:, start_sample:end_sample]
        if wav.shape[1] < SAMPLES_PER_CLIP:
            wav = F.pad(wav, (0, SAMPLES_PER_CLIP - wav.shape[1]))

        # Load MIDI roll & slice
        roll = torch.tensor(PrettyMIDI(midi_path).get_piano_roll(fs=FRAME_RATE), dtype=torch.float32)  # [128, T]
        start_frame = start_sample // samples_per_frame
        end_frame = start_frame + FRAMES_PER_CLIP
        roll_win = roll[:, start_frame:end_frame]
        if roll_win.shape[1] < FRAMES_PER_CLIP:
            roll_win = F.pad(roll_win, (0, FRAMES_PER_CLIP - roll_win.shape[1]))

        # Optional transforms
        if self.transform:
            wav = self.transform(wav)
        if self.target_transform:
            roll_win = self.target_transform(roll_win)

        return wav, roll_win
    
    ##, {
            #"track_idx": row_idx,
            #"start_sample": start_sample,
            #"sr": sr,
        #}