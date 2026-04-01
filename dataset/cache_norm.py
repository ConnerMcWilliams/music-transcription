# dataset_beats.py
import argparse
import os
import numpy as np
from typing import Optional, Callable, Tuple, Dict, Any
from collections import defaultdict

from tqdm.auto import tqdm

import torch
import torchaudio
from torch.utils.data import Dataset
from pretty_midi import PrettyMIDI

from dataset.transforms import linear_time_warp_mel, midi_notes_to_beat_labels, MelTransform
from utils.beat_utils import get_beats_and_downbeats, build_subdivision_times, midi_to_time_roll_window

import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
import re

def cache_data(
                 metadata,                      # pandas.DataFrame with columns: audio_filename, midi_filename
                 root_dir: str,
                 cache_dir: str,
                 mel_tx: Callable[[torch.Tensor, int], Tuple[torch.Tensor, int]],
                 *,
                 n_mels: int,
                 subdivisions: int,
                 beats_per_window: int,
                 hop_beats: Optional[int] = None,
                 hop_length: int,
                 sr_target: Optional[int] = None,
                 waveform_tx: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                 use_parallel: bool = True,
                 num_workers: Optional[int] = None,
                 ):
    # --- config ---
        metadata = metadata.reset_index(drop=True)
        n_mels = int(n_mels)
        S = int(subdivisions)
        beats_per_window = int(beats_per_window)
        hop_beats = int(hop_beats) if hop_beats is not None else beats_per_window

        # seconds per mel frame (for warping math); assume hop_length / sr_target
        if sr_target is None or hop_length is None:
            raise ValueError("sr_target and hop_length must be provided")
        
        time_roll_rate = sr_target // hop_length
        
        index = probe_and_build_index(metadata, root_dir, beats_per_window, hop_beats)
        
        print("Starting transforms.")
        transform_and_pickle(metadata, root_dir, cache_dir, index, mel_tx, 
                    waveform_tx, S, beats_per_window, sr_target, n_mels=n_mels,
                    hop_length=hop_length, target_transform=None, time_roll_rate=time_roll_rate,
                    use_parallel=use_parallel, num_workers=num_workers)

def probe_and_build_index(metadata, root_dir, beats_per_window, hop_beats) -> None:
        idx = []
        for i, row in metadata.iterrows():
            midi_path = os.path.join(root_dir, row["midi_filename"])
            try:
                pm = PrettyMIDI(midi_path)
            except Exception:
                continue

            beats, _ = get_beats_and_downbeats(pm)
            K = max(0, len(beats) - 1)
            if K < beats_per_window:
                continue

            starts = list(range(0, K - beats_per_window + 1, hop_beats))
            last_start = K - beats_per_window
            if not starts or starts[-1] != last_start:
                starts.append(last_start)

            for s in starts:
                idx.append((i, s))

        return idx
    
def find_checkpoint(cache_dir: str, lookback: int = 100):
    """
    Find the last successfully cached window and return a checkpoint to resume from.
    
    Args:
        cache_dir: Directory where pickles are stored
        lookback: How many windows to go back (for safety, in case some weren't pickled)
    
    Returns:
        (start_window_idx, max_completed_idx) - starting window index to process and max index already done
    """
    # Find all spectrogram_*.pkl files
    pattern = re.compile(r'spectrogram_(\d+)\.pkl')
    indices = []
    
    if os.path.exists(cache_dir):
        for filename in os.listdir(cache_dir):
            match = pattern.match(filename)
            if match:
                indices.append(int(match.group(1)))
    
    if not indices:
        print("No cached windows found. Starting from beginning.")
        return 0, -1
    
    max_idx = max(indices)
    start_idx = max(0, max_idx - lookback + 1)
    print(f"Found checkpoint: max completed window={max_idx}, resuming from window {start_idx}")
    return start_idx, max_idx
    
def transform_and_pickle(metadata, root_dir, cache_dir, index, mel_tx, 
                waveform_tx, S, beats_per_window, sr_target, n_mels, hop_length,
                target_transform, time_roll_rate, use_parallel: bool = True, num_workers: Optional[int] = None):
    """
    Transform and pickle dataset windows. Supports both serial and parallel processing.
    Resumes from checkpoint if cache already exists.
    
    Args:
        use_parallel: If True, use multiprocessing to process files in parallel
        num_workers: Number of worker processes (default: CPU count)
    """
    # Find checkpoint
    start_window_idx, max_completed_idx = find_checkpoint(cache_dir)
    
    # Group index by file
    file_to_windows = defaultdict(list)
    for row_idx, start_beat_idx in index:
        file_to_windows[row_idx].append(start_beat_idx)
    
    # Prepare worker arguments, filtering to only include windows after checkpoint
    config = {
        'n_mels': n_mels,
        'S': S,
        'beats_per_window': beats_per_window,
        'sr_target': sr_target,
        'hop_length': hop_length,
    }
    
    worker_args_list = []
    window_counter = 0
    windows_to_process = []
    
    for row_idx in sorted(file_to_windows.keys()):
        start_beat_indices = file_to_windows[row_idx]
        metadata_row = metadata.iloc[row_idx]
        
        audio_path = os.path.join(root_dir, metadata_row["audio_filename"])
        midi_path = os.path.join(root_dir, metadata_row["midi_filename"])
        
        # Only include windows that haven't been processed yet
        filtered_indices = []
        for start_beat_idx in start_beat_indices:
            if window_counter >= start_window_idx:
                filtered_indices.append(start_beat_idx)
                windows_to_process.append(window_counter)
            window_counter += 1
        
        # Only create worker args if this file has unprocessed windows
        if filtered_indices:
            worker_args_list.append((
                row_idx,
                audio_path,
                midi_path,
                filtered_indices,
                cache_dir,
                config,
                max(0, window_counter - len(filtered_indices))  # Start index for this file's windows
            ))
    
    if not windows_to_process:
        print("All windows already cached!")
        return
    
    total_unprocessed = len(windows_to_process)
    
    if use_parallel and len(worker_args_list) > 1:
        # Parallel processing
        if num_workers is None:
            num_workers = os.cpu_count() or 1
        
        print(f"Starting parallel caching with {num_workers} workers ({total_unprocessed} windows remaining)")
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            futures = {executor.submit(_process_file_worker, args): args for args in worker_args_list}
            
            with tqdm(total=total_unprocessed, desc="Caching windows (parallel)", 
                     unit="window", dynamic_ncols=True) as pbar:
                for future in as_completed(futures):
                    try:
                        num_processed, row_idx = future.result()
                        pbar.update(num_processed)
                    except Exception as e:
                        print(f"Worker error: {e}")
    else:
        # Serial processing (fallback or for single file)
        print(f"Starting serial caching ({total_unprocessed} windows remaining)")
        with tqdm(total=total_unprocessed, desc="Caching windows (serial)", 
                 unit="window", dynamic_ncols=True) as pbar:
            for worker_args in worker_args_list:
                try:
                    num_processed, row_idx = _process_file_worker(worker_args)
                    pbar.update(num_processed)
                except Exception as e:
                    print(f"Worker error on row {worker_args[0]}: {e}")

def _process_file_worker(worker_args):
    """
    Worker function for parallel processing. Handles one file and all its windows.
    Must be picklable (defined at module level).
    
    Args:
        worker_args: tuple of (row_idx, audio_path, midi_path, start_beat_indices,
                               cache_dir, config_dict, window_start_idx)
    
    Returns:
        tuple of (num_windows_processed, row_idx) for progress tracking
    """
    (row_idx, audio_path, midi_path, start_beat_indices, cache_dir, config, window_start_idx) = worker_args
    
    # Unpack config
    n_mels = config['n_mels']
    S = config['S']
    beats_per_window = config['beats_per_window']
    sr_target = config['sr_target']
    hop_length = config['hop_length']
    
    try:
        # Create transforms fresh in each worker process
        mel_tx = MelTransform()
        
        # Load MIDI
        pm = PrettyMIDI(midi_path)
        beats, _ = get_beats_and_downbeats(pm)
        
        # Load and process audio
        waveform, sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr_target is not None and sr_target != sr:
            waveform = torchaudio.functional.resample(waveform, sr, sr_target)
            sr = sr_target
        
        # Compute mel spectrogram once per file
        mel_time, _ = mel_tx(waveform, sr)
        
        if mel_time.shape[0] != n_mels:
            raise ValueError(f"mel_tx returned {mel_time.shape[0]} mels, expected {n_mels}")
        
        dt = hop_length / sr_target
        num_beats = beats_per_window
        L = num_beats * S
        
        # Process each window for this file
        windows_processed = 0
        for local_idx, start_beat_idx in enumerate(start_beat_indices):
            window_idx = window_start_idx + local_idx
            
            # Build subdivision times
            ts_target, max_value = build_subdivision_times(beats, S, start_beat_idx, num_beats)
            
            if ts_target is None:
                raise ValueError(
                    f"Invalid beat window: row_idx={row_idx}, start_beat_idx={start_beat_idx}, "
                    f"num_beats={num_beats}, len(beats)={len(beats)}, midi_path={midi_path}"
                )
            
            # Warp mel to beat grid
            mel_beat = linear_time_warp_mel(mel_time, ts_target, dt)
            
            if mel_beat.shape != (n_mels, L):
                raise RuntimeError(
                    f"mel_beat has shape {tuple(mel_beat.shape)}, expected ({n_mels}, {L})"
                )
            
            # Generate labels
            on_b, off_b, frm_b = midi_notes_to_beat_labels(pm, ts_target, S, max_value, num_beats)
            
            # Convert to tensors and transpose
            on_t = torch.as_tensor(on_b, dtype=torch.float32)
            off_t = torch.as_tensor(off_b, dtype=torch.float32)
            frm_t = torch.as_tensor(frm_b, dtype=torch.float32)
            
            if on_t.shape == (128, L):
                on_t = on_t.transpose(0, 1)
                off_t = off_t.transpose(0, 1)
                frm_t = frm_t.transpose(0, 1)
            elif on_t.shape != (L, 128):
                raise RuntimeError(f"Expected targets [L,128], got {tuple(on_t.shape)}")
            
            labels = {
                "on": on_t.contiguous(),
                "off": off_t.contiguous(),
                "frame": frm_t.contiguous()
            }
            
            metadata = {
                "times" : ts_target,
                "max_time" : max_value
            }
            
            mel = mel_beat.to(torch.float32).unsqueeze(0).contiguous()
            pickle_spectrogram(mel, labels, window_idx, cache_dir, metadata)
            windows_processed += 1
        
        return (windows_processed, row_idx)
    
    except Exception as e:
        print(f"Error processing file (row_idx={row_idx}, midi={midi_path}): {e}")
        return (0, row_idx)
        
def pickle_spectrogram(mel, labels, index, cache_dir, extra=None) :
    '''
    Pickle mel spectrogram and labels to cache directory.
    Uses protocol=4 for faster serialization on modern Python.
    '''
    # Pickle the mel spectrogram
    with open(f'{cache_dir}//spectrogram_{index}.pkl', 'wb') as file :
        pickle.dump(mel, file, protocol=4)
    
    # Pickle the labels
    with open(f'{cache_dir}//labels_{index}.pkl', 'wb') as file :
        pickle.dump(labels, file, protocol=4)
        
    if extra :
        with open(f'{cache_dir}//extra_{index}.pkl', 'wb') as file :
            pickle.dump(extra, file, protocol=4)
        
    return

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument('-d_list', help='corpus list directory')
    parser.add_argument('-d_midi', help='midi file directory (input)')
    parser.add_argument('-d_note', help='note file directory (input)')
    parser.add_argument('-d_norm_midi', 
                        help='normalized midi file directory (output)')
    parser.add_argument('-d_norm_note', 
                        help='normalized note file directory (output)')
    parser.add_argument('-d_norm_spec', 
                        help='normalized spectrogram file directory (output)')
    parser.add_argument('-config', help='config file')
    
    args = parser.parse_args()
    
    print('** conv_midi2note: convert midi to note **')
    print(' directory')
    print('  midi (input)  : '+str(args.d_midi))
    print('  note (input) : '+str(args.d_note))
    
    print('  corpus list   : '+str(args.d_list))
    print(' config file    : '+str(args.config))
    
    print('')