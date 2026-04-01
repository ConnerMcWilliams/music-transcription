from dataset.cache_norm import cache_data
from dataset.transforms import MelTransform
from experiment.config import (CSV_PATH, N_MELS, CACHE_PATH, BATCH_SIZE, DEVICE,
                    CSV_PATH, MAESTRO_ROOT, BEATS_PER_CLIP, NUM_EPOCHS,
                    SUBDIVISIONS_PER_BEAT, HOP_LENGTH, SAMPLE_RATE,)
import pandas as pd

import pickle
from utils.display_midi import display_spectrogram, display_midi_from_roll
from pretty_midi import PrettyMIDI

from dataset.cashe_dataset import CashDataset
from torch.utils.data import DataLoader

from components.loss import OnsetsAndFramesLoss, OnsetsAndFramesPaperLoss
from components.schedulers import make_optimizer, make_scheduler
import torch.optim as optim

from models.hFT import HFTModel

import matplotlib.pyplot as plt

import numpy as np

import torch
import os
from tqdm import tqdm
import datetime

LABEL_KEYS = ("on", "off", "frame", "vel")

def load_model_checkpoint(model_path, device=DEVICE):
    """Load a saved model checkpoint."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)

    state_dict = checkpoint.get('model_state_dict', checkpoint)
    config = checkpoint.get('config', {
        'dim': 256,
        'depth': 6,
        'num_heads': 8,
        'n_pitches': 128,
        'n_mels': N_MELS,
    })

    model = HFTModel(
        dim=config['dim'],
        depth=config.get('depth', 6),
        num_heads=config['num_heads'],
        n_pitches=config.get('n_pitches', 128),
        n_mels=config.get('n_mels', N_MELS),
    )
    model.load_state_dict(state_dict, strict=False)
    model.to(device)

    print(f"Loaded model from {model_path}")
    if 'epoch' in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}")
    if 'loss' in checkpoint:
        print(f"  Final loss: {checkpoint['loss']:.6f}")
    print(f"  Config: {config}")

    return model

# Check for available optimizations
USE_AMP = hasattr(torch.cuda, 'amp') and torch.cuda.is_available()
USE_COMPILE = hasattr(torch, 'compile')

# Check if Triton is available for compilation
try:
    import triton
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

USE_COMPILE = USE_COMPILE and TRITON_AVAILABLE

def train_one_epoch(model, loader, criterion, optimizer, scheduler=None, step_per_batch=True, device=DEVICE, use_amp=USE_AMP, show_progress=True, accumulation_steps=1):
    model.train()
    total = 0.0
    lr_track = []
    num_samples = 0
    
    # Setup automatic mixed precision if available
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    
    total_batches = len(loader)
    
    # Pre-allocate for gradient accumulation
    accumulated_loss = 0.0
    accumulation_counter = 0
    
    for step, (x, labels) in enumerate(loader):
        if show_progress and step % 10 == 0:  # Print progress every 10 batches
            print(f"Batch {step+1}/{total_batches}", end='\r')
            
        batch_size = x.size(0)
        num_samples += batch_size
        
        # Move input to device (single transfer)
        x = x.to(device, dtype=torch.float32, non_blocking=True)

        # Prepare labels dict - optimized batching
        y = {}
        label_tensors = []
        shapes_valid = True
        
        for k in LABEL_KEYS:
            if k in labels:
                v = labels[k]
                # Ensure consistent shape: [L, 128] -> [1, L, 128] for batching
                if v.dim() == 2:
                    v = v.unsqueeze(0)
                label_tensors.append(v)
                # Check if all shapes match for batching
                if len(label_tensors) > 1 and v.shape != label_tensors[0].shape:
                    shapes_valid = False
        
        # Single batched transfer for all labels if shapes match
        if label_tensors and shapes_valid:
            label_batch = torch.stack(label_tensors, dim=0).to(device=device, dtype=torch.float32, non_blocking=True)
            y = dict(zip(LABEL_KEYS, label_batch.unbind(0)))
        else:
            # Individual transfers if shapes don't match
            if not shapes_valid and label_tensors:
                print(f"Warning: Label shapes don't match for batching: {[v.shape for v in label_tensors]}")
            for k in LABEL_KEYS:
                if k in labels:
                    v = labels[k]
                    if v.dim() == 2:
                        v = v.unsqueeze(0)
                    y[k] = v.to(device=device, dtype=torch.float32, non_blocking=True)

        # Forward pass
        if use_amp:
            with torch.amp.autocast('cuda'):
                out = model(x)
                loss, _metrics = criterion(out, y)
        else:
            out = model(x)
            loss, _metrics = criterion(out, y)
        
        # Gradient accumulation
        loss = loss / accumulation_steps
        accumulated_loss += loss.item()
        
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        accumulation_counter += 1
        
        # Update weights after accumulation_steps
        if accumulation_counter % accumulation_steps == 0:
            if use_amp:
                # Clip gradients to prevent explosion
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                # Clip gradients to prevent explosion
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            optimizer.zero_grad(set_to_none=True)
            
            # Track accumulated loss
            total += accumulated_loss * batch_size * accumulation_steps
            accumulated_loss = 0.0

        if scheduler is not None and step_per_batch:
            scheduler.step()

        lr_track.append(optimizer.param_groups[0]["lr"])

    # Handle remaining accumulated gradients
    if accumulation_counter % accumulation_steps != 0:
        if use_amp:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        total += accumulated_loss * batch_size * accumulation_steps

    if show_progress:
        print()  # New line after progress
    
    avg_loss = total / max(1, num_samples)
    return avg_loss, lr_track

def show_on_off_overlay(frm, S, pitch_lo=21, pitch_hi=108, title="Frame probability distribution"):
    frm_np = frm[pitch_lo:pitch_hi+1].cpu().numpy().T

    plt.figure(figsize=(12, 4))
    plt.imshow(frm_np, aspect="auto", origin="lower")

    for x in range(0, frm_np.shape[1], S):
        plt.axvline(x, linewidth=0.5, alpha=0.3)

    plt.xlabel("Subdivision index")
    plt.ylabel("Pitch index within range")
    plt.title(title)
    plt.show()

def show_on_off_predictions(on_prob, off_prob, frame_prob, threshold=0.001, S=24, pitch_lo=21, pitch_hi=108):
    """
    Visualize predicted onset, offset, and frame events after thresholding logits.
    
    Args:
        on_prob: Onset logits [T, P]
        off_prob: Offset logits [T, P]
        frame_prob: Frame logits [T, P]
        threshold: Threshold for binarizing predictions
        S: Subdivision spacing for vertical lines
        pitch_lo, pitch_hi: Pitch range to display
    """
    # Convert logits to probabilities so threshold is interpreted consistently.
    on_pred = (torch.sigmoid(on_prob) > threshold).cpu().numpy()
    off_pred = (torch.sigmoid(off_prob) > threshold).cpu().numpy()
    frame_pred = (torch.sigmoid(frame_prob) > threshold).cpu().numpy()
    
    # Slice pitches
    frame_pred = frame_pred[:, pitch_lo:pitch_hi+1].T
    on_pred = on_pred[:, pitch_lo:pitch_hi+1].T
    off_pred = off_pred[:, pitch_lo:pitch_hi+1].T
    
    plt.figure(figsize=(12, 4))
    plt.imshow(frame_pred, aspect="auto", origin="lower", cmap='gray', alpha=0.7)
    
    # Get positions for onsets and offsets
    y_on, x_on = np.nonzero(on_pred)
    y_off, x_off = np.nonzero(off_pred)
    
    # Plot onsets as green circles, offsets as red crosses
    plt.scatter(x_on, y_on, s=8, marker="o", color="green", label="onset")
    plt.scatter(x_off, y_off, s=8, marker="x", color="red", label="offset")
    
    for x in range(0, frame_pred.shape[1], S):
        plt.axvline(x, linewidth=0.5, alpha=0.3)
    
    plt.legend(loc="upper right")
    plt.xlabel("Subdivision index")
    plt.ylabel("Pitch index within range")
    plt.title("Predicted events: frames (gray), onsets (green), offsets (red)")
    plt.show()

def test_training() :
    global USE_COMPILE
    
    # Enable CUDA optimizations
    torch.backends.cudnn.benchmark = True  # Optimize for fixed input sizes
    torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 for faster matmul
    torch.backends.cudnn.allow_tf32 = True  # Allow TF32 for faster convolutions
    
    # Set memory allocator optimizations
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.9)  # Use 90% of GPU memory
    
    cache_dataset = CashDataset(CACHE_PATH)
    
    cache_loader = DataLoader(cache_dataset,
                              batch_size=4,  # Increased from 4 to 16 for better GPU utilization
                                shuffle=True,
                              num_workers=4,  # Use multiple workers for data loading
                              pin_memory=True,  # Faster host->device transfers
                              persistent_workers=True,  # Keep workers alive between epochs
                              prefetch_factor=2)  # Prefetch 2 batches per worker
    
    model = HFTModel(dim=256, num_heads=8, n_pitches=128, n_mels=N_MELS).to(DEVICE)
    
    # Use torch.compile if available (PyTorch 2.0+ and Triton available)
    if USE_COMPILE:
        try:
            model = torch.compile(model, mode='reduce-overhead')  # More aggressive optimization
            print("Using torch.compile with reduce-overhead mode")
        except Exception as e:
            print(f"torch.compile failed: {e}, falling back to eager mode")
            USE_COMPILE = False
    else:
        print("Triton not available, using eager mode")
    
    opt = make_optimizer(model, lr=1e-4)
    # Add learning rate scheduler for better convergence
    scheduler = make_scheduler(
        opt, scheduler_type='onecycle', epochs=NUM_EPOCHS, steps_per_epoch=len(cache_loader),
        max_lr=1e-4, pct_start=0.1, anneal_strategy='cos'
    )
    criterion = OnsetsAndFramesPaperLoss(
        lambda_on=1.0, 
        lambda_frame=1.0, 
        lambda_off=0.5,  # Enable offset loss
        onset_frame_weight=5.0, 
        onset_window=4
    )
    
    # Enable gradient accumulation for larger effective batch size
    accumulation_steps = 2  # Accumulate gradients over 2 steps
    
    for epoch in range(NUM_EPOCHS):
        tr, lr_track = train_one_epoch(model, cache_loader, criterion, opt, scheduler=scheduler, 
                                     use_amp=USE_AMP, accumulation_steps=accumulation_steps)
        print(f"[Onset and Frames] epoch {epoch+1}/{NUM_EPOCHS}  train={tr:.4f}  vlr_end={opt.param_groups[0]['lr']:.2e}")

    # Save the trained model
    os.makedirs('model_weights', exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = f'model_weights/onset_and_frames_{timestamp}.pth'
    
    torch.save({
        'epoch': NUM_EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': tr,  # Final training loss
        'config': {
            'dim': 256,
            'depth': 6,
            'num_heads': 8,
            'batch_size': 16,
            'learning_rate': 1e-4,
            'num_epochs': NUM_EPOCHS,
            'n_pitches': 128
        }
    }, model_path)
    
    print(f"Model saved to: {model_path}")

    model.eval()

    with torch.no_grad():
        # Get a single batch for evaluation
        batch = next(iter(cache_loader))
        x, labels = batch

        # Batch all transfers
        x = x.to(device=DEVICE, dtype=torch.float32, non_blocking=True)
        
        y = {}
        label_tensors = []
        shapes_valid = True
        
        for k in LABEL_KEYS:
            if k in labels:
                v = labels[k]
                if v.dim() == 2:
                    v = v.unsqueeze(0)
                label_tensors.append(v)
                if len(label_tensors) > 1 and v.shape != label_tensors[0].shape:
                    shapes_valid = False
        
        if label_tensors and shapes_valid:
            label_batch = torch.stack(label_tensors, dim=0).to(device=DEVICE, dtype=torch.float32, non_blocking=True)
            y = dict(zip(LABEL_KEYS, label_batch.unbind(0)))
        else:
            if not shapes_valid and label_tensors:
                print(f"Warning: Label shapes don't match: {[v.shape for v in label_tensors]}")
            for k in LABEL_KEYS:
                if k in labels:
                    v = labels[k]
                    if v.dim() == 2:
                        v = v.unsqueeze(0)
                    y[k] = v.to(device=DEVICE, dtype=torch.float32, non_blocking=True)

        out = model(x)

        # Extract predictions (move to CPU once)
        pred_frame = out["frame"][0].detach().cpu()
        true_frame = y["frame"][0].detach().cpu()
        pred_on = out["on"][0].detach().cpu()
        true_on = y["on"][0].detach().cpu()
        pred_off = out["off"][0].detach().cpu()
        true_off = y["off"][0].detach().cpu()

        show_on_off_overlay(pred_frame, 12, 21, 108)
        show_on_off_overlay(true_frame, 12, 21, 108)

def display_transformed_dataset(index) :
    with open(f'{CACHE_PATH}//labels_{index}.pkl', 'rb') as file:
        label = pickle.load(file)
        frm = label['frame']
        on = label['on']
        off = label['off']
        
        #time_roll = label['time_roll']
        
        show_on_off_overlay(frm, 12, 21, 108)
        #display_midi_from_roll(time_roll)
        
        
    with open(f'{CACHE_PATH}//spectrogram_{index}.pkl', 'rb') as file:
        test_spec = pickle.load(file)
        display_spectrogram(test_spec)

def continue_training(model_path, additional_epochs, device=DEVICE):
    """Continue training a saved model for additional epochs."""
    model = load_model_checkpoint(model_path, device)
    
    # Recreate optimizer and scheduler
    optimizer = make_optimizer(model, lr=1e-4)
    scheduler = None  # Not loading scheduler for simplicity
    
    # Load dataset
    cache_dataset = CashDataset(CACHE_PATH)
    cache_loader = DataLoader(cache_dataset,
                              batch_size=16,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True,
                              persistent_workers=True,
                              prefetch_factor=2)
    
    criterion = OnsetsAndFramesLoss(
        lambda_on=0.5,
        lambda_frame=1.0,
        lambda_off=0.5,
        pos_weight_frame=torch.full((128,), 0.5)
    )
    
    # Since checkpoint not returned, start from epoch 0 for additional
    for epoch in range(additional_epochs):
        tr, lr_track = train_one_epoch(model, cache_loader, criterion, optimizer, scheduler=None, use_amp=USE_AMP)
        print(f"[Continue Training] epoch {epoch + 1}  train={tr:.4f}")
    
    # Save updated model
    os.makedirs('model_weights', exist_ok=True)
    
    torch.save({
        'epoch': additional_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': tr,  # Final training loss
        'config': {'dim': 256, 'depth': 6, 'num_heads': 8, 'learning_rate': 1e-4, 'num_epochs': additional_epochs, 'n_pitches': 128}
    }, model_path)
    
    print(f"Continued model saved to: {model_path}")

def test_model(model_or_path, num_samples=5, device=DEVICE, threshold=0.4):
    """Test a saved model path or an already loaded model by displaying predictions."""
    if isinstance(model_or_path, torch.nn.Module):
        model = model_or_path.to(device)
    else:
        model = load_model_checkpoint(model_or_path, device)
    model.eval()
    
    cache_dataset = CashDataset(CACHE_PATH)
    
    with torch.no_grad():
        for i in range(min(num_samples, len(cache_dataset))):
            x, labels = cache_dataset[i]
            x = x.unsqueeze(0).to(device, dtype=torch.float32)
            
            out = model(x)
            
            # Prepare labels
            y = {}
            for k in LABEL_KEYS:
                if k in labels:
                    v = labels[k]
                    if v.dim() == 2:
                        v = v.unsqueeze(0)
                    y[k] = v.to(device, dtype=torch.float32)
            
            # Extract predictions
            pred_frame = out["frame"][0].detach().cpu()
            true_frame = y["frame"][0].detach().cpu()
            pred_on = out["on"][0].detach().cpu()
            true_on = y["on"][0].detach().cpu()
            pred_off = out["off"][0].detach().cpu()
            true_off = y["off"][0].detach().cpu()
            pred_on_binary = (torch.sigmoid(pred_on) > threshold)
            pred_off_binary = (torch.sigmoid(pred_off) > threshold)
            pred_frame_binary = (torch.sigmoid(pred_frame) > threshold)
            
            print(f"Sample {i+1} - True Label:")
            show_on_off_overlay(true_frame, 12, 21, 108, title="True Frame Label")
            
            print(
                f"Sample {i+1} - Thresholded positives @ {threshold}: "
                f"frame={int(pred_frame_binary.sum().item())}, "
                f"onset={int(pred_on_binary.sum().item())}, "
                f"offset={int(pred_off_binary.sum().item())}"
            )
            print(f"Sample {i+1} - Predicted Frame Mask:")
            show_on_off_overlay(
                pred_frame_binary.float(),
                12,
                21,
                108,
                title=f"Predicted Frame Mask (threshold={threshold})",
            )
            
            print(f"Sample {i+1} - Predicted Label:")
            show_on_off_predictions(pred_on, pred_off, pred_frame, threshold=threshold, S=SUBDIVISIONS_PER_BEAT, pitch_lo=21, pitch_hi=108)

def _f1_from_binary_tensors(true_tensor, pred_tensor):
    true_tensor = true_tensor.float().flatten()
    pred_tensor = pred_tensor.float().flatten()

    tp = (true_tensor * pred_tensor).sum().float()
    fp = ((1 - true_tensor) * pred_tensor).sum().float()
    fn = (true_tensor * (1 - pred_tensor)).sum().float()

    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)
    return {
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item(),
    }


def _extract_note_events(on_tensor, off_tensor, frame_tensor):
    on_array = on_tensor.bool().cpu().numpy()
    off_array = off_tensor.bool().cpu().numpy()
    frame_array = frame_tensor.bool().cpu().numpy()
    n_steps, n_pitches = frame_array.shape
    note_events = []

    for pitch in range(n_pitches):
        on_pitch = on_array[:, pitch]
        off_pitch = off_array[:, pitch]
        frame_pitch = frame_array[:, pitch]
        active_start = None
        for step in range(n_steps):
            onset_here = on_pitch[step]
            offset_here = off_pitch[step]
            frame_here = frame_pitch[step]

            if active_start is None and (onset_here or frame_here):
                active_start = step

            if active_start is not None and offset_here:
                note_events.append((pitch, active_start, step))
                active_start = None
                continue

            if active_start is not None:
                next_frame_here = False
                if step + 1 < n_steps:
                    next_frame_here = frame_pitch[step + 1]
                if not frame_here and not onset_here:
                    note_events.append((pitch, active_start, step))
                    active_start = None
                elif frame_here and not next_frame_here:
                    note_events.append((pitch, active_start, step))
                    active_start = None

        if active_start is not None:
            note_events.append((pitch, active_start, n_steps - 1))

    return set(note_events)


def _note_f1_from_events(true_events, pred_events):
    tp = len(true_events & pred_events)
    fp = len(pred_events - true_events)
    fn = len(true_events - pred_events)

    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_notes': len(true_events),
        'pred_notes': len(pred_events),
    }


def compute_f1_score(model, dataloader, threshold=0.5, device=DEVICE):
    """
    Compute per-sample averaged onset, frame, and offset F1 metrics.
    
    Args:
        model: Trained OnsetAndFrames model
        dataloader: DataLoader for evaluation
        threshold: Threshold for binarizing predictions
        device: Device to run on
    
    Returns:
        Dictionary with final per-sample-averaged F1 values.
    """
    model.eval()
    average_sums = {
        'onset': 0.0,
        'frame': 0.0,
        'offset': 0.0,
    }
    num_samples = 0

    with torch.inference_mode():
        for x, labels in tqdm(dataloader, desc="Computing F1", unit="batch"):
            x = x.to(device, dtype=torch.float32)
            out = model(x)

            pred_on = (torch.sigmoid(out['on']) > threshold).cpu()
            pred_off = (torch.sigmoid(out['off']) > threshold).cpu()
            pred_frame = (torch.sigmoid(out['frame']) > threshold).cpu()

            true_on = labels['on'].cpu().bool()
            true_off = labels['off'].cpu().bool()
            true_frame = labels['frame'].cpu().bool()

            for true_on_sample, true_off_sample, true_frame_sample, pred_on_sample, pred_off_sample, pred_frame_sample in zip(
                true_on,
                true_off,
                true_frame,
                pred_on,
                pred_off,
                pred_frame,
            ):
                average_sums['onset'] += _f1_from_binary_tensors(true_on_sample, pred_on_sample)['f1']
                average_sums['frame'] += _f1_from_binary_tensors(true_frame_sample, pred_frame_sample)['f1']
                average_sums['offset'] += _f1_from_binary_tensors(true_off_sample, pred_off_sample)['f1']
                num_samples += 1

    return {
        'onset_f1': average_sums['onset'] / num_samples if num_samples else 0.0,
        'frame_f1': average_sums['frame'] / num_samples if num_samples else 0.0,
        'offset_f1': average_sums['offset'] / num_samples if num_samples else 0.0,
    }


def compute_f1_from_model_path(model_path, threshold=0.5, batch_size=1, device=DEVICE):
    model = load_model_checkpoint(model_path, device=device)
    dataloader = DataLoader(
        CashDataset(CACHE_PATH),
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )
    return compute_f1_score(model, dataloader, threshold=threshold, device=device)

def main() :
    midi_path  = os.path.join(MAESTRO_ROOT, "2018", 
                              "MIDI-Unprocessed_Chamber2_MID--AUDIO_09_R3_2018_wav--1.midi")
    
    pm = PrettyMIDI(midi_path)
    
    metadata = pd.read_csv(CSV_PATH)
    mel_tx = MelTransform()
    
    dataloader = DataLoader(CashDataset(CACHE_PATH), batch_size=1, 
                            shuffle=False, num_workers=1, pin_memory=True,)
    
    # loss: 0.046184, f1: 0.7274966239929199
    #test_model_path = 'model_weights/onset_and_frames_20260309_181510.pth' 
    best_model = 'model_weights/best_model.pth'
    print(best_model)
    test_model(best_model, num_samples=3, device=DEVICE, threshold=0.5)
    #continue_training(test_model_path, additional_epochs=10)
    #test_training()
    #model = load_model_checkpoint(test_model_path)
    # .6 best so far
    # .5
    # 0.1: {'onset_f1': 0.8597313248228593, 'frame_f1': 0.9005319844492063, 'offset_f1': 0.43945023072334954}
    # 0.3: {'onset_f1': 0.8955431835154148, 'frame_f1': 0.9358091985136024, 'offset_f1': 0.45298139032020535}
    # 0.5: {'onset_f1': 0.8963212741189385, 'frame_f1': 0.9474625471840529, 'offset_f1': 0.38025475321433544}
    print(compute_f1_from_model_path(best_model, threshold=0.5, batch_size=1, device=DEVICE))
    #display_transformed_dataset(3)
    """
    cache_data(metadata=metadata, root_dir=MAESTRO_ROOT, cache_dir=CACHE_PATH, mel_tx=mel_tx, n_mels=N_MELS, 
               subdivisions=SUBDIVISIONS_PER_BEAT, 
                 beats_per_window=BEATS_PER_CLIP, hop_length=HOP_LENGTH, sr_target=SAMPLE_RATE)
    #"""
if __name__ == "__main__" :
    main()