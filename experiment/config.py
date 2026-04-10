import os
from typing import Literal, List, TypedDict, Optional, Callable
import torch

# ==========================
# DEVICE / PATHS
# ==========================
DEVICE = "cuda"
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MAESTRO_ROOT = os.path.join(PROJECT_ROOT, "dataset", "maestro")
CSV_PATH = os.path.join(MAESTRO_ROOT, "maestro-v3.0.0.csv")
CACHE_PATH = os.path.join(PROJECT_ROOT, "dataset", "transformed_data")

# ==========================
# DATA / AUDIO PARAMETERS
# ==========================

# ==========================
# DATALOADER PARAMETERS
# ==========================
# These control torch.utils.data.DataLoader
BATCH_SIZE              = 4
NUM_WORKERS             = 4              # set 0 in notebooks or on Windows
DROP_LAST_TRAIN         = True
PIN_MEMORY              = True
PIN_MEMORY_DEVICE       = DEVICE if DEVICE.startswith("cuda") else "cpu"
PREFETCH_FACTOR         = 2              # batches prefetched per worker
TIMEOUT                 = 0              # seconds before a worker times out
SHUFFLE                 = True
PERSISTENT_WORKERS      = True           # keeps workers alive between epochs
MULTIPROCESSING_CONTEXT = None           # e.g., "spawn" or "forkserver"
WORKER_INIT_FN: Optional[Callable] = None
COLLATE_FN: Optional[Callable] = None
BATCH_SAMPLER: Optional[object] = None
SAMPLER: Optional[object] = None
GENERATOR: Optional[torch.Generator] = None  # for deterministic shuffling

# Notes:
# - PREFETCH_FACTOR > 1 improves throughput on GPUs.
# - NUM_WORKERS should roughly equal #CPU cores / 2 on Linux, or 0 on Windows.
# - PIN_MEMORY accelerates H2D transfers when using CUDA.
# - PERSISTENT_WORKERS avoids DataLoader re-spawning overhead.
# - COLLATE_FN can stay None unless using a custom batch merge.
# - MULTIPROCESSING_CONTEXT can fix hangs (e.g., "spawn" for Jupyter).

# ==========================
# AUDIO / FEATURE EXTRACTION
# ==========================
# 320 / 16000 = 0.2 ms vs 160 / 16000 = 0.1 ms
# N ~ 16000 / 320 = 50 vs 16000 / 160 = 100
sample_rate = 16000
coarse_spectrogram = {
    "SAMPLE_RATE"   : sample_rate, 
    "N_FFT"         : 1024,
    "HOP_LENGTH"    : 320,
    "WIN_LENGTH"    : 1024,
    "N_MELS"        : 128, 
    "F_MIN"         : 30.0,
    "F_MAX"         : sample_rate / 2,
    "WINDOW_FN"     : "hann",
    "POWER"         : 2.0,
    "LOG_OFFSET"    : 1e-6
}

# 160 / 22050 < 0.1 ms
# N ~ 
sample_rate = 22050
HIGH_DEF_SPECTROGRAM = {
    "SAMPLE_RATE"   : sample_rate, 
    "N_FFT"         : 2048,
    "HOP_LENGTH"    : 160,
    "WIN_LENGTH"    : 1024,
    "N_MELS"        : 229,
    "F_MIN"         : 30.0,
    "F_MAX"         : sample_rate / 2,
    "WINDOW_FN"     : "hann",
    "POWER"         : 2.0,
    "LOG_OFFSET"    : 1e-6
}

beat_normalized_spectrogram = {
    "BEATS_PER_CLIP"        : 8,
    "SUBDIVISIONS_PER_BEAT" : 12,
    "FRAME_RATE"            : 100
}

coarse_MIDI = {
    
}

beat_normalized_MIDI = {
    
}

# ==========================
# TRAINING HYPERPARAMETERS
# ==========================
NUM_EPOCHS   = 20
WEIGHT_DECAY = 1e-4
SEED         = 0

# ==========================
# MODEL CONFIGURATION
# ==========================
EMBED_DIM = 256
N_HEADS = 8
MLP_RATIO = 2
DROPOUT = 0.2
N_LAYERS = 1
N_NOTES = 128
SMOOTH_K = 5
DETACH_CONDITION = True

# ==== SCALING ASSUMPTIONS FOR COST PROJECTION ====
SUBSET_FRACTION = 1/75    # we trained on ~4% of data

# ==========================
# RESULTS / LOGGING
# ==========================
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
POSW_PATH   = os.path.join(RESULTS_DIR, "pos_weight.pt")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ==========================
# EXPERIMENT VARIANTS
# ==========================

