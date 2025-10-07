import os
from typing import Literal, List, TypedDict

DEVICE = "cuda"
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MAESTRO_ROOT = os.path.join(PROJECT_ROOT, "dataset", "maestro")
CSV_PATH = os.path.join(MAESTRO_ROOT, "maestro-v3.0.0.csv")

# Data
SAMPLES_PER_CLIP = 80_000
FRAMES_PER_CLIP  = 500

BEATS_PER_CLIP = 8
SUBDIVISIONS_PER_BEAT = 12

FRAME_RATE = 100
BATCH_SIZE       = 4
NUM_WORKERS      = 6          # set 0 in notebooks/IDEs on Windows
DROP_LAST_TRAIN  = True
PIN_MEMORY       = True

# Audio / Feature extraction
SAMPLE_RATE     = 16000      # Hz
N_FFT           = 2048
HOP_LENGTH      = 160        # ~10 ms at 16kHz, adjust to match FRAME_RATE
WIN_LENGTH      = 2048
N_MELS          = 229
F_MIN           = 30.0
F_MAX           = SAMPLE_RATE / 2
WINDOW_FN       = "hann"
POWER           = 2.0        # magnitude^2 (torchaudio default)
LOG_OFFSET      = 1e-6       # eps for log scaling

# Training
NUM_EPOCHS       = 5
WEIGHT_DECAY     = 1e-4
SEED             = 0

# Model
EMBED_DIM = 1024

# Results
RESULTS_DIR      = os.path.join(PROJECT_ROOT, "results")
POSW_PATH = os.path.join(RESULTS_DIR, "pos_weight.pt")
os.makedirs(RESULTS_DIR, exist_ok=True)

class ModelConfig(TypedDict):
    type: str  # e.g., "BasicAMTCNN" or "BasicAMTTransformer"

class OptimizerConfig(TypedDict):
    type: str  # e.g., "AdamW"
    lr: float

class SchedulerConfig(TypedDict):
    type: Literal["onecycle", "cosine_warmup", "plateau"]

class VariantConfig(TypedDict):
    name: str
    model: ModelConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig

# Experiment set (name -> dict of model/scheduler configs)
MODEL_VARIANTS = [
    {
        "name": "OAF_max3e-3",
        "model": {"type": "OnsetAndFrames"},
        "optimizer": {"type": "AdamW", "lr": 3e-3},
        "scheduler": {"type": "onecycle"},
    }
]
