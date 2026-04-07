import os
import pickle

from torch.utils.data import Dataset


class CashDataset(Dataset):
    """
    PyTorch Dataset for beat-normalized spectrogram/label pairs produced by cache_spec.py.

    Each sample is a (spectrogram, labels) pair where:
        spectrogram : Tensor  [1, n_mels, L]  float32
        labels      : dict
            "on"       : Tensor  [L, 128]  float32   onset probabilities
            "off"      : Tensor  [L, 128]  float32   offset probabilities
            "frame"    : Tensor  [L, 128]  float32   active-note indicator
            "velocity" : Tensor  [L, 128]  float32   MIDI velocity  (0-127)

    If return_extra=True a third element is returned:
        extra : dict
            "times"    : Tensor  [L]       float32   beat-grid times (seconds)
            "max_time" : float | None               window end time
    """

    def __init__(self, cache_dir: str, return_extra: bool = False):
        self.cache_dir   = cache_dir
        self.return_extra = return_extra
        self.length = sum(
            name.startswith("spectrogram_") and name.endswith(".pkl")
            for name in os.listdir(cache_dir)
        )

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index):
        spec_path  = os.path.join(self.cache_dir, f"spectrogram_{index}.pkl")
        label_path = os.path.join(self.cache_dir, f"labels_{index}.pkl")

        with open(spec_path, "rb") as fh:
            spectrogram = pickle.load(fh)   # [1, n_mels, L]

        with open(label_path, "rb") as fh:
            labels = pickle.load(fh)        # {"on","off","frame","velocity"}

        if not self.return_extra:
            return spectrogram, labels

        extra_path = os.path.join(self.cache_dir, f"extra_{index}.pkl")
        extra = None
        if os.path.exists(extra_path):
            with open(extra_path, "rb") as fh:
                extra = pickle.load(fh)     # {"times": [L], "max_time": float|None}

        return spectrogram, labels, extra