from data import get_splits, make_loader
from dataset.transforms import log_mel
from utils.display_midi import display_spectrogram
from experiment import run_experiment
from config import MODEL_VARIANTS
from train import load_or_compute_pos_weight


def main() :
    
    train, val = get_splits(log_mel)

    train_loader = make_loader(train, train=True)

    val_loader = make_loader(val, train=False)
    
    pos_w = load_or_compute_pos_weight(train_loader=train_loader)

    run_experiment(train_loader=train_loader, val_loader=val_loader, variant=MODEL_VARIANTS[0], pos_weight_vec=pos_w)

if __name__ == "__main__" :
    main()