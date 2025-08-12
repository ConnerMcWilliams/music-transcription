import pandas as pd
from torch.utils.data import DataLoader
from dataset.dataset import MaestroDataset
from config import CSV_PATH, MAESTRO_ROOT, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, DROP_LAST_TRAIN

def get_splits():
    metadata = pd.read_csv(CSV_PATH)
    train_split = metadata[
        ((metadata['year'] == 2018) | (metadata['year'] == 2017)) &
        (metadata['split'] == 'train')
    ]
    val_split = metadata[
        ((metadata['year'] == 2018) | (metadata['year'] == 2017)) &
        (metadata['split'] == 'validation')
    ]
    return (MaestroDataset(train_split, MAESTRO_ROOT),
            MaestroDataset(val_split,   MAESTRO_ROOT))

def make_loader(dataset, train=True):
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=train,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=DROP_LAST_TRAIN if train else False,
        persistent_workers=True if NUM_WORKERS > 0 else False
    )
