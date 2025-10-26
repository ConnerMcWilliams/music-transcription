import pandas as pd
from torch.utils.data import DataLoader

from dataset.beat_dataset import MaestroDatasetWithWindowingInBeats
from dataset.transforms import MelTransform

from config import (
    CSV_PATH,
    MAESTRO_ROOT,
    SAMPLE_RATE,
    N_MELS,
    SUBDIVISIONS_PER_BEAT,
    BEATS_PER_CLIP,
    HOP_LENGTH,
    # dataloader config:
    BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, DROP_LAST_TRAIN,
    BATCH_SAMPLER, SAMPLER, COLLATE_FN, TIMEOUT,
    PIN_MEMORY_DEVICE, WORKER_INIT_FN, PREFETCH_FACTOR,
    SHUFFLE, PERSISTENT_WORKERS, MULTIPROCESSING_CONTEXT,
    GENERATOR,
)


def get_splits(
    transform=None,
    small: bool = False,
    subset_fraction: float = 1.0,
    normalization: bool = True,
):
    """
    Build train/val datasets from MAESTRO metadata.

    Args:
        transform:
            Optional custom mel transform callable with signature
            (waveform, sr) -> (mel_db[n_mels, T], hop_length).
            If None, we'll build a default MelTransform() instance.

        small:
            If True, select only a fraction of each split for timing/experiments.

        subset_fraction:
            Fraction of the split to keep when small=True.
            e.g. 1/25 gives ~4% of the data.

        normalization:
            True  -> tempo-normalized beat warp (use_beat_warp=True)
            False -> plain-time windows (use_beat_warp=False)

    Returns:
        (train_ds, val_ds)
    """

    # IMPORTANT: create an INSTANCE, not the class.
    mel_tx = transform if transform is not None else MelTransform()

    metadata = pd.read_csv(CSV_PATH)

    train_split = metadata[metadata["split"] == "train"].copy()
    val_split   = metadata[metadata["split"] == "validation"].copy()

    if small:
        # --- duration-aware subsampling ---

        # Work on copies so we don't mutate original before printing
        train_tmp = train_split.copy()
        val_tmp   = val_split.copy()

        if "duration" not in train_tmp.columns or "duration" not in val_tmp.columns:
            raise ValueError(
                "Expected 'duration' column in MAESTRO metadata for proportional subsampling."
            )

        # total durations in seconds
        total_train_dur = train_tmp["duration"].sum()
        total_val_dur   = val_tmp["duration"].sum()

        target_train_dur = total_train_dur * subset_fraction
        target_val_dur   = total_val_dur * subset_fraction

        # Sort by duration descending so we hit the target duration quickly
        train_tmp = train_tmp.sort_values("duration", ascending=False).reset_index(drop=True)
        val_tmp   = val_tmp.sort_values("duration",   ascending=False).reset_index(drop=True)

        # Greedily take rows until we reach the target total duration
        def take_until_duration(df, target_seconds):
            picked_rows = []
            dur_accum = 0.0
            for _, row in df.iterrows():
                picked_rows.append(row)
                dur_accum += float(row["duration"])
                if dur_accum >= target_seconds:
                   break
            # turn list[Series] back into DataFrame
            return pd.DataFrame(picked_rows).reset_index(drop=True), dur_accum

        train_subset, train_subset_dur = take_until_duration(train_tmp, target_train_dur)
        val_subset,   val_subset_dur   = take_until_duration(val_tmp,   target_val_dur)

        train_split = train_subset
        val_split   = val_subset

        # Some nice debug prints so you know what you're training on
        print(
            "Retrieving duration-weighted subset of dataset:\n"
            f"  target fraction: {subset_fraction*100:.2f}% of total audio time\n"
            f"  train: {len(train_split)} files "
            f"({train_subset_dur/60:.1f} min out of {total_train_dur/60:.1f} min)\n"
            f"  val:   {len(val_split)} files "
            f"({val_subset_dur/60:.1f} min out of {total_val_dur/60:.1f} min)"
        )
    else:
        print("Retrieved full dataset.")

    use_beat_warp = bool(normalization)

    print(f"Loading {'normalized' if use_beat_warp else 'plain'} training dataset.")
    train_ds = MaestroDatasetWithWindowingInBeats(
        metadata=train_split,
        root_dir=MAESTRO_ROOT,
        mel_tx=mel_tx,  # <-- pass the *instance*
        n_mels=N_MELS,
        subdivisions=SUBDIVISIONS_PER_BEAT,
        beats_per_window=BEATS_PER_CLIP,
        hop_beats=None,                # default: window hop = beats_per_window
        hop_length=HOP_LENGTH,
        sr_target=SAMPLE_RATE,
        waveform_tx=None,
        mel_aug_prewarp=None,
        mel_aug_postwarp=None,
        target_transform=None,
        return_time_labels=False,
        time_roll_rate=None,
        use_beat_warp=use_beat_warp,
    )
    print("Training dataset complete.")

    print(f"Loading {'normalized' if use_beat_warp else 'plain'} validation dataset.")
    val_ds = MaestroDatasetWithWindowingInBeats(
        metadata=val_split,
        root_dir=MAESTRO_ROOT,
        mel_tx=mel_tx,  # same transform instance is fine
        n_mels=N_MELS,
        subdivisions=SUBDIVISIONS_PER_BEAT,
        beats_per_window=BEATS_PER_CLIP,
        hop_beats=None,
        hop_length=HOP_LENGTH,
        sr_target=SAMPLE_RATE,
        waveform_tx=None,
        mel_aug_prewarp=None,
        mel_aug_postwarp=None,
        target_transform=None,
        return_time_labels=False,
        time_roll_rate=None,
        use_beat_warp=use_beat_warp,
    )
    print("Validation dataset loaded.")

    return train_ds, val_ds



def make_loader(dataset, train: bool = True):
    """
    Returns a DataLoader using config settings, with guardrails for PyTorch constraints.

    Rules enforced:
    - If BATCH_SAMPLER is provided, we are not allowed to also provide
      batch_size / shuffle / sampler / drop_last.
    - PREFETCH_FACTOR only applies when NUM_WORKERS > 0.
    - persistent_workers only matters when NUM_WORKERS > 0.
    - pin_memory_device is only passed when we're actually pinning.
    """

    use_batch_sampler = BATCH_SAMPLER is not None

    # shuffle logic
    if use_batch_sampler or (SAMPLER is not None):
        effective_shuffle = False
    else:
        effective_shuffle = SHUFFLE if train else False

    # drop_last logic
    if use_batch_sampler:
        effective_drop_last = False
    else:
        effective_drop_last = (DROP_LAST_TRAIN if train else False)

    # prefetch logic
    use_prefetch = (
        NUM_WORKERS is not None
        and NUM_WORKERS > 0
        and PREFETCH_FACTOR is not None
    )

    # build kwargs dynamically so we don't pass illegal combos
    loader_kwargs = dict(
        num_workers=NUM_WORKERS,
        collate_fn=COLLATE_FN,
        timeout=TIMEOUT,
        worker_init_fn=WORKER_INIT_FN,
        persistent_workers=bool(PERSISTENT_WORKERS) and (NUM_WORKERS > 0),
        multiprocessing_context=MULTIPROCESSING_CONTEXT,
        generator=GENERATOR,
    )

    if not use_batch_sampler:
        loader_kwargs.update({
            "batch_size": BATCH_SIZE,
            "shuffle": effective_shuffle,
            "sampler": SAMPLER,
            "drop_last": effective_drop_last,
        })
    else:
        loader_kwargs.update({
            "batch_sampler": BATCH_SAMPLER,
        })

    # pin_memory and pin_memory_device
    if PIN_MEMORY:
        loader_kwargs["pin_memory"] = True
        # pin_memory_device was added in newer PyTorch; safe to guard it
        if PIN_MEMORY_DEVICE:
            loader_kwargs["pin_memory_device"] = PIN_MEMORY_DEVICE
    else:
        loader_kwargs["pin_memory"] = False

    # prefetch_factor (only valid if workers > 0)
    if use_prefetch:
        loader_kwargs["prefetch_factor"] = max(2, int(PREFETCH_FACTOR))

    return DataLoader(dataset, **loader_kwargs)
