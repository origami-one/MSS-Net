import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from utils import load_recording, normalize_ecg
from scipy import signal
import numpy as np
import random


# =========================================================================
# 1. Augmentation Helpers
# =========================================================================
def apply_masking(x, mask_ratio=0.2):
    """Apply random time-segment masking and lead masking."""
    C, L = x.shape

    # Time Masking (50% probability)
    if random.random() < 0.5:
        mask_len = int(L * mask_ratio)
        if L > mask_len:
            start = random.randint(0, L - mask_len - 1)
            x[:, start:start + mask_len] = 0.0

    # Channel/Lead Masking (30% probability)
    if random.random() < 0.3:
        mask_ch = random.randint(0, C - 1)
        x[mask_ch, :] = 0.0

    return x


# =========================================================================
# 2. Path & Loader Helpers
# =========================================================================
def _get_csv_paths(opt):
    """Generate standard paths for train, validation, and test splits."""
    dataset_name = getattr(opt, "dataset", "ptbxl")
    base_dir = os.path.join(opt.dirs_for_train, dataset_name)

    return (
        os.path.join(base_dir, "train_dataset.csv"),
        os.path.join(base_dir, "val_dataset.csv"),
        os.path.join(base_dir, "test_dataset.csv")
    )


def create_dataloader(opt):
    """Initialize training and validation data loaders."""
    dataset_name = getattr(opt, "dataset", "ptbxl").lower()
    train_csv, valid_csv, _ = _get_csv_paths(opt)

    train = pd.read_csv(train_csv)
    valid = pd.read_csv(valid_csv)

    train_loader = DataLoader(
        dataset=ECG_Dataset(opt, train, mode='train', dataset_name=dataset_name),
        batch_size=opt.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True
    )

    validation_loader = DataLoader(
        dataset=ECG_Dataset(opt, valid, mode='valid', dataset_name=dataset_name),
        batch_size=opt.batch_size, shuffle=False, num_workers=4
    )

    return train_loader, validation_loader


def create_dataloader_for_test(opt):
    """Initialize test data loader."""
    dataset_name = getattr(opt, "dataset", "ptbxl").lower()
    _, _, test_csv = _get_csv_paths(opt)

    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"Test file missing: {test_csv}")

    test = pd.read_csv(test_csv)
    return DataLoader(
        dataset=ECG_Dataset(opt, test, mode='test', dataset_name=dataset_name),
        batch_size=1, shuffle=False, num_workers=4
    )


# =========================================================================
# 3. ECG Dataset Class
# =========================================================================
class ECG_Dataset(Dataset):
    def __init__(self, opt, dataset, mode, dataset_name='ptbxl'):
        self.fs = opt.fs
        self.samples = opt.samples
        self.mode = mode
        self.dataset = dataset.reset_index(drop=True)
        self.dataset_name = dataset_name.lower()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset.iloc[idx]
        fs = float(row['fs']) if 'fs' in self.dataset.columns else float(self.fs)

        # Parse multi-label target string to tensor
        targets_list = list(map(float, row['target'].replace('[', '').replace(']', '').replace(',', ' ').split()))
        targets = torch.tensor(targets_list, dtype=torch.float32)

        # Load recording and handle NaNs
        inputs = load_recording(row['record'])
        inputs = np.nan_to_num(inputs)

        # Z-score Normalization
        if np.std(inputs) > 1e-6:
            inputs = normalize_ecg(inputs)

        # Gain Augmentation (Train only): Random scaling for amplitude invariance
        if self.mode == 'train':
            inputs = inputs * random.uniform(0.9, 1.1)

        # Resampling Logic
        target_fs = float(self.fs)
        if abs(fs - target_fs) > 1e-6:
            # Polyphase resampling for specific integer ratios, else fallback to FFT
            if (fs == 1000.0 and target_fs == 500.0) or (fs == 500.0 and target_fs == 250.0):
                inputs = signal.resample_poly(inputs, up=1, down=2, axis=-1)
            else:
                num_samples = int(inputs.shape[1] * target_fs / fs)
                inputs = signal.resample(inputs, num_samples, axis=1)

        inputs = torch.from_numpy(np.nan_to_num(inputs)).float()

        # Consistent temporal length via random cropping or zero padding
        if inputs.size(1) > self.samples:
            start = random.randint(0, inputs.size(1) - self.samples - 1) if self.mode == 'train' else (inputs.size(
                1) - self.samples) // 2
            inputs = inputs[:, start:start + self.samples]
        else:
            pad_len = self.samples - inputs.size(1)
            inputs = torch.cat([inputs, torch.zeros([inputs.size(0), pad_len])], dim=1)

        # Apply masking augmentation during training
        if self.mode == 'train':
            inputs = apply_masking(inputs, mask_ratio=0.15)

        return inputs, targets