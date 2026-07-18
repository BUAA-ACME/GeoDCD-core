import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class CausalTimeSeriesDataset(Dataset):
    """Sliding-window dataset for time series causal discovery."""

    def __init__(self, data, window_size, stride=1, mode='train', split_ratio=1.0):
        super().__init__()
        self.window_size = window_size
        self.stride = stride

        split_point = int(len(data) * split_ratio)
        if mode == 'train':
            self.data = data[:split_point]
        elif mode == 'val':
            self.data = data[split_point:]
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if len(self.data) < window_size:
            self.n_samples = 0
        else:
            self.n_samples = (len(self.data) - window_size) // stride + 1

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.window_size
        sample = self.data[start:end]
        return torch.from_numpy(sample).float().t()  # (T, N) -> (N, T)


def load_from_disk(base_path, dataset_name, replica_id):
    """Load data, ground truth, and coordinates from .npy files."""
    data_dir = os.path.join(base_path, dataset_name)
    data_path = os.path.join(data_dir, f'data_{replica_id}.npy')
    gt_path = os.path.join(data_dir, f'gt_{replica_id}.npy')
    coords_path = os.path.join(data_dir, f'coords_{replica_id}.npy')

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data not found: {data_path}")

    data_np = np.load(data_path)
    N = data_np.shape[1]

    if os.path.exists(gt_path):
        gt_np = np.load(gt_path)
    else:
        print("Warning: Ground truth not found. Metrics will be skipped.")
        gt_np = None

    if os.path.exists(coords_path):
        coords_np = np.load(coords_path)
    else:
        print("Warning: Coordinates not found. Using random coordinates.")
        np.random.seed(42)
        coords_np = np.random.rand(N, 2)

    return data_np, gt_np, coords_np


def get_data_context(args):
    """Build DataLoaders and metadata from a config namespace."""
    base_path = getattr(args, 'data_path', 'data/synthetic')
    dataset_name = getattr(args, 'dataset', 'lorenz96')
    replica_id = getattr(args, 'replica_id', 0)
    window_size = getattr(args, 'window_size', 100)
    stride = getattr(args, 'stride', 10)
    batch_size = getattr(args, 'batch_size', 32)
    # Windows (os.name == 'nt') multiprocessing DataLoader workers frequently
    # hang/freeze, so default to 0 there. Overridable via --num_workers.
    _nw = getattr(args, 'num_workers', None)
    num_workers = _nw if _nw is not None else (4 if os.name != 'nt' else 0)

    print(f"Loading {dataset_name} (Replica {replica_id})...")
    data_np, gt_np, coords_np = load_from_disk(base_path, dataset_name, replica_id)

    # Standardize
    mean = data_np.mean(axis=0)
    std = data_np.std(axis=0) + 1e-5
    data_np = (data_np - mean) / std

    train_ds = CausalTimeSeriesDataset(data_np, window_size, stride, mode='train')
    val_ds = CausalTimeSeriesDataset(data_np, window_size, stride, mode='val')
    print(f"Data Split: Train={len(train_ds)} samples, Val={len(val_ds)} samples")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    meta = {"coords": coords_np, "gt_fine": gt_np}
    return train_loader, val_loader, meta
