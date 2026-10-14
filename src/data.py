"""Fashion-MNIST loading, splitting and normalization.

Decisions (see project plan):
- Split: 50k train / 10k validation (from the official 60k) / 10k test (official).
- The split uses a FIXED seed, independent of the model seed.
- Normalization uses mean/std computed on the training split only.
- Data are kept as in-memory tensors; batches come from iterate_batches().
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torchvision import datasets

NUM_CLASSES = 10
SPLIT_SEED = 42


@dataclass
class DataBundle:
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_val: torch.Tensor
    y_val: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    mean: float
    std: float


def load_fashion_mnist(
    root: str = "data/raw",
    val_size: int = 10000,
    split_seed: int = SPLIT_SEED,
    device: str = "cpu",
) -> DataBundle:
    """Download (if needed), split, and normalize Fashion-MNIST."""
    train_set = datasets.FashionMNIST(root=root, train=True, download=True)
    test_set = datasets.FashionMNIST(root=root, train=False, download=True)

    # uint8 [N, 28, 28] -> float [N, 784] in [0, 1]
    x_all = train_set.data.float().div(255.0).reshape(len(train_set), -1)
    y_all = train_set.targets.long()
    x_test = test_set.data.float().div(255.0).reshape(len(test_set), -1)
    y_test = test_set.targets.long()

    # Fixed, reproducible split
    g = torch.Generator().manual_seed(split_seed)
    perm = torch.randperm(len(x_all), generator=g)
    val_idx, train_idx = perm[:val_size], perm[val_size:]

    x_train, y_train = x_all[train_idx], y_all[train_idx]
    x_val, y_val = x_all[val_idx], y_all[val_idx]

    # Statistics from the training split ONLY (no leakage)
    mean = x_train.mean().item()
    std = x_train.std().item()

    def norm(x: torch.Tensor) -> torch.Tensor:
        return (x - mean) / std

    return DataBundle(
        x_train=norm(x_train).to(device),
        y_train=y_train.to(device),
        x_val=norm(x_val).to(device),
        y_val=y_val.to(device),
        x_test=norm(x_test).to(device),
        y_test=y_test.to(device),
        mean=mean,
        std=std,
    )


def one_hot(y: torch.Tensor, num_classes: int = NUM_CLASSES) -> torch.Tensor:
    """Class labels -> one-hot float targets (used with the MSE energy)."""
    return F.one_hot(y, num_classes).float()


def iterate_batches(x, y, batch_size, shuffle=True, generator=None):
    """Yield (x_batch, y_batch). Pass a torch.Generator for reproducible shuffling."""
    n = x.shape[0]
    if shuffle:
        idx = torch.randperm(n, generator=generator).to(x.device)
    else:
        idx = torch.arange(n, device=x.device)
    for start in range(0, n, batch_size):
        b = idx[start:start + batch_size]
        yield x[b], y[b]
