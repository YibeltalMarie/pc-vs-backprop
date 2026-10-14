"""Shared training utilities for the backprop and PC trainers.

Rules implemented here (see the Decision record in the project plan):
- model selection uses VALIDATION ACCURACY (comparable across losses and models)
- early stopping with patience; the best epoch's weights are restored
- evaluation runs in batches, never touching the test set unless asked
- each run is saved as one JSON file; finished runs are skipped (resumable)
- diagnostics() reports collapse indicators (output spread, dead ReLU units)
"""

import copy
from pathlib import Path

import torch
import yaml

from src.models.common import loss_fn


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_optimizer(params, name: str, lr: float, momentum: float = 0.9):
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr)
    if name == "sgd_momentum":
        return torch.optim.SGD(params, lr=lr, momentum=momentum)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    raise ValueError(f"Unknown optimizer '{name}'. Options: sgd, sgd_momentum, adam")


@torch.no_grad()
def evaluate(model, x, y, loss: str, batch_size: int = 2000):
    """Return (mean loss, accuracy) of model(x) against labels y."""
    n = x.shape[0]
    total_loss, correct = 0.0, 0
    for s in range(0, n, batch_size):
        xb, yb = x[s:s + batch_size], y[s:s + batch_size]
        out = model(xb)
        total_loss += loss_fn(out, yb, loss).item() * xb.shape[0]
        correct += (out.argmax(dim=1) == yb).sum().item()
    return total_loss / n, correct / n


@torch.no_grad()
def diagnostics(model, x, n: int = 2000) -> dict:
    """Collapse indicators on (up to) n samples.

    output_spread : mean over output units of the std across samples.
                    Near 0 means the model outputs (almost) the same vector for
                    every input, i.e. a constant predictor.
    dead_fractions: per hidden layer, fraction of units inactive for every
                    sample (relu models only; None otherwise).
    """
    x = x[:n]
    out = model(x)
    d = {"output_spread": out.std(dim=0).mean().item()}
    d["dead_fractions"] = model.dead_fractions(x) if hasattr(model, "dead_fractions") else None
    return d


class EarlyStopper:
    """Tracks the best validation metric (higher is better) and its weights."""

    def __init__(self, patience: int, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_metric = float("-inf")
        self.best_epoch = -1
        self.best_state = None
        self.bad_epochs = 0

    def update(self, metric: float, epoch: int, model) -> bool:
        """Record an epoch. Returns True if training should stop now."""
        if metric > self.best_metric + self.min_delta:
            self.best_metric = metric
            self.best_epoch = epoch
            self.best_state = copy.deepcopy(model.state_dict())
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience

    def restore(self, model) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def run_file(results_dir: str, run_name: str) -> Path:
    return Path(results_dir) / f"{run_name}.json"
