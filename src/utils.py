"""Shared helpers: seeding, device choice, environment logging, resumable results."""

import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Fix every random number generator we use."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(preference: str = "auto") -> torch.device:
    """'auto' uses CUDA if available; otherwise CPU."""
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preference)


def environment_info() -> dict:
    """Record the software/hardware used, to save next to every result."""
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def save_json(obj: dict, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path) -> dict:
    with open(path) as f:
        return json.load(f)


def run_done(path) -> bool:
    """Resumable experiments: skip a run if its result file already exists."""
    return Path(path).exists()
