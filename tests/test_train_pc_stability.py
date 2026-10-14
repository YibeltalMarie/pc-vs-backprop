"""Tests for the PC trainer's stability monitor (synthetic data, fast)."""

import copy

from tests.test_train_pc import CFG, make_data
from src.train_pc import train_pc


def test_small_state_lr_keeps_inference_energy_from_rising(tmp_path):
    cfg = copy.deepcopy(CFG)
    cfg["pc"]["state_lr"] = 0.05
    cfg["training"]["max_epochs"] = 3
    res = train_pc(cfg, make_data(), seed=0, results_dir=str(tmp_path), verbose=False)
    assert res["diverged"] is False
    assert all(h["inference_energy_increase_frac"] <= 0.05 for h in res["history"])


def test_huge_state_lr_is_flagged_as_unstable_or_divergent(tmp_path):
    cfg = copy.deepcopy(CFG)
    cfg["pc"]["state_lr"] = 5.0
    cfg["training"]["max_epochs"] = 3
    res = train_pc(cfg, make_data(), seed=0, results_dir=str(tmp_path), verbose=False)
    flagged = res["diverged"] or any(h["inference_energy_increase_frac"] > 0.5 for h in res["history"])
    assert flagged
