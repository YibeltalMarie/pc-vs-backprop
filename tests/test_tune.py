"""Tests for the tuning protocol (synthetic data, tiny budgets)."""

import copy
import os

import pandas as pd

from experiments.tune import (
    LR_RANGE, T_CHOICES, best_under_time_budget, make_cfg, run_tuning, sample_trial, score_runs)
from tests.test_train_pc import CFG, make_data


def test_sampling_is_deterministic_and_paired():
    for k in range(30):
        bp = sample_trial(k, "backprop")
        pc = sample_trial(k, "pc")
        assert bp == sample_trial(k, "backprop")                 # deterministic
        assert bp["lr"] == pc["lr"]                              # paired learning rate
        assert "inference_steps" not in bp                       # backprop has no T
        assert pc["inference_steps"] in T_CHOICES
        assert LR_RANGE[0] * 0.999 <= bp["lr"] <= LR_RANGE[1] * 1.001
    assert len({sample_trial(k, "backprop")["lr"] for k in range(30)}) > 20   # trials differ


def test_make_cfg_sets_trial_values_without_mutating_base():
    base = copy.deepcopy(CFG)
    snapshot = copy.deepcopy(base)
    cfg = make_cfg(base, "pc", {"lr": 0.05, "inference_steps": 30}, state_lr=0.2,
                   max_epochs=7, patience=3)
    assert cfg["training"]["lr"] == 0.05 and cfg["training"]["max_epochs"] == 7
    assert cfg["training"]["patience"] == 3
    assert cfg["pc"]["inference_steps"] == 30 and cfg["pc"]["state_lr"] == 0.2
    assert base == snapshot                                       # base untouched


def test_score_counts_diverged_or_missing_runs_as_zero():
    runs = [{"best_val_acc": 0.9, "diverged": False},
            {"best_val_acc": 0.8, "diverged": True},
            {"best_val_acc": None, "diverged": False}]
    assert abs(score_runs(runs) - 0.3) < 1e-12


def test_run_tuning_saves_interleaves_and_resumes(tmp_path):
    data = make_data()
    kwargs = dict(models=("backprop", "pc"), n_trials=2, seeds=(100,), out_dir=str(tmp_path),
                  state_lr=0.05, max_epochs=2, patience=5, verbose=False)
    df = run_tuning(CFG, data, **kwargs)
    assert len(df) == 4
    assert sorted(df.model.unique()) == ["backprop", "pc"]
    assert df.mean_val_acc.between(0.0, 1.0).all()
    assert df[df.model == "pc"].inference_steps.notna().all()
    assert df[df.model == "backprop"].inference_steps.isna().all()

    files = sorted((tmp_path / "trials").glob("*.json"))
    assert len(files) == 4
    mtimes = [os.stat(f).st_mtime_ns for f in files]
    df2 = run_tuning(CFG, data, **kwargs)                         # resume: nothing recomputed
    assert [os.stat(f).st_mtime_ns for f in files] == mtimes
    assert len(df2) == 4


def test_best_under_time_budget_uses_cumulative_time_in_trial_order():
    df = pd.DataFrame({
        "model": ["pc"] * 4,
        "trial": [0, 1, 2, 3],
        "mean_val_acc": [0.70, 0.85, 0.80, 0.90],
        "total_time_s": [10.0, 10.0, 10.0, 10.0],
    })
    assert best_under_time_budget(df, "pc", 5.0) is None
    assert best_under_time_budget(df, "pc", 10.0).trial == 0
    assert best_under_time_budget(df, "pc", 25.0).trial == 1      # trials 0 and 1 fit
    assert best_under_time_budget(df, "pc", 100.0).trial == 3
