"""Tests for the backprop divergence guard and the stage-2 tuning options."""

import copy

from experiments.tune import load_runs, run_trial, sample_trial
from src.train_backprop import train_backprop
from tests.test_train_pc import CFG, make_data


def test_backprop_divergence_is_detected_and_saved(tmp_path):
    cfg = copy.deepcopy(CFG)
    cfg["training"]["lr"] = 1e6
    cfg["training"]["max_epochs"] = 5
    res = train_backprop(cfg, make_data(), seed=0, results_dir=str(tmp_path), verbose=False)
    assert res["diverged"] is True
    assert res["epochs_run"] < 5
    assert list(tmp_path.glob("*.json"))            # the run was still saved


def test_custom_ranges_are_respected_and_diverged_trial_scores_zero(tmp_path):
    for k in range(20):
        p = sample_trial(k, "pc", lr_range=(0.02, 0.08), t_choices=(20, 30, 50))
        assert 0.02 * 0.999 <= p["lr"] <= 0.08 * 1.001 and p["inference_steps"] in (20, 30, 50)

    data = make_data()
    rec = run_trial("pc", 0, CFG, data, seeds=(0,), out_dir=str(tmp_path / "a"), state_lr=0.05,
                    max_epochs=1, lr_range=(0.01, 0.02), t_choices=(3,), verbose=False)
    assert rec["params"]["inference_steps"] == 3
    assert 0.01 * 0.999 <= rec["params"]["lr"] <= 0.02 * 1.001

    bad = run_trial("backprop", 0, CFG, data, seeds=(0,), out_dir=str(tmp_path / "b"),
                    max_epochs=3, lr_range=(1e5, 1e6), verbose=False)
    assert bad["per_seed"][0]["diverged"] is True and bad["mean_val_acc"] == 0.0


def test_load_runs_has_one_row_per_seed(tmp_path):
    run_trial("pc", 0, CFG, make_data(), seeds=(0, 1), out_dir=str(tmp_path), state_lr=0.05,
              max_epochs=2, verbose=False)
    df = load_runs(str(tmp_path))
    assert len(df) == 2 and set(df.model) == {"pc"}
    assert {"lr", "T", "seed", "best_epoch", "epochs_run", "best_val_acc", "diverged"} <= set(df.columns)
