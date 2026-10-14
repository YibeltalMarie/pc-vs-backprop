"""Main comparison: tuned BP vs tuned PC, fresh seeds 0..4, test set evaluated once per run.

Settings were fixed from stage-2 validation tuning BEFORE any test number was seen:
best mean validation accuracy per model. Do not edit FINAL after looking at test results.
"""
import numpy as np
import pandas as pd

from experiments.tune import make_cfg
from src.train_backprop import train_backprop
from src.train_pc import train_pc

FINAL = {
    "backprop": {"lr": 0.0361},
    "pc": {"lr": 0.0161, "inference_steps": 50},
}
SEEDS = (0, 1, 2, 3, 4)

def run_main(base_cfg, data, out_dir="results/tables/main", seeds=SEEDS, settings=FINAL,
             state_lr=0.2, max_epochs=30, patience=5, verbose=True):
    rows = []
    for seed in seeds:
        for model, params in settings.items():
            cfg = make_cfg(base_cfg, model, params, state_lr, max_epochs, patience)
            fn = train_pc if model == "pc" else train_backprop
            res = fn(cfg, data, seed=seed, results_dir=f"{out_dir}/{model}",
                     evaluate_test=True, verbose=False)
            row = {
                "model": model, "seed": seed,
                "val_acc": res["best_val_acc"], "test_acc": res.get("test_acc"),
                "best_epoch": res["best_epoch"], "epochs_run": res["epochs_run"],
                "epoch_time_s": res["mean_epoch_time_s"], "train_time_s": res["total_train_time_s"],
                "diverged": bool(res.get("diverged", False)),
                "collapsed": (res["best_val_acc"] is None) or (res["best_val_acc"] < 0.2),
            }
            rows.append(row)
            if verbose:
                print(f"[done] {model:8s} seed {seed} val {row['val_acc']} test {row['test_acc']} "
                      f"best_epoch {row['best_epoch']} | {row['train_time_s']:.0f}s")
    return pd.DataFrame(rows)

def summarize_main(df):
    print("\n=== test accuracy, mean +- std over seeds (ddof=1) ===")
    for model, g in df.groupby("model"):
        print(f"{model:8s} test {g.test_acc.mean():.4f} +- {g.test_acc.std(ddof=1):.4f} | "
              f"val {g.val_acc.mean():.4f} | epoch {g.epoch_time_s.mean():.2f}s | "
              f"train {g.train_time_s.mean():.0f}s | diverged {int(g.diverged.sum())}"
              f"| collapsed {int(g.collapsed.sum())}")
    piv = df.pivot(index="seed", columns="model", values="test_acc")
    diff = piv["backprop"] - piv["pc"]
    print(f"\npaired gap BP - PC (same seed): mean {diff.mean():.4f}, "
          f"min {diff.min():.4f}, max {diff.max():.4f}")
