"""Fair hyperparameter tuning for backprop and PC (random search).

Protocol (see the Decision record in the project plan):
- Same number of trials for each model, on the VALIDATION set only.
- Tuning seeds differ from the final-run seeds (final runs use 0..4).
- PAIRED sampling: trial k gets the same learning rate for both models; PC's number of
  inference steps T comes from a separate random stream.
- Trials are interleaved (trial 0 for both models, then trial 1, ...), so an interrupted
  session leaves both models with the same number of finished trials.
- A trial's score = mean best-validation-accuracy over the tuning seeds; a diverged run
  counts as 0.0.
- Each trial is saved as its own JSON file; finished trials are skipped (resumable).
"""

import copy
import math
import random
import time
from pathlib import Path

import pandas as pd

from src.train_backprop import train_backprop
from src.train_pc import train_pc
from src.utils import load_json, run_done, save_json

LR_RANGE = (0.003, 0.3)
T_CHOICES = (10, 20, 30)
BASE_SEED = 12345


def sample_trial(trial_idx, model, base_seed=BASE_SEED, lr_range=LR_RANGE, t_choices=T_CHOICES):
    """Hyperparameters for one trial. Deterministic and PAIRED across models."""
    r_lr = random.Random(f"{base_seed}-lr-{trial_idx}")
    lr = math.exp(r_lr.uniform(math.log(lr_range[0]), math.log(lr_range[1])))
    params = {"lr": float(f"{lr:.4g}")}
    if model == "pc":
        r_t = random.Random(f"{base_seed}-T-{trial_idx}")
        params["inference_steps"] = r_t.choice(list(t_choices))
    return params


def make_cfg(base_cfg, model, params, state_lr, max_epochs, patience):
    """Copy of the base config with this trial's settings (the base is not modified)."""
    cfg = copy.deepcopy(base_cfg)
    cfg["training"]["lr"] = params["lr"]
    cfg["training"]["max_epochs"] = max_epochs
    cfg["training"]["patience"] = patience
    if model == "pc":
        cfg["pc"]["inference_steps"] = params["inference_steps"]
        cfg["pc"]["state_lr"] = state_lr
    return cfg


def score_runs(per_seed):
    """Mean validation accuracy over seeds; missing or diverged runs count as 0.0."""
    scores = []
    for r in per_seed:
        bad = r.get("diverged") or r.get("best_val_acc") is None
        scores.append(0.0 if bad else float(r["best_val_acc"]))
    return sum(scores) / len(scores)


def run_trial(model, trial_idx, base_cfg, data, seeds, out_dir, state_lr=0.2,
              max_epochs=15, patience=5, force=False, verbose=True, base_seed=BASE_SEED):
    path = Path(out_dir) / "trials" / f"{model}_trial{trial_idx:02d}.json"
    if run_done(path) and not force:
        rec = load_json(path)
        rec["resumed"] = True
        if verbose:
            print(f"[skip] {model} trial {trial_idx} already done "
                  f"(mean val acc {rec['mean_val_acc']:.4f})")
        return rec

    params = sample_trial(trial_idx, model, base_seed)
    cfg = make_cfg(base_cfg, model, params, state_lr, max_epochs, patience)
    runs_dir = str(Path(out_dir) / "runs" / model)

    per_seed = []
    for seed in seeds:
        if model == "pc":
            res = train_pc(cfg, data, seed=seed, results_dir=runs_dir, verbose=False)
        else:
            res = train_backprop(cfg, data, seed=seed, results_dir=runs_dir, verbose=False)
        per_seed.append({
            "seed": seed,
            "best_val_acc": res["best_val_acc"],
            "epochs_run": res["epochs_run"],
            "mean_epoch_time_s": res["mean_epoch_time_s"],
            "total_train_time_s": res["total_train_time_s"],
            "diverged": bool(res.get("diverged", False)),
        })

    rec = {
        "model": model,
        "trial": trial_idx,
        "params": params,
        "state_lr": state_lr if model == "pc" else None,
        "max_epochs": max_epochs,
        "patience": patience,
        "seeds": list(seeds),
        "per_seed": per_seed,
        "mean_val_acc": score_runs(per_seed),
        "total_time_s": sum(r["total_train_time_s"] for r in per_seed),
        "resumed": False,
    }
    save_json(rec, path)
    if verbose:
        extra = f" T={params['inference_steps']}" if model == "pc" else ""
        print(f"[done] {model:8s} trial {trial_idx:2d} lr={params['lr']:<8}{extra:6s} "
              f"mean val acc {rec['mean_val_acc']:.4f} | {rec['total_time_s']:.0f}s")
    return rec


def load_trials(out_dir) -> pd.DataFrame:
    rows = []
    for p in sorted((Path(out_dir) / "trials").glob("*.json")):
        r = load_json(p)
        rows.append({
            "model": r["model"],
            "trial": r["trial"],
            "lr": r["params"]["lr"],
            "inference_steps": r["params"].get("inference_steps"),
            "mean_val_acc": r["mean_val_acc"],
            "total_time_s": r["total_time_s"],
            "n_diverged": sum(s["diverged"] for s in r["per_seed"]),
        })
    return pd.DataFrame(rows)


def run_tuning(base_cfg, data, models=("backprop", "pc"), n_trials=12, seeds=(100, 101),
               out_dir="results/tables/tuning", state_lr=0.2, max_epochs=15, patience=5,
               force=False, verbose=True) -> pd.DataFrame:
    """Run (or resume) the search. Trials are interleaved across models."""
    for k in range(n_trials):
        for model in models:
            run_trial(model, k, base_cfg, data, seeds, out_dir, state_lr=state_lr,
                      max_epochs=max_epochs, patience=patience, force=force, verbose=verbose)
    return load_trials(out_dir)


# ----------------------------------------------------------------------------
# Views of the results
# ----------------------------------------------------------------------------


def best_by_trials(df, model, n=None):
    """Best trial among the first n trials of a model (equal-trials view)."""
    sub = df[df.model == model].sort_values("trial")
    if n is not None:
        sub = sub[sub.trial < n]
    return None if sub.empty else sub.loc[sub.mean_val_acc.idxmax()]


def best_under_time_budget(df, model, budget_s):
    """Best trial among those finished within a cumulative time budget (equal-wall-clock
    view). Trials are taken in index order; None if not even the first fits."""
    sub = df[df.model == model].sort_values("trial").copy()
    sub["cum_time_s"] = sub.total_time_s.cumsum()
    sub = sub[sub.cum_time_s <= budget_s]
    return None if sub.empty else sub.loc[sub.mean_val_acc.idxmax()]


def summarize(df, top_k=5):
    for model in sorted(df.model.unique()):
        sub = df[df.model == model].sort_values("mean_val_acc", ascending=False)
        cols = ["trial", "lr", "inference_steps", "mean_val_acc", "total_time_s", "n_diverged"]
        cols = [c for c in cols if not (c == "inference_steps" and model != "pc")]
        print(f"\n=== {model}: top {top_k} of {len(sub)} trials (validation accuracy, tuning seeds) ===")
        print(sub[cols].head(top_k).round(4).to_string(index=False))
    print("\n=== equal-trials view: best mean val acc in the first n trials ===")
    n_max = int(df.trial.max()) + 1
    for n in sorted({3, 6, 9, n_max}):
        if n > n_max:
            continue
        parts = []
        for model in sorted(df.model.unique()):
            b = best_by_trials(df, model, n)
            parts.append(f"{model} {b.mean_val_acc:.4f}")
        print(f"first {n:2d} trials: " + " | ".join(parts))
    total = {m: df[df.model == m].total_time_s.sum() for m in df.model.unique()}
    print("\ntotal training time per model (s):", {m: round(v) for m, v in total.items()})
