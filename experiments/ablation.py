"""Ablations: one factor at a time from the main configuration. VALIDATION ONLY (test untouched).

Protocol
- Main settings (FINAL: BP lr 0.0361; PC lr 0.0161, T=50) with eta=0.2, max 30 epochs, patience 5.
- Seeds 0,1,2. The baseline is read from the finished main-comparison runs (same seeds, no retraining).
- Learning rates are NOT retuned per variant (except where the variant says so): a variant that
  loses may simply be untuned. Report that.
- A run counts as bad if it diverged, or its best validation accuracy is None or < 0.2 (collapse);
  bad runs score 0.0 in means, and are counted separately.
- Each run is saved as its own JSON under out_dir/<variant>/<model>/ and skipped if it exists.
Variant spec keys: models, model, training, pc (config overrides), seeds, max_epochs, compare_to.
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.main_comparison import FINAL
from experiments.tune import make_cfg
from src.train_backprop import train_backprop
from src.train_pc import train_pc

SEEDS = (0, 1, 2)
STATE_LR = 0.2
BOTH = ["backprop", "pc"]

SHORT = {
    # --- weak output clamp (the one test of a gap explanation); lr 0.0361 = BP's lr (exploratory)
    "lam0.1": {"models": ["pc"], "pc": {"output_weight": 0.1}},
    "lam0.1_lr0.0361": {"models": ["pc"], "pc": {"output_weight": 0.1}, "training": {"lr": 0.0361}},
    # --- open decisions
    "init_zeros": {"models": ["pc"], "pc": {"state_init": "zeros"}},
    "init_random": {"models": ["pc"], "pc": {"state_init": "random"}},
    "tanh_xavier": {"models": BOTH, "model": {"activation": "tanh", "init": "xavier"}},
    "scale0.01": {"models": BOTH, "model": {"last_layer_scale": 0.01}},
    # --- literature claim: small batches (own reference, same 15-epoch cap, seed 0 only: cost)
    "bs256_cap15": {"models": BOTH, "max_epochs": 15, "seeds": (0,)},
    "bs32_cap15": {"models": BOTH, "training": {"batch_size": 32}, "max_epochs": 15,
                   "seeds": (0,), "compare_to": "bs256_cap15"},
}
EXTRA = {
    "T100": {"models": ["pc"], "pc": {"inference_steps": 100}},
    "eta0.1": {"models": ["pc"], "pc": {"state_lr": 0.1}},
    "eta0.15": {"models": ["pc"], "pc": {"state_lr": 0.15}},
}


def build_cfg(base_cfg, model, spec, settings=FINAL, max_epochs=30, patience=5):
    cfg = make_cfg(base_cfg, model, settings[model], STATE_LR, max_epochs, patience)
    for sec in ("model", "training", "pc"):
        cfg[sec].update(spec.get(sec, {}))
    return cfg


def _row(variant, model, seed, res):
    acc = res["best_val_acc"]
    diverged = bool(res.get("diverged", False))
    collapsed = acc is None or acc < 0.2
    return {"variant": variant, "model": model, "seed": seed,
            "val_acc": acc, "score": 0.0 if (diverged or collapsed) else float(acc),
            "diverged": diverged, "collapsed": collapsed,
            "best_epoch": res["best_epoch"], "epochs_run": res["epochs_run"],
            "epoch_time_s": res["mean_epoch_time_s"]}


def load_baseline(main_dir, models=BOTH, seeds=SEEDS):
    rows = []
    for model in models:
        by_seed = {}
        for p in sorted((Path(main_dir) / model).glob("*.json")):
            r = json.load(open(p))
            by_seed[r["seed"]] = r
        for s in seeds:
            if s not in by_seed:
                raise FileNotFoundError(f"baseline run missing: {main_dir}/{model} seed {s}")
            rows.append(_row("baseline", model, s, by_seed[s]))
    return rows


def run_variant(name, spec, base_cfg, data, out_dir, settings=FINAL, seeds=SEEDS,
                max_epochs=30, verbose=True):
    seeds = spec.get("seeds", seeds)
    max_epochs = spec.get("max_epochs", max_epochs)
    rows = []
    for model in spec["models"]:
        cfg = build_cfg(base_cfg, model, spec, settings, max_epochs)
        fn = train_pc if model == "pc" else train_backprop
        for seed in seeds:
            res = fn(cfg, data, seed=seed, results_dir=str(Path(out_dir) / name / model),
                     verbose=False)
            row = _row(name, model, seed, res)
            rows.append(row)
            if verbose:
                print(f"[done] {name:16s} {model:8s} seed {seed} val {row['val_acc']} "
                      f"diverged {row['diverged']} collapsed {row['collapsed']} "
                      f"best_epoch {row['best_epoch']} {row['epoch_time_s'] or 0:.1f}s/epoch")
    return rows


def run_ablation(base_cfg, data, variants=None, out_dir="results/tables/ablation",
                 main_dir="results/tables/main", settings=FINAL, seeds=SEEDS,
                 max_epochs=30, verbose=True):
    variants = SHORT if variants is None else variants
    rows = load_baseline(main_dir, seeds=seeds)
    for name, spec in variants.items():
        rows += run_variant(name, spec, base_cfg, data, out_dir, settings, seeds,
                            max_epochs, verbose)
    df = pd.DataFrame(rows)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(out_dir) / "runs.csv", index=False)
    return df


def summarize_ablation(df, variants=None):
    """One line per (variant, model): mean +- std validation score, paired delta against its
    reference (baseline unless the spec says compare_to), bad-run counts, epoch time."""
    variants = SHORT if variants is None else variants
    out = []
    order = ["baseline"] + [v for v in variants if v in set(df.variant)]
    for v in order:
        ref_name = "baseline" if v == "baseline" else variants[v].get("compare_to", "baseline")
        for model in BOTH:
            g = df[(df.variant == v) & (df.model == model)]
            if g.empty:
                continue
            ref = df[(df.variant == ref_name) & (df.model == model)]
            common = sorted(set(g.seed) & set(ref.seed))
            if v == "baseline" or not common:
                delta = float("nan")
            else:
                a = g.set_index("seed").loc[common, "score"]
                b = ref.set_index("seed").loc[common, "score"]
                delta = float((a - b).mean())
            out.append({"variant": v, "model": model, "n_seeds": len(g),
                        "val_mean": g.score.mean(),
                        "val_std": g.score.std(ddof=1) if len(g) > 1 else float("nan"),
                        "delta_vs_ref": delta, "ref": "-" if v == "baseline" else ref_name,
                        "diverged": int(g.diverged.sum()), "collapsed": int(g.collapsed.sum()),
                        "epoch_s": g.epoch_time_s.mean()})
    table = pd.DataFrame(out)
    print(table.round(4).to_string(index=False))
    return table
