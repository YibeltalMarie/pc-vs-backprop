"""Gradient-equivalence experiment: how close are PC's weight gradients to backprop's?

For the SAME weights and the SAME batch, per weight matrix W_l, we compare
  g_bp : the backprop gradient (hand-derived, verified against autograd)
  g_pc : the PC weight gradient after T inference steps
using cosine similarity (direction) and norm ratio ||g_pc|| / ||g_bp|| (size).
Both have the same sign convention (dE/dW vs dL/dW), so no sign flip is needed.

Layer index l = 0 is the matrix next to the input; the last index is the output layer.
Settings compared: output-layer scale, state learning rate (eta), and T.
This version compares at INITIALIZATION (untrained weights).
Each (scale, eta) setting is saved to its own JSON; finished settings are skipped.
"""

import hashlib
import json
import time
from pathlib import Path

import pandas as pd
import torch

from src.metrics import cosine_similarity, norm_ratio
from src.models.backprop_mlp import BackpropMLP
from src.models.common import to_targets
from src.models.pc_network import PCNetwork
from src.utils import run_done, save_json

NUM_CLASSES = 10


@torch.no_grad()
def compare_gradients(bp, pc, x, y, steps, state_lr, state_init="forward"):
    """Return {T: [(cos, ratio), ...]} with one pair per weight matrix, for each T in steps.

    bp and pc must hold the same weights. The PC states are advanced once, up to
    max(steps), and the weight gradient is read off at every requested T.
    """
    targets = to_targets(y, bp.layer_sizes[-1], x.dtype)
    _, cache = bp.forward_cached(x)
    bp_grads = bp.manual_gradients(cache, y, "mse")

    zs = pc.init_states(x, targets, init=state_init)
    result, done = {}, 0
    for T in sorted(set(steps)):
        while done < T:
            grads = pc.state_gradients(zs)
            for k, g in enumerate(grads, start=1):
                zs[k] = zs[k] - state_lr * g
            done += 1
        pc_grads = pc.weight_gradients(zs)
        result[T] = [
            (cosine_similarity(p[0], b[0]), norm_ratio(p[0], b[0]))
            for p, b in zip(pc_grads, bp_grads)
        ]
    return result


def _settings_tag(cfg, steps, seeds, batch_ids, batch_size) -> str:
    m = cfg["model"]
    key = json.dumps([m["activation"], m["init"], list(m["hidden_sizes"]),
                      list(steps), list(seeds), list(batch_ids), batch_size])
    return hashlib.md5(key.encode()).hexdigest()[:8]


def _records_from_files(paths) -> pd.DataFrame:
    rows = []
    for p in paths:
        with open(p) as f:
            rows.extend(json.load(f)["records"])
    return pd.DataFrame(rows)


def load_records(results_dir="results/tables/gradient_equivalence") -> pd.DataFrame:
    """Load every saved setting in a folder (may mix runs with different sweeps)."""
    return _records_from_files(sorted(Path(results_dir).glob("ge_*.json")))


def run_gradient_equivalence(
    cfg, data,
    scales=(1.0, 0.1), etas=(0.05, 0.1, 0.2),
    steps=(0, 1, 2, 3, 5, 10, 20, 50, 100, 200),
    seeds=(0, 1), batch_ids=(0, 1), batch_size=256,
    results_dir="results/tables/gradient_equivalence",
    force=False, verbose=True,
) -> pd.DataFrame:
    """Run (or resume) the sweep and return a DataFrame of the requested settings.

    Columns: scale, eta, seed, batch, steps, layer, cos, ratio.
    """
    m = cfg["model"]
    device = data.x_train.device
    sizes = [data.x_train.shape[1]] + list(m["hidden_sizes"]) + [NUM_CLASSES]
    tag = _settings_tag(cfg, steps, seeds, batch_ids, batch_size)
    paths = []

    for scale in scales:
        for eta in etas:
            path = Path(results_dir) / f"ge_scale{scale}_eta{eta}_{tag}.json"
            paths.append(path)
            if run_done(path) and not force:
                if verbose:
                    print(f"[skip] scale={scale} eta={eta} already done")
                continue
            t0 = time.time()
            records = []
            for seed in seeds:
                kw = dict(activation=m["activation"], init=m["init"], seed=seed,
                          last_layer_scale=scale)
                bp = BackpropMLP(sizes, **kw).to(device)
                pc = PCNetwork(sizes, **kw).to(device)
                for b in batch_ids:
                    x = data.x_train[b * batch_size:(b + 1) * batch_size]
                    y = data.y_train[b * batch_size:(b + 1) * batch_size]
                    for T, per_layer in compare_gradients(bp, pc, x, y, steps, eta).items():
                        for l, (c, r) in enumerate(per_layer):
                            records.append({"scale": scale, "eta": eta, "seed": seed,
                                            "batch": b, "steps": T, "layer": l,
                                            "cos": c, "ratio": r})
            save_json({"scale": scale, "eta": eta, "tag": tag, "steps": list(steps),
                       "seeds": list(seeds), "batch_ids": list(batch_ids),
                       "records": records}, path)
            if verbose:
                print(f"[done] scale={scale} eta={eta} in {time.time() - t0:.1f}s")
    return _records_from_files(paths)


# ----------------------------------------------------------------------------
# Summaries and plots
# ----------------------------------------------------------------------------


def report_table(df, scale, eta, steps=(1, 3, 10, 20, 50, 100, 200)):
    """Print mean cosine and mean norm ratio per layer (rows) and T (columns)."""
    sub = df[(df.scale == scale) & (df.eta == eta) & (df.steps.isin(steps))]
    cos = sub.pivot_table(index="layer", columns="steps", values="cos", aggfunc="mean")
    ratio = sub.pivot_table(index="layer", columns="steps", values="ratio", aggfunc="mean")
    print(f"scale={scale}, eta={eta}\nmean cosine similarity (rows = weight matrix, 0 = input side)")
    print(cos.round(3).to_string())
    print("\nmean norm ratio ||g_pc|| / ||g_bp||")
    print(ratio.round(4).to_string())
    return cos, ratio


def _layer_label(l, n):
    return f"W{l}" + (" (input side)" if l == 0 else " (output)" if l == n - 1 else "")


def _agg(sub):
    return (sub.groupby(["layer", "steps"])
               .agg(cos_mean=("cos", "mean"), cos_std=("cos", "std"),
                    ratio_mean=("ratio", "mean"), ratio_std=("ratio", "std"))
               .reset_index().fillna({"cos_std": 0.0, "ratio_std": 0.0}))


def _style(axes):
    for ax in axes:
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xlabel("inference steps T")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("cosine similarity with backprop gradient")
    axes[0].set_ylim(-0.05, 1.05)
    axes[1].set_ylabel("norm ratio  ||g_PC|| / ||g_BP||")
    axes[1].set_yscale("symlog", linthresh=1e-3)
    axes[1].axhline(1.0, color="k", ls="--", lw=1)


def plot_equivalence(df, scale, eta, out_dir="results/figures"):
    """Cosine and norm ratio versus T, one line per weight matrix."""
    import matplotlib.pyplot as plt
    g = _agg(df[(df.scale == scale) & (df.eta == eta)])
    n = int(g.layer.max()) + 1
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for l in range(n):
        d = g[g.layer == l].sort_values("steps")
        axes[0].errorbar(d.steps, d.cos_mean, yerr=d.cos_std, marker="o", capsize=2, label=_layer_label(l, n))
        axes[1].errorbar(d.steps, d.ratio_mean, yerr=d.ratio_std, marker="o", capsize=2, label=_layer_label(l, n))
    _style(axes)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"PC vs backprop gradients at initialization (output scale {scale}, state lr {eta})")
    fig.tight_layout()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(out_dir) / f"gradient_equivalence_scale{scale}_eta{eta}.png", dpi=150)
    return fig


def plot_by_eta(df, scale, layer=0, out_dir="results/figures"):
    """One weight matrix, one line per state learning rate."""
    import matplotlib.pyplot as plt
    sub = df[(df.scale == scale) & (df.layer == layer)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for eta in sorted(sub.eta.unique()):
        g = _agg(sub[sub.eta == eta])
        g = g[g.layer == layer].sort_values("steps")
        axes[0].errorbar(g.steps, g.cos_mean, yerr=g.cos_std, marker="o", capsize=2, label=f"eta={eta}")
        axes[1].errorbar(g.steps, g.ratio_mean, yerr=g.ratio_std, marker="o", capsize=2, label=f"eta={eta}")
    _style(axes)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Weight matrix W{layer}: effect of state learning rate (output scale {scale})")
    fig.tight_layout()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(out_dir) / f"gradient_equivalence_by_eta_scale{scale}_W{layer}.png", dpi=150)
    return fig
