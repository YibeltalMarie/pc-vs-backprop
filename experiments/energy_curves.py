"""Layer-wise energy figures: (A) over training, from saved run histories;
(B) over inference steps, for an untrained and a trained network.
Layers are numbered 1..4: E1..E3 hidden, E4 output (the layer clamped to the target)."""
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from experiments.main_comparison import FINAL
from experiments.tune import make_cfg
from src.models.pc_network import PCNetwork
from src.train_pc import train_pc

def plot_over_training(main_dir, out_png):
    hists = [json.load(open(p))["history"] for p in sorted(glob.glob(f"{main_dir}/pc/*.json"))]
    n = min(len(h) for h in hists)
    start = np.mean([[h[e]["train_energy_start"] for e in range(n)] for h in hists], axis=0)
    end = np.mean([[h[e]["train_energy_end"] for e in range(n)] for h in hists], axis=0)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for a, arr, title in ((ax[0], start, "inference step 0"), (ax[1], end, "after inference (T=50)")):
        for k in range(arr.shape[1]):
            a.plot(arr[:, k], label=f"E{k+1}")
        a.plot(arr.sum(1), "k--", label="total")
        a.set_yscale("log"); a.set_xlabel("epoch"); a.set_title(title)
    ax[0].set_ylabel("batch-mean energy"); ax[0].legend()
    fig.suptitle(f"PC layer energies over training (mean of {len(hists)} seeds, first {n} epochs)")
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)
    print(f"over training, epoch 0 -> {n-1}:")
    for name, arr in (("step 0", start), ("after T", end)):
        print(f"  {name:7s} first {np.round(arr[0], 5)} | last {np.round(arr[-1], 5)}")

@torch.no_grad()
def inference_curve(net, x, y, steps, state_lr):
    target = F.one_hot(y, 10).float()
    _, hist = net.infer(x, target, steps=steps, state_lr=state_lr, record=True)
    return np.array(hist)                      # (steps+1, n_layers)

def plot_over_inference(curves, out_png, steps):
    fig, ax = plt.subplots(1, len(curves), figsize=(11, 4), sharey=True)
    for a, (name, arr) in zip(ax, curves.items()):
        for k in range(arr.shape[1]):
            a.plot(arr[:, k], label=f"E{k+1}")
        a.plot(arr.sum(1), "k--", label="total")
        a.set_yscale("log"); a.set_xlabel("inference step"); a.set_title(name)
    ax[0].set_ylabel("batch-mean energy"); ax[0].legend()
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)
    for name, arr in curves.items():
        print(f"\n{name}: energy per layer at selected steps (columns E1..E4, total)")
        for t in (0, 1, 2, 5, 10, 20, steps):
            if t <= steps:
                print(f"  step {t:3d}: {np.round(arr[t], 5)}  total {arr[t].sum():.5f}")

def run_energy(base_cfg, data, main_dir, fig_dir="results/figures",
               run_dir="results/tables/energy_run", steps=50, state_lr=0.2, n_eval=512):
    Path(fig_dir).mkdir(parents=True, exist_ok=True)
    plot_over_training(main_dir, f"{fig_dir}/energy_over_training.png")

    cfg = make_cfg(base_cfg, "pc", FINAL["pc"], state_lr, 30, 5)
    res = train_pc(cfg, data, seed=0, results_dir=run_dir, save_model=True, verbose=False)
    m = cfg["model"]
    sizes = [data.x_train.shape[1]] + list(m["hidden_sizes"]) + [10]
    kw = dict(activation=m["activation"], init=m["init"], seed=0,
              last_layer_scale=m.get("last_layer_scale", 1.0), loss=m["loss"])
    device = data.x_train.device
    untrained = PCNetwork(sizes, **kw).to(device)
    trained = PCNetwork(sizes, **kw).to(device)
    trained.load_state_dict(torch.load(str(Path(run_dir) / (res["run_name"] + ".pt")), map_location=device))
    x, y = data.x_val[:n_eval], data.y_val[:n_eval]
    curves = {"untrained (seed 0 init)": inference_curve(untrained, x, y, steps, state_lr),
              "trained (best epoch)": inference_curve(trained, x, y, steps, state_lr)}
    plot_over_inference(curves, f"{fig_dir}/energy_over_inference.png", steps)
    return curves
