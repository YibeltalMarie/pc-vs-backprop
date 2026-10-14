"""Train the backprop MLP for one (config, seed) and save a result file.

Gradient source:
- default: autograd (loss.backward())
- use_manual_grads=True: the hand-derived gradients (verified against autograd
  in tests); used to show both give the same training curve.
The test set is only evaluated when evaluate_test=True (final runs only).
Every epoch also records collapse diagnostics (output spread, dead units).

Divergence guard (same idea as the PC trainer): if a batch loss or the validation
loss is non-finite, training stops and the run is saved with diverged=True.
"""

import math
import time

import torch

from src.data import iterate_batches
from src.models.backprop_mlp import BackpropMLP
from src.models.common import loss_fn
from src.trainer_utils import EarlyStopper, diagnostics, evaluate, make_optimizer, run_file
from src.utils import environment_info, load_json, run_done, save_json

NUM_CLASSES = 10


def default_run_name(cfg: dict, seed: int, use_manual_grads: bool = False) -> str:
    m, t = cfg["model"], cfg["training"]
    tag = "manual" if use_manual_grads else "autograd"
    ls = m.get("last_layer_scale", 1.0)
    return (f"backprop_{m['activation']}_{m['loss']}_{t['weight_optimizer']}"
            f"_lr{t['lr']}_bs{t['batch_size']}_ls{ls}_{tag}_seed{seed}")


def train_backprop(
    cfg: dict,
    data,
    seed: int,
    results_dir: str = "results/tables/backprop",
    run_name: str = None,
    use_manual_grads: bool = False,
    evaluate_test: bool = False,
    force: bool = False,
    verbose: bool = True,
) -> dict:
    run_name = run_name or default_run_name(cfg, seed, use_manual_grads)
    path = run_file(results_dir, run_name)
    if run_done(path) and not force:
        result = load_json(path)
        result["resumed"] = True
        if verbose:
            print(f"[skip] {run_name} already done (best val acc {result['best_val_acc']})")
        return result

    m, t = cfg["model"], cfg["training"]
    device = data.x_train.device
    layer_sizes = [data.x_train.shape[1]] + list(m["hidden_sizes"]) + [NUM_CLASSES]

    model = BackpropMLP(layer_sizes, activation=m["activation"], init=m["init"], seed=seed,
                        last_layer_scale=m.get("last_layer_scale", 1.0)).to(device)
    opt = make_optimizer(model.parameters(), t["weight_optimizer"], t["lr"], t.get("momentum", 0.9))
    stopper = EarlyStopper(patience=t["patience"])
    shuffle_gen = torch.Generator().manual_seed(seed)
    loss_name = m["loss"]
    n_train = data.x_train.shape[0]

    history = []
    t_start = time.time()
    stopped_early = False
    diverged = False

    for epoch in range(t["max_epochs"]):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()

        running = 0.0
        for xb, yb in iterate_batches(data.x_train, data.y_train, t["batch_size"],
                                      shuffle=True, generator=shuffle_gen):
            opt.zero_grad()
            if use_manual_grads:
                with torch.no_grad():
                    out, cache = model.forward_cached(xb)
                    batch_loss = loss_fn(out, yb, loss_name)
                    model.assign_gradients(model.manual_gradients(cache, yb, loss_name))
            else:
                out = model(xb)
                batch_loss = loss_fn(out, yb, loss_name)
                batch_loss.backward()
            batch_loss_value = batch_loss.item()
            if not math.isfinite(batch_loss_value):
                diverged = True
                break
            opt.step()
            running += batch_loss_value * xb.shape[0]

        if diverged:
            if verbose:
                print(f"epoch {epoch:3d} | batch loss became non-finite: DIVERGED (lr too large?)")
            break

        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_time = time.time() - t0

        val_loss, val_acc = evaluate(model, data.x_val, data.y_val, loss_name)
        if not math.isfinite(val_loss):
            diverged = True
            if verbose:
                print(f"epoch {epoch:3d} | validation loss became non-finite: DIVERGED")
            break
        diag = diagnostics(model, data.x_val)
        history.append({
            "epoch": epoch,
            "train_loss": running / n_train,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "epoch_time_s": epoch_time,
            **diag,
        })
        if verbose:
            dead = diag["dead_fractions"]
            dead_s = f" | dead {[round(d, 2) for d in dead]}" if dead is not None else ""
            print(f"epoch {epoch:3d} | train loss {running / n_train:.4f} | "
                  f"val loss {val_loss:.4f} | val acc {val_acc:.4f} | "
                  f"spread {diag['output_spread']:.3f}{dead_s} | {epoch_time:.1f}s")

        if stopper.update(val_acc, epoch, model):
            stopped_early = True
            if verbose:
                print(f"early stop at epoch {epoch} (best epoch {stopper.best_epoch})")
            break

    stopper.restore(model)
    best = history[stopper.best_epoch] if history else None

    result = {
        "run_name": run_name,
        "model": "backprop",
        "seed": seed,
        "grad_source": "manual" if use_manual_grads else "autograd",
        "config": cfg,
        "layer_sizes": layer_sizes,
        "n_parameters": sum(p.numel() for p in model.parameters()),
        "history": history,
        "best_epoch": stopper.best_epoch if history else None,
        "best_val_acc": best["val_acc"] if best else None,
        "best_val_loss": best["val_loss"] if best else None,
        "epochs_run": len(history),
        "stopped_early": stopped_early,
        "diverged": diverged,
        "total_train_time_s": time.time() - t_start,
        "mean_epoch_time_s": (sum(h["epoch_time_s"] for h in history) / len(history)) if history else None,
        "environment": environment_info(),
        "resumed": False,
    }
    if evaluate_test and history:
        test_loss, test_acc = evaluate(model, data.x_test, data.y_test, loss_name)
        result["test_loss"], result["test_acc"] = test_loss, test_acc

    save_json(result, path)
    return result
