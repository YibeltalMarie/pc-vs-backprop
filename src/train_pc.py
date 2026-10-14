"""Train the Predictive Coding network for one (config, seed) and save a result file.

One training step per batch:
  1. clamp input to x and output to the one-hot target
  2. initialize hidden states (cfg pc.state_init)
  3. record layer energies at inference step 0
  4. run pc.inference_steps state updates with the weights frozen
  5. record layer energies after inference
  6. compute the weight gradient from the settled states and take one optimizer step

Validation/test prediction is the FEEDFORWARD pass: with the output free, the energy
(a sum of squares) is minimal (zero) when every state equals its own prediction.
MSE energy only; cross-entropy raises NotImplementedError (see PCNetwork).

Stability guard: for a small enough state learning rate, inference is gradient descent
on the energy, so the energy must not rise. Each epoch logs the fraction of batches
where it did (inference_energy_increase_frac). If the energy becomes non-finite the
run stops and is saved with diverged=True.
"""

import math
import time
import warnings

import torch

from src.data import iterate_batches
from src.models.common import to_targets
from src.models.pc_network import PCNetwork
from src.trainer_utils import EarlyStopper, diagnostics, evaluate, make_optimizer, run_file
from src.utils import environment_info, load_json, run_done, save_json

NUM_CLASSES = 10


def default_run_name(cfg: dict, seed: int) -> str:
    m, t, p = cfg["model"], cfg["training"], cfg["pc"]
    ls = m.get("last_layer_scale", 1.0)
    return (f"pc_{m['activation']}_{m['loss']}_{t['weight_optimizer']}"
            f"_lr{t['lr']}_bs{t['batch_size']}_ls{ls}"
            f"_T{p['inference_steps']}_eta{p['state_lr']}_{p['state_init']}_seed{seed}")


@torch.no_grad()
def settle(net, x, targets, steps, state_lr, init="forward", generator=None):
    """Initialize states, run `steps` state updates, return (states, energies_start, energies_end).

    Energies are per-layer batch means (tensors of length num_layers).
    """
    zs = net.init_states(x, targets, init=init, generator=generator)
    e_start = net.layer_energies(zs)
    for _ in range(steps):
        grads = net.state_gradients(zs)
        for k, g in enumerate(grads, start=1):
            zs[k] = zs[k] - state_lr * g
    e_end = net.layer_energies(zs)
    return zs, e_start, e_end


def train_pc(
    cfg: dict,
    data,
    seed: int,
    results_dir: str = "results/tables/pc",
    run_name: str = None,
    evaluate_test: bool = False,
    force: bool = False,
    verbose: bool = True,
) -> dict:
    run_name = run_name or default_run_name(cfg, seed)
    path = run_file(results_dir, run_name)
    if run_done(path) and not force:
        result = load_json(path)
        result["resumed"] = True
        if verbose:
            print(f"[skip] {run_name} already done (best val acc {result['best_val_acc']})")
        return result

    m, t, p = cfg["model"], cfg["training"], cfg["pc"]
    n_hidden = len(m["hidden_sizes"])
    if p["inference_steps"] < n_hidden:
        warnings.warn(
            f"inference_steps={p['inference_steps']} is smaller than the number of hidden layers "
            f"({n_hidden}): the weights next to the input receive exactly zero gradient.",
            UserWarning)

    device = data.x_train.device
    layer_sizes = [data.x_train.shape[1]] + list(m["hidden_sizes"]) + [NUM_CLASSES]

    net = PCNetwork(layer_sizes, activation=m["activation"], init=m["init"], seed=seed,
                    last_layer_scale=m.get("last_layer_scale", 1.0), loss=m["loss"]).to(device)
    opt = make_optimizer(net.parameters(), t["weight_optimizer"], t["lr"], t.get("momentum", 0.9))
    stopper = EarlyStopper(patience=t["patience"])
    shuffle_gen = torch.Generator().manual_seed(seed)
    state_gen = torch.Generator().manual_seed(seed + 1000)    # only used by state_init="random"
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

        e_start_sum = torch.zeros(net.num_layers)
        e_end_sum = torch.zeros(net.num_layers)
        n_batches, n_increase = 0, 0
        for xb, yb in iterate_batches(data.x_train, data.y_train, t["batch_size"],
                                      shuffle=True, generator=shuffle_gen):
            targets = to_targets(yb, NUM_CLASSES, xb.dtype)
            opt.zero_grad()
            zs, e_start, e_end = settle(net, xb, targets, p["inference_steps"], p["state_lr"],
                                        init=p["state_init"], generator=state_gen)
            total_end = e_end.sum().item()
            if not math.isfinite(total_end):
                diverged = True
                break
            n_batches += 1
            n_increase += int(total_end > e_start.sum().item() * (1.0 + 1e-9))
            net.assign_gradients(net.weight_gradients(zs))
            opt.step()
            e_start_sum += e_start.cpu() * xb.shape[0]
            e_end_sum += e_end.cpu() * xb.shape[0]

        if diverged:
            if verbose:
                print(f"epoch {epoch:3d} | energy became non-finite: DIVERGED (state_lr too large?)")
            break

        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_time = time.time() - t0

        e_start_mean = (e_start_sum / n_train).tolist()
        e_end_mean = (e_end_sum / n_train).tolist()
        increase_frac = n_increase / max(n_batches, 1)
        val_loss, val_acc = evaluate(net, data.x_val, data.y_val, loss_name)
        diag = diagnostics(net, data.x_val)
        history.append({
            "epoch": epoch,
            "train_loss": sum(e_start_mean),            # total energy at step 0
            "train_energy_start": e_start_mean,         # per layer, step 0
            "train_energy_end": e_end_mean,             # per layer, after inference
            "inference_energy_increase_frac": increase_frac,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "epoch_time_s": epoch_time,
            **diag,
        })
        if verbose:
            dead = diag["dead_fractions"]
            dead_s = f" | dead {[round(d, 2) for d in dead]}" if dead is not None else ""
            print(f"epoch {epoch:3d} | energy {sum(e_start_mean):.4f} -> {sum(e_end_mean):.4f} | "
                  f"energy rose in {increase_frac:.0%} of batches | "
                  f"val loss {val_loss:.4f} | val acc {val_acc:.4f} | "
                  f"spread {diag['output_spread']:.3f}{dead_s} | {epoch_time:.1f}s")

        if stopper.update(val_acc, epoch, net):
            stopped_early = True
            if verbose:
                print(f"early stop at epoch {epoch} (best epoch {stopper.best_epoch})")
            break

    stopper.restore(net)
    best = history[stopper.best_epoch] if history else None

    result = {
        "run_name": run_name,
        "model": "pc",
        "seed": seed,
        "config": cfg,
        "layer_sizes": layer_sizes,
        "n_parameters": sum(q.numel() for q in net.parameters()),
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
        test_loss, test_acc = evaluate(net, data.x_test, data.y_test, loss_name)
        result["test_loss"], result["test_acc"] = test_loss, test_acc

    save_json(result, path)
    return result
