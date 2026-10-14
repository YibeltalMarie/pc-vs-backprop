"""Tests for the PC trainer (synthetic data, fast)."""

import copy
import math
from types import SimpleNamespace

import pytest
import torch

from src.models.backprop_mlp import BackpropMLP
from src.models.common import to_targets
from src.models.pc_network import PCNetwork
from src.train_pc import settle, train_pc
from src.trainer_utils import diagnostics, relu_dead_fractions

CFG = {
    "model": {"hidden_sizes": [16, 8], "activation": "relu", "init": "he",
              "loss": "mse", "last_layer_scale": 0.1},
    "training": {"batch_size": 64, "max_epochs": 15, "patience": 5,
                 "weight_optimizer": "sgd_momentum", "lr": 0.01, "momentum": 0.9},
    "pc": {"inference_steps": 10, "state_lr": 0.1, "state_init": "forward"},
}


def make_data(n_train=600, n_val=200, d=20, k=10, seed=0):
    g = torch.Generator().manual_seed(seed)
    centers = torch.randn(k, d, generator=g) * 2.0

    def make(n):
        y = torch.randint(0, k, (n,), generator=g)
        x = centers[y] + 0.5 * torch.randn(n, d, generator=g)
        return x, y

    x_tr, y_tr = make(n_train)
    x_va, y_va = make(n_val)
    x_te, y_te = make(n_val)
    mu, sd = x_tr.mean(), x_tr.std()
    return SimpleNamespace(x_train=(x_tr - mu) / sd, y_train=y_tr,
                           x_val=(x_va - mu) / sd, y_val=y_va,
                           x_test=(x_te - mu) / sd, y_test=y_te)


def test_pc_training_learns_and_does_not_collapse(tmp_path):
    res = train_pc(CFG, make_data(), seed=0, results_dir=str(tmp_path), verbose=False)
    last = res["history"][-1]
    assert res["best_val_acc"] > 0.5                              # chance is 0.1
    assert sum(last["train_energy_end"]) < sum(last["train_energy_start"])   # inference lowers energy
    assert last["output_spread"] > 0.05                           # not a constant predictor
    assert max(last["dead_fractions"]) < 1.0


def test_settle_matches_net_infer():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(8, 20, generator=g, dtype=torch.float64)
    t = to_targets(torch.randint(0, 10, (8,), generator=g), 10, torch.float64)
    net = PCNetwork([20, 16, 12, 10], activation="tanh", seed=1).double()
    zs_a, e_start, e_end = settle(net, x, t, steps=5, state_lr=0.05)
    zs_b, _ = net.infer(x, t, steps=5, state_lr=0.05)
    for a, b in zip(zs_a, zs_b):
        assert torch.allclose(a, b, atol=1e-12)
    assert e_start[:-1].abs().max() < 1e-12            # hidden energies are 0 at step 0
    assert e_end.sum() < e_start.sum()


def test_runs_are_resumable(tmp_path):
    data = make_data()
    first = train_pc(CFG, data, seed=2, results_dir=str(tmp_path), verbose=False)
    second = train_pc(CFG, data, seed=2, results_dir=str(tmp_path), verbose=False)
    assert first["resumed"] is False and second["resumed"] is True
    assert first["best_val_acc"] == second["best_val_acc"]


def test_same_seed_is_reproducible(tmp_path):
    data = make_data()
    a = train_pc(CFG, data, seed=3, results_dir=str(tmp_path / "a"), verbose=False)
    b = train_pc(CFG, data, seed=3, results_dir=str(tmp_path / "b"), verbose=False)
    assert [h["val_acc"] for h in a["history"]] == [h["val_acc"] for h in b["history"]]


@pytest.mark.parametrize("state_init", ["forward", "zeros", "random"])
def test_state_init_options_train_without_nan(tmp_path, state_init):
    cfg = copy.deepcopy(CFG)
    cfg["pc"]["state_init"] = state_init
    cfg["training"]["max_epochs"] = 2
    res = train_pc(cfg, make_data(), seed=0, results_dir=str(tmp_path), verbose=False)
    assert res["epochs_run"] == 2
    assert all(math.isfinite(h["val_loss"]) and math.isfinite(h["train_loss"]) for h in res["history"])


def test_warns_when_inference_steps_fewer_than_hidden_layers(tmp_path):
    cfg = copy.deepcopy(CFG)
    cfg["pc"]["inference_steps"] = 1               # 2 hidden layers need at least 2 steps
    cfg["training"]["max_epochs"] = 1
    with pytest.warns(UserWarning, match="inference_steps"):
        train_pc(cfg, make_data(), seed=0, results_dir=str(tmp_path), verbose=False)


def test_generic_dead_fraction_matches_backprop_method_and_works_for_pc():
    x = torch.randn(64, 20)
    bp = BackpropMLP([20, 16, 8, 10], activation="relu", seed=0)
    pc = PCNetwork([20, 16, 8, 10], activation="relu", seed=0)
    assert relu_dead_fractions(pc, x) == bp.dead_fractions(x)
    assert diagnostics(pc, x)["dead_fractions"] is not None
    assert relu_dead_fractions(PCNetwork([20, 16, 8, 10], activation="tanh", seed=0), x) is None
