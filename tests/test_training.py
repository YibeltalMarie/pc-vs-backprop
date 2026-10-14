"""Tests for the training machinery (synthetic data, fast)."""

import copy
from types import SimpleNamespace

import pytest
import torch

from src.train_backprop import train_backprop
from src.trainer_utils import EarlyStopper

BASE_CFG = {
    "model": {"hidden_sizes": [16, 8], "activation": "relu", "init": "he", "loss": "mse"},
    "training": {"batch_size": 64, "max_epochs": 15, "patience": 5,
                 "weight_optimizer": "sgd_momentum", "lr": 0.05, "momentum": 0.9},
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
    return SimpleNamespace(x_train=x_tr, y_train=y_tr, x_val=x_va, y_val=y_va,
                           x_test=x_te, y_test=y_te)


def test_training_learns(tmp_path):
    res = train_backprop(BASE_CFG, make_data(), seed=0, results_dir=str(tmp_path), verbose=False)
    assert res["best_val_acc"] > 0.5            # chance is 0.1
    assert res["history"][-1]["train_loss"] < res["history"][0]["train_loss"]


def test_manual_gradient_training_matches_autograd(tmp_path):
    cfg = copy.deepcopy(BASE_CFG)
    cfg["training"].update({"weight_optimizer": "sgd", "max_epochs": 3, "patience": 100})
    data = make_data()
    a = train_backprop(cfg, data, seed=1, results_dir=str(tmp_path / "a"), verbose=False)
    m = train_backprop(cfg, data, seed=1, results_dir=str(tmp_path / "m"),
                       use_manual_grads=True, verbose=False)
    for ha, hm in zip(a["history"], m["history"]):
        assert abs(ha["train_loss"] - hm["train_loss"]) < 1e-4
        assert abs(ha["val_loss"] - hm["val_loss"]) < 1e-4


def test_runs_are_resumable(tmp_path):
    data = make_data()
    first = train_backprop(BASE_CFG, data, seed=2, results_dir=str(tmp_path), verbose=False)
    second = train_backprop(BASE_CFG, data, seed=2, results_dir=str(tmp_path), verbose=False)
    assert first["resumed"] is False and second["resumed"] is True
    assert first["best_val_acc"] == second["best_val_acc"]


def test_same_seed_is_reproducible(tmp_path):
    data = make_data()
    a = train_backprop(BASE_CFG, data, seed=3, results_dir=str(tmp_path / "a"), verbose=False)
    b = train_backprop(BASE_CFG, data, seed=3, results_dir=str(tmp_path / "b"), verbose=False)
    assert [h["val_acc"] for h in a["history"]] == [h["val_acc"] for h in b["history"]]


def test_early_stopper_stops_and_restores_best():
    model = torch.nn.Linear(2, 2)
    stopper = EarlyStopper(patience=2)
    assert stopper.update(0.5, 0, model) is False
    best_weights = model.weight.detach().clone()
    with torch.no_grad():
        model.weight.add_(1.0)                       # model changes after the best epoch
    assert stopper.update(0.4, 1, model) is False    # 1 bad epoch
    assert stopper.update(0.3, 2, model) is True     # 2 bad epochs -> stop
    stopper.restore(model)
    assert torch.equal(model.weight.detach(), best_weights)
    assert stopper.best_epoch == 0
