"""Tests for the weak output clamp (output_weight) and the ablation runner."""

import copy

import pytest
import torch

from experiments.ablation import build_cfg, run_ablation, summarize_ablation
from src.models.common import to_targets
from src.models.pc_network import PCNetwork
from src.train_backprop import train_backprop
from src.train_pc import default_run_name, settle, train_pc
from tests.test_train_pc import CFG, make_data

SIZES = [20, 16, 12, 10]


def make_states(init="random"):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(7, 20, generator=g, dtype=torch.float64)
    t = to_targets(torch.randint(0, 10, (7,), generator=g), 10, torch.float64)
    net = PCNetwork(SIZES, activation="tanh", seed=1).double()
    return net, net.init_states(x, t, init=init, generator=torch.Generator().manual_seed(3))


def test_default_output_weight_changes_nothing():
    net, zs = make_states()
    for a, b in zip(net.state_gradients(zs), net.state_gradients(zs, 1.0)):
        assert torch.equal(a, b)
    for (a, c), (b, d) in zip(net.weight_gradients(zs), net.weight_gradients(zs, 1.0)):
        assert torch.equal(a, b) and torch.equal(c, d)


@pytest.mark.parametrize("lam", [0.5, 0.1])
def test_weighted_state_gradient_matches_autograd(lam):
    net, zs = make_states()
    hidden = [z.detach().clone().requires_grad_(True) for z in zs[1:-1]]
    full = [zs[0]] + hidden + [zs[-1]]
    eps, _ = net._layer_terms(full, need_deriv=False)
    energy = sum(0.5 * (e ** 2).sum() * (lam if i == len(eps) - 1 else 1.0)
                 for i, e in enumerate(eps))
    auto = torch.autograd.grad(energy, hidden)
    for m, a in zip(net.state_gradients(zs, lam), auto):
        assert torch.allclose(m, a, atol=1e-10)


@pytest.mark.parametrize("lam", [0.5, 0.1])
def test_weighted_weight_gradient_is_autograd_of_weighted_energy_over_lambda(lam):
    net, zs = make_states()
    full = [z.detach() for z in zs]
    e = net.layer_energies(full)
    weighted = e[:-1].sum() + lam * e[-1]
    params = list(net.weights) + list(net.biases)
    g = torch.autograd.grad(weighted, params)
    n = net.num_layers
    for l, (dW, db) in enumerate(net.weight_gradients(zs, lam)):
        assert torch.allclose(dW, g[l] / lam, atol=1e-10)
        assert torch.allclose(db, g[n + l] / lam, atol=1e-10)


def test_weakly_clamped_inference_lowers_weighted_energy():
    net, _ = make_states()
    g = torch.Generator().manual_seed(5)
    x = torch.randn(16, 20, generator=g, dtype=torch.float64)
    t = to_targets(torch.randint(0, 10, (16,), generator=g), 10, torch.float64)
    lam = 0.1
    w = torch.ones(net.num_layers, dtype=torch.float64)
    w[-1] = lam
    _, e0, e1 = settle(net, x, t, steps=20, state_lr=0.05, output_weight=lam)
    assert (e1 * w).sum() < (e0 * w).sum()


def test_run_name_unchanged_for_standard_pc_and_marked_for_lambda():
    base = default_run_name(CFG, 0)
    assert "lam" not in base
    cfg = copy.deepcopy(CFG)
    cfg["pc"]["output_weight"] = 0.1
    assert default_run_name(cfg, 0) == base.replace("_seed0", "_lam0.1_seed0")


def test_train_pc_with_weak_clamp_runs(tmp_path):
    cfg = copy.deepcopy(CFG)
    cfg["pc"].update({"output_weight": 0.5, "state_lr": 0.05})
    cfg["training"]["max_epochs"] = 2
    res = train_pc(cfg, make_data(), seed=0, results_dir=str(tmp_path), verbose=False)
    assert res["epochs_run"] == 2 and not res["diverged"]


def test_ablation_runner_end_to_end(tmp_path):
    data = make_data()
    settings = {"backprop": {"lr": 0.01}, "pc": {"lr": 0.01, "inference_steps": 3}}
    base_spec = {}
    for model in ("backprop", "pc"):          # fake "main" runs in the expected layout
        cfg = build_cfg(CFG, model, base_spec, settings, max_epochs=2)
        fn = train_pc if model == "pc" else train_backprop
        fn(cfg, data, seed=0, results_dir=str(tmp_path / "main" / model), verbose=False)
    variants = {
        "lam": {"models": ["pc"], "pc": {"output_weight": 0.5}},
        "tanh": {"models": ["backprop", "pc"], "model": {"activation": "tanh", "init": "xavier"}},
        "bs16": {"models": ["backprop"], "training": {"batch_size": 16}, "compare_to": "tanh"},
    }
    df = run_ablation(CFG, data, variants=variants, out_dir=str(tmp_path / "abl"),
                      main_dir=str(tmp_path / "main"), settings=settings, seeds=(0,),
                      max_epochs=2, verbose=False)
    assert set(df.variant) == {"baseline", "lam", "tanh", "bs16"}
    assert len(df) == 2 + 1 + 2 + 1
    table = summarize_ablation(df, variants)
    row = table[(table.variant == "bs16") & (table.model == "backprop")].iloc[0]
    assert row.ref == "tanh"
    again = run_ablation(CFG, data, variants=variants, out_dir=str(tmp_path / "abl"),
                         main_dir=str(tmp_path / "main"), settings=settings, seeds=(0,),
                         max_epochs=2, verbose=False)
    assert len(again) == len(df)               # resumed from saved run files
