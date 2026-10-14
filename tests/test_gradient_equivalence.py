"""Tests for the gradient metrics and the gradient-equivalence experiment."""

import math
import os
from types import SimpleNamespace

import pytest
import torch

from experiments.gradient_equivalence import compare_gradients, run_gradient_equivalence
from src.metrics import cosine_similarity, norm_ratio
from src.models.backprop_mlp import BackpropMLP
from src.models.pc_network import PCNetwork

SIZES = [20, 16, 12, 10]


def make_pair(seed=1, scale=1.0, dtype=torch.float64):
    kw = dict(activation="relu", init="he", seed=seed, last_layer_scale=scale)
    return BackpropMLP(SIZES, **kw).to(dtype), PCNetwork(SIZES, **kw).to(dtype)


def make_batch(dtype=torch.float64, n=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, 20, generator=g, dtype=dtype), torch.randint(0, 10, (n,), generator=g)


@pytest.mark.parametrize("g_pc, cos, ratio", [
    ([3.0, 4.0], 1.0, 1.0),        # identical
    ([0.3, 0.4], 1.0, 0.1),        # right direction, 10x too small
    ([4.0, 3.0], 0.96, 1.0),       # same length, slightly rotated
    ([4.0, -3.0], 0.0, 1.0),       # perpendicular
    ([-3.0, -4.0], -1.0, 1.0),     # opposite
])
def test_metrics_worked_examples(g_pc, cos, ratio):
    g_bp = torch.tensor([3.0, 4.0])
    g = torch.tensor(g_pc)
    assert abs(cosine_similarity(g, g_bp) - cos) < 1e-6
    assert abs(norm_ratio(g, g_bp) - ratio) < 1e-6


def test_metrics_zero_vectors():
    zero, ones = torch.zeros(5), torch.ones(5)
    assert cosine_similarity(zero, ones) == 0.0
    assert norm_ratio(zero, ones) == 0.0
    assert math.isnan(norm_ratio(ones, zero))


def test_step0_output_layer_matches_backprop_and_others_are_zero():
    bp, pc = make_pair()
    x, y = make_batch()
    res = compare_gradients(bp, pc, x, y, steps=[0], state_lr=0.1)[0]
    assert abs(res[-1][0] - 1.0) < 1e-9 and abs(res[-1][1] - 1.0) < 1e-9
    for _, ratio in res[:-1]:
        assert ratio < 1e-9


def test_first_step_gradient_is_exactly_eta_times_backprop():
    """After ONE step, the second-to-last matrix gets exactly eta * the backprop
    gradient (its input states have not moved yet); the input-side matrix gets nothing."""
    eta = 0.05
    bp, pc = make_pair()
    x, y = make_batch()
    res = compare_gradients(bp, pc, x, y, steps=[1], state_lr=eta)[1]
    cos1, ratio1 = res[1]
    assert abs(cos1 - 1.0) < 1e-9
    assert abs(ratio1 - eta) < 1e-9
    assert res[0][1] < 1e-9


def test_error_reaches_the_input_side_matrix_after_two_steps():
    bp, pc = make_pair()
    x, y = make_batch()
    res = compare_gradients(bp, pc, x, y, steps=[2], state_lr=0.05)[2]
    assert res[0][1] > 1e-6


def fake_data(n=64, d=20, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g)
    y = torch.randint(0, 10, (n,), generator=g)
    return SimpleNamespace(x_train=x, y_train=y)


def test_run_saves_resumes_and_matches_theory(tmp_path):
    cfg = {"model": {"hidden_sizes": [16, 12], "activation": "relu", "init": "he"}}
    kwargs = dict(scales=(0.1,), etas=(0.05,), steps=(0, 1, 2, 3), seeds=(0,),
                  batch_ids=(0,), batch_size=16, results_dir=str(tmp_path), verbose=False)
    df = run_gradient_equivalence(cfg, fake_data(), **kwargs)
    assert len(df) == 1 * 1 * 1 * 4 * 3              # settings x seeds x batches x steps x layers

    row = df[(df.steps == 1) & (df.layer == 1)].iloc[0]
    assert abs(row.ratio - 0.05) < 1e-3 and abs(row.cos - 1.0) < 1e-3

    files = list(tmp_path.glob("ge_*.json"))
    assert len(files) == 1
    mtime = os.stat(files[0]).st_mtime_ns
    df2 = run_gradient_equivalence(cfg, fake_data(), **kwargs)
    assert os.stat(files[0]).st_mtime_ns == mtime     # resumed, not recomputed
    assert len(df2) == len(df)
