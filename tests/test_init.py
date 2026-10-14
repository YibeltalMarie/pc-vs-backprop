"""Initialization tests, including the measured large-initial-loss finding.

Finding (Oct 2026): with He init on a linear output layer and 0/1 targets,
the initial MSE is far above the constant-predictor loss (0.45 for 10 classes).
Scaling only the last layer by 0.1 brings it close. Both models share this
initialization, and PC's step-0 output error equals this initial error.
"""

import torch

from src.models.backprop_mlp import BackpropMLP
from src.models.common import loss_fn, make_initial_weights

SIZES = [784, 512, 256, 128, 10]


def _mean_initial_loss(scale, seeds=(0, 1, 2)):
    g = torch.Generator().manual_seed(123)
    x = torch.randn(256, 784, generator=g)               # standardized-like inputs
    y = torch.randint(0, 10, (256,), generator=g)
    vals = []
    for s in seeds:
        m = BackpropMLP(SIZES, activation="relu", init="he", seed=s, last_layer_scale=scale)
        with torch.no_grad():
            vals.append(loss_fn(m(x), y, "mse").item())
    return sum(vals) / len(vals)


def test_full_he_output_layer_starts_with_large_loss():
    assert _mean_initial_loss(1.0) > 3.0


def test_scaled_output_layer_starts_near_constant_predictor_loss():
    assert _mean_initial_loss(0.1) < 1.0


def test_last_layer_scale_only_changes_last_layer():
    a = make_initial_weights(SIZES, init="he", seed=5, last_layer_scale=1.0)
    b = make_initial_weights(SIZES, init="he", seed=5, last_layer_scale=0.1)
    for (Wa, _), (Wb, _) in zip(a[:-1], b[:-1]):
        assert torch.equal(Wa, Wb)
    assert torch.allclose(a[-1][0] * 0.1, b[-1][0])


def test_dead_fractions_shape_and_non_relu():
    x = torch.randn(32, 20)
    relu_model = BackpropMLP([20, 16, 8, 10], activation="relu", seed=0)
    fr = relu_model.dead_fractions(x)
    assert len(fr) == 2 and all(0.0 <= f <= 1.0 for f in fr)
    assert BackpropMLP([20, 16, 8, 10], activation="tanh", seed=0).dead_fractions(x) is None
