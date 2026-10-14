"""Verification: hand-derived backprop gradients must equal torch.autograd.

Run from the repo root:  python -m pytest tests -q
Float64 is used so agreement is checked to ~1e-10, not just 'close enough'.
"""

import pytest
import torch

from src.models.backprop_mlp import BackpropMLP
from src.models.common import (
    ACTIVATIONS,
    ACTIVATION_DERIVS,
    loss_fn,
    make_initial_weights,
    output_blame,
)


@pytest.mark.parametrize("activation", ["relu", "tanh", "gelu"])
def test_activation_derivative_matches_autograd(activation):
    z = torch.randn(1000, dtype=torch.float64, requires_grad=True)
    ACTIVATIONS[activation](z).sum().backward()
    manual = ACTIVATION_DERIVS[activation](z.detach())
    assert torch.allclose(z.grad, manual, atol=1e-10)


@pytest.mark.parametrize("activation", ["relu", "tanh", "gelu"])
@pytest.mark.parametrize("loss", ["mse", "cross_entropy"])
def test_manual_gradients_match_autograd(activation, loss):
    torch.manual_seed(0)
    model = BackpropMLP([20, 16, 12, 8, 10], activation=activation, init="he", seed=1).double()
    x = torch.randn(7, 20, dtype=torch.float64)
    y = torch.randint(0, 10, (7,))

    out, cache = model.forward_cached(x)
    loss_fn(out, y, loss).backward()
    manual = model.manual_gradients(cache, y, loss)

    for l, (dW, db) in enumerate(manual):
        assert torch.allclose(dW, model.weights[l].grad, atol=1e-10, rtol=1e-8), f"dW mismatch, layer {l}"
        assert torch.allclose(db, model.biases[l].grad, atol=1e-10, rtol=1e-8), f"db mismatch, layer {l}"


def test_slide_output_values():
    """Output-layer numbers from the mentor's example: Yhat = 0.2283, y = 1.0."""
    out = torch.tensor([[0.2283]])
    y = torch.tensor([0])  # with one output unit, class 0 -> target 1.0
    assert abs(loss_fn(out, y, "mse").item() - 0.2977) < 1e-3
    assert abs(output_blame(out, y, "mse").item() - (-0.7717)) < 1e-4


def test_same_seed_gives_identical_weights():
    a = make_initial_weights([10, 8, 4], init="he", seed=3)
    b = make_initial_weights([10, 8, 4], init="he", seed=3)
    c = make_initial_weights([10, 8, 4], init="he", seed=4)
    assert all(torch.equal(Wa, Wb) for (Wa, _), (Wb, _) in zip(a, b))
    assert not torch.equal(a[0][0], c[0][0])


def test_forward_and_forward_cached_agree():
    model = BackpropMLP([6, 5, 3], activation="tanh", init="xavier", seed=0)
    x = torch.randn(4, 6)
    out, _ = model.forward_cached(x)
    assert torch.allclose(out, model(x))
