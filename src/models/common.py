"""Shared building blocks for the backprop and Predictive Coding models.

Contents:
- activations and their derivatives (relu, tanh, gelu)
- weight initialization from a seeded generator (so both models can start
  from IDENTICAL weights for a given seed)
- losses and the output-layer "blame" dL/dZ_out used by manual backprop

Conventions (see the Decision record in the project plan):
- loss is averaged over the batch
- MSE carries a factor 1/2:  L = mean_b( 1/2 * ||out - onehot||^2 )
- relu'(0) = 0, the same convention PyTorch uses
"""

import math

import torch
import torch.nn.functional as F

# ----------------------------------------------------------------------------
# Activations and derivatives
# ----------------------------------------------------------------------------


def relu_deriv(z: torch.Tensor) -> torch.Tensor:
    return (z > 0).to(z.dtype)


def tanh_deriv(z: torch.Tensor) -> torch.Tensor:
    return 1.0 - torch.tanh(z) ** 2


def gelu_deriv(z: torch.Tensor) -> torch.Tensor:
    """Derivative of the exact GELU: z * Phi(z)  ->  Phi(z) + z * phi(z)."""
    cdf = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    pdf = torch.exp(-0.5 * z ** 2) / math.sqrt(2.0 * math.pi)
    return cdf + z * pdf


ACTIVATIONS = {"relu": F.relu, "tanh": torch.tanh, "gelu": F.gelu}
ACTIVATION_DERIVS = {"relu": relu_deriv, "tanh": tanh_deriv, "gelu": gelu_deriv}


def get_activation(name: str):
    """Return (function, derivative) for an activation name."""
    if name not in ACTIVATIONS:
        raise ValueError(f"Unknown activation '{name}'. Options: {list(ACTIVATIONS)}")
    return ACTIVATIONS[name], ACTIVATION_DERIVS[name]


# ----------------------------------------------------------------------------
# Initialization
# ----------------------------------------------------------------------------


def make_initial_weights(layer_sizes, init: str = "he", seed: int = 0):
    """Create [(W, b), ...] with W of shape [out, in].

    Uses its own torch.Generator, so it does not touch the global RNG and the
    same (layer_sizes, init, seed) always gives the same weights.
      he     : std = sqrt(2 / fan_in)            (designed for relu-type units)
      xavier : std = sqrt(2 / (fan_in + fan_out)) (designed for tanh-type units)
    Biases start at zero.
    """
    if init not in ("he", "xavier"):
        raise ValueError(f"Unknown init '{init}'. Options: he, xavier")
    g = torch.Generator().manual_seed(seed)
    params = []
    for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        if init == "he":
            std = math.sqrt(2.0 / fan_in)
        else:
            std = math.sqrt(2.0 / (fan_in + fan_out))
        W = torch.randn(fan_out, fan_in, generator=g) * std
        b = torch.zeros(fan_out)
        params.append((W, b))
    return params


# ----------------------------------------------------------------------------
# Losses
# ----------------------------------------------------------------------------


def to_targets(y: torch.Tensor, num_classes: int, dtype) -> torch.Tensor:
    return F.one_hot(y, num_classes).to(dtype)


def loss_fn(out: torch.Tensor, y: torch.Tensor, loss: str) -> torch.Tensor:
    """Batch-mean loss. `out` are raw outputs [B, C]; `y` are class indices [B]."""
    if loss == "mse":
        t = to_targets(y, out.shape[1], out.dtype)
        return 0.5 * ((out - t) ** 2).sum(dim=1).mean()
    if loss == "cross_entropy":
        return F.cross_entropy(out, y)
    raise ValueError(f"Unknown loss '{loss}'. Options: mse, cross_entropy")


def output_blame(out: torch.Tensor, y: torch.Tensor, loss: str) -> torch.Tensor:
    """dL/dZ_out for the batch-mean loss (the output layer is linear).

    mse           : (out - onehot) / B
    cross_entropy : (softmax(out) - onehot) / B
    The 1/B comes from averaging the loss over the batch.
    """
    B = out.shape[0]
    t = to_targets(y, out.shape[1], out.dtype)
    if loss == "mse":
        return (out - t) / B
    if loss == "cross_entropy":
        return (torch.softmax(out, dim=1) - t) / B
    raise ValueError(f"Unknown loss '{loss}'. Options: mse, cross_entropy")


def accuracy(out: torch.Tensor, y: torch.Tensor) -> float:
    return (out.argmax(dim=1) == y).float().mean().item()
