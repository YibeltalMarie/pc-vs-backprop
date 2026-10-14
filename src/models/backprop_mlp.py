"""Standard backpropagation MLP, with BOTH autograd and hand-derived gradients.

Layers l = 0 .. L-1:   Z_l = A_l @ W_l^T + b_l ,   A_{l+1} = f(Z_l)   (l < L-1)
The last layer is linear: output = Z_{L-1}. A_0 is the input batch.

Manual backward (batch version of the formulas in the project plan):
    dZ_{L-1} = output_blame(out, y)                (includes the 1/B factor)
    dW_l     = dZ_l^T @ A_l
    db_l     = sum over batch of dZ_l
    dA_l     = dZ_l @ W_l
    dZ_{l-1} = dA_l * f'(Z_{l-1})
The tests check these against torch.autograd.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.common import (
    get_activation,
    loss_fn,
    make_initial_weights,
    output_blame,
)


class BackpropMLP(nn.Module):
    def __init__(self, layer_sizes, activation: str = "relu", init: str = "he", seed: int = 0):
        super().__init__()
        self.layer_sizes = list(layer_sizes)
        self.activation_name = activation
        self.act, self.act_deriv = get_activation(activation)

        initial = make_initial_weights(self.layer_sizes, init=init, seed=seed)
        self.weights = nn.ParameterList([nn.Parameter(W) for W, _ in initial])
        self.biases = nn.ParameterList([nn.Parameter(b) for _, b in initial])

    @property
    def num_layers(self) -> int:
        return len(self.weights)

    # ------------------------------------------------------------------ forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = x
        for l in range(self.num_layers):
            z = F.linear(a, self.weights[l], self.biases[l])
            a = self.act(z) if l < self.num_layers - 1 else z
        return a

    def forward_cached(self, x: torch.Tensor):
        """Forward pass that also stores what manual backprop needs.

        cache["a"][l] is the input to layer l (cache["a"][0] = x);
        cache["z"][l] is the pre-activation of layer l.
        """
        a_list, z_list = [x], []
        a = x
        for l in range(self.num_layers):
            z = F.linear(a, self.weights[l], self.biases[l])
            z_list.append(z)
            a = self.act(z) if l < self.num_layers - 1 else z
            if l < self.num_layers - 1:
                a_list.append(a)
        return a, {"a": a_list, "z": z_list}

    # ----------------------------------------------------------------- backward
    @torch.no_grad()
    def manual_gradients(self, cache, y: torch.Tensor, loss: str):
        """Hand-derived gradients. Returns a list of (dW, db), one per layer."""
        out = cache["z"][-1]
        dZ = output_blame(out, y, loss)
        grads = [None] * self.num_layers
        for l in reversed(range(self.num_layers)):
            a_in = cache["a"][l]
            dW = dZ.T @ a_in
            db = dZ.sum(dim=0)
            grads[l] = (dW, db)
            if l > 0:
                dA = dZ @ self.weights[l]
                dZ = dA * self.act_deriv(cache["z"][l - 1])
        return grads

    @torch.no_grad()
    def assign_gradients(self, grads) -> None:
        """Put manual gradients into .grad so any torch optimizer can use them."""
        for l, (dW, db) in enumerate(grads):
            self.weights[l].grad = dW.clone()
            self.biases[l].grad = db.clone()

    # --------------------------------------------------------------------- loss
    def loss(self, out: torch.Tensor, y: torch.Tensor, loss: str) -> torch.Tensor:
        return loss_fn(out, y, loss)
