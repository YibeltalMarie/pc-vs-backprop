"""Predictive Coding network (MSE energy).

Layout. Layers l = 0 .. L-1 have weights W_l (shape [out, in]) and biases b_l.
States are z_0 = x (clamped), z_1 .. z_{L-1} (hidden, free), z_L = target
(clamped during training). Each layer predicts the next one:

    S_l      = z_l @ W_l^T + b_l
    mu_{l+1} = f(S_l)            (l < L-1)       mu_L = S_{L-1}  (linear output)
    eps_{l+1} = z_{l+1} - mu_{l+1}
    E_{l+1}  = 1/2 * ||eps_{l+1}||^2             total energy = sum of layer energies

In the code, eps[l] and fprime[l] hold the error and slope of layer l's
prediction (i.e. of z_{l+1}).

Averaging conventions (important for comparing with backprop):
- STATES use the per-sample energy, summed over the batch with NO 1/B. Each
  sample's states only affect that sample's energy, so the update of one
  sample must not depend on the batch size.
- WEIGHTS use the batch MEAN of the energy (1/B), the same convention as the
  backprop loss. With these two choices, at equilibrium PC's weight gradient
  has the same scale as backprop's.

Updates:
    state:   dE/dz_k = eps_k - (eps_{k+1} * f'(S_k)) @ W_k           (k = 1..L-1)
             z_k <- z_k - state_lr * dE/dz_k
    weights: dE/dW_l = -(eps_{l+1} * f'(S_l))^T @ z_l / B
             dE/db_l = -sum_b(eps_{l+1} * f'(S_l)) / B

Only the MSE energy is implemented. Cross-entropy variants will be added after
the literature check.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.common import get_activation, make_initial_weights


class PCNetwork(nn.Module):
    def __init__(self, layer_sizes, activation: str = "relu", init: str = "he",
                 seed: int = 0, last_layer_scale: float = 1.0, loss: str = "mse"):
        super().__init__()
        if loss != "mse":
            raise NotImplementedError(
                "Only the MSE energy is implemented. Cross-entropy variants will be "
                "added after the literature check.")
        self.layer_sizes = list(layer_sizes)
        self.activation_name = activation
        self.act, self.act_deriv = get_activation(activation)

        # Same initializer as BackpropMLP: identical weights for the same seed.
        initial = make_initial_weights(self.layer_sizes, init=init, seed=seed,
                                       last_layer_scale=last_layer_scale)
        self.weights = nn.ParameterList([nn.Parameter(W) for W, _ in initial])
        self.biases = nn.ParameterList([nn.Parameter(b) for _, b in initial])

    @property
    def num_layers(self) -> int:
        return len(self.weights)

    # ------------------------------------------------------- test-time prediction
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Feedforward pass = prediction at test time with forward initialization
        (every state starts at its own prediction, all errors are 0)."""
        a = x
        for l in range(self.num_layers):
            z = F.linear(a, self.weights[l], self.biases[l])
            a = self.act(z) if l < self.num_layers - 1 else z
        return a

    # ------------------------------------------------------------- state set-up
    @torch.no_grad()
    def init_states(self, x, target, init: str = "forward", generator=None, random_std: float = 0.1):
        """Return [z_0, ..., z_L] with z_0 = x and z_L = target (both clamped).

        init = forward : hidden states equal their own predictions
               zeros   : all hidden states 0
               random  : hidden states ~ N(0, random_std^2), seeded by `generator`
        """
        if init not in ("forward", "zeros", "random"):
            raise ValueError(f"Unknown state init '{init}'. Options: forward, zeros, random")
        B = x.shape[0]
        zs = [x]
        a = x
        for l in range(self.num_layers - 1):
            if init == "forward":
                a = self.act(F.linear(a, self.weights[l], self.biases[l]))
                hidden = a
            elif init == "zeros":
                hidden = torch.zeros(B, self.layer_sizes[l + 1], dtype=x.dtype, device=x.device)
            else:
                g = generator if generator is not None else torch.Generator().manual_seed(0)
                hidden = (torch.randn(B, self.layer_sizes[l + 1], generator=g, dtype=x.dtype)
                          * random_std).to(x.device)
            zs.append(hidden)
        zs.append(target)
        return zs

    # ----------------------------------------------------- errors and energies
    def _layer_terms(self, zs, need_deriv: bool = True):
        """Return (eps, fprime): lists of length L, entry l belongs to the prediction of z_{l+1}."""
        eps, fprime = [], []
        for l in range(self.num_layers):
            S = F.linear(zs[l], self.weights[l], self.biases[l])
            if l < self.num_layers - 1:
                mu = self.act(S)
                fp = self.act_deriv(S) if need_deriv else None
            else:
                mu = S
                fp = torch.ones_like(S) if need_deriv else None
            eps.append(zs[l + 1] - mu)
            fprime.append(fp)
        return eps, fprime

    def layer_energies(self, zs) -> torch.Tensor:
        """Per-layer energies E_1..E_L, each the batch mean of 1/2 * ||eps||^2."""
        eps, _ = self._layer_terms(zs, need_deriv=False)
        return torch.stack([0.5 * (e ** 2).sum(dim=1).mean() for e in eps])

    def total_energy(self, zs) -> torch.Tensor:
        return self.layer_energies(zs).sum()

    # ------------------------------------------------------ state update rules
    @torch.no_grad()
    def state_gradients(self, zs, output_weight: float = 1.0):
        """Hand-derived dE/dz_k for the hidden states k = 1..L-1 (per-sample, no 1/B)."""
        eps, fprime = self._layer_terms(zs)
        grads = []
        for k in range(1, self.num_layers):
            own = eps[k - 1]
            w = output_weight if k == self.num_layers - 1 else 1.0
            from_above = w * ((eps[k] * fprime[k]) @ self.weights[k])
            grads.append(own - from_above)
        return grads

    def state_gradients_autograd(self, zs):
        """The same gradients computed by autograd (used to verify the formulas)."""
        hidden = [z.detach().clone().requires_grad_(True) for z in zs[1:-1]]
        full = [zs[0].detach()] + hidden + [zs[-1].detach()]
        with torch.enable_grad():
            eps, _ = self._layer_terms(full, need_deriv=False)
            energy = sum(0.5 * (e ** 2).sum() for e in eps)      # per-sample sum, NO 1/B
            grads = torch.autograd.grad(energy, hidden)
        return [g.detach() for g in grads]

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def infer(self, x, target, steps: int, state_lr: float, init: str = "forward",
              generator=None, record: bool = False, use_autograd: bool = False):
        """Run `steps` inference steps with the weights frozen.

        Returns (zs, history). history is None unless record=True, in which case it
        is a list with steps+1 entries (step 0 = right after initialization), each a
        list of the L layer energies.
        """
        zs = self.init_states(x, target, init=init, generator=generator)
        hist = [self.layer_energies(zs)] if record else None
        for _ in range(steps):
            grads = self.state_gradients_autograd(zs) if use_autograd else self.state_gradients(zs)
            for k, g in enumerate(grads, start=1):
                zs[k] = zs[k] - state_lr * g
            if record:
                hist.append(self.layer_energies(zs))
        history = torch.stack(hist).tolist() if record else None
        return zs, history

    # --------------------------------------------------------------- weight update
    @torch.no_grad()
    def weight_gradients(self, zs, output_weight: float = 1.0):
        """Hand-derived dE/dW, dE/db from the (settled) states; batch-mean convention."""
        B = zs[0].shape[0]
        eps, fprime = self._layer_terms(zs)
        grads = []
        for l in range(self.num_layers):
            delta = eps[l] * fprime[l]
            if l == self.num_layers - 1:
                delta = delta * output_weight
            dW = -(delta.T @ zs[l]) / B / output_weight
            db = -delta.sum(dim=0) / B / output_weight
            grads.append((dW, db))
        return grads

    def weight_gradients_autograd(self, zs):
        """The same gradients computed by autograd (states detached)."""
        full = [z.detach() for z in zs]
        params = list(self.weights) + list(self.biases)
        with torch.enable_grad():
            energy = self.layer_energies(full).sum()             # batch-mean convention
            g = torch.autograd.grad(energy, params)
        n = self.num_layers
        return [(g[l].detach(), g[n + l].detach()) for l in range(n)]

    @torch.no_grad()
    def assign_gradients(self, grads) -> None:
        """Put weight gradients into .grad so any torch optimizer can use them."""
        for l, (dW, db) in enumerate(grads):
            self.weights[l].grad = dW.clone()
            self.biases[l].grad = db.clone()
