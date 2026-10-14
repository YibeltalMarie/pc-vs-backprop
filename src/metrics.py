"""Metrics for comparing two gradients (or any two tensors of the same shape).

cosine_similarity : do they point the same way?   1 = same direction, 0 = unrelated,
                    -1 = opposite. Returns 0.0 if either vector is (numerically) zero,
                    meaning "no alignment"; the norm ratio then shows the size problem.
norm_ratio        : is PC's gradient the same size as backprop's?  ||a|| / ||b||.
                    Returns NaN if b is (numerically) zero (ratio undefined).
Both flatten the tensors and compute in float64.
"""

import torch


def _flat(a: torch.Tensor) -> torch.Tensor:
    return a.detach().reshape(-1).double()


def cosine_similarity(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-30) -> float:
    a, b = _flat(a), _flat(b)
    na, nb = a.norm().item(), b.norm().item()
    if na < eps or nb < eps:
        return 0.0
    return float((a @ b).item() / (na * nb))


def norm_ratio(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-30) -> float:
    a, b = _flat(a), _flat(b)
    nb = b.norm().item()
    if nb < eps:
        return float("nan")
    return float(a.norm().item() / nb)
