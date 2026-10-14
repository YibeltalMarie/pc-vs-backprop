"""Tests for the Predictive Coding network (float64, small networks)."""

import pytest
import torch

from src.models.backprop_mlp import BackpropMLP
from src.models.common import loss_fn, to_targets
from src.models.pc_network import PCNetwork

SIZES = [20, 16, 12, 10]
ACTS = ["relu", "tanh", "gelu"]


def make_batch(n=7, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 20, generator=g, dtype=torch.float64)
    y = torch.randint(0, 10, (n,), generator=g)
    t = to_targets(y, 10, torch.float64)
    return x, y, t


def make_pc(activation="relu", seed=1, scale=1.0, sizes=SIZES):
    return PCNetwork(sizes, activation=activation, init="he", seed=seed,
                     last_layer_scale=scale).double()


def make_bp(activation="relu", seed=1, scale=1.0, sizes=SIZES):
    return BackpropMLP(sizes, activation=activation, init="he", seed=seed,
                       last_layer_scale=scale).double()


def test_initial_weights_identical_to_backprop():
    bp = BackpropMLP(SIZES, activation="relu", init="he", seed=7, last_layer_scale=0.1)
    pc = PCNetwork(SIZES, activation="relu", init="he", seed=7, last_layer_scale=0.1)
    for l in range(len(SIZES) - 1):
        assert torch.equal(bp.weights[l], pc.weights[l])
        assert torch.equal(bp.biases[l], pc.biases[l])


def test_forward_init_has_zero_hidden_errors_and_matches_backprop_loss():
    x, y, t = make_batch()
    pc, bp = make_pc(), make_bp()
    zs = pc.init_states(x, t, init="forward")
    eps, _ = pc._layer_terms(zs, need_deriv=False)
    for e in eps[:-1]:
        assert e.abs().max() < 1e-12
    assert torch.allclose(pc.layer_energies(zs).sum(), loss_fn(bp(x), y, "mse"))


@pytest.mark.parametrize("activation", ACTS)
@pytest.mark.parametrize("init", ["forward", "random"])
def test_state_gradients_manual_vs_autograd(activation, init):
    x, y, t = make_batch()
    net = make_pc(activation)
    zs = net.init_states(x, t, init=init, generator=torch.Generator().manual_seed(3))
    manual = net.state_gradients(zs)
    auto = net.state_gradients_autograd(zs)
    assert len(manual) == len(auto) == 2
    for m, a in zip(manual, auto):
        assert torch.allclose(m, a, atol=1e-10)


@pytest.mark.parametrize("activation", ACTS)
def test_weight_gradients_manual_vs_autograd(activation):
    x, y, t = make_batch()
    net = make_pc(activation)
    zs = net.init_states(x, t, init="random", generator=torch.Generator().manual_seed(3))
    manual = net.weight_gradients(zs)
    auto = net.weight_gradients_autograd(zs)
    for l, ((dW, db), (aW, ab)) in enumerate(zip(manual, auto)):
        assert torch.allclose(dW, aW, atol=1e-10), f"dW mismatch, layer {l}"
        assert torch.allclose(db, ab, atol=1e-10), f"db mismatch, layer {l}"


@pytest.mark.parametrize("activation", ACTS)
def test_step0_output_layer_gradient_equals_backprop_and_hidden_layers_get_none(activation):
    x, y, t = make_batch()
    pc, bp = make_pc(activation), make_bp(activation)
    zs = pc.init_states(x, t, init="forward")
    pc_grads = pc.weight_gradients(zs)
    _, cache = bp.forward_cached(x)
    bp_grads = bp.manual_gradients(cache, y, "mse")
    assert torch.allclose(pc_grads[-1][0], bp_grads[-1][0], atol=1e-10)
    assert torch.allclose(pc_grads[-1][1], bp_grads[-1][1], atol=1e-10)
    for dW, db in pc_grads[:-1]:                 # hidden errors are 0 at step 0
        assert dW.abs().max() < 1e-12 and db.abs().max() < 1e-12


def slide_network():
    """HL-2 -> output part of the mentor's example (input = HL-1 states)."""
    net = PCNetwork([4, 2, 1], activation="relu").double()
    d = torch.float64
    with torch.no_grad():
        net.weights[0].copy_(torch.tensor([[-0.1, 0.2, 0.3, 0.1], [0.0, 0.0, 0.0, 0.0]], dtype=d))
        net.biases[0].copy_(torch.tensor([0.1, 0.1], dtype=d))
        net.weights[1].copy_(torch.tensor([[0.5, 0.5]], dtype=d))
        net.biases[1].copy_(torch.tensor([0.1], dtype=d))
    return net


def test_slide_numbers_forward_and_first_inference_step():
    net = slide_network()
    x = torch.tensor([[0.5, 0.433, 0.0, 0.2]], dtype=torch.float64)
    t = torch.tensor([[1.0]], dtype=torch.float64)

    zs = net.init_states(x, t, init="forward")
    assert torch.allclose(zs[1], torch.tensor([[0.1566, 0.1]], dtype=torch.float64))
    assert abs(net(x).item() - 0.2283) < 1e-9
    e0 = net.layer_energies(zs)
    assert abs(e0[0].item()) < 1e-12 and abs(e0[1].item() - 0.2977) < 1e-4

    zs, hist = net.infer(x, t, steps=1, state_lr=0.1, record=True)
    # exact values; the slide shows 0.1945 for the first state because it rounds 0.1566 to 0.156
    assert abs(zs[1][0, 0].item() - 0.195185) < 1e-6
    assert abs(zs[1][0, 1].item() - 0.138585) < 1e-6
    assert abs(zs[1][0, 0].item() - 0.1945) < 1e-3
    assert abs(hist[1][0] - 0.0015) < 1e-4       # hidden-layer energy after step 1
    assert abs(hist[1][1] - 0.269) < 1e-3        # output energy after step 1


def test_energy_decreases_during_inference():
    x, y, t = make_batch(n=16)
    net = make_pc("relu", scale=0.1)
    _, hist = net.infer(x, t, steps=20, state_lr=0.05, record=True)
    totals = [sum(h) for h in hist]
    assert all(b <= a + 1e-12 for a, b in zip(totals, totals[1:]))
    assert totals[-1] < totals[0]
    assert hist[-1][-1] < hist[0][-1]            # output energy fell
    assert max(hist[-1][:-1]) > 0                # error spread into the hidden layers


def test_inference_with_manual_and_autograd_updates_agree():
    x, y, t = make_batch()
    net = make_pc("tanh")
    a, _ = net.infer(x, t, steps=5, state_lr=0.05)
    b, _ = net.infer(x, t, steps=5, state_lr=0.05, use_autograd=True)
    for za, zb in zip(a, b):
        assert torch.allclose(za, zb, atol=1e-10)


def test_state_updates_do_not_depend_on_batch_size():
    x, y, t = make_batch(n=7)
    net = make_pc("tanh")
    full, _ = net.infer(x, t, steps=5, state_lr=0.05)
    single, _ = net.infer(x[:1], t[:1], steps=5, state_lr=0.05)
    for zf, zs in zip(full, single):
        assert torch.allclose(zf[:1], zs, atol=1e-10)


def test_weight_gradient_is_a_batch_mean():
    x, y, t = make_batch()
    net = make_pc("relu")
    zs, _ = net.infer(x, t, steps=3, state_lr=0.05)
    g1 = net.weight_gradients(zs)
    g2 = net.weight_gradients([torch.cat([z, z], dim=0) for z in zs])
    for (a, b), (c, d) in zip(g1, g2):
        assert torch.allclose(a, c, atol=1e-10) and torch.allclose(b, d, atol=1e-10)


def test_cross_entropy_is_not_implemented_yet():
    with pytest.raises(NotImplementedError):
        PCNetwork(SIZES, loss="cross_entropy")


def test_prediction_equals_feedforward_pass():
    x, y, t = make_batch()
    assert torch.allclose(make_pc()(x), make_bp()(x))


def test_error_reaches_a_layer_only_after_enough_inference_steps():
    """Layers [20,16,12,10] have 2 hidden layers: W_0 gets no gradient after
    1 inference step, and a nonzero one after 2."""
    x, y, t = make_batch()
    net = make_pc("relu")
    g1 = net.weight_gradients(net.infer(x, t, steps=1, state_lr=0.05)[0])
    g2 = net.weight_gradients(net.infer(x, t, steps=2, state_lr=0.05)[0])
    assert g1[0][0].abs().max() < 1e-12
    assert g1[1][0].abs().max() > 1e-9
    assert g2[0][0].abs().max() > 1e-9
