# Backpropagation vs Predictive Coding (PyTorch)

iCog Labs training task. A standard backpropagation (BP) MLP and a Predictive Coding (PC) network are implemented from scratch in PyTorch, trained on the same data with the same architecture, initial weights, optimizer and tuning budget, and compared. Hand-derived gradients are verified against `torch.autograd`, and the mentor's worked example is reproduced in tests.

**Status: work in progress.** The implementation, verification, tuning, main comparison, energy analysis, gradient-equivalence analysis and first ablations are done. Still open: the final test-set run of the weak-clamp variant, depth study, Adam and cross-entropy ablations, input-noise robustness and prediction-time measurement. The report (Google Drive) is the main write-up and holds the derivations, decision record and literature discussion.

## Main result (Fashion-MNIST, test set, 5 seeds)

| Model | Test accuracy (mean ± std) | Epoch time | Train time per run |
| --- | --- | --- | --- |
| Backprop | 0.8879 ± 0.0043 | 0.33 s | 8 s |
| PC (T = 50, η = 0.2) | 0.8517 ± 0.0039 | 6.79 s | 177 s |

BP beats PC on all 5 seeds (paired gap 3.0 to 4.0 points, mean 3.6). Settings were fixed from validation tuning before any test number was seen. Timings are from a Colab GPU. This is a result for this setup: η, activation, loss and output scale were not tuned for PC, so it is not a general claim about PC.

## What else has been measured

- **Gradient equivalence.** Cosine similarity and norm ratio between PC's and BP's weight gradients, for the same weights and batch. At initialization the direction matches (cosine ≥ 0.97 from T = 3) but size is below BP's. At trained weights (one seed, 4 batches) cosine falls to 0.77 to 0.87 and size to 6 to 12 percent of BP's. Weakening the output clamp (weight λ on the output energy, gradient divided by λ) restores cosine ≈ 0.99 at λ = 0.01.
- **Weak output clamp in training (validation only, seeds 0 to 2).** PC with λ = 0.1 reaches 0.880 to 0.882 validation accuracy against 0.860 for standard PC, closing more than half of the gap to BP (0.896). It has not been run on the test set yet, and λ was not tuned.
- **Layer-wise energy.** Hidden-layer energies are exactly 0 at inference step 0 (forward initialization). Error then spreads one layer per step. Figures are in `results/figures/`.
- **Other ablations (validation only, no per-variant retuning).** Zeros or random state initialization fail badly (0.28 to 0.61). Tanh and output scale 0.01 show no measurable difference. A batch-size test (one seed) found no small-batch advantage for PC in this setup.
- **Stability.** η ≥ 0.25 is unstable for the trained network, and weight learning rates around 0.07 fail for both models in a seed-dependent way.

All numbers above are in `results/tables/` as JSON or CSV, one file per run.

## Setup

Fixed choices (see the report's decision record for reasons):

- Data: Fashion-MNIST, 50k train / 10k validation (fixed split seed 42) / official 10k test. Global mean and std from the training split only. No augmentation.
- Architecture, both models: 784-512-256-128-10 (567,434 parameters), ReLU, linear output, MSE loss with factor 1/2 and mean over the batch.
- Both models start from identical weights for a given seed. The output layer is scaled by 0.1 (`last_layer_scale`) because a full-size He draw gave an initial loss of about 11 against 0.45 for a constant predictor.
- Optimizer for both: SGD with momentum 0.9. Early stopping on validation accuracy (patience 5), best-epoch weights restored.
- Tuned values: BP lr 0.0361; PC lr 0.0161 with T = 50, η = 0.2. Tuning used random search on the validation set, the same number of trials per model, seeds 100 and 101 (final runs use 0 to 4).
- PC state updates use per-sample energy (no 1/B); PC weight updates use the batch mean, matching BP.

## Repository layout

```text
configs/default.yaml        provisional config; open decisions are marked OPEN
src/
  data.py                   loading, split, train-only normalization, batches
  models/common.py          activations and derivatives, initialization, losses
  models/backprop_mlp.py    BP MLP: autograd and hand-derived gradients
  models/pc_network.py      PC network: states, errors, energies, state and weight updates
  train_backprop.py         BP trainer (early stopping, divergence guard)
  train_pc.py               PC trainer (inference loop, per-layer energy logging, stability monitor)
  trainer_utils.py          early stopping, evaluation, diagnostics
  metrics.py                cosine similarity and norm ratio
experiments/
  gradient_equivalence.py   PC vs BP gradients (initialization)
  tune.py                   paired random-search tuning
  main_comparison.py        5-seed comparison, test set evaluated once
  energy_curves.py          layer-wise energy figures
  ablation.py               one-factor-at-a-time ablations (validation only)
tests/                      pytest suite
results/figures, results/tables   outputs (one JSON per run)
```

## Reproduce

```bash
git clone https://github.com/YibeltalMarie/pc-vs-backprop.git
cd pc-vs-backprop
pip install -r requirements.txt        # versions are not pinned yet
python -m pytest tests -q              # tests must pass first
```

Experiments are Python functions, run from a notebook or script with the repo root on `sys.path`:

```python
from src.data import load_fashion_mnist
from src.trainer_utils import load_config
from src.utils import get_device
import experiments.main_comparison as mc

cfg = load_config("configs/default.yaml")
data = load_fashion_mnist(root="data/raw", device=str(get_device(cfg["device"])))
df = mc.run_main(cfg, data, out_dir="results/tables/main")
mc.summarize_main(df)
```

Tuning, energy figures and ablations follow the same pattern (`experiments.tune.run_tuning`, `experiments.energy_curves.run_energy`, `experiments.ablation.run_ablation`). Every run saves one JSON file with its config, per-epoch history and environment, and finished runs are skipped, so interrupted sessions resume. The ablation baseline is read from the finished main-comparison runs, so run the main comparison first. The dataset downloads on first use and is not committed.

## Verification

The tests check, among other things:

- hand-derived BP gradients equal `torch.autograd` (ReLU, tanh, GELU; MSE and cross-entropy);
- PC state and weight gradients equal their autograd versions, including with the weak output clamp;
- the mentor's example is reproduced (output 0.2283, loss 0.2977, blame -0.7717, states 0.195185 and 0.138585 after one step, output energy about 0.269 after the second step);
- at inference step 0 the PC output-layer gradient equals BP's and hidden layers get exactly zero gradient;
- runs are reproducible for a fixed seed and resumable.

Two inconsistencies in the slides, noted in the report: the first updated state is shown as 0.1945 but is 0.195185 (the slide rounds 0.1566 to 0.156), and one step-2 state is shown as 0.1174 where the derivation gives 0.1385.

## Known limitations

- Tuning used 2 seeds, and several choices are still provisional (marked OPEN in the config and report).
- Gradient-equivalence results at trained weights come from one seed and 4 batches.
- Ablations use validation accuracy on seeds 0 to 2, with learning rates not retuned per variant.
- Only the MSE energy is implemented for PC; cross-entropy for PC is not implemented.
- PC training is about 20 times slower than BP in this sequential implementation. Prediction time (a forward pass for both) has not been measured.
- GPU results can differ slightly between runs. `requirements.txt` is not pinned yet.
