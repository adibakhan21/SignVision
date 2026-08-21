# Verified claims

Every claim below names the metric, the experiment that produced it and the file it is read
from. Nothing here is estimated, rounded up, or carried over from the previous version of the
project. Regenerate all of it with the commands in the README's Reproducibility section.

Protocol shared by every number: 29 classes, 29,000 images (1,000/class stratified from 87,000),
deterministic 70/15/15 split (`split_seed=1234`), **4,350 held-out test images**, Adam lr 1e-3,
batch 32, best epoch selected on validation macro F1. 16 training runs total.

---

## 1. The previous headline number was not a test result

**Claim.** The previously reported 97.93% was training-set accuracy, and the accompanying
classification report was computed on 28 images — one per class.

| Evidence | Value |
|---|---|
| Source | `Project/2_230061_240792(1).ipynb`, stored cell outputs |
| Printed line | `Epoch 10 Loss 192.30 Accuracy 97.93431655420551`, emitted inside the training loop |
| Report support | `accuracy 1.00, support 28`; `Labels in test set: [0..25, 27, 28]` |
| Missing class | `del` (index 26) has zero test images and scored 0.00 |
| Training set size | 2,719 batches × 32 = 87,008 images |

**Replacement.** A real held-out evaluation on 4,350 images gives **93.13% ± 1.65%** for the same
architecture (`results/baseline_seed{42,43,44}/results.json`).

## 2. Restoring a missing non-linearity is worth +3.5 points

**Hypothesis.** `self.drop(self.f1(x))` feeds `f2` with no activation between, so the two linear
layers collapse to a single affine map and the 270-unit layer adds no capacity.
**Variable.** One `F.relu`. **Constant.** Parameter count, conv stack, optimiser, budget, split.

| Arm | Test accuracy | Macro F1 |
|---|--:|--:|
| A · ASLNet (original) | 93.13% ± 1.65 | 0.9313 ± 0.0165 |
| C · ASLNet-ReLU | **96.67% ± 1.00** | **0.9666 ± 0.0102** |

Δ = **+3.54 points accuracy**, 3 seeds each.
Files: `results/baseline_seed*/results.json`, `results/aslnet_relu_seed*/results.json`.
Test asserting the two models are otherwise identical: `tests/test_pipeline.py::test_relu_variant_differs_only_in_activation`.

## 3. Augmentation trades clean accuracy for geometric robustness

**Variable.** `aug.enabled`. **Constant.** Everything else. Perturbations are fixed, not sampled.

Δ accuracy in percentage points, augmented arm minus its non-augmented twin:

| Contrast | Budget | Clean | Rot 10° | Rot 20° | Shift 5% | Shift 10% |
|---|--:|--:|--:|--:|--:|--:|
| B − A | 10 ep | **−8.8** | +30.6 | +33.7 | +32.5 | **+45.0** |
| D − C | 10 ep | **−10.0** | +15.3 | +30.2 | +19.3 | **+43.2** |
| B30 − A30 | 30 ep | **−2.8** | +41.5 | +47.1 | +40.0 | **+56.3** |

Files: `results/TRADEOFF.md`, `results/*/robustness.json`, produced by `experiments/03_robustness.py`
and aggregated by `experiments/06_tradeoff.py`.

## 4. The clean-accuracy cost is largely under-convergence

**Hypothesis.** Augmentation makes the training distribution harder, so at a fixed 10-epoch budget
the augmented model simply has not converged — the "cost" should shrink with a longer budget.
**Variable.** Epoch budget (10 → 30). **Constant.** Everything else.

| Budget | Clean-accuracy cost of augmentation | Robustness gain at 10% shift |
|---|--:|--:|
| 10 epochs | −8.8 points | +45.0 points |
| 30 epochs | **−2.8 points** | **+56.3 points** |

Tripling the budget cuts the cost by 68% while *increasing* the benefit. Confirms the hypothesis.

**Caveat that must be stated with this claim:** the 30-epoch augmented arm is **n = 1 seed**
(`results/augmented_e30_seed42/`), against n = 3 for its baseline. Direction only; not a
measured mean.

## 5. Errors are concentrated and linguistically coherent

**Claim.** The top-5 confusion pairs account for **27.4% of all errors** (32 of 117) on the best
model, and the dominant pairs are handshapes differing only in finger count or separation.

| Rank | True → Pred | Errors | % of all errors |
|--:|---|--:|--:|
| 1 | V → W | 11 | 9.4% |
| 2 | U → R | 7 | 6.0% |
| 3 | G → H | 7 | 6.0% |
| 4 | W → K | 4 | 3.4% |
| 5 | S → T | 3 | 2.6% |

Run: `aslnet_relu_seed42`, 117 errors on 4,350 images (2.69% error rate).
File: `results/aslnet_relu_seed42/error_analysis.json`, produced by `experiments/02_error_analysis.py`.

## 6. The model is far more brittle to geometric than photometric shift

**Claim.** A 5-pixel (5%) translation costs **32.8 points** of accuracy, while a brightness change
to ×0.8 costs **1.2 points**, on the same checkpoint and the same images.

| Perturbation | Accuracy | Δ from clean |
|---|--:|--:|
| clean | 97.31% | — |
| brightness ×0.8 | 96.14% | −1.2 |
| blur σ0.5 | 96.83% | −0.5 |
| **translation 5%** | **65.36%** | **−32.8** |
| **rotation 20°** | **35.38%** | **−61.9** |
| translation 15% | 18.55% | −78.8 |

Run: `aslnet_relu_seed42`. File: `results/aslnet_relu_seed42/robustness.json`.

**Mechanism** (architectural, checkable by reading `models.py`): the network flattens a 10×10
spatial map directly into a fully connected layer with no global pooling, so absolute position is
encoded in the FC weights and a shift moves every feature onto weights that never learned it.

---

# Do not claim yet

Nothing in this repository measures any of the following. Do not put them on a resume.

| Claim | Why not |
|---|---|
| **Latency / FPS / real-time** | No timing instrumentation exists. Training wall-clock is recorded; inference latency is not. |
| **Deployment / edge / quantisation** | Not implemented. No export, no quantisation, no serving path. |
| **Statistical significance** | 3 seeds per arm at most. Arm D spans 0.75–0.93 (σ = 10.1%), larger than several between-arm gaps. No significance test was run. |
| **"Augmentation improved accuracy"** | It reduces clean accuracy at every budget tested (−8.8 at 10 epochs, −2.8 at 30). The honest claim is the trade-off. |
| **"Robust model"** | Accuracy still falls to 18.6% at 15% translation. The model is *more* robust with augmentation, not robust. |
| **"96.7% on ASL recognition"** without qualification | Single signer, single background, single capture session. Frames are near-duplicates, so a random split is optimistic for an unseen signer. |
| **Model superiority over any published work** | No external baseline was reproduced or compared against. |
| **Grad-CAM "proves" the model looks at the hand** | Grad-CAM is a diagnostic. Several confident errors attend to forearm and wall corner, and one attends to nothing at all (all-zero map at p = 0.94). |
| **Transfer learning / architecture comparison** | The ResNet-18 arm was implemented but never run. |
| **Any number from the 30-epoch augmented arm as a mean** | n = 1 seed. |
