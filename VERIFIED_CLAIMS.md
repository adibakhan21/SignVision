# Verified claims

Every claim below names the metric, the experiment that produced it and the file it is read
from. Nothing here is estimated, rounded up, or carried over from the previous version of the
project. Regenerate all of it with the commands in the README's Reproducibility section.

**Every finding below was run twice**, on Apple MPS locally and on an NVIDIA T4 via Kaggle —
identical code, split and seeds, but a different RNG stream. Contrasts that keep their sign in both
environments are reported as findings; one that flipped (§2) is retracted. See `results/CROSS_ENV.md`.

Protocol shared by every number: 29 classes, 29,000 images (1,000/class stratified from 87,000),
deterministic 70/15/15 split (`split_seed=1234`), **4,350 held-out test images**, Adam lr 1e-3,
batch 32, best epoch selected on validation macro F1. **40 training runs total** (16 local + 24 on Kaggle).

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

## 2. RETRACTED — the FC non-linearity fix does not replicate

**What was claimed.** That restoring the missing activation between the two fully connected
layers raised accuracy from 93.13% to 96.67%, a +3.5 point gain at identical parameter count.

**Why it is withdrawn.** Re-running the identical code, split and seeds on a second device
(NVIDIA T4 vs Apple MPS) reverses the sign of the effect:

| Budget | Baseline | ASLNet-ReLU | Δ | Pooled sd |
|---|--:|--:|--:|--:|
| Local, 10 ep (MPS) | 93.13% ± 1.65 | 96.67% ± 1.00 | **+3.54** | 1.37 |
| Kaggle, 10 ep (T4) | 93.85% ± 1.73 | 92.85% ± 4.60 | **−1.00** | 3.48 |
| Kaggle, 30 ep (T4) | 97.30% ± 0.43 | 97.49% ± 1.38 | **+0.19** | 1.03 |

At 30 epochs — where both arms have converged and variance is lowest — the difference is
**+0.19 points against a pooled sd of 1.03**, i.e. indistinguishable from zero. The `aslnet_relu`
arm is also the most unstable in the study (Kaggle seeds: 0.9648 / 0.8768 / 0.9439, sd 4.6 points),
so a 3-seed mean on one device was never enough to establish a 3.5-point effect.

**The observation that remains true.** The architectural fact is still a fact: `self.drop(self.f1(x))`
feeds `f2` with no activation between them, so at inference the two linear layers collapse to a
single affine map and the 270-unit hidden layer contributes no representational capacity. What is
*not* true is that fixing it measurably improves accuracy on this task. The most likely reason is
that the task does not need the extra capacity — the conv stack already separates the classes.

**How this was caught.** Cross-environment replication (`results/CROSS_ENV.md`). This is the
single most useful thing in the project: it killed a finding that three seeds on one machine had
made look solid.

## 3. Augmentation trades clean accuracy for geometric robustness

**Variable.** `aug.enabled`. **Constant.** Everything else. Perturbations are fixed, not sampled.

Δ accuracy in percentage points, augmented arm minus its non-augmented twin. Every arm is n = 3.

| Environment | Budget | Clean | Rot 10° | Rot 20° | Shift 5% | Shift 10% | Blur σ2 |
|---|--:|--:|--:|--:|--:|--:|--:|
| Local (MPS) | 10 ep | −8.8 | +30.6 | +33.7 | +32.5 | **+45.0** | +4.1 |
| Kaggle (T4) | 10 ep | −8.0 | +35.8 | +32.5 | +39.3 | **+46.6** | +4.2 |
| Kaggle (T4) | 30 ep | −4.3 | +41.5 | +45.1 | +40.2 | **+55.6** | +11.4 |
| Kaggle (T4), ReLU variant | 30 ep | −6.8 | +20.6 | +34.8 | +25.0 | **+52.9** | +11.2 |

**All four contrasts agree in sign and rough magnitude across two devices and two budgets.**
This is the project's most robust finding.

Files: `results/TRADEOFF.md`, `results/*/robustness.json`, produced by `experiments/03_robustness.py`
and aggregated by `experiments/06_tradeoff.py`.

## 4. The clean-accuracy cost is largely under-convergence

**Hypothesis.** Augmentation makes the training distribution harder, so at a fixed 10-epoch budget
the augmented model simply has not converged — the "cost" should shrink with a longer budget.
**Variable.** Epoch budget (10 → 30). **Constant.** Everything else.

| Environment | 10 epochs | 30 epochs | Change |
|---|--:|--:|--:|
| Local (MPS), clean cost | −8.8 | −2.8 (n=1) | −68% |
| Kaggle (T4), clean cost | −8.0 | **−4.3** | **−46%** |
| Kaggle (T4), gain at 10% shift | +46.6 | **+55.6** | **+19%** |

Tripling the budget roughly halves the cost while *increasing* the benefit. Confirms the hypothesis
in both environments.

Replicated at n = 3 per arm on the T4 (`results_kaggle/`): the cost falls from −8.0 to −4.3 points
while the gain at 10% translation rises from +46.6 to +55.6. Both environments agree.

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
| **"Fixing the FC non-linearity improved accuracy"** | Retracted — see §2. The effect reverses sign across devices and is +0.19 ± 1.03 at 30 epochs. |
| **Any single-environment result with n ≤ 3 seeds** | The retraction in §2 is the proof that three seeds on one machine can manufacture a 3.5-point effect that does not exist. |
| **A best accuracy above ~97.3%** | The best verified arm is `baseline_e30` at 97.30% ± 0.43 (T4, n=3). `aslnet_relu_e30` reads 97.49% but with sd 1.38 and no replication. |
