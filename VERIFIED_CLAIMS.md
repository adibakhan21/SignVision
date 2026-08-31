# Verified claims

Every claim below names the metric, the experiment that produced it and the file it is read
from. Nothing here is estimated, rounded up, or carried over from the previous version of the
project. Regenerate all of it with the commands in the README's Reproducibility section.

**Every finding in §1–§6 was run twice**, in two independent execution environments —
identical code, split and seeds, but a different RNG stream. Contrasts that keep their sign in both
environments are reported as findings; one that flipped (§2) is retracted. See `results/CROSS_ENV.md`.
**§7 is the one exception and is marked pending**: it is sign-stable across three seeds but has been
run in one environment only, so by the rule that produced §2 it is not yet a finding.

Protocol shared by every number: 29 classes, 29,000 images (1,000/class stratified from 87,000),
deterministic 70/15/15 split (`split_seed=1234`), **4,350 held-out test images**, Adam lr 1e-3,
batch 32, best epoch selected on validation macro F1. **43 training runs total** — 40 across the two
environments for §1–§6, plus the 3-seed §7 arm in Environment A only.

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

**Why it is withdrawn.** Re-running the identical code, split and seeds in a second environment
reverses the sign of the effect:

| Budget | Baseline | ASLNet-ReLU | Δ | Pooled sd |
|---|--:|--:|--:|--:|
| Environment A, 10 ep | 93.13% ± 1.65 | 96.67% ± 1.00 | **+3.54** | 1.37 |
| Environment B, 10 ep | 93.85% ± 1.73 | 92.85% ± 4.60 | **−1.00** | 3.48 |
| Environment B, 30 ep | 97.30% ± 0.43 | 97.49% ± 1.38 | **+0.19** | 1.03 |

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
| Environment A | 10 ep | −8.8 | +30.6 | +33.7 | +32.5 | **+45.0** | +4.1 |
| Environment B | 10 ep | −8.0 | +35.8 | +32.5 | +39.3 | **+46.6** | +4.2 |
| Environment B | 30 ep | −4.3 | +41.5 | +45.1 | +40.2 | **+55.6** | +11.4 |
| Environment B, ReLU variant | 30 ep | −6.8 | +20.6 | +34.8 | +25.0 | **+52.9** | +11.2 |

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
| Environment A, clean cost | −8.8 | −2.8 (n=1) | −68% |
| Environment B, clean cost | −8.0 | **−4.3** | **−46%** |
| Environment B, gain at 10% shift | +46.6 | **+55.6** | **+19%** |

Tripling the budget roughly halves the cost while *increasing* the benefit. Confirms the hypothesis
in both environments.

Replicated at n = 3 per arm in Environment B (`results_kaggle/`): the cost falls from −8.0 to −4.3 points
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

## 7. Removing the flatten buys rotation robustness and sells photometric robustness

> **Single environment only (n = 3 seeds, Environment A).** Every contrast below is sign-stable
> across three seeds, but has **not** yet been replicated in a second environment. Under the rule
> that produced the retraction in §2, that is not enough to state as a finding. Treat this section
> as pending until the arm is re-run on Environment B.

**Hypothesis.** §6 proposed a mechanism for the geometric brittleness: `ASLNetOriginal` flattens the
27x10x10 map straight into a fully connected layer, so absolute position is encoded in that weight
matrix. If that is the cause, removing the flatten should recover geometric robustness.

**Variable.** `flatten -> Linear(2700, 270)` replaced by global average pooling -> `Linear(27, 270)`.
**Constant.** Conv stack, head widths, the missing FC non-linearity, epoch budget, split, seeds,
optimiser, evaluation set. Parameters fall from **757,433 to 35,723** (21.2x) as a consequence.

Δ accuracy in percentage points, `aslnet_gap_e30` minus `baseline_e30`, both n = 3 at 30 epochs.

| Perturbation | `baseline_e30` | `aslnet_gap_e30` | Δ | Per-seed Δ | Sign stable? |
|---|--:|--:|--:|---|:--:|
| clean | 96.18% ± 0.16 | 94.04% ± 1.59 | −2.1 | −1.4 / −3.8 / −1.2 | yes |
| rotation 5° | 66.51% ± 5.47 | 78.80% ± 1.34 | **+12.3** | +5.4 / +15.8 / +15.6 | yes |
| rotation 10° | 47.33% ± 7.88 | 61.14% ± 2.18 | **+13.8** | +8.1 / +8.9 / +24.4 | yes |
| rotation 20° | 19.52% ± 7.13 | 28.92% ± 3.82 | **+9.4** | +1.2 / +7.8 / +19.1 | yes |
| blur σ1.0 | 83.43% ± 1.08 | 88.02% ± 1.95 | +4.6 | +6.2 / +1.1 / +6.4 | yes |
| blur σ0.5 | 95.36% ± 0.41 | 93.47% ± 1.58 | −1.9 | −0.5 / −4.2 / −1.0 | yes |
| brightness ×0.8 | 92.80% ± 1.91 | 87.09% ± 2.96 | −5.7 | −3.8 / −10.7 / −2.7 | yes |
| brightness ×0.6 | 80.04% ± 5.25 | 52.05% ± 3.75 | **−28.0** | −24.4 / −32.8 / −26.7 | yes |
| **translation 5%** | 51.23% ± 5.95 | 59.26% ± 4.08 | +8.0 | −1.4 / +17.0 / +8.4 | **no** |
| **translation 10%** | 27.12% ± 2.49 | 28.87% ± 8.60 | +1.7 | +5.4 / −7.4 / +7.2 | **no** |
| **translation 15%** | 16.85% ± 5.97 | 15.47% ± 7.06 | −1.4 | +13.5 / −10.5 / −7.2 | **no** |
| brightness ×1.4 | 83.89% ± 1.39 | 81.59% ± 6.07 | −2.3 | +1.0 / +0.3 / −8.2 | **no** |
| blur σ2.0 | 60.63% ± 6.16 | 56.47% ± 2.70 | −4.2 | −10.1 / −8.1 / +5.7 | **no** |

**What the hypothesis got right.** Rotation robustness improves at all three severities, sign-stable,
and with markedly tighter seed spread than the baseline (±1.34 vs ±5.47 at 10°). The flatten does
encode orientation-sensitive position information, and removing it recovers some of that — at 21x
fewer parameters and 2.1 points of clean accuracy.

**What the hypothesis got wrong.** **Translation does not improve.** All three translation contrasts
flip sign across seeds and cannot be claimed in either direction. Since the flatten is demonstrably
not the fix, the dominant cause of translation brittleness must sit **upstream of it** — the two
strided max-pools (4x4 then 2x2) change which pixels each window samples when the input shifts, and
that information is already lost before the classifier head is reached. Global average pooling
cannot repair damage done two layers earlier.

**The trade-off appears a third time.** Augmentation trades clean accuracy for geometric robustness
(§3). Global average pooling trades **photometric** robustness for geometric robustness: −28.0 points
at brightness ×0.6, the largest single regression in the study. Pooling collapses each channel to one
average, so a global intensity change scales all 27 inputs together with no spatial pattern left to
disambiguate. On this task, robustness is consistently bought rather than gained.

Files: `results/aslnet_gap_e30_seed{42,43,44}/robustness.json`, `results/TRADEOFF.md`.
Reproduce with `./scripts/run_grid_gap.sh`.

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
| **"Global average pooling fixes the translation brittleness"** | It does not. All three translation contrasts flip sign across seeds (§7). The mechanism in §6 explains rotation, not translation. |
| **Anything in §7, until it is re-run on a second environment** | Sign-stable across 3 seeds in one environment only. §2 is the proof that this is not sufficient. |
| **Any single-environment result with n ≤ 3 seeds** | The retraction in §2 is the proof that three seeds on one machine can manufacture a 3.5-point effect that does not exist. |
| **A best accuracy above ~97.3%** | The best verified arm is `baseline_e30` at 97.30% ± 0.43 (n=3). `aslnet_relu_e30` reads 97.49% but with sd 1.38 and no replication. |
