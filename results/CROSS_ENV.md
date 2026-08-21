# Cross-environment replication

The same code, split and seeds were run in two environments:

| | Environment A | Environment B |
|---|---|---|
| runs | 16 | 24 |

The RNG stream differs between environments, so weight initialisation and shuffle
order diverge. Any gap below is environment + seed noise, not a code difference.

## Test accuracy by arm

| Arm | Environment A (n) | Environment B (n) | Δ mean |
|---|--:|--:|--:|
| `aslnet_relu` | 0.9667 ± 0.0100 (3) | 0.9285 ± 0.0460 (3) | -0.0382 |
| `aslnet_relu_aug` | 0.8665 ± 0.1009 (3) | 0.8167 ± 0.1463 (3) | -0.0498 |
| `aslnet_relu_aug_e30` | — (0) | 0.9073 ± 0.1022 (3) | — |
| `aslnet_relu_e30` | — (0) | 0.9749 ± 0.0138 (3) | — |
| `augmented` | 0.8434 ± 0.0259 (3) | 0.8585 ± 0.0304 (3) | +0.0151 |
| `augmented_e30` | 0.9343 (1) | 0.9299 ± 0.0246 (3) | -0.0044 |
| `baseline` | 0.9313 ± 0.0165 (3) | 0.9385 ± 0.0173 (3) | +0.0072 |
| `baseline_e30` | 0.9618 ± 0.0016 (3) | 0.9730 ± 0.0043 (3) | +0.0112 |

## Do the findings survive?

| Contrast | Env A Δ | Env B Δ | same sign? |
|---|--:|--:|:--:|
| ReLU fix (aslnet_relu − baseline) | +0.0354 | -0.0100 | **no** |
| Augmentation cost, 10 ep (augmented − baseline) | -0.0880 | -0.0801 | yes |
| Augmentation cost, 30 ep (augmented_e30 − baseline_e30) | -0.0275 | -0.0431 | yes |

A contrast that keeps its sign across two independent environments is a finding.
One that flips is within noise and must not be claimed.

