# SignVision — verified experimental results

Every number below was written by a script in `experiments/` and read back from
`results/*/results.json`. Regenerate with `python experiments/05_report.py`.

## Per-run results (held-out test set)

| Run | Model | Aug | Seed | Params | Best ep | Test acc | Macro F1 | Weighted F1 | Macro P | Macro R |
|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| `baseline_e30_seed42` | aslnet_original | no | 42 | 757,433 | 24 | 0.9634 | 0.9634 | 0.9634 | 0.9639 | 0.9634 |
| `baseline_seed42` | aslnet_original | no | 42 | 757,433 | 10 | 0.9501 | 0.9501 | 0.9501 | 0.9515 | 0.9501 |
| `baseline_e30_seed43` | aslnet_original | no | 43 | 757,433 | 30 | 0.9602 | 0.9603 | 0.9603 | 0.9613 | 0.9602 |
| `baseline_seed43` | aslnet_original | no | 43 | 757,433 | 10 | 0.9193 | 0.9192 | 0.9192 | 0.9212 | 0.9193 |
| `baseline_e30_seed44` | aslnet_original | no | 44 | 757,433 | 28 | 0.9616 | 0.9617 | 0.9617 | 0.9623 | 0.9616 |
| `baseline_seed44` | aslnet_original | no | 44 | 757,433 | 10 | 0.9246 | 0.9244 | 0.9244 | 0.9262 | 0.9246 |
| `augmented_e30_seed42` | aslnet_original | yes | 42 | 757,433 | 24 | 0.9343 | 0.9347 | 0.9347 | 0.9373 | 0.9343 |
| `augmented_seed42` | aslnet_original | yes | 42 | 757,433 | 9 | 0.8674 | 0.8687 | 0.8687 | 0.8764 | 0.8674 |
| `augmented_seed43` | aslnet_original | yes | 43 | 757,433 | 10 | 0.8469 | 0.8468 | 0.8468 | 0.8587 | 0.8469 |
| `augmented_seed44` | aslnet_original | yes | 44 | 757,433 | 10 | 0.8159 | 0.8145 | 0.8145 | 0.8229 | 0.8159 |
| `aslnet_relu_seed42` | aslnet_relu | no | 42 | 757,433 | 10 | 0.9731 | 0.9730 | 0.9730 | 0.9735 | 0.9731 |
| `aslnet_relu_seed43` | aslnet_relu | no | 43 | 757,433 | 10 | 0.9552 | 0.9548 | 0.9548 | 0.9562 | 0.9552 |
| `aslnet_relu_seed44` | aslnet_relu | no | 44 | 757,433 | 10 | 0.9720 | 0.9720 | 0.9720 | 0.9722 | 0.9720 |
| `aslnet_relu_aug_seed42` | aslnet_relu | yes | 42 | 757,433 | 10 | 0.9209 | 0.9213 | 0.9213 | 0.9253 | 0.9209 |
| `aslnet_relu_aug_seed43` | aslnet_relu | yes | 43 | 757,433 | 9 | 0.7501 | 0.7446 | 0.7446 | 0.7579 | 0.7501 |
| `aslnet_relu_aug_seed44` | aslnet_relu | yes | 44 | 757,433 | 10 | 0.9285 | 0.9294 | 0.9294 | 0.9336 | 0.9285 |

## Arm summary (mean ± sd across seeds)

| Arm | Seeds | Test acc | Macro F1 | Weighted F1 |
|---|--:|--:|--:|--:|
| aslnet_original|aug=False | 6 | 0.9466 ± 0.0197 | 0.9465 ± 0.0198 | 0.9465 ± 0.0198 |
| aslnet_original|aug=True | 4 | 0.8661 ± 0.0501 | 0.8662 ± 0.0508 | 0.8662 ± 0.0508 |
| aslnet_relu|aug=False | 3 | 0.9667 ± 0.0100 | 0.9666 ± 0.0102 | 0.9666 ± 0.0102 |
| aslnet_relu|aug=True | 3 | 0.8665 ± 0.1009 | 0.8651 ± 0.1044 | 0.8651 ± 0.1044 |

## Augmentation ablation (identical split, architecture, optimiser, budget)

| Metric | Baseline | + Augmentation | Δ (abs) | Δ (rel) |
|---|--:|--:|--:|--:|
| Accuracy | 0.9466 | 0.8661 | -0.0805 | -8.50% |
| Macro F1 | 0.9465 | 0.8662 | -0.0804 | -8.49% |
| Weighted F1 | 0.9465 | 0.8662 | -0.0804 | -8.49% |

Averaged over 4 seed(s) per arm. No significance test was run; treat the direction, not the magnitude, as the finding.

## Robustness — `aslnet_relu_aug_seed42`

Clean test accuracy 0.9209, macro F1 0.9213.

| Perturbation | Accuracy | Δ acc | Macro F1 | Δ macro F1 | Rel. acc drop |
|---|--:|--:|--:|--:|--:|
| rotation 5deg | 0.9060 | -0.0149 | 0.9066 | -0.0147 | 1.62% |
| rotation 10deg | 0.8722 | -0.0487 | 0.8732 | -0.0481 | 5.29% |
| rotation 20deg | 0.6722 | -0.2487 | 0.6725 | -0.2488 | 27.01% |
| brightness x0.6 | 0.8913 | -0.0297 | 0.8924 | -0.0289 | 3.22% |
| brightness x0.8 | 0.9193 | -0.0016 | 0.9198 | -0.0015 | 0.17% |
| brightness x1.4 | 0.9009 | -0.0200 | 0.9011 | -0.0202 | 2.17% |
| blur sigma0.5 | 0.9186 | -0.0023 | 0.9191 | -0.0022 | 0.25% |
| blur sigma1.0 | 0.8949 | -0.0260 | 0.8952 | -0.0261 | 2.82% |
| blur sigma2.0 | 0.7616 | -0.1593 | 0.7594 | -0.1619 | 17.30% |
| translation 5pct | 0.8846 | -0.0363 | 0.8852 | -0.0362 | 3.94% |
| translation 10pct | 0.7954 | -0.1255 | 0.7972 | -0.1241 | 13.63% |
| translation 15pct | 0.6315 | -0.2894 | 0.6288 | -0.2925 | 31.43% |

## Robustness — `aslnet_relu_aug_seed43`

Clean test accuracy 0.7501, macro F1 0.7446.

| Perturbation | Accuracy | Δ acc | Macro F1 | Δ macro F1 | Rel. acc drop |
|---|--:|--:|--:|--:|--:|
| rotation 5deg | 0.7287 | -0.0214 | 0.7228 | -0.0218 | 2.85% |
| rotation 10deg | 0.6853 | -0.0648 | 0.6809 | -0.0637 | 8.64% |
| rotation 20deg | 0.4584 | -0.2917 | 0.4555 | -0.2891 | 38.89% |
| brightness x0.6 | 0.7271 | -0.0230 | 0.7271 | -0.0174 | 3.06% |
| brightness x0.8 | 0.7552 | +0.0051 | 0.7506 | +0.0061 | -0.67% |
| brightness x1.4 | 0.7345 | -0.0156 | 0.7267 | -0.0178 | 2.08% |
| blur sigma0.5 | 0.7455 | -0.0046 | 0.7404 | -0.0041 | 0.61% |
| blur sigma1.0 | 0.7267 | -0.0234 | 0.7227 | -0.0218 | 3.13% |
| blur sigma2.0 | 0.6108 | -0.1393 | 0.6126 | -0.1320 | 18.57% |
| translation 5pct | 0.7097 | -0.0405 | 0.7059 | -0.0387 | 5.39% |
| translation 10pct | 0.5657 | -0.1844 | 0.5679 | -0.1767 | 24.58% |
| translation 15pct | 0.3959 | -0.3543 | 0.3895 | -0.3551 | 47.23% |

## Robustness — `aslnet_relu_aug_seed44`

Clean test accuracy 0.9285, macro F1 0.9294.

| Perturbation | Accuracy | Δ acc | Macro F1 | Δ macro F1 | Rel. acc drop |
|---|--:|--:|--:|--:|--:|
| rotation 5deg | 0.9207 | -0.0078 | 0.9218 | -0.0075 | 0.84% |
| rotation 10deg | 0.8857 | -0.0428 | 0.8874 | -0.0420 | 4.61% |
| rotation 20deg | 0.6936 | -0.2349 | 0.6940 | -0.2354 | 25.30% |
| brightness x0.6 | 0.8917 | -0.0368 | 0.8938 | -0.0356 | 3.96% |
| brightness x0.8 | 0.9274 | -0.0011 | 0.9284 | -0.0009 | 0.12% |
| brightness x1.4 | 0.9138 | -0.0147 | 0.9137 | -0.0156 | 1.58% |
| blur sigma0.5 | 0.9267 | -0.0018 | 0.9277 | -0.0017 | 0.20% |
| blur sigma1.0 | 0.9115 | -0.0170 | 0.9131 | -0.0163 | 1.83% |
| blur sigma2.0 | 0.8039 | -0.1246 | 0.8071 | -0.1223 | 13.42% |
| translation 5pct | 0.9011 | -0.0274 | 0.9025 | -0.0269 | 2.95% |
| translation 10pct | 0.8087 | -0.1198 | 0.8114 | -0.1180 | 12.90% |
| translation 15pct | 0.6301 | -0.2984 | 0.6339 | -0.2955 | 32.14% |

## Robustness — `aslnet_relu_seed42`

Clean test accuracy 0.9731, macro F1 0.9730.

| Perturbation | Accuracy | Δ acc | Macro F1 | Δ macro F1 | Rel. acc drop |
|---|--:|--:|--:|--:|--:|
| rotation 5deg | 0.9078 | -0.0653 | 0.9086 | -0.0644 | 6.71% |
| rotation 10deg | 0.7168 | -0.2563 | 0.7216 | -0.2515 | 26.34% |
| rotation 20deg | 0.3538 | -0.6193 | 0.3286 | -0.6444 | 63.64% |
| brightness x0.6 | 0.9053 | -0.0678 | 0.9061 | -0.0670 | 6.97% |
| brightness x0.8 | 0.9614 | -0.0117 | 0.9612 | -0.0118 | 1.20% |
| brightness x1.4 | 0.9441 | -0.0290 | 0.9434 | -0.0296 | 2.98% |
| blur sigma0.5 | 0.9683 | -0.0048 | 0.9681 | -0.0049 | 0.50% |
| blur sigma1.0 | 0.9372 | -0.0359 | 0.9368 | -0.0363 | 3.69% |
| blur sigma2.0 | 0.7977 | -0.1754 | 0.7977 | -0.1754 | 18.03% |
| translation 5pct | 0.6536 | -0.3195 | 0.6790 | -0.2941 | 32.84% |
| translation 10pct | 0.3080 | -0.6651 | 0.2999 | -0.6732 | 68.34% |
| translation 15pct | 0.1855 | -0.7876 | 0.1721 | -0.8009 | 80.94% |

## Robustness — `aslnet_relu_seed43`

Clean test accuracy 0.9552, macro F1 0.9548.

| Perturbation | Accuracy | Δ acc | Macro F1 | Δ macro F1 | Rel. acc drop |
|---|--:|--:|--:|--:|--:|
| rotation 5deg | 0.8108 | -0.1444 | 0.8106 | -0.1442 | 15.11% |
| rotation 10deg | 0.5692 | -0.3860 | 0.5656 | -0.3892 | 40.41% |
| rotation 20deg | 0.2616 | -0.6936 | 0.2379 | -0.7169 | 72.61% |
| brightness x0.6 | 0.7823 | -0.1729 | 0.7794 | -0.1754 | 18.10% |
| brightness x0.8 | 0.9186 | -0.0366 | 0.9173 | -0.0375 | 3.83% |
| brightness x1.4 | 0.9301 | -0.0251 | 0.9297 | -0.0251 | 2.62% |
| blur sigma0.5 | 0.9568 | +0.0016 | 0.9566 | +0.0018 | -0.17% |
| blur sigma1.0 | 0.9476 | -0.0076 | 0.9476 | -0.0072 | 0.79% |
| blur sigma2.0 | 0.8379 | -0.1172 | 0.8358 | -0.1190 | 12.27% |
| translation 5pct | 0.5191 | -0.4361 | 0.5177 | -0.4371 | 45.66% |
| translation 10pct | 0.2200 | -0.7352 | 0.2027 | -0.7521 | 76.97% |
| translation 15pct | 0.1099 | -0.8453 | 0.0948 | -0.8600 | 88.50% |

## Robustness — `aslnet_relu_seed44`

Clean test accuracy 0.9720, macro F1 0.9720.

| Perturbation | Accuracy | Δ acc | Macro F1 | Δ macro F1 | Rel. acc drop |
|---|--:|--:|--:|--:|--:|
| rotation 5deg | 0.9000 | -0.0720 | 0.9006 | -0.0714 | 7.40% |
| rotation 10deg | 0.6970 | -0.2749 | 0.6969 | -0.2751 | 28.29% |
| rotation 20deg | 0.3018 | -0.6701 | 0.2611 | -0.7108 | 68.95% |
| brightness x0.6 | 0.8239 | -0.1480 | 0.8326 | -0.1394 | 15.23% |
| brightness x0.8 | 0.9457 | -0.0262 | 0.9460 | -0.0260 | 2.70% |
| brightness x1.4 | 0.9285 | -0.0434 | 0.9285 | -0.0435 | 4.47% |
| blur sigma0.5 | 0.9724 | +0.0005 | 0.9724 | +0.0005 | -0.05% |
| blur sigma1.0 | 0.9593 | -0.0126 | 0.9595 | -0.0125 | 1.30% |
| blur sigma2.0 | 0.8437 | -0.1283 | 0.8434 | -0.1285 | 13.20% |
| translation 5pct | 0.7448 | -0.2271 | 0.7415 | -0.2304 | 23.37% |
| translation 10pct | 0.3453 | -0.6267 | 0.3527 | -0.6193 | 64.47% |
| translation 15pct | 0.1823 | -0.7897 | 0.1759 | -0.7961 | 81.24% |

## Robustness — `augmented_e30_seed42`

Clean test accuracy 0.9343, macro F1 0.9347.

| Perturbation | Accuracy | Δ acc | Macro F1 | Δ macro F1 | Rel. acc drop |
|---|--:|--:|--:|--:|--:|
| rotation 5deg | 0.9278 | -0.0064 | 0.9283 | -0.0064 | 0.69% |
| rotation 10deg | 0.8878 | -0.0464 | 0.8880 | -0.0467 | 4.97% |
| rotation 20deg | 0.6667 | -0.2676 | 0.6654 | -0.2693 | 28.64% |
| brightness x0.6 | 0.9002 | -0.0340 | 0.9008 | -0.0339 | 3.64% |
| brightness x0.8 | 0.9331 | -0.0011 | 0.9335 | -0.0012 | 0.12% |
| brightness x1.4 | 0.9117 | -0.0225 | 0.9115 | -0.0233 | 2.41% |
| blur sigma0.5 | 0.9333 | -0.0009 | 0.9338 | -0.0010 | 0.10% |
| blur sigma1.0 | 0.9083 | -0.0260 | 0.9085 | -0.0263 | 2.78% |
| blur sigma2.0 | 0.7039 | -0.2303 | 0.7033 | -0.2314 | 24.66% |
| translation 5pct | 0.9122 | -0.0221 | 0.9126 | -0.0221 | 2.36% |
| translation 10pct | 0.8345 | -0.0998 | 0.8350 | -0.0997 | 10.68% |
| translation 15pct | 0.6805 | -0.2538 | 0.6795 | -0.2552 | 27.17% |

## Robustness — `augmented_seed42`

Clean test accuracy 0.8674, macro F1 0.8687.

| Perturbation | Accuracy | Δ acc | Macro F1 | Δ macro F1 | Rel. acc drop |
|---|--:|--:|--:|--:|--:|
| rotation 5deg | 0.8451 | -0.0223 | 0.8464 | -0.0223 | 2.57% |
| rotation 10deg | 0.8067 | -0.0607 | 0.8072 | -0.0616 | 7.00% |
| rotation 20deg | 0.5756 | -0.2917 | 0.5746 | -0.2941 | 33.63% |
| brightness x0.6 | 0.8434 | -0.0239 | 0.8453 | -0.0234 | 2.76% |
| brightness x0.8 | 0.8657 | -0.0016 | 0.8677 | -0.0011 | 0.19% |
| brightness x1.4 | 0.8529 | -0.0145 | 0.8538 | -0.0149 | 1.67% |
| blur sigma0.5 | 0.8623 | -0.0051 | 0.8640 | -0.0048 | 0.58% |
| blur sigma1.0 | 0.8393 | -0.0280 | 0.8411 | -0.0277 | 3.23% |
| blur sigma2.0 | 0.6586 | -0.2087 | 0.6614 | -0.2073 | 24.07% |
| translation 5pct | 0.8382 | -0.0292 | 0.8401 | -0.0286 | 3.37% |
| translation 10pct | 0.7347 | -0.1326 | 0.7354 | -0.1333 | 15.29% |
| translation 15pct | 0.5533 | -0.3140 | 0.5572 | -0.3115 | 36.20% |

## Robustness — `augmented_seed43`

Clean test accuracy 0.8469, macro F1 0.8468.

| Perturbation | Accuracy | Δ acc | Macro F1 | Δ macro F1 | Rel. acc drop |
|---|--:|--:|--:|--:|--:|
| rotation 5deg | 0.8366 | -0.0103 | 0.8368 | -0.0100 | 1.22% |
| rotation 10deg | 0.7871 | -0.0598 | 0.7886 | -0.0582 | 7.06% |
| rotation 20deg | 0.5710 | -0.2759 | 0.5728 | -0.2740 | 32.57% |
| brightness x0.6 | 0.7811 | -0.0657 | 0.7813 | -0.0654 | 7.76% |
| brightness x0.8 | 0.8409 | -0.0060 | 0.8410 | -0.0058 | 0.71% |
| brightness x1.4 | 0.8202 | -0.0267 | 0.8178 | -0.0290 | 3.15% |
| blur sigma0.5 | 0.8471 | +0.0002 | 0.8466 | -0.0002 | -0.03% |
| blur sigma1.0 | 0.8147 | -0.0322 | 0.8132 | -0.0336 | 3.80% |
| blur sigma2.0 | 0.6402 | -0.2067 | 0.6380 | -0.2088 | 24.40% |
| translation 5pct | 0.8177 | -0.0292 | 0.8184 | -0.0284 | 3.45% |
| translation 10pct | 0.7028 | -0.1441 | 0.7021 | -0.1447 | 17.02% |
| translation 15pct | 0.5267 | -0.3202 | 0.5310 | -0.3157 | 37.81% |

## Robustness — `augmented_seed44`

Clean test accuracy 0.8159, macro F1 0.8145.

| Perturbation | Accuracy | Δ acc | Macro F1 | Δ macro F1 | Rel. acc drop |
|---|--:|--:|--:|--:|--:|
| rotation 5deg | 0.7841 | -0.0317 | 0.7815 | -0.0330 | 3.89% |
| rotation 10deg | 0.7260 | -0.0899 | 0.7261 | -0.0884 | 11.02% |
| rotation 20deg | 0.4614 | -0.3545 | 0.4602 | -0.3543 | 43.45% |
| brightness x0.6 | 0.7995 | -0.0163 | 0.7978 | -0.0167 | 2.00% |
| brightness x0.8 | 0.8211 | +0.0053 | 0.8197 | +0.0052 | -0.65% |
| brightness x1.4 | 0.7761 | -0.0398 | 0.7712 | -0.0433 | 4.87% |
| blur sigma0.5 | 0.8122 | -0.0037 | 0.8110 | -0.0035 | 0.45% |
| blur sigma1.0 | 0.7982 | -0.0177 | 0.7990 | -0.0155 | 2.17% |
| blur sigma2.0 | 0.6485 | -0.1674 | 0.6533 | -0.1612 | 20.51% |
| translation 5pct | 0.7754 | -0.0405 | 0.7733 | -0.0412 | 4.96% |
| translation 10pct | 0.6674 | -0.1485 | 0.6612 | -0.1533 | 18.20% |
| translation 15pct | 0.5299 | -0.2860 | 0.5236 | -0.2909 | 35.05% |

## Robustness — `baseline_e30_seed42`

Clean test accuracy 0.9634, macro F1 0.9634.

| Perturbation | Accuracy | Δ acc | Macro F1 | Δ macro F1 | Rel. acc drop |
|---|--:|--:|--:|--:|--:|
| rotation 5deg | 0.7264 | -0.2370 | 0.7367 | -0.2267 | 24.60% |
| rotation 10deg | 0.5366 | -0.4269 | 0.5618 | -0.4016 | 44.31% |
| rotation 20deg | 0.2747 | -0.6887 | 0.2727 | -0.6907 | 71.49% |
| brightness x0.6 | 0.8000 | -0.1634 | 0.8084 | -0.1550 | 16.96% |
| brightness x0.8 | 0.9324 | -0.0310 | 0.9329 | -0.0305 | 3.22% |
| brightness x1.4 | 0.8336 | -0.1299 | 0.8333 | -0.1301 | 13.48% |
| blur sigma0.5 | 0.9501 | -0.0133 | 0.9501 | -0.0133 | 1.38% |
| blur sigma1.0 | 0.8308 | -0.1326 | 0.8299 | -0.1336 | 13.77% |
| blur sigma2.0 | 0.6568 | -0.3067 | 0.6571 | -0.3064 | 31.83% |
| translation 5pct | 0.5802 | -0.3832 | 0.6020 | -0.3614 | 39.78% |
| translation 10pct | 0.2986 | -0.6648 | 0.3164 | -0.6471 | 69.01% |
| translation 15pct | 0.1000 | -0.8634 | 0.0764 | -0.8870 | 89.62% |

## Robustness — `baseline_e30_seed43`

Clean test accuracy 0.9602, macro F1 0.9603.

| Perturbation | Accuracy | Δ acc | Macro F1 | Δ macro F1 | Rel. acc drop |
|---|--:|--:|--:|--:|--:|
| rotation 5deg | 0.6214 | -0.3389 | 0.6294 | -0.3309 | 35.29% |
| rotation 10deg | 0.4982 | -0.4621 | 0.4981 | -0.4622 | 48.12% |
| rotation 20deg | 0.1740 | -0.7862 | 0.1645 | -0.7958 | 81.88% |
| brightness x0.6 | 0.8531 | -0.1071 | 0.8550 | -0.1053 | 11.16% |
| brightness x0.8 | 0.9446 | -0.0156 | 0.9446 | -0.0157 | 1.63% |
| brightness x1.4 | 0.8547 | -0.1055 | 0.8527 | -0.1077 | 10.99% |
| blur sigma0.5 | 0.9582 | -0.0021 | 0.9583 | -0.0020 | 0.22% |
| blur sigma1.0 | 0.8464 | -0.1138 | 0.8476 | -0.1127 | 11.85% |
| blur sigma2.0 | 0.6244 | -0.3359 | 0.6277 | -0.3326 | 34.98% |
| translation 5pct | 0.4692 | -0.4910 | 0.4768 | -0.4835 | 51.14% |
| translation 10pct | 0.2651 | -0.6952 | 0.2552 | -0.7051 | 72.40% |
| translation 15pct | 0.2092 | -0.7510 | 0.1957 | -0.7647 | 78.21% |

## Robustness — `baseline_e30_seed44`

Clean test accuracy 0.9616, macro F1 0.9617.

| Perturbation | Accuracy | Δ acc | Macro F1 | Δ macro F1 | Rel. acc drop |
|---|--:|--:|--:|--:|--:|
| rotation 5deg | 0.6476 | -0.3140 | 0.6522 | -0.3096 | 32.66% |
| rotation 10deg | 0.3851 | -0.5766 | 0.3710 | -0.5907 | 59.96% |
| rotation 20deg | 0.1370 | -0.8246 | 0.1143 | -0.8474 | 85.75% |
| brightness x0.6 | 0.7480 | -0.2136 | 0.7511 | -0.2106 | 22.21% |
| brightness x0.8 | 0.9071 | -0.0545 | 0.9080 | -0.0538 | 5.67% |
| brightness x1.4 | 0.8285 | -0.1331 | 0.8277 | -0.1340 | 13.84% |
| blur sigma0.5 | 0.9526 | -0.0090 | 0.9528 | -0.0089 | 0.93% |
| blur sigma1.0 | 0.8257 | -0.1359 | 0.8266 | -0.1351 | 14.13% |
| blur sigma2.0 | 0.5377 | -0.4239 | 0.5332 | -0.4285 | 44.08% |
| translation 5pct | 0.4876 | -0.4740 | 0.5008 | -0.4609 | 49.29% |
| translation 10pct | 0.2499 | -0.7117 | 0.2287 | -0.7330 | 74.01% |
| translation 15pct | 0.1963 | -0.7653 | 0.1753 | -0.7865 | 79.58% |

## Robustness — `baseline_seed42`

Clean test accuracy 0.9501, macro F1 0.9501.

| Perturbation | Accuracy | Δ acc | Macro F1 | Δ macro F1 | Rel. acc drop |
|---|--:|--:|--:|--:|--:|
| rotation 5deg | 0.6968 | -0.2533 | 0.7088 | -0.2413 | 26.66% |
| rotation 10deg | 0.5154 | -0.4347 | 0.5446 | -0.4055 | 45.75% |
| rotation 20deg | 0.2740 | -0.6761 | 0.2815 | -0.6686 | 71.16% |
| brightness x0.6 | 0.7648 | -0.1853 | 0.7698 | -0.1803 | 19.50% |
| brightness x0.8 | 0.9216 | -0.0285 | 0.9220 | -0.0281 | 3.00% |
| brightness x1.4 | 0.8476 | -0.1025 | 0.8467 | -0.1034 | 10.79% |
| blur sigma0.5 | 0.9441 | -0.0060 | 0.9442 | -0.0059 | 0.63% |
| blur sigma1.0 | 0.8717 | -0.0784 | 0.8735 | -0.0766 | 8.25% |
| blur sigma2.0 | 0.6917 | -0.2584 | 0.6946 | -0.2555 | 27.20% |
| translation 5pct | 0.5221 | -0.4280 | 0.5460 | -0.4041 | 45.05% |
| translation 10pct | 0.2338 | -0.7163 | 0.2359 | -0.7142 | 75.39% |
| translation 15pct | 0.1030 | -0.8471 | 0.0763 | -0.8738 | 89.16% |

## Robustness — `baseline_seed43`

Clean test accuracy 0.9193, macro F1 0.9192.

| Perturbation | Accuracy | Δ acc | Macro F1 | Δ macro F1 | Rel. acc drop |
|---|--:|--:|--:|--:|--:|
| rotation 5deg | 0.6416 | -0.2777 | 0.6570 | -0.2622 | 30.21% |
| rotation 10deg | 0.4754 | -0.4439 | 0.4774 | -0.4419 | 48.29% |
| rotation 20deg | 0.1729 | -0.7464 | 0.1612 | -0.7580 | 81.20% |
| brightness x0.6 | 0.8471 | -0.0722 | 0.8498 | -0.0694 | 7.85% |
| brightness x0.8 | 0.9087 | -0.0106 | 0.9088 | -0.0105 | 1.15% |
| brightness x1.4 | 0.8234 | -0.0959 | 0.8220 | -0.0972 | 10.43% |
| blur sigma0.5 | 0.9170 | -0.0023 | 0.9171 | -0.0021 | 0.25% |
| blur sigma1.0 | 0.8260 | -0.0933 | 0.8286 | -0.0907 | 10.15% |
| blur sigma2.0 | 0.6184 | -0.3009 | 0.6234 | -0.2958 | 32.73% |
| translation 5pct | 0.4428 | -0.4766 | 0.4390 | -0.4803 | 51.84% |
| translation 10pct | 0.2922 | -0.6271 | 0.2845 | -0.6347 | 68.22% |
| translation 15pct | 0.1901 | -0.7292 | 0.1780 | -0.7413 | 79.32% |

## Robustness — `baseline_seed44`

Clean test accuracy 0.9246, macro F1 0.9244.

| Perturbation | Accuracy | Δ acc | Macro F1 | Δ macro F1 | Rel. acc drop |
|---|--:|--:|--:|--:|--:|
| rotation 5deg | 0.6354 | -0.2892 | 0.6483 | -0.2762 | 31.28% |
| rotation 10deg | 0.4106 | -0.5140 | 0.4101 | -0.5143 | 55.59% |
| rotation 20deg | 0.1503 | -0.7743 | 0.1207 | -0.8037 | 83.74% |
| brightness x0.6 | 0.7372 | -0.1874 | 0.7384 | -0.1861 | 20.26% |
| brightness x0.8 | 0.8775 | -0.0471 | 0.8770 | -0.0474 | 5.10% |
| brightness x1.4 | 0.8244 | -0.1002 | 0.8282 | -0.0963 | 10.84% |
| blur sigma0.5 | 0.9205 | -0.0041 | 0.9206 | -0.0038 | 0.45% |
| blur sigma1.0 | 0.7837 | -0.1409 | 0.7894 | -0.1350 | 15.24% |
| blur sigma2.0 | 0.5129 | -0.4117 | 0.5138 | -0.4107 | 44.53% |
| translation 5pct | 0.4929 | -0.4317 | 0.5038 | -0.4207 | 46.69% |
| translation 10pct | 0.2276 | -0.6970 | 0.2251 | -0.6993 | 75.39% |
| translation 15pct | 0.1393 | -0.7853 | 0.1227 | -0.8017 | 84.93% |

## Error analysis — `aslnet_relu_aug_seed42`

344 errors out of 4350 test images (7.91% error rate).

* Top-1 confusion pairs account for **9.9%** of all errors (34/344).
* Top-3 confusion pairs account for **20.9%** of all errors (72/344).
* Top-5 confusion pairs account for **27.0%** of all errors (93/344).
* Top-10 confusion pairs account for **38.1%** of all errors (131/344).

| True | Predicted | Count | % of all errors | % of class support |
|---|---|--:|--:|--:|
| R | U | 34 | 9.9% | 22.7% |
| K | V | 21 | 6.1% | 14.0% |
| M | N | 17 | 4.9% | 11.3% |
| Y | T | 11 | 3.2% | 7.3% |
| I | E | 10 | 2.9% | 6.7% |
| A | S | 8 | 2.3% | 5.3% |
| W | V | 8 | 2.3% | 5.3% |
| K | W | 8 | 2.3% | 5.3% |
| V | U | 7 | 2.0% | 4.7% |
| U | R | 7 | 2.0% | 4.7% |

Lowest-recall classes:

| Class | Support | Recall | Precision | Errors |
|---|--:|--:|--:|--:|
| R | 150 | 0.7533 | 0.8626 | 37 |
| K | 150 | 0.7800 | 0.9669 | 33 |
| U | 150 | 0.8267 | 0.7209 | 26 |
| M | 150 | 0.8333 | 0.9766 | 25 |
| V | 150 | 0.8533 | 0.7758 | 22 |
| E | 150 | 0.8600 | 0.8958 | 21 |
| I | 150 | 0.8600 | 1.0000 | 21 |
| A | 150 | 0.8867 | 0.9301 | 17 |

## Error analysis — `aslnet_relu_aug_seed43`

1087 errors out of 4350 test images (24.99% error rate).

* Top-1 confusion pairs account for **6.1%** of all errors (66/1087).
* Top-3 confusion pairs account for **12.1%** of all errors (131/1087).
* Top-5 confusion pairs account for **17.3%** of all errors (188/1087).
* Top-10 confusion pairs account for **26.4%** of all errors (287/1087).

| True | Predicted | Count | % of all errors | % of class support |
|---|---|--:|--:|--:|
| U | R | 66 | 6.1% | 44.0% |
| V | R | 34 | 3.1% | 22.7% |
| W | R | 31 | 2.9% | 20.7% |
| U | V | 30 | 2.8% | 20.0% |
| G | H | 27 | 2.5% | 18.0% |
| X | R | 22 | 2.0% | 14.7% |
| R | K | 21 | 1.9% | 14.0% |
| N | M | 20 | 1.8% | 13.3% |
| X | S | 19 | 1.7% | 12.7% |
| U | K | 17 | 1.6% | 11.3% |

Lowest-recall classes:

| Class | Support | Recall | Precision | Errors |
|---|--:|--:|--:|--:|
| U | 150 | 0.1000 | 0.4545 | 135 |
| X | 150 | 0.4200 | 0.6495 | 87 |
| W | 150 | 0.4467 | 0.4855 | 83 |
| V | 150 | 0.5200 | 0.4535 | 72 |
| E | 150 | 0.5667 | 0.7456 | 65 |
| N | 150 | 0.6200 | 0.7881 | 57 |
| S | 150 | 0.6267 | 0.5562 | 56 |
| T | 150 | 0.6400 | 0.6809 | 54 |

## Error analysis — `aslnet_relu_aug_seed44`

311 errors out of 4350 test images (7.15% error rate).

* Top-1 confusion pairs account for **6.1%** of all errors (19/311).
* Top-3 confusion pairs account for **16.7%** of all errors (52/311).
* Top-5 confusion pairs account for **25.1%** of all errors (78/311).
* Top-10 confusion pairs account for **41.2%** of all errors (128/311).

| True | Predicted | Count | % of all errors | % of class support |
|---|---|--:|--:|--:|
| U | R | 19 | 6.1% | 12.7% |
| N | M | 18 | 5.8% | 12.0% |
| K | R | 15 | 4.8% | 10.0% |
| E | S | 13 | 4.2% | 8.7% |
| V | K | 13 | 4.2% | 8.7% |
| U | X | 11 | 3.5% | 7.3% |
| O | S | 10 | 3.2% | 6.7% |
| W | V | 10 | 3.2% | 6.7% |
| R | U | 10 | 3.2% | 6.7% |
| A | S | 9 | 2.9% | 6.0% |

Lowest-recall classes:

| Class | Support | Recall | Precision | Errors |
|---|--:|--:|--:|--:|
| U | 150 | 0.7200 | 0.8571 | 42 |
| V | 150 | 0.7667 | 0.8456 | 35 |
| K | 150 | 0.8600 | 0.9085 | 21 |
| W | 150 | 0.8600 | 0.9556 | 21 |
| N | 150 | 0.8667 | 0.9155 | 20 |
| E | 150 | 0.8800 | 0.8742 | 18 |
| O | 150 | 0.9000 | 0.9926 | 15 |
| A | 150 | 0.9133 | 0.9786 | 13 |

## Error analysis — `aslnet_relu_seed42`

117 errors out of 4350 test images (2.69% error rate).

* Top-1 confusion pairs account for **9.4%** of all errors (11/117).
* Top-3 confusion pairs account for **21.4%** of all errors (25/117).
* Top-5 confusion pairs account for **27.4%** of all errors (32/117).
* Top-10 confusion pairs account for **40.2%** of all errors (47/117).

| True | Predicted | Count | % of all errors | % of class support |
|---|---|--:|--:|--:|
| V | W | 11 | 9.4% | 7.3% |
| U | R | 7 | 6.0% | 4.7% |
| G | H | 7 | 6.0% | 4.7% |
| W | K | 4 | 3.4% | 2.7% |
| S | T | 3 | 2.6% | 2.0% |
| M | N | 3 | 2.6% | 2.0% |
| Y | T | 3 | 2.6% | 2.0% |
| D | O | 3 | 2.6% | 2.0% |
| W | V | 3 | 2.6% | 2.0% |
| U | B | 3 | 2.6% | 2.0% |

Lowest-recall classes:

| Class | Support | Recall | Precision | Errors |
|---|--:|--:|--:|--:|
| U | 150 | 0.8800 | 0.9496 | 18 |
| V | 150 | 0.8867 | 0.9638 | 17 |
| W | 150 | 0.9267 | 0.9145 | 11 |
| G | 150 | 0.9533 | 0.9862 | 7 |
| I | 150 | 0.9533 | 0.9795 | 7 |
| X | 150 | 0.9533 | 0.9662 | 7 |
| M | 150 | 0.9600 | 0.9863 | 6 |
| Y | 150 | 0.9600 | 0.9863 | 6 |

## Error analysis — `aslnet_relu_seed43`

195 errors out of 4350 test images (4.48% error rate).

* Top-1 confusion pairs account for **14.4%** of all errors (28/195).
* Top-3 confusion pairs account for **26.7%** of all errors (52/195).
* Top-5 confusion pairs account for **32.3%** of all errors (63/195).
* Top-10 confusion pairs account for **44.6%** of all errors (87/195).

| True | Predicted | Count | % of all errors | % of class support |
|---|---|--:|--:|--:|
| U | R | 28 | 14.4% | 18.7% |
| V | W | 13 | 6.7% | 8.7% |
| M | N | 11 | 5.6% | 7.3% |
| U | V | 6 | 3.1% | 4.0% |
| E | B | 5 | 2.6% | 3.3% |
| Q | P | 5 | 2.6% | 3.3% |
| W | V | 5 | 2.6% | 3.3% |
| R | U | 5 | 2.6% | 3.3% |
| Y | T | 5 | 2.6% | 3.3% |
| S | Y | 4 | 2.1% | 2.7% |

Lowest-recall classes:

| Class | Support | Recall | Precision | Errors |
|---|--:|--:|--:|--:|
| U | 150 | 0.7133 | 0.9304 | 43 |
| V | 150 | 0.8600 | 0.8958 | 21 |
| S | 150 | 0.9000 | 0.9247 | 15 |
| M | 150 | 0.9133 | 0.9514 | 13 |
| R | 150 | 0.9200 | 0.8023 | 12 |
| E | 150 | 0.9267 | 0.9653 | 11 |
| X | 150 | 0.9267 | 0.9392 | 11 |
| Y | 150 | 0.9533 | 0.9470 | 7 |

## Error analysis — `aslnet_relu_seed44`

122 errors out of 4350 test images (2.80% error rate).

* Top-1 confusion pairs account for **5.7%** of all errors (7/122).
* Top-3 confusion pairs account for **16.4%** of all errors (20/122).
* Top-5 confusion pairs account for **26.2%** of all errors (32/122).
* Top-10 confusion pairs account for **43.4%** of all errors (53/122).

| True | Predicted | Count | % of all errors | % of class support |
|---|---|--:|--:|--:|
| V | U | 7 | 5.7% | 4.7% |
| U | V | 7 | 5.7% | 4.7% |
| W | V | 6 | 4.9% | 4.0% |
| R | U | 6 | 4.9% | 4.0% |
| U | R | 6 | 4.9% | 4.0% |
| M | N | 5 | 4.1% | 3.3% |
| G | H | 5 | 4.1% | 3.3% |
| S | T | 4 | 3.3% | 2.7% |
| R | V | 4 | 3.3% | 2.7% |
| U | X | 3 | 2.5% | 2.0% |

Lowest-recall classes:

| Class | Support | Recall | Precision | Errors |
|---|--:|--:|--:|--:|
| U | 150 | 0.8667 | 0.8904 | 20 |
| R | 150 | 0.9200 | 0.9517 | 12 |
| V | 150 | 0.9200 | 0.8679 | 12 |
| S | 150 | 0.9400 | 0.9527 | 9 |
| E | 150 | 0.9533 | 0.9862 | 7 |
| M | 150 | 0.9533 | 0.9795 | 7 |
| W | 150 | 0.9533 | 0.9533 | 7 |
| G | 150 | 0.9600 | 0.9863 | 6 |

## Error analysis — `augmented_e30_seed42`

286 errors out of 4350 test images (6.57% error rate).

* Top-1 confusion pairs account for **7.7%** of all errors (22/286).
* Top-3 confusion pairs account for **20.6%** of all errors (59/286).
* Top-5 confusion pairs account for **31.8%** of all errors (91/286).
* Top-10 confusion pairs account for **49.0%** of all errors (140/286).

| True | Predicted | Count | % of all errors | % of class support |
|---|---|--:|--:|--:|
| G | H | 22 | 7.7% | 14.7% |
| R | U | 19 | 6.6% | 12.7% |
| U | R | 18 | 6.3% | 12.0% |
| N | M | 17 | 5.9% | 11.3% |
| W | V | 15 | 5.2% | 10.0% |
| K | V | 14 | 4.9% | 9.3% |
| V | W | 11 | 3.8% | 7.3% |
| X | U | 9 | 3.1% | 6.0% |
| M | N | 8 | 2.8% | 5.3% |
| V | U | 7 | 2.4% | 4.7% |

Lowest-recall classes:

| Class | Support | Recall | Precision | Errors |
|---|--:|--:|--:|--:|
| U | 150 | 0.8333 | 0.7485 | 25 |
| W | 150 | 0.8333 | 0.8993 | 25 |
| G | 150 | 0.8467 | 0.9769 | 23 |
| V | 150 | 0.8467 | 0.7791 | 23 |
| X | 150 | 0.8467 | 0.9137 | 23 |
| K | 150 | 0.8733 | 0.9776 | 19 |
| N | 150 | 0.8733 | 0.9225 | 19 |
| R | 150 | 0.8733 | 0.8086 | 19 |

## Error analysis — `augmented_seed42`

577 errors out of 4350 test images (13.26% error rate).

* Top-1 confusion pairs account for **6.2%** of all errors (36/577).
* Top-3 confusion pairs account for **15.3%** of all errors (88/577).
* Top-5 confusion pairs account for **21.5%** of all errors (124/577).
* Top-10 confusion pairs account for **33.4%** of all errors (193/577).

| True | Predicted | Count | % of all errors | % of class support |
|---|---|--:|--:|--:|
| R | U | 36 | 6.2% | 24.0% |
| N | M | 29 | 5.0% | 19.3% |
| G | H | 23 | 4.0% | 15.3% |
| W | V | 20 | 3.5% | 13.3% |
| S | X | 16 | 2.8% | 10.7% |
| A | B | 15 | 2.6% | 10.0% |
| V | U | 14 | 2.4% | 9.3% |
| U | R | 14 | 2.4% | 9.3% |
| V | W | 13 | 2.3% | 8.7% |
| A | S | 13 | 2.3% | 8.7% |

Lowest-recall classes:

| Class | Support | Recall | Precision | Errors |
|---|--:|--:|--:|--:|
| V | 150 | 0.6867 | 0.7410 | 47 |
| R | 150 | 0.7133 | 0.7810 | 43 |
| N | 150 | 0.7400 | 0.8952 | 39 |
| W | 150 | 0.7400 | 0.8222 | 39 |
| A | 150 | 0.7467 | 0.9032 | 38 |
| G | 150 | 0.7467 | 0.9573 | 38 |
| S | 150 | 0.8000 | 0.6857 | 30 |
| X | 150 | 0.8067 | 0.7076 | 29 |

## Error analysis — `augmented_seed43`

666 errors out of 4350 test images (15.31% error rate).

* Top-1 confusion pairs account for **6.8%** of all errors (45/666).
* Top-3 confusion pairs account for **16.5%** of all errors (110/666).
* Top-5 confusion pairs account for **24.0%** of all errors (160/666).
* Top-10 confusion pairs account for **36.0%** of all errors (240/666).

| True | Predicted | Count | % of all errors | % of class support |
|---|---|--:|--:|--:|
| R | U | 45 | 6.8% | 30.0% |
| M | N | 34 | 5.1% | 22.7% |
| S | X | 31 | 4.7% | 20.7% |
| V | W | 25 | 3.8% | 16.7% |
| W | V | 25 | 3.8% | 16.7% |
| S | T | 19 | 2.9% | 12.7% |
| K | W | 19 | 2.9% | 12.7% |
| E | A | 16 | 2.4% | 10.7% |
| U | X | 15 | 2.3% | 10.0% |
| Q | P | 11 | 1.7% | 7.3% |

Lowest-recall classes:

| Class | Support | Recall | Precision | Errors |
|---|--:|--:|--:|--:|
| S | 150 | 0.5400 | 0.7941 | 69 |
| R | 150 | 0.5533 | 0.9222 | 67 |
| M | 150 | 0.5867 | 0.8544 | 62 |
| V | 150 | 0.7133 | 0.6859 | 43 |
| U | 150 | 0.7333 | 0.6627 | 40 |
| W | 150 | 0.7667 | 0.6216 | 35 |
| X | 150 | 0.7800 | 0.5707 | 33 |
| K | 150 | 0.7933 | 0.9444 | 31 |

## Error analysis — `augmented_seed44`

801 errors out of 4350 test images (18.41% error rate).

* Top-1 confusion pairs account for **5.6%** of all errors (45/801).
* Top-3 confusion pairs account for **12.9%** of all errors (103/801).
* Top-5 confusion pairs account for **18.0%** of all errors (144/801).
* Top-10 confusion pairs account for **28.2%** of all errors (226/801).

| True | Predicted | Count | % of all errors | % of class support |
|---|---|--:|--:|--:|
| U | R | 45 | 5.6% | 30.0% |
| V | W | 36 | 4.5% | 24.0% |
| M | N | 22 | 2.7% | 14.7% |
| Y | space | 21 | 2.6% | 14.0% |
| N | M | 20 | 2.5% | 13.3% |
| V | K | 17 | 2.1% | 11.3% |
| O | D | 17 | 2.1% | 11.3% |
| G | H | 17 | 2.1% | 11.3% |
| M | S | 16 | 2.0% | 10.7% |
| R | U | 15 | 1.9% | 10.0% |

Lowest-recall classes:

| Class | Support | Recall | Precision | Errors |
|---|--:|--:|--:|--:|
| U | 150 | 0.4467 | 0.6634 | 83 |
| V | 150 | 0.4867 | 0.6822 | 77 |
| M | 150 | 0.6333 | 0.7252 | 55 |
| W | 150 | 0.6933 | 0.6797 | 46 |
| T | 150 | 0.7133 | 0.9640 | 43 |
| N | 150 | 0.7333 | 0.7190 | 40 |
| X | 150 | 0.7467 | 0.7724 | 38 |
| E | 150 | 0.7600 | 0.7550 | 36 |

## Error analysis — `baseline_e30_seed42`

159 errors out of 4350 test images (3.66% error rate).

* Top-1 confusion pairs account for **7.5%** of all errors (12/159).
* Top-3 confusion pairs account for **18.9%** of all errors (30/159).
* Top-5 confusion pairs account for **28.9%** of all errors (46/159).
* Top-10 confusion pairs account for **43.4%** of all errors (69/159).

| True | Predicted | Count | % of all errors | % of class support |
|---|---|--:|--:|--:|
| V | W | 12 | 7.5% | 8.0% |
| V | U | 9 | 5.7% | 6.0% |
| U | R | 9 | 5.7% | 6.0% |
| T | L | 8 | 5.0% | 5.3% |
| W | V | 8 | 5.0% | 5.3% |
| G | H | 6 | 3.8% | 4.0% |
| N | M | 5 | 3.1% | 3.3% |
| O | D | 4 | 2.5% | 2.7% |
| X | R | 4 | 2.5% | 2.7% |
| I | K | 4 | 2.5% | 2.7% |

Lowest-recall classes:

| Class | Support | Recall | Precision | Errors |
|---|--:|--:|--:|--:|
| V | 150 | 0.8467 | 0.9203 | 23 |
| U | 150 | 0.9067 | 0.8947 | 14 |
| W | 150 | 0.9200 | 0.9200 | 12 |
| T | 150 | 0.9333 | 0.9655 | 10 |
| I | 150 | 0.9400 | 0.9658 | 9 |
| N | 150 | 0.9467 | 0.9660 | 8 |
| X | 150 | 0.9467 | 0.9660 | 8 |
| M | 150 | 0.9533 | 0.9226 | 7 |

## Error analysis — `baseline_e30_seed43`

173 errors out of 4350 test images (3.98% error rate).

* Top-1 confusion pairs account for **9.8%** of all errors (17/173).
* Top-3 confusion pairs account for **22.5%** of all errors (39/173).
* Top-5 confusion pairs account for **32.4%** of all errors (56/173).
* Top-10 confusion pairs account for **46.2%** of all errors (80/173).

| True | Predicted | Count | % of all errors | % of class support |
|---|---|--:|--:|--:|
| W | V | 17 | 9.8% | 11.3% |
| G | H | 13 | 7.5% | 8.7% |
| V | W | 9 | 5.2% | 6.0% |
| R | U | 9 | 5.2% | 6.0% |
| U | V | 8 | 4.6% | 5.3% |
| M | N | 7 | 4.0% | 4.7% |
| U | W | 5 | 2.9% | 3.3% |
| F | D | 5 | 2.9% | 3.3% |
| V | U | 4 | 2.3% | 2.7% |
| X | U | 3 | 1.7% | 2.0% |

Lowest-recall classes:

| Class | Support | Recall | Precision | Errors |
|---|--:|--:|--:|--:|
| W | 150 | 0.8733 | 0.8851 | 19 |
| G | 150 | 0.8933 | 0.9710 | 16 |
| U | 150 | 0.8933 | 0.8701 | 16 |
| V | 150 | 0.9000 | 0.8133 | 15 |
| F | 150 | 0.9200 | 1.0000 | 12 |
| M | 150 | 0.9267 | 0.9653 | 11 |
| R | 150 | 0.9267 | 0.9858 | 11 |
| X | 150 | 0.9267 | 0.9653 | 11 |

## Error analysis — `baseline_e30_seed44`

167 errors out of 4350 test images (3.84% error rate).

* Top-1 confusion pairs account for **5.4%** of all errors (9/167).
* Top-3 confusion pairs account for **14.4%** of all errors (24/167).
* Top-5 confusion pairs account for **21.6%** of all errors (36/167).
* Top-10 confusion pairs account for **37.1%** of all errors (62/167).

| True | Predicted | Count | % of all errors | % of class support |
|---|---|--:|--:|--:|
| M | N | 9 | 5.4% | 6.0% |
| W | V | 8 | 4.8% | 5.3% |
| A | M | 7 | 4.2% | 4.7% |
| T | S | 6 | 3.6% | 4.0% |
| R | U | 6 | 3.6% | 4.0% |
| V | W | 6 | 3.6% | 4.0% |
| N | M | 6 | 3.6% | 4.0% |
| G | H | 5 | 3.0% | 3.3% |
| U | V | 5 | 3.0% | 3.3% |
| S | T | 4 | 2.4% | 2.7% |

Lowest-recall classes:

| Class | Support | Recall | Precision | Errors |
|---|--:|--:|--:|--:|
| T | 150 | 0.9133 | 0.9648 | 13 |
| A | 150 | 0.9200 | 0.9857 | 12 |
| U | 150 | 0.9200 | 0.9324 | 12 |
| M | 150 | 0.9267 | 0.8854 | 11 |
| R | 150 | 0.9267 | 0.9586 | 11 |
| S | 150 | 0.9333 | 0.9333 | 10 |
| V | 150 | 0.9333 | 0.8805 | 10 |
| W | 150 | 0.9333 | 0.9211 | 10 |

## Error analysis — `baseline_seed42`

217 errors out of 4350 test images (4.99% error rate).

* Top-1 confusion pairs account for **8.8%** of all errors (19/217).
* Top-3 confusion pairs account for **19.4%** of all errors (42/217).
* Top-5 confusion pairs account for **27.6%** of all errors (60/217).
* Top-10 confusion pairs account for **39.2%** of all errors (85/217).

| True | Predicted | Count | % of all errors | % of class support |
|---|---|--:|--:|--:|
| H | G | 19 | 8.8% | 12.7% |
| V | U | 13 | 6.0% | 8.7% |
| M | N | 10 | 4.6% | 6.7% |
| R | U | 9 | 4.1% | 6.0% |
| V | W | 9 | 4.1% | 6.0% |
| U | R | 7 | 3.2% | 4.7% |
| W | V | 6 | 2.8% | 4.0% |
| space | Y | 4 | 1.8% | 2.7% |
| G | H | 4 | 1.8% | 2.7% |
| P | Q | 4 | 1.8% | 2.7% |

Lowest-recall classes:

| Class | Support | Recall | Precision | Errors |
|---|--:|--:|--:|--:|
| V | 150 | 0.8333 | 0.9259 | 25 |
| H | 150 | 0.8600 | 0.9556 | 21 |
| M | 150 | 0.8733 | 0.9850 | 19 |
| W | 150 | 0.8933 | 0.9241 | 16 |
| U | 150 | 0.9000 | 0.8385 | 15 |
| R | 150 | 0.9200 | 0.9262 | 12 |
| I | 150 | 0.9333 | 0.9929 | 10 |
| O | 150 | 0.9333 | 0.9459 | 10 |

## Error analysis — `baseline_seed43`

351 errors out of 4350 test images (8.07% error rate).

* Top-1 confusion pairs account for **6.6%** of all errors (23/351).
* Top-3 confusion pairs account for **17.9%** of all errors (63/351).
* Top-5 confusion pairs account for **23.1%** of all errors (81/351).
* Top-10 confusion pairs account for **32.2%** of all errors (113/351).

| True | Predicted | Count | % of all errors | % of class support |
|---|---|--:|--:|--:|
| U | V | 23 | 6.6% | 15.3% |
| M | N | 20 | 5.7% | 13.3% |
| V | W | 20 | 5.7% | 13.3% |
| H | G | 9 | 2.6% | 6.0% |
| G | H | 9 | 2.6% | 6.0% |
| Q | P | 8 | 2.3% | 5.3% |
| E | B | 6 | 1.7% | 4.0% |
| T | S | 6 | 1.7% | 4.0% |
| O | N | 6 | 1.7% | 4.0% |
| U | R | 6 | 1.7% | 4.0% |

Lowest-recall classes:

| Class | Support | Recall | Precision | Errors |
|---|--:|--:|--:|--:|
| U | 150 | 0.7600 | 0.9048 | 36 |
| M | 150 | 0.7800 | 0.9000 | 33 |
| V | 150 | 0.8067 | 0.7516 | 29 |
| X | 150 | 0.8600 | 0.9021 | 21 |
| O | 150 | 0.8733 | 0.9357 | 19 |
| R | 150 | 0.8733 | 0.8851 | 19 |
| W | 150 | 0.8800 | 0.8098 | 18 |
| E | 150 | 0.8933 | 0.8874 | 16 |

## Error analysis — `baseline_seed44`

328 errors out of 4350 test images (7.54% error rate).

* Top-1 confusion pairs account for **3.7%** of all errors (12/328).
* Top-3 confusion pairs account for **11.0%** of all errors (36/328).
* Top-5 confusion pairs account for **17.4%** of all errors (57/328).
* Top-10 confusion pairs account for **29.3%** of all errors (96/328).

| True | Predicted | Count | % of all errors | % of class support |
|---|---|--:|--:|--:|
| U | V | 12 | 3.7% | 8.0% |
| U | R | 12 | 3.7% | 8.0% |
| E | B | 12 | 3.7% | 8.0% |
| V | W | 11 | 3.4% | 7.3% |
| T | S | 10 | 3.0% | 6.7% |
| T | X | 9 | 2.7% | 6.0% |
| Q | P | 8 | 2.4% | 5.3% |
| R | U | 8 | 2.4% | 5.3% |
| W | V | 7 | 2.1% | 4.7% |
| M | N | 7 | 2.1% | 4.7% |

Lowest-recall classes:

| Class | Support | Recall | Precision | Errors |
|---|--:|--:|--:|--:|
| T | 150 | 0.7667 | 0.9200 | 35 |
| U | 150 | 0.7800 | 0.8667 | 33 |
| E | 150 | 0.8267 | 0.9323 | 26 |
| M | 150 | 0.8667 | 0.9420 | 20 |
| R | 150 | 0.8733 | 0.8675 | 19 |
| S | 150 | 0.8733 | 0.8618 | 19 |
| V | 150 | 0.8733 | 0.7988 | 19 |
| X | 150 | 0.8733 | 0.8239 | 19 |
