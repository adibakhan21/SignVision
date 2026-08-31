#!/bin/bash
# Mechanism test: is the translation brittleness caused by the flatten?
#
# ASLNet-GAP is ASLNetOriginal with the 27x10x10 flatten replaced by global
# average pooling. Same conv stack, same head widths, same 30-epoch budget as
# baseline_e30 - so the contrast against baseline_e30 isolates the flatten.
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY="${PY:-python}"
for seed in 42 43 44; do
  [ -f "results/aslnet_gap_e30_seed${seed}/results.json" ] && { echo "skip gap ${seed}"; continue; }
  echo "===== aslnet_gap_e30 seed ${seed} ====="
  $PY experiments/01_train.py --config configs/aslnet_gap_e30.json --seed $seed 2>&1 | grep -E "^\[done\] [a-z]"
  $PY experiments/03_robustness.py     --run aslnet_gap_e30_seed${seed} >/dev/null 2>&1
  $PY experiments/02_error_analysis.py --run aslnet_gap_e30_seed${seed} >/dev/null 2>&1
done
echo "GAP ARM COMPLETE"
