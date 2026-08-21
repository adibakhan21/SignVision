#!/bin/bash
# Longer-budget replication: identical arms at 30 epochs instead of 10.
set -e
cd /Users/dhruvgaur/Desktop/claude/SignVision
export PYTHONPATH=src
PY=../.venv/bin/python
for cfg in baseline_e30 augmented_e30 aslnet_relu_e30 aslnet_relu_aug_e30; do
  for seed in 42 43 44; do
    [ -f "results/${cfg}_seed${seed}/results.json" ] && { echo "skip ${cfg} ${seed}"; continue; }
    echo "===== ${cfg} seed ${seed} ====="
    $PY experiments/01_train.py --config configs/${cfg}.json --seed $seed 2>&1 | grep -E "^\[done\] [a-z]"
    $PY experiments/03_robustness.py --run ${cfg}_seed${seed} >/dev/null 2>&1
    $PY experiments/02_error_analysis.py --run ${cfg}_seed${seed} >/dev/null 2>&1
  done
done
echo "E30 GRID COMPLETE"
