#!/bin/bash
# Full experimental grid: every arm x 3 seeds, identical protocol throughout.
set -e
cd /Users/dhruvgaur/Desktop/claude/SignVision
export PYTHONPATH=src
PY=../.venv/bin/python
for cfg in baseline augmented aslnet_relu aslnet_relu_aug; do
  for seed in 42 43 44; do
    if [ -f "results/${cfg}_seed${seed}/results.json" ]; then
      echo "== skip ${cfg} seed ${seed} (done)"; continue
    fi
    echo "===== ${cfg} seed ${seed} ====="
    $PY experiments/01_train.py --config configs/${cfg}.json --seed $seed 2>&1 | grep -E "^\[train\]|^\[done\]"
  done
done
echo "GRID COMPLETE"
