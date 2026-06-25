#!/usr/bin/env bash
# Rerun all RND/LPM MiniGrid arms under the shared, per-rollout IntrinsicVecWrapper
# (fixes claims 4+5). Stale rnd/lpm artifacts are archived OUT OF BAND before this
# runs (see archive_pre_shared_intrinsic_*). cell_complete() skips finished cells,
# so the round loop converges; train_one.py trains one chunk per process (reaper
# workaround) and this idle meta-loop survives the ~18-min reaper.
#
# Budgets match the original runs: DoorKey/FourRooms 1M, MultiRoom 2M, fig2
# FourRooms beta-sweep 500k. 170 cells total.
set -uo pipefail
cd "$(dirname "$0")"
JOBS="${JOBS:-12}"
CHUNK="${CHUNK:-300000}"
# run_grid resolves its own venv python for the child train_one.py processes; the
# meta-loop itself only needs to import argparse/config, so use the venv python.
PY=../LPM_exploration/.venv/bin/python
run () { PYTHONPATH=. "$PY" run_grid.py --jobs "$JOBS" --chunk-steps "$CHUNK" "$@"; }

for r in $(seq 1 14); do
  echo "============ ROUND $r  $(date) ============"
  # A) DoorKey-5x5, 8 seeds, clean + noisy@0.10, 1M
  run --envs MiniGrid-DoorKey-5x5-v0 --variants intrinsic_no_noise intrinsic_noise \
      --methods rnd lpm --seeds 1 2 3 4 5 6 7 8 --steps 1000000
  # B) FourRooms, 3 seeds, clean + noise sweep np0.01..0.10, 1M
  run --envs MiniGrid-FourRooms-v0 --variants intrinsic_no_noise intrinsic_noise \
      --methods rnd lpm --noise-probs 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10 \
      --seeds 1 2 3 --steps 1000000
  # C) MultiRoom-N6, 3 seeds, clean + noisy@0.10, MLP + LSTM arms, 2M
  run --envs MiniGrid-MultiRoom-N6-v0 --variants intrinsic_no_noise intrinsic_noise \
      --methods rnd lpm rnd_lstm lpm_lstm --seeds 1 2 3 --steps 2000000
  # D) MultiRoom-N6 beta robustness (clean), 2M
  run --envs MiniGrid-MultiRoom-N6-v0 --variants intrinsic_no_noise \
      --methods rnd lpm --betas 0.005 0.01 0.05 --seeds 1 2 3 --steps 2000000
  # E) FourRooms beta sweep for fig2, 500k
  run --envs MiniGrid-FourRooms-v0 --variants intrinsic_no_noise \
      --methods rnd lpm --betas 0.0005 0.001 0.005 0.01 0.05 --seeds 1 2 3 \
      --steps 500000 --chunk-steps 250000
done
echo "============ RERUN META-LOOP DONE  $(date) ============"
