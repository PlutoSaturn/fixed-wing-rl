#!/usr/bin/env bash
# Fly the speed lead test (see speed_lead_test.py): dense_1 x 3 setups, unattended, at 5x. About 5 minutes.
# Run in WSL; close any running SITL first.
source ~/venv-ardupilot/bin/activate
ROOT=/mnt/c/Users/isaia/fixed-wing-rl
for b in spd0 spd4 spd8; do
  echo "=================== $b ==================="
  python3 $ROOT/ardupilot/run_batch.py $ROOT/localstore/batches/$b/manifest.json \
      --only tier3_dense_envs_1 --speedup 5
done
echo "All done. Now in PyCharm:  python -m ardupilot.speed_lead_test analyze"
