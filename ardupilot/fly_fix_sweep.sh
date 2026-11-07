#!/usr/bin/env bash
# Fly the fix sweep (see fix_sweep.py): 4 setups x 3 plans, unattended, at 5x. About 45 minutes.
# Run in WSL; close any running SITL first.
source ~/venv-ardupilot/bin/activate
ROOT=/mnt/c/Users/isaia/fixed-wing-rl
for b in fixS1 fixS2 fixS3 fixS4; do
  echo "=================== $b ==================="
  python3 $ROOT/ardupilot/run_batch.py $ROOT/localstore/batches/$b/manifest.json \
      --only tier3_easy_envs_0 tier3_nominal_envs_2 tier3_dense_envs_1 --speedup 5
done
echo "All done. Now in PyCharm:  python -m ardupilot.fix_sweep analyze"
