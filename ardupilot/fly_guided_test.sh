#!/usr/bin/env bash
# Fly the GUIDED follower test (see guided_test.py). Round 3: dense_1 only, 3 setups. About 10 minutes.
# Runs at --speedup 3 (not 5) so the follower's 20 Hz stream keeps up comfortably.
# Run in WSL; close any running SITL first.
source ~/venv-ardupilot/bin/activate
ROOT=/mnt/c/Users/isaia/fixed-wing-rl
for b in k0 k1 k2; do
  echo "=================== $b ==================="
  python3 $ROOT/ardupilot/run_batch.py $ROOT/localstore/batches/$b/manifest.json \
      --only tier3_dense_envs_1 --speedup 3
done
echo "All done. Now in PyCharm:  python -m ardupilot.guided_test analyze"
