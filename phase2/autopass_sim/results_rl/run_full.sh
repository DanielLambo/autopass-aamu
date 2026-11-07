#!/usr/bin/env bash
source ~/venvs/apsim/bin/activate
cd ~/autopass/phase2/autopass_sim
for s in 0 1 2; do
  env -u PYTHONPATH PYTHONUNBUFFERED=1 python -m autopass_rl.ppo_lag --total-steps 500000 --envs 18 --seed $s --lam-lr 0.5 --out results_rl/ppolag_s$s > results_rl/ppolag_s$s.log 2>&1
  env -u PYTHONPATH PYTHONUNBUFFERED=1 python -m autopass_rl.ppo_lag --total-steps 500000 --envs 18 --seed $s --lam-lr 0.5 --shield --out results_rl/ppolag_shield_s$s > results_rl/ppolag_shield_s$s.log 2>&1
done
echo DONE > results_rl/full_batch.done
