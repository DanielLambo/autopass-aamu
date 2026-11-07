#!/usr/bin/env bash
source ~/venvs/apsim/bin/activate
cd ~/autopass/phase2/autopass_sim
for s in 0 1 2; do
  env -u PYTHONPATH PYTHONUNBUFFERED=1 python -m autopass_rl.ppo_lag --total-steps 500000 --envs 18 --seed $s --lam-lr 0.5 --shield --shield-mode mask --budget-dist --dwell-cap 20 --out results_rl/v3/A_mask_budget_cap20_s$s > results_rl/v3/A_mask_budget_cap20_s$s.log 2>&1
done
echo DONE > results_rl/v3/train.done
