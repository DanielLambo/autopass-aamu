#!/usr/bin/env bash
# Report 3 v2 training, code 2ad8d1d. A = mask shield + budget_dist, B = no shield + budget_dist.
source ~/venvs/apsim/bin/activate
cd ~/autopass/phase2/autopass_sim
for s in 0 1 2; do
  env -u PYTHONPATH PYTHONUNBUFFERED=1 python -m autopass_rl.ppo_lag --total-steps 500000 --envs 18 --seed $s --lam-lr 0.5 --shield --shield-mode mask --budget-dist --out results_rl/v2/A_mask_budget_s$s > results_rl/v2/A_mask_budget_s$s.log 2>&1
  env -u PYTHONPATH PYTHONUNBUFFERED=1 python -m autopass_rl.ppo_lag --total-steps 500000 --envs 18 --seed $s --lam-lr 0.5 --budget-dist --out results_rl/v2/B_noshield_budget_s$s > results_rl/v2/B_noshield_budget_s$s.log 2>&1
done
echo DONE > results_rl/v2/train.done
