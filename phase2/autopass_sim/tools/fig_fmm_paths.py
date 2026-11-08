"""fig_fmm_paths.png: v2-A on episode seed 5081 with A* paths vs FMM paths (simulated).

Same renderer and layout as the rule-vs-v2-A replay figure (render2d.Panel, final frame), one panel per
path_mode. Same trained agent (v2-A training seed 0, mask shield), same world and sensor seed.

    python tools/fig_fmm_paths.py --out results_rl/fmm/fig_fmm_paths.png
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from autopass_sim import render2d
from tools.replay_rl import record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=5081)
    ap.add_argument("--agent", default="results_rl/v2/A_mask_budget_s0/agent.pt")
    ap.add_argument("--out", default="results_rl/fmm/fig_fmm_paths.png")
    a = ap.parse_args()
    runs = []
    for mode, name in (("astar", "A* (8-connected + shortcut)"), ("fmm", "FMM (fast marching, gradient descent)")):
        env, res, frames, ev, info = record(a.seed, a.agent, path_mode=mode)
        title = f"v2-A, {name}: {res.outcome}, {res.time:.1f} s, {info['path_len']:.2f} m, {info['scans']} sweeps"
        runs.append((title, env, res, frames, ev))
        print(title)
    env = runs[0][1]
    world, model, w_A = env.world, env.model, env.w_A
    h = 12.8 / 2 * world.H / world.W + 1.3
    fig, axes = plt.subplots(1, 2, figsize=(12.8, h), dpi=150)
    fig.patch.set_facecolor("#FBFBF8")
    for ax, (title, _, res, frames, ev) in zip(axes, runs):
        p = render2d.Panel(ax, world, model, title, res, frames, ev, w_A)
        p.update(len(p.frames) - 1)
        ax.set_title(title, fontsize=9.5, color=render2d.INK, loc="left")
    fig.text(0.01, 1 - 0.12 / h, f"AutoPass Report 3: simulated, same v2-A policy with A* vs FMM paths, main setting, "
             f"episode seed {a.seed}", fontsize=12, fontweight="bold", color=render2d.INK, va="top")
    fig.text(0.01, 1 - 0.38 / h, "Same world, sensor seed and trained agent (training seed 0, mask shield). Only the path "
             f"planner differs. Agent: wheelchair, needs {w_A:.1f} cm.", fontsize=8.5, color=render2d.GRAY, va="top")
    fig.subplots_adjust(left=0.01, right=0.99, top=1 - 0.95 / h, bottom=0.45 / h, wspace=0.04)
    fig.savefig(a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
