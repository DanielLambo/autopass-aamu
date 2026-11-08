"""Side-by-side 2D replay in the RL env: rule vs v2-A (mask shield), one main-setting world (simulated).

Reuses autopass_sim.render2d (Panel, drawing) with frames recorded by the same mission.Tracer hook
that Report 2's replays use; the tracer has its own RNG, so recording does not change the episode.

    python tools/replay_rl.py --seed 5081 --agent results_rl/v2/A_mask_budget_s0/agent.pt \
        --out results_rl/v2/addendum/replay_rule_vs_v2A_5081.mp4 --png results_rl/v2/addendum/replay_rule_vs_v2A_5081.png
"""
import argparse
from types import SimpleNamespace
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

from autopass_sim import render2d
from autopass_sim.mission import Tracer
from autopass_sim.planners import resample
from autopass_rl.env import AutoPassEnv
from autopass_rl.evaluate import load_agent
from autopass_rl.policies import RulePolicy


def record(seed, agent_path, path_mode="astar"):
    env = AutoPassEnv(shield=agent_path is not None, shield_mode="mask", path_mode=path_mode)
    obs, info = env.reset(seed=seed)
    tr = Tracer()
    env.cfg.tracer = tr
    agent = load_agent(agent_path) if agent_path else None
    rule = RulePolicy(); rule._dwell = 0
    decisions, final_leg = [], None
    while True:
        if agent is None:
            a = rule(env, info)
        else:
            with torch.no_grad():
                lg = agent.pi(torch.as_tensor(obs[None, :agent.n_obs]))[0]
                a = int(lg.masked_fill(~torch.as_tensor(info["action_mask"]), -torch.inf).argmax())
        if env.A_COMMIT <= a < env.A_REJECT + len(env.doors):
            commit = a < env.A_REJECT
            i = a - (env.A_COMMIT if commit else env.A_REJECT)
            d = env.doors[i]
            dec = "PASS" if commit else env.ztest(i)
            m, s, b, n = env._fused(d)
            tr.event(d.id, dec)
            decisions.append((d.id, round(d.w * 100 - env.w_A, 2), dec, round(m, 2) if n else None, 0.0))
            if commit:
                final_leg = env._commit_path(i)[0]
        obs, r, term, trunc, info = env.step(a)
        if term or trunc:
            break
    env.cfg.tracer = None
    frames = list(tr.frames)
    if final_leg is not None and len(final_leg) > 1:
        pts, _ = resample(final_leg, 0.25)
        for p in pts[1:]:
            frames.append(dict(pos=p, pts=np.zeros((0, 3)), est=frames[-1]["est"] if frames else {}))
    res = SimpleNamespace(outcome=info["outcome"], time=info["sim_time"], decisions=decisions)
    return env, res, frames, tr.events, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=5081)
    ap.add_argument("--agent", default="results_rl/v2/A_mask_budget_s0/agent.pt")
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--png", required=True)
    a = ap.parse_args()
    import imageio_ffmpeg
    plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

    runs = [("Rule (z-test, square-on viewpoint, dwell ≤ 20)", record(a.seed, None)),
            ("PPO-Lag v2-A + mask shield (training seed 0)", record(a.seed, a.agent))]
    env = runs[0][1][0]
    world, model, w_A = env.world, env.model, env.w_A
    for title, (_, res, frames, ev, info) in runs:
        print(f"{title}: {res.outcome}, {res.time:.1f} s, {info['scans']} sweeps, {len(frames)} frames, decisions {res.decisions}")

    def build():
        h = 12.8 / 2 * world.H / world.W + 1.3
        fig, axes = plt.subplots(1, 2, figsize=(12.8, h), dpi=150)
        fig.patch.set_facecolor("#FBFBF8")
        panels = []
        for ax, (title, (_, res, frames, ev, info)) in zip(axes, runs):
            p = render2d.Panel(ax, world, model, title, res, frames, ev, w_A)
            ax.set_title(title, fontsize=11, color=render2d.INK, loc="left")
            panels.append(p)
        fig.text(0.01, 1 - 0.12 / h, f"AutoPass Report 3: simulated replay in the RL environment (not robot data), "
                 f"main setting, episode seed {a.seed}", fontsize=12, fontweight="bold", color=render2d.INK, va="top")
        fig.text(0.01, 1 - 0.38 / h, "Shaded: trusted viewing zone. Dots: LiDAR hits on door frames (teal = trusted, "
                 f"orange = oblique/occluded). Agent: wheelchair, needs {w_A:.1f} cm. Same world and sensor seed in both panels.",
                 fontsize=8.5, color=render2d.GRAY, va="top")
        fig.subplots_adjust(left=0.01, right=0.99, top=1 - 0.95 / h, bottom=0.45 / h, wspace=0.04)
        return fig, panels

    fig, panels = build()
    n = max(len(p.frames) for p in panels) + a.fps * 2
    anim = FuncAnimation(fig, lambda i: [p.update(i) for p in panels] and [], frames=n, blit=False)
    anim.save(a.out, writer=FFMpegWriter(fps=a.fps, bitrate=4000))
    plt.close(fig)
    print("wrote", a.out, n, "frames")

    fig, panels = build()
    for p in panels:
        p.update(len(p.frames) - 1)
    fig.savefig(a.png)
    plt.close(fig)
    print("wrote", a.png)


if __name__ == "__main__":
    main()
