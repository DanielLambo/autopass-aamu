"""Non-learned policies for sanity checks and as in-env baselines."""
import numpy as np
from .env import K


def random_policy(env, info):
    """Uniform over valid actions (uses the mask)."""
    return int(env.np_random.choice(np.flatnonzero(info["action_mask"])))


class RulePolicy:
    """Report 2's decision rule driven inside the env (roughly B2 with a square-on viewpoint):
    take doors in order of route length; go to the square-on commit-point viewpoint;
    dwell until the z-test decides (max `dwell_max` scans); PASS -> COMMIT, else REJECT.
    """

    def __init__(self, dwell_max=20, viewpoint=0):
        self.dwell_max, self.vp = dwell_max, viewpoint

    def __call__(self, env, info):
        mask = info["action_mask"]
        open_doors = [i for i in range(len(env.doors)) if env.doors[i].id not in env.rejected]
        i = min(open_doors, key=lambda j: env.route[j][1])
        dec = env.ztest(i)
        if dec == "PASS" and mask[env.A_COMMIT + i]:
            return env.A_COMMIT + i
        if dec == "FAIL":
            return env.A_REJECT + i
        a_go = i * K + self.vp
        target = env.vp[i, self.vp]
        if target is not None and np.linalg.norm(target - env.pos) > 0.05 and mask[a_go]:
            self._dwell = 0
            return a_go
        self._dwell = getattr(self, "_dwell", 0) + 1
        return env.A_DWELL if self._dwell <= self.dwell_max else env.A_REJECT + i


def rollout(env, policy, seed, options=None):
    obs, info = env.reset(seed=seed, options=options)
    if hasattr(policy, "_dwell"):
        policy._dwell = 0
    R = C = 0.0
    traj = []
    while True:
        a = policy(env, info)
        obs, r, term, trunc, info = env.step(a)
        R += r; C += info["cost"]
        traj.append((a, round(r, 6), info["cost"]))
        if term or trunc:
            return dict(ret=R, cost=C, outcome=info["outcome"], steps=len(traj), path_len=info["path_len"],
                        sim_time=info["sim_time"], scans=info["scans"], shield_blocks=info["shield_blocks"],
                        traj=traj, final_obs=obs)
