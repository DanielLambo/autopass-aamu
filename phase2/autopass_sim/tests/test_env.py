"""AutoPassEnv tests. Run: python -m pytest -q tests"""
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from autopass_rl.env import AutoPassEnv, SafeAutoPassEnv, N_MAX, K
from autopass_rl.policies import RulePolicy, random_policy, rollout


@pytest.fixture(scope="module")
def env():
    return AutoPassEnv()


def test_gymnasium_api(env):
    check_env(AutoPassEnv(), skip_render_check=True)


def test_obs_shape_and_finite(env):
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape and np.all(np.isfinite(obs))
    assert info["action_mask"].shape == (env.action_space.n,) and info["action_mask"][env.A_DWELL]


def test_seeded_runs_repeat_exactly(env):
    a = rollout(env, random_policy, seed=7)
    b = rollout(env, random_policy, seed=7)
    assert a["traj"] == b["traj"] and a["outcome"] == b["outcome"]
    assert np.array_equal(a["final_obs"], b["final_obs"])


def test_different_seeds_differ(env):
    env.reset(seed=1); w1 = [d.w for d in env.doors]
    env.reset(seed=2); w2 = [d.w for d in env.doors]
    assert w1 != w2


def test_cost_only_on_unsafe_commit():
    """Commit to a door narrower than w_A -> cost 1, reward unchanged by safety."""
    env = AutoPassEnv(commit_needs_trusted=False)
    for seed in range(50):
        env.reset(seed=seed, options={"w_A": 81.3})
        bad = [i for i, d in enumerate(env.doors) if d.w * 100 < env.w_A and env.action_masks()[env.A_COMMIT + i]]
        good = [i for i, d in enumerate(env.doors) if d.w * 100 >= env.w_A and env.action_masks()[env.A_COMMIT + i]]
        if bad and good:
            t0 = env.sim_time
            _, r_bad, term, _, info = env.step(env.A_COMMIT + bad[0])
            r_bad += env.time_cost * (info["sim_time"] - t0)
            assert term and info["cost"] == 1.0 and info["outcome"] == "unsafe"
            env.reset(seed=seed, options={"w_A": 81.3})
            t0 = env.sim_time
            _, r_good, term, _, info = env.step(env.A_COMMIT + good[0])
            r_good += env.time_cost * (info["sim_time"] - t0)
            assert term and info["cost"] == 0.0 and info["outcome"] == "success"
            assert abs(r_bad - r_good) < 1e-9                      # safety lives in the cost, not the reward
            return
    pytest.skip("no seed with both a visible narrow and a visible wide door at reset")


def test_commit_masked_without_trusted_view(env):
    env.reset(seed=3)
    for i, d in enumerate(env.doors):
        if env._fused(d)[3] == 0:
            assert not env.action_masks()[env.A_COMMIT + i]


def test_rejected_door_is_masked(env):
    env.reset(seed=4)
    env.step(env.A_REJECT + 0)
    m = env.action_masks()
    assert not m[env.A_COMMIT] and not m[env.A_REJECT] and not m[0:K].any()


def test_invalid_action_becomes_dwell(env):
    env.reset(seed=5)
    env.step(env.A_REJECT + 0)
    scans = env.scans
    _, _, term, _, info = env.step(env.A_COMMIT + 0)   # masked -> DWELL
    assert not term and info["scans"] == scans + 1


def test_shield_blocks_unconfirmed_commit():
    """post mode (Report 3 v1): a non-PASS COMMIT is valid but blocked and costs a DWELL."""
    e = AutoPassEnv(shield=True, shield_mode="post", commit_needs_trusted=False)
    e.reset(seed=6)
    i = next(i for i in range(len(e.doors)) if e.ztest(i) != "PASS")
    _, _, term, _, info = e.step(e.A_COMMIT + i)
    assert not term and info["shield_blocks"] == 1 and info["cost"] == 0.0


def test_all_rejected_ends_as_flag(env):
    env.reset(seed=8)
    for i in range(len(env.doors)):
        _, _, term, _, info = env.step(env.A_REJECT + i)
    assert term and info["outcome"] == "flag" and info["cost"] == 0.0


def test_safe_env_six_tuple():
    e = SafeAutoPassEnv()
    e.reset(seed=0)
    out = e.step(e.A_DWELL)
    assert len(out) == 6 and out[2] == 0.0


def test_rule_policy_is_safe_on_small_batch(env):
    res = [rollout(env, RulePolicy(), seed=s) for s in range(20)]
    assert sum(r["cost"] for r in res) == 0
    assert sum(r["outcome"] == "success" for r in res) >= 15


def test_scan_budget_masks_goto_and_dwell():
    """Once the sweep budget is spent only COMMIT/REJECT remain, sweeps never exceed the budget,
    and the rule policy rejects instead of committing without a z-test PASS."""
    e = AutoPassEnv(scan_budget=12)
    e.reset(seed=0)
    term = trunc = False
    while not e.budget_spent() and not (term or trunc):
        _, _, term, trunc, _ = e.step(e.A_DWELL)
    m = e.action_masks()
    assert e.scans == 12 and not m[:e.A_COMMIT].any() and m[e.A_REJECT:].any()
    _, _, _, _, info = e.step(e.A_DWELL)                 # masked -> no extra sweep
    assert info["scans"] == 12
    for seed in range(20):
        r = rollout(e, RulePolicy(), seed=seed)
        assert r["scans"] <= 12 and r["cost"] == 0 and r["outcome"] != "timeout"


def test_mask_shield_masks_non_pass_commit():
    """mask mode: COMMIT(i) is valid only on a z-test PASS, so no commit is ever blocked."""
    e = AutoPassEnv(shield=True, shield_mode="mask", commit_needs_trusted=False)
    e.reset(seed=6)
    m = e.action_masks()
    for i in range(len(e.doors)):
        if e.ztest(i) != "PASS":
            assert not m[e.A_COMMIT + i]
    i = next(i for i in range(len(e.doors)) if e.ztest(i) != "PASS")
    _, _, term, _, info = e.step(e.A_COMMIT + i)          # masked -> plain DWELL, not a shield block
    assert not term and info["shield_blocks"] == 0


def test_mask_shield_no_timeouts_from_blocked_commits():
    """A commit-greedy policy (always COMMIT when any COMMIT is valid) is never blocked in mask mode,
    so it cannot time out by repeating blocked commits."""
    def greedy(env, info):
        m = info["action_mask"]
        c = np.flatnonzero(m[env.A_COMMIT:env.A_REJECT])
        if len(c):
            return env.A_COMMIT + int(c[0])
        return env.A_DWELL if m[env.A_DWELL] else env.A_REJECT + int(np.flatnonzero(m[env.A_REJECT:])[0])
    em = AutoPassEnv(shield=True, shield_mode="mask")
    for seed in range(30):
        r = rollout(em, greedy, seed=seed)
        assert r["shield_blocks"] == 0 and r["cost"] == 0     # nothing blocked -> no blocked-commit loop


def test_budget_observation_and_sampling():
    e = AutoPassEnv()
    obs, _ = e.reset(seed=0)
    assert obs.shape == (e.observation_space.shape[0],) and tuple(obs[-2:]) == (0.0, 1.0)
    e = AutoPassEnv(scan_budget=20)
    obs, _ = e.reset(seed=0)
    assert obs[-2] == 1.0 and np.isclose(obs[-1], (20 - e.scans) / 40)
    e.step(e.A_DWELL)
    assert np.isclose(e._obs()[-1], (20 - e.scans) / 40)
    from autopass_rl.env import BUDGET_DIST_V2
    e = AutoPassEnv(budget_dist=BUDGET_DIST_V2)
    seen = set()
    for s in range(200):
        e.reset(seed=s); seen.add(e.scan_budget)
    assert seen == {None, 30, 20, 12}
    # the budget draw comes last: worlds and agents per seed match the no-budget env
    a, b = AutoPassEnv(), AutoPassEnv(budget_dist=BUDGET_DIST_V2)
    for s in range(10):
        a.reset(seed=s); b.reset(seed=s)
        assert a.w_A == b.w_A and [d.w for d in a.doors] == [d.w for d in b.doors]


def test_dwell_cap_per_door_and_per_stop():
    """dwell_cap: DWELL is masked after `cap` DWELLs at one door's viewpoints (summed over its viewpoints),
    and after `cap` consecutive DWELLs at a stop that is not a viewpoint. Off by default."""
    assert AutoPassEnv().dwell_cap is None
    cap = 3
    e = AutoPassEnv(dwell_cap=cap)
    e.reset(seed=1)
    for _ in range(cap):                                   # start position: per-stop count
        assert e.action_masks()[e.A_DWELL]
        e.step(e.A_DWELL)
    if e._vp_door() is None:
        assert not e.action_masks()[e.A_DWELL]
    ks = [k for k in range(K) if e.vp[0, k] is not None]
    e.step(0 * K + ks[0])                                  # move to a door-0 viewpoint: stop count resets
    assert e.stop_dwells == 0 and e._vp_door() == 0
    n = e.door_dwells[0]
    while e.door_dwells[0] < cap:
        assert e.action_masks()[e.A_DWELL]
        e.step(e.A_DWELL)
    assert not e.action_masks()[e.A_DWELL]                 # door 0 capped
    e.step(0 * K + ks[1])                                  # another viewpoint of the same door: still capped
    assert e._vp_door() == 0 and not e.action_masks()[e.A_DWELL]
    e.step(e.A_REJECT + 0)                                 # REJECT door 0 frees DWELL again
    assert e.action_masks()[e.A_DWELL] or e.budget_spent()


def test_dwell_cap_20_leaves_rule_unchanged():
    """The rule never dwells more than 20 times per door, so dwell_cap=20 must not change it."""
    for seed in range(40):
        a = rollout(AutoPassEnv(), RulePolicy(), seed=seed)
        b = rollout(AutoPassEnv(dwell_cap=20), RulePolicy(), seed=seed)
        fa, fb = a.pop("final_obs"), b.pop("final_obs")
        assert a == b and np.array_equal(fa, fb)
