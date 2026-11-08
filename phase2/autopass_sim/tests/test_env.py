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


def test_hide_wA_zeroes_only_the_wA_feature():
    a, b = AutoPassEnv(), AutoPassEnv(hide_wA=True)
    oa, _ = a.reset(seed=3); ob, _ = b.reset(seed=3)
    assert oa[4] != 0 and ob[4] == 0
    assert np.array_equal(np.delete(oa, 4), np.delete(ob, 4))


def test_world_seed_option_rebuilds_report2_world():
    from autopass_sim.world import make_world
    e = AutoPassEnv()
    e.reset(seed=0, options=dict(world_seed=1003, layout="corridor", w_A=81.3))
    w = make_world(np.random.default_rng(1003), "corridor")
    assert [round(d.w, 6) for d in e.world.doors] == [round(d.w, 6) for d in w.doors]
    assert np.array_equal(e.world.free, w.free)


def test_viewpoints_option_masks_other_viewpoints_and_keeps_rule():
    e = AutoPassEnv(viewpoints=(0,))
    _, info = e.reset(seed=2)
    m = info["action_mask"]
    assert not any(m[i * K + k] for i in range(len(e.doors)) for k in range(1, K))
    for seed in range(20):
        a = rollout(AutoPassEnv(), RulePolicy(), seed=seed); b = rollout(AutoPassEnv(viewpoints=(0,)), RulePolicy(), seed=seed)
        a.pop("final_obs"); b.pop("final_obs")
        assert a == b


# ---------------------------------------------------------------- path_mode="fmm"
def _segment_free(free, res, a, b, ds=0.02):
    n = max(1, int(np.ceil(np.linalg.norm(b - a) / ds)))
    for t in np.linspace(0, 1, n + 1):
        q = a + t * (b - a)
        i, j = int(q[0] / res), int(q[1] / res)
        if not (0 <= j < free.shape[0] and 0 <= i < free.shape[1] and free[j, i]):
            return False
    return True


def test_fmm_paths_collision_free_and_reach_target():
    from autopass_rl import fmm
    env = AutoPassEnv(path_mode="fmm")
    n_paths = 0
    for seed in range(5000, 5012):
        env.reset(seed=seed)
        res = env.world.res
        for i in range(len(env.doors)):
            for k in range(K):
                t = env.vp[i, k]
                if t is None:
                    continue
                p = env.fc.path(("vp", i, k), env.free_near, env.pos, t)
                assert p is not None, (seed, i, k)
                assert np.allclose(p[-1], t) and np.allclose(p[0], env.pos)
                assert all(_segment_free(env.free_near, res, a, b) for a, b in zip(p[:-1], p[1:])), (seed, i, k)
                n_paths += 1
            path, L = env._commit_path(i)
            assert path is not None and np.allclose(path[-1], env.world.goal)
            fr = env.field[i][1]
            assert all(_segment_free(fr, res, a, b) for a, b in zip(path[:-1], path[1:])), (seed, i)
            assert env.world.door_crossed(path).id == env.doors[i].id
            n_paths += 1
    assert n_paths > 100


def test_fmm_deterministic_for_a_seed():
    a = rollout(AutoPassEnv(path_mode="fmm"), RulePolicy(), seed=5003)
    b = rollout(AutoPassEnv(path_mode="fmm"), RulePolicy(), seed=5003)
    assert a["traj"] == b["traj"] and a["outcome"] == b["outcome"]
    assert np.array_equal(a["final_obs"], b["final_obs"])
    c = rollout(AutoPassEnv(path_mode="fmm"), random_policy, seed=11)
    d = rollout(AutoPassEnv(path_mode="fmm"), random_policy, seed=11)
    assert c["traj"] == d["traj"] and np.array_equal(c["final_obs"], d["final_obs"])


def test_fmm_route_features_come_from_the_fmm_field():
    from autopass_rl import fmm
    from autopass_sim.planners import path_length
    env = AutoPassEnv(path_mode="fmm")
    env.reset(seed=5081)
    for i in range(len(env.doors)):
        v = fmm.value_at(env.field[i][0], env.pos, env.world.res)
        assert env.route[i][1] == pytest.approx(v)
        path, L = env._commit_path(i)                  # following the same field: length close to its value
        assert L == pytest.approx(v, rel=0.03)


def test_astar_is_the_default_path_mode():
    a = rollout(AutoPassEnv(), RulePolicy(), seed=5003)
    b = rollout(AutoPassEnv(path_mode="astar"), RulePolicy(), seed=5003)
    assert a["traj"] == b["traj"] and np.array_equal(a["final_obs"], b["final_obs"])
