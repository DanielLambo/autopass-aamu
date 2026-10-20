"""Sanity tests. Run: python -m pytest -q"""
import os
import numpy as np
from autopass_sim.geometry import raycast
from autopass_sim.world import World, Door, make_world
from autopass_sim.sensor import scan, estimate_width, view_geometry
from autopass_sim.planners import astar, path_length
from autopass_sim.mission import _decide, Obs, Config

CAL = os.path.join(os.path.dirname(__file__), "..", "calib.json")


def test_raycast_hits_wall():
    segs = np.array([[1.0, -1.0, 1.0, 1.0]])
    t = raycast(np.array([0.0, 0.0]), np.array([[1.0, 0.0], [-1.0, 0.0]]), segs)
    assert abs(t[0] - 1.0) < 1e-9 and np.isinf(t[1])


def test_square_on_width_close():
    rng = np.random.default_rng(0)
    door = Door(0, 4.0, 5.0, 0.837, 0.0159, 0.0492)
    w = World("t", 8, 8, 5.0, [door], [], np.array([4, 1.0]), np.array([4, 7.0])).build()
    est = [estimate_width(scan(w, (4.0, 3.5), rng, doors=[door]), door) for _ in range(30)]
    assert abs(np.nanmean(est) - door.w) < 0.01          # within 1 cm square-on at 1.5 m


def test_hidden_stop_reads_wider():
    rng = np.random.default_rng(1)
    door = Door(0, 4.0, 5.0, 0.837, 0.0159, 0.0492)
    w = World("t", 8, 8, 5.0, [door], [], np.array([4, 1.0]), np.array([4, 7.0])).build()
    on = np.nanmean([estimate_width(scan(w, (4.0, 3.0), rng, doors=[door]), door) for _ in range(40)])
    off = np.nanmean([estimate_width(scan(w, (2.6, 3.4), rng, doors=[door]), door) for _ in range(40)])
    assert view_geometry(door, (2.6, 3.4))[1] > 20
    assert off - on > 0.008                               # oblique view reads ~1+ cm wider


def test_astar_finds_path_and_crosses_a_door():
    w = make_world(np.random.default_rng(3), "room")
    p = astar(w, w.start, w.goal, np.ones_like(w.free, float))
    assert p is not None and w.door_crossed(p) is not None
    assert path_length(p) >= np.linalg.norm(w.goal - w.start) - 1e-6


def test_decision_rule_bias_allowance():
    class M:
        b_res, d_max, density = 0.5, 4.0, False
    obs = [Obs(m=0.6, sig=0.2, d=1.0, psi=0.0, trusted=True)] * 4    # sigma_fused = 0.1
    assert _decide(obs, Config(protocol="trusted"), M)[0] == "FLAG"   # 0.6-0.5 < 2*0.1
    assert _decide(obs, Config(protocol="naive"), M)[0] == "PASS"     # 0.6 >= 2*0.1


def test_density_model_monotone():
    from autopass_sim.observability import ObsModel
    M = ObsModel("calib.json", rho_calib="calib_rho.json")
    ds = np.array([0.75, 1.5, 2.5, 3.5])
    q = M.q_grid(ds, np.zeros(4))
    assert np.all(np.diff(q) < 0)                                  # quality falls with distance
    assert np.all(np.diff(M.bias_rho(np.array([5, 10, 20, 80]))) <= 0)   # bias bound falls with density
    assert M.q_grid(np.array(1.0), np.array(13.0)) == 0            # outside the angle gate
