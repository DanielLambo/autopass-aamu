"""Observation model shared by the planner and the decision rule.

q(d, psi) = 1 / sigma_total(d)^2  if the view is trusted (psi <= psi_safe, d <= d_max,
                                  and both jambs in line of sight)
          = 0                     otherwise
sigma_total^2 = sigma_scan(d)^2 + floor^2, with sigma_scan(d) = a + c d^p from calibration
and `floor` standing in for GAIT's other error sources (plane fit, registration, odometry).

Density-aware variant (Report 2 rev., pass rho_calib="calib_rho.json"):
  sigma_scan(rho, psi) = exp(s0) rho^s_rho exp(s_psi psi)   rho = edge-support density per sweep
  B(rho)               = max(b_min, kB / sqrt(rho))          per-sweep bias bound
  q                    = 1 / (sigma_scan^2 + floor^2 + B^2)  inverse mean-squared error of one sweep
  the planner, which cannot see future sweeps, uses rho_hat(d, psi) = exp(r0) d^r_d exp(r_psi psi).

The planner's cost field turns this into a per-metre penalty in each door's approach zone:
  g(x) = 1 - q(x) / q_max  (0 = best possible view, 1 = untrusted view)
"""
import json
import numpy as np
from .sensor import jambs_visible


class ObsModel:
    def __init__(self, calib_path, floor=1.0, psi_safe=12.5, d_max=4.0, b_res=0.5, b_max=1.6, rho_calib=None):
        c = json.load(open(calib_path))
        self.ds = np.array(c["d"])
        self.b0_tab = np.array(c["b0"])
        f = c["sigma_fit"]
        self.a, self.c, self.p = f["a"], f["c"], f["p"]
        self.floor, self.psi_safe, self.d_max = floor, psi_safe, d_max
        self.b_res, self.b_max = b_res, b_max          # cm: bias bounds, trusted / untrusted views
        self.density = rho_calib is not None
        if self.density:
            r = json.load(open(rho_calib))
            self.s0, self.s_rho, self.s_psi = r["sigma"]["s0"], r["sigma"]["s_rho"], r["sigma"]["s_psi"]
            self.r0, self.r_d, self.r_psi = r["rho"]["r0"], r["rho"]["r_d"], r["rho"]["r_psi"]
            self.kB, self.b_min = r["bias"]["kB"], r["bias"]["b_min"]
            self.q_max = float(self.q_grid(np.array(0.75), np.array(0.0)))
        else:
            self.q_max = 1.0 / self.sigma(0.75) ** 2

    # ---- density-aware model -------------------------------------------------------------
    def rho_hat(self, d, psi):
        return np.exp(self.r0) * np.asarray(d, float) ** self.r_d * np.exp(self.r_psi * np.asarray(psi, float))

    def sigma_rho(self, rho, psi):                     # cm, includes the GAIT floor
        rho = np.maximum(np.asarray(rho, float), 1.0)
        return np.hypot(np.exp(self.s0) * rho ** self.s_rho * np.exp(self.s_psi * np.asarray(psi, float)), self.floor)

    def bias_rho(self, rho):                           # cm
        return np.maximum(self.b_min, self.kB / np.sqrt(np.maximum(np.asarray(rho, float), 1.0)))

    def q_rho(self, rho, psi):
        return 1.0 / (self.sigma_rho(rho, psi) ** 2 + self.bias_rho(rho) ** 2)

    def q_grid(self, d, psi):
        """Expected quality of one sweep from (d, psi), before line-of-sight checks."""
        if not self.density:
            sig = np.hypot(self.a + self.c * np.asarray(d) ** self.p, self.floor)
            return np.where((psi <= self.psi_safe) & (d <= self.d_max), 1 / sig ** 2, 0.0)
        q = self.q_rho(self.rho_hat(d, psi), psi)
        return np.where((psi <= self.psi_safe) & (d <= self.d_max), q, 0.0)

    def b0(self, d):
        return float(np.interp(d, self.ds, self.b0_tab))

    def sigma(self, d):                                 # cm
        return float(np.hypot(self.a + self.c * d ** self.p, self.floor))

    def trusted(self, d, psi):          # geometry only; mission.py adds the line-of-sight check
        return d <= self.d_max and psi <= self.psi_safe

    def q(self, d, psi):
        if self.density:
            return float(self.q_grid(np.array(d), np.array(psi)))
        return 1.0 / self.sigma(d) ** 2 if self.trusted(d, psi) else 0.0


def cost_field(world, model, lam, zone=3.0, active=None):
    """Per-cell cost multiplier (>= 1) for the planning grid.

    Cells in front of a door (within `zone` m) cost 1 + lam * g; everything else costs 1.
    `active` limits the penalty to doors whose status is still unknown.
    """
    ny, nx = world.free.shape
    cost = np.ones((ny, nx))
    if lam <= 0:
        return cost
    xs = (np.arange(nx) + 0.5) * world.res
    ys = (np.arange(ny) + 0.5) * world.res
    X, Y = np.meshgrid(xs, ys)
    for door in world.doors:
        if active is not None and door.id not in active:
            continue
        dx, dy = X - door.xc, door.yw - Y
        d = np.hypot(dx, dy)
        zone_mask = (d <= zone) & (dy > 0)
        psi = np.degrees(np.arctan(np.maximum(0, np.abs(dx) - door.w / 2) / (door.r + np.maximum(dy, 1e-6))))
        q = model.q_grid(d, psi)
        # occluded views count as untrusted (furniture between robot and a jamb)
        jj, ii = np.nonzero(zone_mask & (q > 0))
        for j, i in zip(jj, ii):
            if not jambs_visible(world, door, (X[j, i], Y[j, i])):
                q[j, i] = 0.0
        g = 1 - np.clip(q / model.q_max, 0, 1)
        cost = np.where(zone_mask, np.maximum(cost, 1 + lam * g), cost)
    return cost
