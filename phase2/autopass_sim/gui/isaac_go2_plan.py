"""Isaac Lab GUI run: the frozen Go2 walking policy executes v2-A's plan in a matching scene (simulated).

Scene: flat floor, outer walls, the door wall with openings at the exported soffit width and door stops
(so the clear width is the true stop-to-stop width), lintels, furniture boxes, all static colliders.
Robot: Unitree-Go2-Velocity (unitree_rl_lab play config), frozen checkpoint model_800.pt, 1 env.
Control: the RL policy only sees a velocity command (vx, vy, wz); a pure-pursuit waypoint controller
turns the exported plan into commands. At each viewpoint the robot stops, turns to face the door and
holds for the plan's dwell time plus a fixed settle pause (v2-A's plan has 0 dwell sweeps: it scans
while walking). After the last viewpoint it follows the COMMIT path through the chosen door to the goal.

Run with the Isaac Lab env (GUI, not headless):
    env -u PYTHONPATH OMNI_KIT_ACCEPT_EULA=YES ~/anaconda3/envs/env_isaaclab/bin/python \
        gui/isaac_go2_plan.py --plan results_rl/gui/plan_5081.json --out results_rl/gui
"""
import argparse
import json
import math
import os

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--plan", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--checkpoint", default=os.path.expanduser(
    "~/sim/unitree_rl_lab/logs/rsl_rl/unitree_go2_velocity/2026-11-06_14-31-38/model_800.pt"))
ap.add_argument("--vmax", type=float, default=0.5, help="m/s, same as the RL env's walking speed")
ap.add_argument("--settle", type=float, default=1.0, help="s, pause at each viewpoint (not in the env plan)")
ap.add_argument("--fps", type=int, default=25)
ap.add_argument("--res", type=int, nargs=2, default=(1280, 720))
ap.add_argument("--tag", default="faithful", help="output name tag")
ap.add_argument("--keep-open", action="store_true", help="keep the GUI window open after the run (close it to exit)")
ap.add_argument("--door-straight", action="store_true",
                help="executor variant: replace the commit path near the door by a straight pass through the door centre (0.5 m before to 0.7 m after the wall); door choice and order unchanged")
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import imageio  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import unitree_rl_lab.tasks  # noqa: E402,F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg  # noqa: E402

TASK = "Unitree-Go2-Velocity"
LABEL = "Simulated — Isaac Lab, frozen Go2 policy"
OUTER_H, WALL_H, T_OUT = 1.0, 2.4, 0.10   # outer walls kept low so the cameras see over them (m)
FURN_H = 0.75                              # furniture height: not in the 2D world, assumed (m)


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


# ---------------------------------------------------------------- scene geometry (world frame, m)
def scene_boxes(w):
    """Static boxes (name, cx, cy, cz, sx, sy, sz, color, collide)."""
    W, H, y0, D = w["W"], w["H"], w["wall_y"], w["wall_depth"]
    gray, dark, wood = (0.80, 0.80, 0.78), (0.35, 0.36, 0.40), (0.55, 0.40, 0.25)
    B = []
    # outer walls (outside the [0,W]x[0,H] footprint)
    B += [("outer_s", W / 2, -T_OUT / 2, OUTER_H / 2, W + 2 * T_OUT, T_OUT, OUTER_H, gray, True),
          ("outer_n", W / 2, H + T_OUT / 2, OUTER_H / 2, W + 2 * T_OUT, T_OUT, OUTER_H, gray, True),
          ("outer_w", -T_OUT / 2, H / 2, OUTER_H / 2, T_OUT, H, OUTER_H, gray, True),
          ("outer_e", W + T_OUT / 2, H / 2, OUTER_H / 2, T_OUT, H, OUTER_H, gray, True)]
    doors = sorted(w["doors"], key=lambda d: d["xc"])
    edges = [0.0] + [e for d in doors for e in (d["xc"] - d["hf"], d["xc"] + d["hf"])] + [W]
    for k, (a, b) in enumerate(zip(edges[0::2], edges[1::2])):     # door-wall pieces between openings
        if b - a > 1e-4:
            B.append((f"wall_{k}", (a + b) / 2, y0 + D / 2, WALL_H / 2, b - a, D, WALL_H, gray, True))
    for d in doors:
        i, xc, hf, s, r = d["id"], d["xc"], d["hf"], d["s"], d["r"]
        ws = D - 2 * r
        dh = w["door_height"]
        B.append((f"lintel_{i}", xc, y0 + D / 2, (dh + WALL_H) / 2, 2 * hf, D, WALL_H - dh, gray, True))
        for side, x in (("l", xc - hf + s / 2), ("r", xc + hf - s / 2)):   # stops: clear width = w
            B.append((f"stop_{i}{side}", x, y0 + r + ws / 2, dh / 2, s, ws, dh, dark, True))
    for k, (x0, yb0, x1, yb1) in enumerate(w["boxes"]):
        B.append((f"furniture_{k}", (x0 + x1) / 2, (yb0 + yb1) / 2, FURN_H / 2, x1 - x0, yb1 - yb0, FURN_H, wood, True))
    return B


def floor_marks(w, plan):
    """Visual-only floor markers (no collision): door margin colour, viewpoints, goal."""
    M = []
    for d in w["doors"]:
        m = d["margin_cm"]
        col = (0.2, 0.7, 0.3) if m >= 3 else (0.95, 0.6, 0.1) if m >= 0 else (0.85, 0.15, 0.15)
        M.append((f"mark_door_{d['id']}", d["xc"], d["yw"] - 0.15, 0.003, d["w"], 0.06, 0.006, col, False))
    for k, st in enumerate(plan):
        if st["kind"] == "walk":
            x, y = st["pts"][-1]
            M.append((f"mark_vp_{k}", x, y, 0.003, 0.2, 0.2, 0.006, (0.2, 0.45, 0.85), False))
    gx, gy = w["goal"]
    M.append(("mark_goal", gx, gy, 0.003, 0.4, 0.4, 0.006, (0.2, 0.7, 0.3), False))
    return M


def box_cfg(name, cx, cy, cz, sx, sy, sz, color, collide):
    return AssetBaseCfg(
        prim_path=f"/World/autopass/{name}",
        spawn=sim_utils.CuboidCfg(size=(sx, sy, sz),
                                  collision_props=sim_utils.CollisionPropertiesCfg() if collide else None,
                                  visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.8)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(cx, cy, cz)))


def dist_to_boxes(p, boxes):
    """2D distance from point p to the nearest collider footprint."""
    best = np.inf
    for name, cx, cy, _, sx, sy, _, _, collide in boxes:
        if not collide or name.startswith("lintel"):
            continue
        dx = max(abs(p[0] - cx) - sx / 2, 0.0)
        dy = max(abs(p[1] - cy) - sy / 2, 0.0)
        best = min(best, math.hypot(dx, dy))
    return best


# ---------------------------------------------------------------- env
def make_env(plan_d, boxes, marks):
    cfg = parse_env_cfg(TASK, device=args.device, num_envs=1, use_fabric=True,
                        entry_point_key="play_env_cfg_entry_point")
    w = plan_d["world"]
    first = np.array(plan_d["plan"][0]["pts"][1]) - np.array(w["start"])
    yaw0 = math.atan2(first[1], first[0])
    cfg.scene.robot.init_state.pos = (w["start"][0], w["start"][1], 0.4)
    cfg.scene.robot.init_state.rot = (math.cos(yaw0 / 2), 0.0, 0.0, math.sin(yaw0 / 2))
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    cfg.scene.terrain.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.62, 0.60, 0.56), roughness=0.9)
    cfg.scene.height_scanner = None                     # not a policy input; needs a mesh terrain
    cfg.scene.sky_light = AssetBaseCfg(prim_path="/World/skyLight",
                                       spawn=sim_utils.DomeLightCfg(intensity=900.0, color=(0.95, 0.95, 1.0)))
    cfg.scene.sun = AssetBaseCfg(prim_path="/World/sun",
                                 spawn=sim_utils.DistantLightCfg(intensity=2500.0, angle=1.0),
                                 init_state=AssetBaseCfg.InitialStateCfg(rot=(0.92, 0.2, 0.3, 0.1)))
    for b in boxes + marks:
        setattr(cfg.scene, b[0], box_cfg(*b))
    # deterministic, no disturbances
    cfg.events.push_robot = None
    cfg.events.add_base_mass = None
    cfg.events.physics_material = None
    cfg.events.base_external_force_torque = None
    cfg.events.reset_base.params["pose_range"] = {}
    cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
    cfg.observations.policy.enable_corruption = False
    cfg.curriculum.terrain_levels = None
    cfg.curriculum.lin_vel_cmd_levels = None
    cfg.commands.base_velocity.resampling_time_range = (1e6, 1e6)
    cfg.commands.base_velocity.rel_standing_envs = 0.0
    cfg.commands.base_velocity.debug_vis = False
    cfg.episode_length_s = 600.0
    cfg.viewer.resolution = tuple(args.res)
    cfg.viewer.origin_type = "world"
    return ManagerBasedRLEnv(cfg, render_mode="rgb_array")


# ---------------------------------------------------------------- waypoint controller
class Follower:
    """Pure pursuit on a polyline; in-place turn when the heading error is large."""

    def __init__(self, pts, look=0.25, tol=0.10):
        P = [np.asarray(pts[0], float)]
        for a, b in zip(pts[:-1], pts[1:]):
            a, b = np.asarray(a, float), np.asarray(b, float)
            n = max(1, int(np.ceil(np.linalg.norm(b - a) / 0.05)))
            P += [a + (b - a) * (j / n) for j in range(1, n + 1)]
        self.P, self.i, self.look, self.tol = np.array(P), 0, look, tol

    def __call__(self, p, yaw):
        d = np.linalg.norm(self.P[self.i:] - p, axis=1)
        self.i += int(np.argmin(d[: max(1, int(1.0 / 0.05))]))       # progress, search 1 m ahead
        j = self.i
        while j < len(self.P) - 1 and np.linalg.norm(self.P[j] - p) < self.look:
            j += 1
        tgt, end = self.P[j], self.P[-1]
        dist_end = np.linalg.norm(end - p)
        if dist_end < self.tol:
            return np.zeros(3), True
        e = wrap(math.atan2(tgt[1] - p[1], tgt[0] - p[0]) - yaw)
        wz = float(np.clip(2.0 * e, -1.0, 1.0))
        vx = 0.0 if abs(e) > 0.8 else args.vmax * math.cos(e) ** 2
        vx = min(vx, 0.8 * dist_end + 0.08)
        return np.array([vx, 0.0, wz]), False


# ---------------------------------------------------------------- main
def main():
    plan_d = json.load(open(args.plan))
    w, plan = plan_d["world"], plan_d["plan"]
    os.makedirs(args.out, exist_ok=True)
    boxes, marks = scene_boxes(w), floor_marks(w, plan)
    env = make_env(plan_d, boxes, marks)
    venv = RslRlVecEnvWrapper(env)
    agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
    runner = OnPolicyRunner(venv, agent_cfg.to_dict(), log_dir=None, device=env.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=env.device)
    print("[autopass] loaded", args.checkpoint)
    origin = env.scene.env_origins[0].cpu().numpy()
    print("[autopass] env origin", origin)
    assert np.allclose(origin[:2], 0, atol=1e-6), "env origin must be at the world origin"

    robot = env.scene["robot"]
    cmd_term = env.command_manager.get_term("base_velocity")
    contact = env.scene["contact_forces"]
    foot_ids = [k for k, n in enumerate(contact.body_names) if n.endswith("_foot")]
    other_ids = [k for k, n in enumerate(contact.body_names) if not n.endswith("_foot")]
    dt = env.step_dt
    frame_every = max(1, round(1.0 / (args.fps * dt)))
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    font2 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    writer = imageio.get_writer(os.path.join(args.out, f"go2_v2A_{plan_d['episode_seed']}_{args.tag}.mp4"), fps=args.fps,
                                codec="libx264", quality=7, macro_block_size=8)

    state = dict(t=0.0, status="", frames=0)
    log, events, shots = [], [], {}
    cam = dict(eye=None, tgt=None)

    def label(img, status):
        im = Image.fromarray(img[..., :3])
        dr = ImageDraw.Draw(im, "RGBA")
        dr.rectangle([12, 12, 12 + 18 + dr.textlength(LABEL, font=font), 48], fill=(0, 0, 0, 150))
        dr.text((21, 17), LABEL, font=font, fill=(255, 255, 255, 255))
        if status:
            dr.rectangle([12, 54, 12 + 16 + dr.textlength(status, font=font2), 80], fill=(0, 0, 0, 120))
            dr.text((20, 58), status, font=font2, fill=(235, 235, 235, 255))
        dr.text((im.width - 260, im.height - 26), f"t = {state['t']:5.1f} s (sim)", font=font2, fill=(20, 20, 20, 255))
        return np.asarray(im)

    def pose():
        p = robot.data.root_pos_w[0].cpu().numpy()
        yaw = float(robot.data.heading_w[0])
        return p, yaw

    def follow_cam(p, yaw):
        past = p[1] > w["wall_y"] + w["wall_depth"] + 0.3
        off = np.array([0.9, 2.6, 2.1]) if past else np.array([-1.7, -2.3, 2.0])
        eye_t, tgt_t = p + off, p + np.array([0.0, 0.0, 0.1])
        if cam["eye"] is None:
            cam["eye"], cam["tgt"] = eye_t, tgt_t
        a = 0.04
        cam["eye"] = (1 - a) * cam["eye"] + a * eye_t
        cam["tgt"] = (1 - 0.2) * cam["tgt"] + 0.2 * tgt_t
        env.sim.set_camera_view(cam["eye"].tolist(), cam["tgt"].tolist())

    obs = venv.get_observations()
    fallen = [False]

    def step(cmd, status, rec=True):
        nonlocal obs
        cmd_term.vel_command_b[0] = torch.as_tensor(cmd, dtype=torch.float32, device=env.device)
        with torch.inference_mode():
            act = policy(obs)
            obs, _, dones, _ = venv.step(act)
        state["t"] += dt
        p, yaw = pose()
        quat = robot.data.root_quat_w[0].cpu().numpy()
        tilt = math.degrees(math.acos(float(np.clip(-robot.data.projected_gravity_b[0, 2].item(), -1, 1))))
        f = contact.data.net_forces_w[0].norm(dim=-1).cpu().numpy()
        body_xy = robot.data.body_pos_w[0, :, :2].cpu().numpy()
        clear = min(dist_to_boxes(q, boxes) for q in body_xy)
        if bool(dones[0]) or p[2] < 0.15:
            fallen[0] = True
        kb = other_ids[int(np.argmax(f[other_ids]))]
        log.append(dict(t=round(state["t"], 3), x=p[0], y=p[1], z=p[2], yaw=yaw, tilt_deg=tilt,
                        vx_cmd=cmd[0], wz_cmd=cmd[2], vx=float(robot.data.root_lin_vel_b[0, 0]),
                        wz=float(robot.data.root_ang_vel_b[0, 2]), nonfoot_contact_N=float(f[other_ids].max()), nonfoot_body=contact.body_names[kb],
                        body_clearance_m=clear, done=bool(dones[0]), status=status))
        if rec:
            follow_cam(p, yaw)
            if int(round(state["t"] / dt)) % frame_every == 0:
                img = env.render()
                if img is not None:
                    writer.append_data(label(img, status)); state["frames"] += 1
                    state["last"] = (img, status)
        return p, yaw

    def snapshot(name):
        img, status = state["last"]
        Image.fromarray(label(img, status)).save(os.path.join(args.out, f"shot_{args.tag}_{name}.png"))
        shots[name] = round(state["t"], 2)

    # --- overview still (world camera), then let the robot settle into its stance
    env.sim.set_camera_view([w["W"] / 2 - 1.0, -4.5, 9.5], [w["W"] / 2, w["H"] / 2 - 0.5, 0.0])
    for _ in range(int(1.0 / dt)):
        step(np.zeros(3), "", rec=False)
    img = None
    for _ in range(60):                       # the first renders can be blank while the viewport warms up
        img = env.render()
        if img is not None and img[..., :3].mean() > 20:
            break
    if img is None or img[..., :3].mean() <= 20:
        print("[autopass] WARNING: overview frame still blank")
    Image.fromarray(label(img, f"Overview: seed {plan_d['episode_seed']}, doors 0/1/2 margins "
                    + " / ".join(f"{d['margin_cm']:+.2f}" for d in w["doors"]) + " cm (w_A 81.3 cm); "
                    "orange/green/red floor strips")).save(os.path.join(args.out, f"shot_{args.tag}_overview.png"))
    shots["overview"] = round(state["t"], 2)
    t_start = state["t"]

    # --- execute the plan
    wp_log = []
    for k, st in enumerate(plan):
        if fallen[0]:
            break
        if st["kind"] == "walk":
            status = f"v2-A step {k + 1}: GOTO door {st['door']} viewpoint {st['viewpoint']} (scanning while walking)"
            fol, t0, ok = Follower(st["pts"]), state["t"], False
            limit = 3 * sum(np.linalg.norm(np.diff(np.array(st["pts"]), axis=0), axis=1)) / args.vmax + 10
            while state["t"] - t0 < limit and not fallen[0]:
                p, yaw = pose()
                cmd, ok = fol(p[:2], yaw)
                if ok:
                    break
                step(cmd, status)
            p, yaw = pose()
            err = float(np.linalg.norm(p[:2] - np.array(st["pts"][-1])))
            wp_log.append(dict(step=k + 1, kind="viewpoint", door=st["door"], viewpoint=st["viewpoint"],
                               target=st["pts"][-1], reached=bool(ok), pos_error_m=round(err, 3),
                               t_arrive=round(state["t"] - t_start, 2), walk_time_s=round(state["t"] - t0, 2),
                               env_walk_time_s=st["t_env"]))
            # face the door, then hold (dwell from the plan + settle pause)
            d = next(dd for dd in w["doors"] if dd["id"] == st["door"])
            face = math.atan2(d["yw"] - p[1], d["xc"] - p[0])
            t1 = state["t"]
            while state["t"] - t1 < 6.0 and not fallen[0]:
                p, yaw = pose()
                e = wrap(face - yaw)
                if abs(e) < 0.12:
                    break
                wz = math.copysign(min(1.0, max(0.5, 2.0 * abs(e))), e)
                step(np.array([0.0, 0.0, wz]), status + " - turn to door")
            hold = args.settle
            t1 = state["t"]
            while state["t"] - t1 < hold and not fallen[0]:
                step(np.zeros(3), f"v2-A step {k + 1}: at door {st['door']} viewpoint {st['viewpoint']} "
                                  f"(pause {hold:.1f} s; env plan dwell 0 s)")
            if "at_viewpoint" not in shots:
                snapshot("at_viewpoint")
        elif st["kind"] == "dwell":
            t1 = state["t"]
            while state["t"] - t1 < st["seconds"] and not fallen[0]:
                step(np.zeros(3), f"v2-A step {k + 1}: DWELL ({st['sweeps']} sweeps)")
        elif st["kind"] == "reject":
            events.append(dict(step=k + 1, kind="reject", door=st["door"], t=round(state["t"] - t_start, 2)))
        elif st["kind"] == "commit":
            d = next(dd for dd in w["doors"] if dd["id"] == st["door"])
            status = (f"v2-A step {k + 1}: COMMIT door {st['door']} (z-test {st['ztest']}, fused margin "
                      f"{st['fused_margin_cm']:+.1f} ± {st['sigma_cm']:.1f} cm) -> goal")
            events.append(dict(step=k + 1, kind="commit", door=st["door"], t=round(state["t"] - t_start, 2)))
            pts = st["pts"]
            if args.door_straight:
                y_out = d["yw"] + d["D"] + 0.7
                pts = [pts[0], [d["xc"], d["yw"] - 0.5], [d["xc"], y_out]] + [q for q in pts[1:] if q[1] > y_out]
                events.append(dict(kind="executor_variant", commit_pts=pts))
            fol, t0, ok = Follower(pts, tol=0.15), state["t"], False
            limit = 3 * sum(np.linalg.norm(np.diff(np.array(pts), axis=0), axis=1)) / args.vmax + 10
            crossed = False
            while state["t"] - t0 < limit and not fallen[0]:
                p, yaw = pose()
                cmd, ok = fol(p[:2], yaw)
                if ok:
                    break
                p, yaw = step(cmd, status)
                in_door = abs(p[0] - d["xc"]) < d["hf"] + 0.3 and d["yw"] - 0.05 < p[1] < d["yw"] + d["D"] + 0.05
                if in_door and "at_commit_door" not in shots:
                    snapshot("at_commit_door")
                if p[1] > d["yw"] + d["D"] and not crossed:
                    crossed = True
                    events.append(dict(kind="crossed_wall", door_x_at_wall=round(float(p[0]), 3),
                                       offset_from_door_centre_m=round(float(p[0] - d["xc"]), 3),
                                       t=round(state["t"] - t_start, 2)))
                if p[1] > d["yw"] + 1.2 and "through_door" not in shots:
                    snapshot("through_door")
            p, yaw = pose()
            # intermediate waypoints of the commit path: closest approach
            traj = np.array([[r["x"], r["y"]] for r in log])
            for j, q in enumerate(pts[1:-1], 1):
                wp_log.append(dict(step=k + 1, kind=f"commit waypoint {j}", target=q,
                                   reached=bool(np.min(np.linalg.norm(traj - q, axis=1)) < 0.25),
                                   closest_m=round(float(np.min(np.linalg.norm(traj - q, axis=1))), 3)))
            err = float(np.linalg.norm(p[:2] - np.array(pts[-1])))
            wp_log.append(dict(step=k + 1, kind="goal", target=pts[-1], reached=bool(ok), pos_error_m=round(err, 3),
                               t_arrive=round(state["t"] - t_start, 2), walk_time_s=round(state["t"] - t0, 2),
                               env_walk_time_s=st["t_env"], crossed_door_wall=crossed))
    # short hold at the goal for the end of the video
    for _ in range(int(1.5 / dt)):
        if fallen[0]:
            break
        step(np.zeros(3), "Goal reached" if wp_log and wp_log[-1].get("kind") == "goal" and wp_log[-1]["reached"]
             else "stopped")
    writer.close()

    import csv
    with open(os.path.join(args.out, f"trajectory_{args.tag}.csv"), "w", newline="") as fh:
        cw = csv.DictWriter(fh, fieldnames=list(log[0].keys()))
        cw.writeheader(); cw.writerows(log)
    run = [r for r in log if r["t"] > t_start]
    contacts = [r for r in run if r["nonfoot_contact_N"] > 1.0]
    summary = dict(
        label="simulated (Isaac Lab sim-to-sim walkability check; not robot data)",
        plan=args.plan, checkpoint=args.checkpoint, isaac_task=TASK, gui=not args.headless,
        settings=dict(vmax=args.vmax, settle_s=args.settle, physics_dt=env.physics_dt, control_dt=dt,
                      obs_noise=False, pushes=False, outer_wall_h=OUTER_H, door_wall_h=WALL_H, furniture_h=FURN_H),
        all_waypoints_reached=all(x["reached"] for x in wp_log) and not fallen[0],
        fell=fallen[0], max_tilt_deg=round(max(r["tilt_deg"] for r in run), 2),
        min_base_height_m=round(min(r["z"] for r in run), 3),
        tag=args.tag, door_straight=args.door_straight,
        nonfoot_contact_steps=len(contacts),
        nonfoot_contact_bodies=sorted({r["nonfoot_body"] for r in contacts}),
        nonfoot_contact_where=[dict(t=round(r["t"] - t_start, 2), x=round(float(r["x"]), 3), y=round(float(r["y"]), 3),
                                    body=r["nonfoot_body"], N=round(r["nonfoot_contact_N"], 1)) for r in contacts][:40],
        max_nonfoot_contact_N=round(max(r["nonfoot_contact_N"] for r in run), 2),
        min_body_clearance_m=round(min(r["body_clearance_m"] for r in run), 3),
        min_body_clearance_note="2D distance from any Go2 link origin to the nearest collider footprint",
        final_position_error_m=wp_log[-1].get("pos_error_m") if wp_log else None,
        run_time_s=round(state["t"] - t_start, 2), env_plan_time_s=plan_d["env_time_s"],
        video_frames=state["frames"], video_fps=args.fps, screenshots=shots, waypoints=wp_log, events=events)
    json.dump(summary, open(os.path.join(args.out, f"walkability_{args.tag}.json"), "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("waypoints", "nonfoot_contact_where")}, indent=1, default=float))
    for x in wp_log:
        print(x)
    if args.keep_open:
        print("[autopass] run finished; window stays open until you close it")
        while app.is_running():
            env.sim.render()
    env.close()


if __name__ == "__main__":
    main()
    app.close()
