#!/usr/bin/env python3
"""Kick lab: how strong / how sensitive are the frozen kick policies?

  uv run python -m microduck_football.rl.kick_lab grid        # ball-offset grid around KICK_SPOT -> speed, distance, direction
  uv run python -m microduck_football.rl.kick_lab rolling     # kick distance vs floor rolling friction
  uv run python -m microduck_football.rl.kick_lab aim         # aimed-kick skill from random approach setups

Places the ball at (spot + offset) in the duck's yaw frame after a 1 s settle, triggers the kick,
measures the ball over 2 s.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from microduck_football.sim import duck_sim as ds  # noqa: E402


def place_ball_rel(sim, fx, fy):
    x, y, yaw = sim.duck_pose()
    bx = x + fx * math.cos(yaw) - fy * math.sin(yaw)
    by = y + fx * math.sin(yaw) + fy * math.cos(yaw)
    a = sim.ball_adr
    sim.data.qpos[a:a + 7] = [bx, by, ds.BALL_R, 1, 0, 0, 0]
    jid = sim.model.body("ball").jntadr[0]
    v = sim.model.jnt_dofadr[jid]
    sim.data.qvel[v:v + 6] = 0.0
    mujoco.mj_forward(sim.model, sim.data)


def open_reset(sim, x=0.2, y=-0.8, yaw=math.pi / 2):
    """reset with the duck in open space (default: on the side line facing across, ~1.8 m of free pitch
    ahead) so kicks are not stopped by the goal/net. Goes through reset() itself (layout
    override) - teleporting a standing duck leaves it tilted."""
    orig = sim.sample_layout
    sim.sample_layout = lambda difficulty="full": ((x, y, yaw), (x - 0.5, y + 0.2))
    try:
        sim.reset(0)
    finally:
        sim.sample_layout = orig


def kick_trial(sim, name, fx, fy, seconds=2.0):
    """Returns (max ball speed, distance moved, direction in duck frame [deg], fallen)."""
    _, _, yaw = sim.duck_pose()
    p0 = sim.ball_pos()[:2].copy()
    vmax = 0.0
    sim.apply_primitive(ds.P[name])
    for _ in range(int(seconds * ds.CONTROL_HZ)):
        sim.step()
        vmax = max(vmax, float(np.linalg.norm(sim.ball_vel()[:2])))
    d = sim.ball_pos()[:2] - p0
    ang = math.degrees(ds.wrap(math.atan2(d[1], d[0]) - yaw)) if np.linalg.norm(d) > 0.02 else float("nan")
    return vmax, float(np.linalg.norm(d)), ang, sim.fallen()


# ── aimed kick: a 50 Hz closed-loop skill on top of the frozen walk + kick policies ──────────
# Window measured by `grid` (kick works with the ball 0-4 cm closer than the trained spot and
# within +-3 cm laterally; 1 cm further out already fails).
AIM_X_FIRE = 0.085        # fire when the ball's forward offset drops to this (spot is 0.09)
AIM_X_MIN = 0.05          # too close (the foot would be on top of it)
AIM_Y_OUT = 0.03          # lateral tolerance, outward (away from the body centre) ...
AIM_Y_IN = 0.015          # ... and inward (the grid is weaker there)
AIM_MAX_S = 2.0
AIM_REACH = 0.40          # ball further than this ahead (or behind / wide) -> not kickable


def settle_until_still(sim, max_s=1.2, min_s=0.3, gyro_tol=0.15, hold=0.2):
    """Stand until the trunk is still (|gyro| < gyro_tol for `hold` s): the kick policy only
    connects from a settled stance (measured: weak kicks had |gyro| ~0.6-0.9, good ones ~0.05)."""
    t0, still_since = sim.t, None
    while sim.t - t0 < max_s:
        sim.step()
        if np.linalg.norm(sim.state_vector()[:3]) < gyro_tol:
            still_since = sim.t if still_since is None else still_since
            if sim.t - t0 >= min_s and sim.t - still_since >= hold:
                return True
        else:
            still_since = None
    return False


def aimed_kick(sim, max_s=5.0, lead=-0.005, settle=1.2):
    """Closed-loop kick skill (50 Hz) on top of the frozen walk + kick policies. The kick policy
    only connects from a settled stance, and the walk policy cannot make tiny moves, so:
    walk in (lateral correction by strafing), stop `lead` m before the fire line, settle, then
    kick if the ball is in the window, otherwise walk in again / back off. Foot = ball's side.
    Returns (fired, foot, seconds used)."""
    t0 = sim.t
    _, by = sim.ball_in_duck_frame()
    foot = "kick_left" if by > 0 else "kick_right"
    tx, ty = ds.KICK_SPOT[foot]
    out = 1.0 if ty > 0 else -1.0

    burst = 16                                  # control steps of the "creep in" walk burst (the walk
                                                # policy needs ~0.4 s before it translates at all)

    def lat_ok(ey):
        return -AIM_Y_IN < out * ey < AIM_Y_OUT

    bx, by = sim.ball_in_duck_frame()
    if bx < AIM_X_FIRE + 0.05 and lat_ok(by - ty):         # close already: settle, maybe kick now
        settle_until_still(sim, max_s=settle)
        bx, by = sim.ball_in_duck_frame()
        if AIM_X_MIN < bx <= AIM_X_FIRE + 0.005 and lat_ok(by - ty):
            sim.apply_primitive(ds.P[foot])
            return True, foot, sim.t - t0
    while sim.t - t0 < max_s:
        bx, by = sim.ball_in_duck_frame()
        if not (0.0 < bx < AIM_REACH and abs(by) < 0.15):
            break
        ey = by - ty
        in_window = AIM_X_MIN < bx <= AIM_X_FIRE and lat_ok(ey)
        if in_window or bx <= AIM_X_FIRE + lead:
            sim.apply_primitive(ds.P["stand"])          # stop and settle, then decide
            settle_until_still(sim, max_s=settle)
            bx, by = sim.ball_in_duck_frame()
            ey = by - ty
            if AIM_X_MIN < bx <= AIM_X_FIRE + 0.005 and lat_ok(ey):
                sim.apply_primitive(ds.P[foot])
                return True, foot, sim.t - t0
            if AIM_X_FIRE < bx < AIM_X_FIRE + 0.05 and lat_ok(ey):
                # stopped short. Walking off from a stance lurches the trunk forward (the ball
                # *looks* 2 cm closer for an instant), so a threshold would stop us at once:
                # walk a fixed burst instead, longer on every retry.
                burst += 5
                for _ in range(burst):
                    sim.apply_twist(0.3, float(np.clip(4.0 * (ey - out * 0.007), -0.3, 0.3)), 0.0)
                    sim.step()
                sim.apply_primitive(ds.P["stand"])
                settle_until_still(sim, max_s=settle)
                bx, by = sim.ball_in_duck_frame()
                ey = by - ty
                if AIM_X_MIN < bx <= AIM_X_FIRE + 0.005 and lat_ok(ey):
                    sim.apply_primitive(ds.P[foot])
                    return True, foot, sim.t - t0
                continue
            if bx <= AIM_X_MIN or not lat_ok(ey):          # too close / misaligned: step back
                for _ in range(int(0.6 * ds.CONTROL_HZ)):   # < ~0.4 s would not move at all
                    sim.apply_twist(-0.4, float(np.clip(4.0 * ey, -0.3, 0.3)), 0.0)
                    sim.step()
                continue
        sim.apply_twist(0.3, float(np.clip(4.0 * (ey - out * 0.007), -0.3, 0.3)), 0.0)
        sim.step()
    sim.apply_primitive(ds.P["stand"])
    return False, foot, sim.t - t0


def aim(args):
    """Random approach setups (default ball 0.12-0.35 m ahead, +-10 cm lateral); aimed kick, then
    measure the ball."""
    sim = ds.DuckSim(seed=0)
    g = sim.model.geom("ball_geom").id
    rng = np.random.default_rng(1)
    for lead in args.leads:
        res = []
        for k in range(args.n):
            open_reset(sim)
            sim.model.geom_friction[g, 2] = args.rolling
            fx, fy = rng.uniform(*args.x_range), rng.uniform(-args.y_max, args.y_max)
            place_ball_rel(sim, fx, fy)
            _, _, yaw = sim.duck_pose()
            p0 = sim.ball_pos()[:2].copy()
            fired, foot, used = aimed_kick(sim, lead=lead)
            vmax = 0.0
            for _ in range(int(4 * ds.CONTROL_HZ)):
                sim.step()
                vmax = max(vmax, float(np.linalg.norm(sim.ball_vel()[:2])))
            d = sim.ball_pos()[:2] - p0
            ang = math.degrees(ds.wrap(math.atan2(d[1], d[0]) - yaw))
            res.append((fired, used, float(np.linalg.norm(d)), vmax, ang, sim.fallen()))
        r = np.array(res, float)
        good = (r[:, 0] > 0) & (r[:, 2] > 0.5)
        print(f"lead={lead:.2f}: fired {r[:, 0].mean():.0%}, ball >0.5 m {good.mean():.0%}, "
              f"median distance {np.median(r[:, 2]):.2f} m, median time to fire {np.median(r[r[:, 0] > 0, 1]):.2f} s, "
              f"|direction| median {np.median(np.abs(r[good, 4])) if good.any() else float('nan'):.0f} deg, falls {int(r[:, 5].sum())}")


def grid(args):
    sim = ds.DuckSim(seed=0)
    dxs = np.arange(-0.04, 0.041, 0.01)
    dys = np.arange(-0.03, 0.031, 0.01)
    for name in ("kick_right", "kick_left"):
        sx, sy = ds.KICK_SPOT[name]
        print(f"\n{name}: ball distance moved [m] (rows: forward offset, cols: lateral offset, outward = away from body centre)")
        print("  dx\\dy " + " ".join(f"{dy:+.2f}" for dy in dys))
        best = (0, None)
        for dx in dxs:
            row = []
            for dy in dys:
                open_reset(sim)
                lat = dy if sy > 0 else -dy           # + = outward for both feet
                place_ball_rel(sim, sx + dx, sy + lat)
                v, dist, ang, fell = kick_trial(sim, name, sx + dx, sy + lat)
                row.append(f"{dist:5.2f}" + ("F" if fell else " "))
                if dist > best[0]:
                    best = (dist, (dx, dy, v, ang))
            print(f" {dx:+.2f} " + "".join(row))
        d, (dx, dy, v, ang) = best
        print(f" best: {d:.2f} m at dx={dx:+.2f} dy(out)={dy:+.2f}, max speed {v:.2f} m/s, direction {ang:+.0f} deg")


def rolling(args):
    """Kick distance vs ball rolling friction (the floor): well-placed kicks (ball 0-3 cm closer than
    the trained spot, +-2 cm lateral, both feet) from a duck standing in open space."""
    import json
    sim = ds.DuckSim(seed=0)
    g = sim.model.geom("ball_geom").id
    out = {}
    for roll in args.values:
        dists = []
        for name in ("kick_right", "kick_left"):
            sx, sy = ds.KICK_SPOT[name]
            for dx in (-0.03, -0.02, -0.01, 0.0):
                for dy in (-0.02, 0.0, 0.02):
                    open_reset(sim)
                    sim.model.geom_friction[g, 2] = roll
                    place_ball_rel(sim, sx + dx, sy + dy)
                    _, d, _, _ = kick_trial(sim, name, 0, 0, seconds=6)
                    dists.append(d)
        out[str(roll)] = dists
        print(f"rolling {roll:.4f}: median {np.median(dists):.2f} m, >0.8 m in {np.mean(np.array(dists) > 0.8):.0%}",
              flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(required=True)
    sub.add_parser("grid").set_defaults(func=grid)
    r = sub.add_parser("rolling", help="kick distance vs ball rolling friction")
    r.add_argument("--values", type=float, nargs="+", default=[0.003, 0.0015, 0.001, 0.0005])
    r.add_argument("--json", type=str, default=None)
    r.set_defaults(func=rolling)
    a = sub.add_parser("aim")
    a.add_argument("--n", type=int, default=40)
    a.add_argument("--rolling", type=float, default=0.0015)
    a.add_argument("--leads", type=float, nargs="+", default=[-0.005])
    a.add_argument("--x-range", type=float, nargs=2, default=[0.12, 0.35], help="ball start, ahead [m]")
    a.add_argument("--y-max", type=float, default=0.10, help="ball start, |lateral| [m]")
    a.set_defaults(func=aim)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
