#!/usr/bin/env python3
"""Sanity checks for the opponent curriculum - run BEFORE training on it.

1. physics: a ball rolled at a defender / the keeper must bounce, the duck walking into one must be
   blocked, contacts must be recorded; patrol defenders must move; the keeper must follow the ball.
2. layouts: one overview image per stage (montage).
3. difficulty: the open-play teacher (blind to opponents) on every stage - a stage it still wins
   is decorative.

  uv run python -m microduck_football.rl.check_obstacles --episodes 12 --out /tmp/obstacles.png
"""
import argparse, multiprocessing as mp, os, sys
from pathlib import Path
os.environ.setdefault("MUJOCO_GL", "egl")
from microduck_football import paths
import numpy as np


def physics():
    import mujoco, duck_sim as ds
    s = ds.DuckSim(seed=0); m, d = s.model, s.data
    ok = True
    def contacts(gid):
        out = set()
        for i in range(d.ncon):
            a, b = d.contact[i].geom1, d.contact[i].geom2
            if gid in (a, b):
                o = b if a == gid else a
                out.add(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, o) or "duck")
        return out
    for name, gid, mid, face in [("defender", s.def_gids[0], s.def_mocap[0], ds.DEF_R),
                                 ("keeper", s.keeper_gid, s.keeper_mocap, 0.03)]:
        # ball rolled at it
        s.reset("full"); s._clear_opponents()
        d.mocap_pos[mid] = [0.8, 0.0, 0.1]; a = s.trunk_adr; d.qpos[a:a + 2] = [-0.5, 0.6]
        d.qpos[s.ball_adr:s.ball_adr + 7] = [0.3, 0.0, ds.BALL_R, 1, 0, 0, 0]
        d.qvel[s.ball_vadr:s.ball_vadr + 6] = [1.0, 0, 0, 0, 1 / ds.BALL_R, 0]; mujoco.mj_forward(m, d)
        hits, xs = set(), []
        for _ in range(150):
            s.step(); hits |= contacts(gid); xs.append(s.ball_pos()[0])
        ball_ok = max(xs) < 0.8 - face - ds.BALL_R + 0.01 and "ball_geom" in hits
        # duck walked into it
        s.reset("full"); s._clear_opponents(); d.mocap_pos[mid] = [0.5, 0.0, 0.1]
        d.qpos[a:a + 3] = [0.1, 0.0, 0.125]; d.qpos[a + 3:a + 7] = [1, 0, 0, 0]
        d.qpos[s.ball_adr:s.ball_adr + 3] = [-0.8, 0.8, ds.BALL_R]; d.qvel[:] = 0; mujoco.mj_forward(m, d)
        s.apply_primitive(ds.P["walk"]); hits2, xs2 = set(), []
        for _ in range(int(8 / s.dt)):
            s.step(); hits2 |= contacts(gid); xs2.append(s.duck_pose()[0])
        duck_ok = max(xs2) < 0.5 and len(hits2) > 0
        ok &= ball_ok and duck_ok
        print(f"  {name:8s}: ball stops at x={max(xs):.3f} ({'OK' if ball_ok else 'FAIL'}), "
              f"duck stops at x={max(xs2):.3f} ({'OK' if duck_ok else 'FAIL'})")
    # moving defenders / keeper follow
    s.reset(6); p0 = [q["pos"].copy() for q in s.defenders]
    for _ in range(100): s.step()
    moved = [float(np.linalg.norm(q["pos"] - p)) for q, p in zip(s.defenders, p0)]
    s.reset(9); y0 = s.keeper["y"]
    d.qpos[s.ball_adr + 1] = 0.6 if y0 < 0 else -0.6; mujoco.mj_forward(m, d)
    for _ in range(150): s.step()
    follow = abs(s.keeper["y"] - y0)
    mv_ok, kp_ok = all(x > 0.05 for x in moved), follow > 0.05
    ok &= mv_ok and kp_ok
    print(f"  patrol defenders moved {np.round(moved, 2).tolist()} m in 2 s ({'OK' if mv_ok else 'FAIL'}); "
          f"keeper moved {follow:.2f} m toward the ball in 3 s ({'OK' if kp_ok else 'FAIL'})")
    return ok


def layouts(out):
    import imageio.v3 as iio, duck_sim as ds
    s = ds.DuckSim(seed=3); imgs = []
    for st in [4, 5, 6, 7, 9, 10]:
        s.reset(st); imgs.append(s.render_overview(240, 320))
    iio.imwrite(out, np.concatenate([np.concatenate(imgs[:3], 1), np.concatenate(imgs[3:], 1)], 0))


def blind(args):
    stage, seeds = args
    from stable_baselines3 import PPO
    from microduck_football.envs import football_env as E
    from microduck_football.envs.football_env import FootballEnv
    E.REPLAY_EARLIER = 0.0
    model = PPO.load(paths.TEACHER_OPEN_PLAY, device="cpu")
    env = FootballEnv(seed=0, stage=stage)
    res = []
    for sd in seeds:
        obs, _ = env.reset(seed=sd); done = False
        while not done:
            obs, r, te, tr, info = env.step(model.predict(obs, deterministic=True)[0]); done = te or tr
        res.append((info["is_success"], info["sim_time"]))
    return stage, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--out", default="obstacle_layouts.png")
    a = ap.parse_args()
    print("1) physics"); ok = physics()
    print(f"   -> {'ALL OK' if ok else 'PHYSICS FAILURE'}")
    layouts(a.out); print(f"2) layouts -> {a.out}")
    print("3) open-play teacher (blind to opponents) per stage:")
    jobs = [(st, list(range(42000 + 100 * st, 42000 + 100 * st + a.episodes))) for st in [7, 8, 9, 10]]
    with mp.get_context("fork").Pool(6) as pool:
        for st, res in sorted(pool.map(blind, jobs)):
            g = [t for okk, t in res if okk]
            print(f"   stage {st}: goals {len(g)}/{len(res)}" + (f" ({np.mean(g):.1f}s)" if g else ""))


if __name__ == "__main__":
    main()
