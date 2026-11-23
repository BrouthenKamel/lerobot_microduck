#!/usr/bin/env python3
"""Does head tracking keep the trained RL teacher working? Compare head modes on the same seeds.

  uv run python -m microduck_football.rl.eval_tracking models/ppo_teacher_open_play.zip --eye detector --episodes 50

Reports goals / time / falls, the share of 10 Hz camera frames in which the ball is truly inside
the image (ground truth from the simulator, for every mode), and for the detector eye how often
it found the ball when it was visible.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from microduck_football.envs import football_env as env_module  # noqa: E402
from microduck_football.envs.football_env import FootballEnv  # noqa: E402
from microduck_football.sim.head_tracking import UPDATE_EVERY, detect_ball, project_ball  # noqa: E402
from microduck_football.sim import duck_sim as ds  # noqa: E402


def goal_in_view(sim) -> bool:
    """Goal mouth centre (on the line, mid-height) inside the head image."""
    pos = sim.data.cam_xpos[sim.cam_id]
    rot = sim.data.cam_xmat[sim.cam_id].reshape(3, 3)
    pc = rot.T @ (np.array([ds.GOAL_X, 0.0, 0.15]) - pos)
    if -pc[2] <= 0.02:
        return False
    col = 159 + 160 * pc[1] / -pc[2]
    row = 120 + 160 * pc[0] / -pc[2]
    return 0 <= col < ds.IMG_W and 0 <= row < ds.IMG_H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--eye", choices=["none", "privileged", "detector"], default="none")
    ap.add_argument("--glance", action="store_true", help="tracker also glances at the goal")
    ap.add_argument("--stage", type=int, default=3)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    from stable_baselines3 import PPO
    model = PPO.load(args.model, device="cpu")
    env_module.REPLAY_EARLIER = 0.0
    env = FootballEnv(seed=args.seed, stage=args.stage,
                      head_tracking=None if args.eye == "none" else args.eye, glance=args.glance)

    stats = {"frames": 0, "visible": 0, "goal": 0, "det_hits": 0, "det_err": []}

    def hook(sim, _n=[0]):
        _n[0] += 1
        if _n[0] % UPDATE_EVERY:
            return
        stats["frames"] += 1
        stats["goal"] += goal_in_view(sim)
        truth = project_ball(sim)
        if truth is not None:
            stats["visible"] += 1
            if args.eye == "detector":
                d = detect_ball(sim.render_head())
                if d is not None:
                    stats["det_hits"] += 1
                    stats["det_err"].append(np.hypot(d[0] - truth[0], d[1] - truth[1]))
    env.on_control_step = hook

    res = []
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        done = False
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, te, tr, info = env.step(a)
            done = te or tr
        res.append(info)

    goals = [r for r in res if r["is_success"]]
    line = (f"eye={args.eye + ('+glance' if args.glance else ''):16s} stage {args.stage}: goals {len(goals)}/{len(res)}"
            + (f", time-to-goal {np.mean([r['sim_time'] for r in goals]):.1f}s" if goals else "")
            + f", falls/ep {np.mean([r['n_falls'] for r in res]):.2f}"
            + f", ball in view {100 * stats['visible'] / max(stats['frames'], 1):.0f}% of frames"
            + f", goal in view {100 * stats['goal'] / max(stats['frames'], 1):.0f}%")
    if args.eye == "detector":
        line += (f", detector found it {100 * stats['det_hits'] / max(stats['visible'], 1):.0f}% of visible frames"
                 f" (centroid err {np.median(stats['det_err']) if stats['det_err'] else float('nan'):.1f} px median)")
    print(line, flush=True)


if __name__ == "__main__":
    main()
