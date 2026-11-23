#!/usr/bin/env python3
"""Does a SMOOTHED teacher still score? Teacher primitive -> twist command, low-pass (EMA, time
constant --tau s), executed as a continuous command (FootballEnv action_mode="twist").

  uv run python -m microduck_football.rl.smooth_teacher_test --tau 0.4 --episodes 20
"""
import argparse, os, sys
from pathlib import Path
os.environ.setdefault("MUJOCO_GL", "egl")
from microduck_football import paths
import numpy as np

TWIST = {"stand": (0, 0, 0), "walk": (0.4, 0, 0), "walk_left": (0.4, 0, 1.0), "walk_right": (0.4, 0, -1.0),
         "strafe_left": (0.3, 0.3, 0), "strafe_right": (0.3, -0.3, 0), "back": (-0.4, 0, 0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=0.4, help="EMA time constant in seconds (0 = no smoothing)")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--stage", type=int, default=3)
    ap.add_argument("--seed", type=int, default=555000)
    args = ap.parse_args()
    from stable_baselines3 import PPO
    from microduck_football.sim import duck_sim as ds
    from microduck_football.envs import football_env as E
    from microduck_football.envs.football_env import FootballEnv, OBS_SLICES
    E.REPLAY_EARLIER = 0.0
    teacher = PPO.load(paths.TEACHER_OPEN_PLAY, device="cpu")
    env = FootballEnv(seed=0, stage=args.stage, head_tracking="privileged", glance=True, action_mode="twist")
    alpha = 1.0 if args.tau <= 0 else min(1.0, 0.2 / args.tau)
    res, yaw_switches = [], []
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        cmd = np.zeros(3, np.float32); done = False; prev = None; sw = 0
        while not done:
            a = int(teacher.predict(obs, deterministic=True)[0])
            name = ds.PRIMITIVE_NAMES[a]
            kick = name.startswith("kick")
            target = np.array(TWIST.get(name, (0, 0, 0)), np.float32)
            cmd = cmd + alpha * (target - cmd)
            if prev is not None and np.sign(cmd[2]) != np.sign(prev):
                sw += 1
            prev = cmd[2]
            obs, r, te, tr, info = env.step(np.array([*cmd, float(kick)], np.float32))
            obs[OBS_SLICES["prev_action"]] = 0.0          # the teacher expects ITS last primitive here
            obs[OBS_SLICES["prev_action"].start + a] = 1.0
            done = te or tr
        res.append(info); yaw_switches.append(sw / max(info["sim_time"], 1e-6))
    g = [r for r in res if r["is_success"]]
    print(f"smoothed teacher tau={args.tau:.1f}s stage {args.stage}: goals {len(g)}/{len(res)}"
          + (f", time-to-goal {np.mean([r['sim_time'] for r in g]):.1f}s" if g else "")
          + f", falls/ep {np.mean([r['n_falls'] for r in res]):.2f}"
          + f", turn-direction switches {np.mean(yaw_switches):.1f}/s", flush=True)


if __name__ == "__main__":
    main()
