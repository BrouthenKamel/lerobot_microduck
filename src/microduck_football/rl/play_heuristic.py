#!/usr/bin/env python3
"""Run the heuristic player (or a trained PPO checkpoint) in FootballEnv: per-episode stats,
optional video / live viewer.

  uv run python -m microduck_football.rl.play_heuristic --model runs/<run>/checkpoints/ppo_XXX_steps.zip --stage 3 --episodes 3 --video out.mp4

  uv run python -m microduck_football.rl.play_heuristic --episodes 20 --difficulty easy
  uv run python -m microduck_football.rl.play_heuristic --episodes 3 --difficulty full --video heuristic_demo.mp4
  uv run python -m microduck_football.rl.play_heuristic --episodes 1 --viewer
  uv run python -m microduck_football.rl.play_heuristic --episodes 20 --random        # baseline
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from microduck_football.sim import duck_sim as ds  # noqa: E402
from microduck_football.envs.football_env import FootballEnv  # noqa: E402
from microduck_football.agents.heuristic import HeuristicPlayer  # noqa: E402


def caption(frame: np.ndarray, text: str) -> np.ndarray:
    from PIL import Image, ImageDraw
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, img.width, 16], fill=(0, 0, 0))
    d.text((4, 2), text, fill=(255, 255, 255))
    return np.asarray(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--difficulty", choices=["easy", "full"], default="full")
    ap.add_argument("--max-seconds", type=float, default=90.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=str, default=None,
                    help="comma-separated episode seeds (overrides --episodes/--seed)")
    ap.add_argument("--random", action="store_true", help="random actions (baseline)")
    ap.add_argument("--model", type=Path, default=None, help="play a trained PPO checkpoint (.zip)")
    ap.add_argument("--stage", type=int, default=None, help="curriculum stage layout (0-3) instead of --difficulty")
    ap.add_argument("--head-tracking", choices=["privileged", "detector"], default=None,
                    help="ball-tracking head reflex (see head_tracking.py); default: head level")
    ap.add_argument("--glance", action="store_true", help="head tracker also glances at the goal")
    ap.add_argument("--video", type=Path, default=None, help="write overview + head-camera mp4")
    ap.add_argument("--viewer", action="store_true", help="watch live in the MuJoCo viewer")
    args = ap.parse_args()

    model = None
    if args.model is not None:
        from stable_baselines3 import PPO
        model = PPO.load(args.model, device="cpu")
    from microduck_football.envs import football_env as env_module
    obstacles = model is not None and model.observation_space.shape[0] > env_module.OBS_DIM
    env = FootballEnv(difficulty=args.difficulty, max_seconds=args.max_seconds, seed=args.seed,
                      stage=args.stage, head_tracking=args.head_tracking, glance=args.glance,
                      obstacles=obstacles)
    if args.stage is not None:
        from microduck_football.envs import football_env as env_module
        env_module.REPLAY_EARLIER = 0.0
    player = HeuristicPlayer()
    who = "random" if args.random else ("ppo" if model is not None else "heuristic")
    rng = np.random.default_rng(args.seed)

    frames, label = [], {"text": ""}
    if args.video:
        def grab(sim, _n=[0]):
            _n[0] += 1
            if _n[0] % 2 == 0:   # 25 fps
                head = sim.render_head().copy()
                cy, cx = head.shape[0] // 2, head.shape[1] // 2      # centre crosshair
                head[cy - 1:cy + 2, cx - 8:cx + 9] = (255, 255, 255)
                head[cy - 8:cy + 9, cx - 1:cx + 2] = (255, 255, 255)
                text = label["text"]
                if env.tracker is not None:
                    t = env.tracker
                    look = {"to_goal": "->GOAL", "hold_goal": "GOAL", "to_ball": "->ball"}.get(
                        getattr(t, "mode", "ball"), "BALL" if t.seen else "search")
                    text += f"  head yaw={t.yaw:+.2f} pitch={t.pitch:+.2f} look:{look}"
                frames.append(caption(np.concatenate([sim.render_overview(), head], axis=1), text))
        env.on_control_step = grab

    viewer = None
    if args.viewer:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(env.sim.model, env.sim.data,
                                              show_left_ui=False, show_right_ui=False)
        viewer.cam.lookat[:] = (0.4, 0.0, 0.0)
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 3.6, 90, -55

        def sync(sim, _last=[time.perf_counter()]):
            viewer.sync()
            _last[0] += sim.dt
            time.sleep(max(0.0, _last[0] - time.perf_counter()))
            if time.perf_counter() - _last[0] > 0.5:
                _last[0] = time.perf_counter()
        env.on_control_step = sync

    seeds = ([int(x) for x in args.seeds.split(",")] if args.seeds
             else [args.seed + i for i in range(args.episodes)])
    results = []
    for ep, ep_seed in enumerate(seeds):
        obs, info = env.reset(seed=ep_seed)
        player.reset()
        ret, done = 0.0, False
        while not done:
            if args.random:
                a = int(rng.integers(len(ds.PRIMITIVES)))
            elif model is not None:
                a = int(model.predict(obs, deterministic=True)[0])
            else:
                a = player.act(obs)
            state = "" if model is not None or args.random else f"  state={player.state}"
            label["text"] = f"{who}  ep {ep}  t={env.sim.t:5.1f}s  action={ds.PRIMITIVE_NAMES[a]:12s}{state}"
            obs, r, term, trunc, info = env.step(a)
            ret += r
            done = term or trunc
            if viewer is not None and not viewer.is_running():
                done = True
        results.append(info | {"return": ret})
        print(f"ep {ep:3d}: {'GOAL   ' if info['is_success'] else 'timeout'} t={info['sim_time']:5.1f}s "
              f"kicks={info['n_kicks']:2d} falls={info['n_falls']:2d} return={ret:6.2f} "
              f"ball->goal={info['ball_dist_to_goal']:.2f} m", flush=True)
        if viewer is not None and not viewer.is_running():
            break

    goals = [r for r in results if r["is_success"]]
    print(f"\n{who} / {'stage ' + str(args.stage) if args.stage is not None else args.difficulty}: "
          f"goals {len(goals)}/{len(results)}"
          + (f", mean time-to-goal {np.mean([r['sim_time'] for r in goals]):.1f}s"
             f", mean kicks/goal {np.mean([r['n_kicks'] for r in goals]):.1f}" if goals else "")
          + f", falls/episode {np.mean([r['n_falls'] for r in results]):.1f}")

    if args.video and frames:
        import imageio.v3 as iio
        iio.imwrite(args.video, np.stack(frames), fps=25, codec="libx264")
        print(f"video: {args.video} ({len(frames) / 25:.0f} s)")
    if viewer is not None:
        viewer.close()


if __name__ == "__main__":
    main()
