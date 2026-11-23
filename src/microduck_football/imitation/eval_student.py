#!/usr/bin/env python3
"""Closed-loop evaluation of a LeRobot policy (e.g. ACT) in FootballEnv, from pixels + state only.

  uv run python -m microduck_football.imitation.eval_student runs/act_e2e_v0/checkpoints/last/pretrained_model \
      --stage 3 --episodes 20 [--video assets/media/act_e2e_v0.mp4]

The student sees what the dataset recorded (head camera + 20-D state) and outputs a 9-D score per
primitive; we execute the argmax. Head tracking + glances run as part of the robot, as during data
generation. Also reports agreement with the RL teacher on the same states (imitation quality,
independent of where the student drives the duck).
"""

from __future__ import annotations

import argparse
import collections
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
from microduck_football import paths  # noqa: E402

import numpy as np  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path, help=".../checkpoints/last/pretrained_model")
    ap.add_argument("--stage", type=int, default=3)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=777000)
    ap.add_argument("--teacher", type=Path, default=paths.TEACHER_OPEN_PLAY)
    ap.add_argument("--video", type=Path, default=None, help="mp4 of the first --video-episodes episodes")
    ap.add_argument("--video-episodes", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-action-steps", type=int, default=None,
                    help="override how many chunk actions run before re-planning (1 = every decision)")
    ap.add_argument("--temporal-ensemble", type=float, default=None,
                    help="ACT temporal ensembling coefficient (e.g. 0.01); re-plans every decision")
    args = ap.parse_args()

    import torch
    from stable_baselines3 import PPO
    from lerobot.common.control_utils import predict_action
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    from microduck_football.sim import duck_sim as ds
    from microduck_football.envs import football_env as env_module
    from microduck_football.imitation import student_io as sio
    from microduck_football.envs.football_env import FootballEnv
    from microduck_football.rl.play_heuristic import caption

    ckpt = str(args.checkpoint.resolve())
    device = torch.device(args.device)
    policy = ACTPolicy.from_pretrained(ckpt)
    policy.to(device)
    policy.eval()
    if args.temporal_ensemble is not None:
        from lerobot.policies.act.modeling_act import ACTTemporalEnsembler
        policy.config.temporal_ensemble_coeff = args.temporal_ensemble
        policy.config.n_action_steps = 1
        policy.temporal_ensembler = ACTTemporalEnsembler(args.temporal_ensemble, policy.config.chunk_size)
    elif args.n_action_steps is not None:
        policy.config.n_action_steps = args.n_action_steps
    mode = (f"temporal-ensemble {args.temporal_ensemble}" if args.temporal_ensemble is not None
            else f"n_action_steps={policy.config.n_action_steps}")
    pre, post = make_pre_post_processors(
        policy.config, pretrained_path=ckpt,
        preprocessor_overrides={"device_processor": {"device": str(device)}})
    teacher = PPO.load(args.teacher, device="cpu")

    act_dim = int(policy.config.output_features["action"].shape[0])
    state_dim = int(policy.config.input_features["observation.state"].shape[0])
    memory = state_dim == len(sio.STATE_NAMES_MEM)
    twist = act_dim == 4                           # twist students (student_io) vs one-hot primitive
    env_module.REPLAY_EARLIER = 0.0
    env = FootballEnv(seed=args.seed, stage=args.stage, head_tracking="privileged", glance=True,
                      action_mode="twist" if twist else "primitive")
    smoothed = sio.SmoothedTeacher(teacher)
    lo, hi = env.action_space.low if twist else None, env.action_space.high if twist else None

    frames, label = [], {"text": ""}
    if args.video:
        def grab(sim, _n=[0]):
            if label.get("ep", 0) >= args.video_episodes:
                return
            _n[0] += 1
            if _n[0] % 2 == 0:
                head = sim.render_head().copy()
                frames.append(caption(np.concatenate([sim.render_overview(), head], axis=1), label["text"]))
        env.on_control_step = grab

    res, agree, hist = [], [], collections.Counter()
    t0 = time.time()
    for ep in range(args.episodes):
        obs_priv, info = env.reset(seed=args.seed + ep)
        policy.reset(); pre.reset(); post.reset(); smoothed.reset()
        prev_cmd = np.zeros(3, np.float32)
        mem = sio.GoalMemory(seed=args.seed + ep) if memory else None
        done = False
        while not done:
            state = sio.student_state(env.sim, prev_cmd, mem) if twist else env.sim.state_vector()
            obs = {"observation.images.head": env.sim.render_head(), "observation.state": state}
            with torch.inference_mode():
                out = predict_action(obs, policy, device, pre, post, use_amp=False)
            out = out.squeeze(0).float().cpu().numpy()
            if twist:
                a = np.clip(out, lo, hi).astype(np.float32)
                t_cmd = smoothed.act(obs_priv)        # teacher's smoothed command on this state
                if env.sim.policy.behavior_mode is None:
                    agree.append(abs(float(a[2]) - float(t_cmd[2])))     # yaw-rate error
                    hist["turn left" if a[2] > 0.2 else "turn right" if a[2] < -0.2 else "straight"] += 1
                label.update(ep=ep, text=f"ACT student  ep {ep}  t={env.sim.t:5.1f}s  cmd vx={a[0]:+.2f} "
                                         f"vy={a[1]:+.2f} yaw={a[2]:+.2f}  teacher yaw={t_cmd[2]:+.2f}")
                obs_priv, r, term, trunc, info = env.step(a)
                obs_priv = smoothed.fix_obs(obs_priv)
                prev_cmd = a[:3].copy()
            else:
                a = int(out.argmax())
                t_a = int(teacher.predict(obs_priv, deterministic=True)[0])
                if env.sim.policy.behavior_mode is None:     # only count decisions the env executes
                    agree.append(a == t_a)
                    hist[ds.PRIMITIVE_NAMES[a]] += 1
                label.update(ep=ep, text=f"ACT student  ep {ep}  t={env.sim.t:5.1f}s  action={ds.PRIMITIVE_NAMES[a]:12s}"
                                         f"  teacher={ds.PRIMITIVE_NAMES[t_a]}")
                obs_priv, r, term, trunc, info = env.step(a)
            done = term or trunc
        res.append(info)
        print(f"ep {ep:3d}: {'GOAL   ' if info['is_success'] else 'timeout'} t={info['sim_time']:5.1f}s "
              f"falls={info['n_falls']} ball->goal={info['ball_dist_to_goal']:.2f} m", flush=True)

    goals = [r for r in res if r["is_success"]]
    stalls = [r for r in res if not r["is_success"] and r["ball_dist_to_goal"] < 0.5]
    print(f"\nACT student [{mode}] / stage {args.stage}: goals {len(goals)}/{len(res)}"
          + (f", time-to-goal {np.mean([r['sim_time'] for r in goals]):.1f}s" if goals else "")
          + f", falls/ep {np.mean([r['n_falls'] for r in res]):.2f}"
          + (f", |yaw-rate error| vs smoothed teacher {np.mean(agree):.2f} rad/s" if twist
             else f", agreement with teacher {100 * np.mean(agree):.0f}% of decisions")
          + f", near-goal stalls {len(stalls)} (timeouts with the ball < 0.5 m from goal)"
          + f", eval wall time {time.time() - t0:.0f}s")
    print("student action histogram:", dict(hist.most_common()))
    if args.video and frames:
        import imageio.v3 as iio
        iio.imwrite(args.video, np.stack(frames), fps=25, codec="libx264")
        print(f"video: {args.video} ({len(frames) / 25:.0f} s)")


if __name__ == "__main__":
    main()
