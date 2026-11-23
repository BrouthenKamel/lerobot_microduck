#!/usr/bin/env python3
"""One DAgger round for the camera student: the ACT student plays, the RL teacher labels every
state it visits, the new frames are appended to the LeRobot dataset (as a new merged dataset).

  uv run python -m microduck_football.imitation.dagger_camera --student runs/act_v2/checkpoints/last/pretrained_model \
      --base data/rl_football_v2 --out data/rl_football_v2_d1 --episodes 80 --workers 3
then fine-tune:  lerobot-train --policy.path=<student> --dataset.root=<out> ...

All episodes are kept (failures included: those are exactly the states plain imitation never saw).
Labels: student_io.teacher_label (teacher target smoothed from the command actually executed).
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
from microduck_football import paths  # noqa: E402

TEACHER = paths.TEACHER_OPEN_PLAY


def worker(wid, n_eps, student, shard_root, repo_id, seed0, q, max_seconds=None, near_goal=0.3):
    import numpy as np
    import torch
    from stable_baselines3 import PPO
    from lerobot.common.control_utils import predict_action
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    from microduck_football.envs import football_env as env_module
    from microduck_football.imitation import student_io as sio
    from microduck_football.envs.football_env import FootballEnv

    torch.set_num_threads(1)
    device = torch.device("cuda")
    policy = ACTPolicy.from_pretrained(student).to(device).eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=student,
                                         preprocessor_overrides={"device_processor": {"device": "cuda"}})
    memory = int(policy.config.input_features["observation.state"].shape[0]) == len(sio.STATE_NAMES_MEM)
    teacher = sio.SmoothedTeacher(PPO.load(TEACHER, device="cpu"))
    env_module.REPLAY_EARLIER = 0.0
    env = FootballEnv(seed=wid, stage=3, head_tracking="privileged", glance=True, action_mode="twist")
    lo, hi = env.action_space.low, env.action_space.high
    ds_ = LeRobotDataset.create(repo_id, fps=5, features=sio.features(memory), root=shard_root,
                                robot_type="microduck", use_videos=True)
    rng = np.random.default_rng(seed0 + wid)
    stats = {"eps": 0, "goals": 0, "frames": 0}
    for ep in range(n_eps):
        r = rng.random()           # near_goal share starts at stage 0/1 (ball close to the goal)
        env.set_stage(int(rng.integers(0, 2)) if r < near_goal else (2 if r < near_goal + 0.15 else 3))
        seed = seed0 + wid * 1000 + ep
        obs_priv, _ = env.reset(seed=seed)
        if max_seconds:
            env._limit = min(env._limit, max_seconds)
        policy.reset(); pre.reset(); post.reset(); teacher.reset()
        mem = sio.GoalMemory(seed=seed) if memory else None
        prev = np.zeros(3, np.float32)
        done = False
        while not done:
            label = sio.teacher_label(teacher, obs_priv, prev)
            state = sio.student_state(env.sim, prev, mem)
            img = env.sim.render_head()
            ds_.add_frame({"observation.images.head": img, "observation.state": state,
                           "action": label, "task": sio.TASK})
            stats["frames"] += 1
            with torch.inference_mode():
                out = predict_action({"observation.images.head": img, "observation.state": state},
                                     policy, device, pre, post, use_amp=False)
            a = np.clip(out.squeeze(0).float().cpu().numpy(), lo, hi).astype(np.float32)
            obs_priv, r, te, tr, info = env.step(a)
            obs_priv = teacher.fix_obs(obs_priv)
            prev = a[:3].copy()
            done = te or tr
        ds_.save_episode()
        stats["eps"] += 1
        stats["goals"] += int(info["is_success"])
        q.put(("progress", wid, ep + 1))
    ds_.finalize()
    q.put(("done", wid, stats))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--student", type=Path, required=True)
    ap.add_argument("--base", type=Path, required=True, help="existing dataset root to extend")
    ap.add_argument("--base-repo-id", default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repo-id", default=None)
    ap.add_argument("--episodes", type=int, default=80)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=900000)
    ap.add_argument("--max-seconds", type=float, default=None, help="cap student episodes (sim seconds)")
    ap.add_argument("--near-goal", type=float, default=0.3,
                    help="share of rollouts starting near the goal (stages 0-1), to target finishing")
    args = ap.parse_args()
    base, out = args.base.resolve(), args.out.resolve()
    base_repo = args.base_repo_id or f"local/{base.name}"
    repo = args.repo_id or f"local/{out.name}"
    if out.exists():
        sys.exit(f"{out} exists")
    shard_dir = out.parent / f"{out.name}_shards"
    shutil.rmtree(shard_dir, ignore_errors=True)
    shard_dir.mkdir(parents=True)
    per = [args.episodes // args.workers + (i < args.episodes % args.workers) for i in range(args.workers)]
    ctx = mp.get_context("spawn")       # CUDA in the workers
    q = ctx.Queue()
    t0 = time.time()
    procs, roots, repos = [], [], []
    for i, n in enumerate(per):
        r = shard_dir / f"shard{i}"
        roots.append(r)
        repos.append(f"{repo}_shard{i}")
        p = ctx.Process(target=worker, args=(i, n, str(args.student.resolve()), str(r), repos[-1], args.seed, q, args.max_seconds,
                                                     args.near_goal))
        p.start()
        procs.append(p)
    done, prog, tot = 0, [0] * len(procs), {"eps": 0, "goals": 0, "frames": 0}
    while done < len(procs):
        kind, wid, payload = q.get()
        if kind == "progress":
            prog[wid] = payload
            print(f"\r student episodes {sum(prog)}/{args.episodes} ({time.time() - t0:4.0f}s)", end="", flush=True)
        else:
            done += 1
            for k in tot:
                tot[k] += payload[k]
    for p in procs:
        p.join()
    print(f"\nstudent rollouts: {tot['eps']} episodes, {tot['goals']} goals, {tot['frames']} teacher-labelled "
          f"frames in {time.time() - t0:.0f}s", flush=True)
    from lerobot.datasets.aggregate import aggregate_datasets
    aggregate_datasets(repo_ids=[base_repo] + repos, aggr_repo_id=repo, roots=[base] + roots, aggr_root=out)
    shutil.rmtree(shard_dir)
    print(f"merged dataset: {out} ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
