#!/usr/bin/env python3
"""Generate a LeRobotDataset from RL-teacher rollouts (duck camera + state -> teacher primitive).

  uv run python -m microduck_football.imitation.gen_dataset --episodes 200 --workers 6 --out data/rl_football_v0

Each worker plays episodes with the PPO teacher in FootballEnv (head tracking + goal glances on),
records at the decision rate (5 Hz):
  observation.images.head  (240, 320, 3) duck head camera, video
  observation.state        (20,)  gyro, projected gravity, 14 joint positions (incl. head)
  action                   (9,)   one-hot over duck_sim.PRIMITIVE_NAMES (the teacher's choice)
and writes its own shard; shards are merged with lerobot's aggregate_datasets at the end.
Noise injection: in a share of episodes a random non-kick primitive is EXECUTED now and then
(the label stays the teacher's), so the data contains recoveries from off-teacher states.
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

ROOT = paths.REPO_ROOT
TEACHER = paths.TEACHER_OPEN_PLAY
TASK = "dribble the ball into the goal"


def features():
    from microduck_football.sim import duck_sim as ds
    return {
        "observation.images.head": {"dtype": "video", "shape": (ds.IMG_H, ds.IMG_W, 3),
                                    "names": ["height", "width", "channels"]},
        "observation.state": {"dtype": "float32", "shape": (len(ds.STATE_NAMES),),
                              "names": list(ds.STATE_NAMES)},
        "action": {"dtype": "float32", "shape": (len(ds.PRIMITIVES),),
                   "names": list(ds.PRIMITIVE_NAMES)},
    }


def worker(wid: int, n_episodes: int, shard_root: str, repo_id: str, args_dict: dict, out_q):
    import numpy as np
    import torch
    from stable_baselines3 import PPO
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from microduck_football.sim import duck_sim as ds
    from microduck_football.envs import football_env as env_module
    from microduck_football.imitation import student_io as sio
    from microduck_football.envs.football_env import FootballEnv

    torch.set_num_threads(1)
    twist = args_dict["action_mode"] == "twist"
    env_module.REPLAY_EARLIER = 0.0
    rng = np.random.default_rng(1000 + wid)
    teacher = PPO.load(args_dict["teacher"], device="cpu")
    env = FootballEnv(seed=wid, stage=3, head_tracking="privileged", glance=True,
                      action_mode="twist" if twist else "primitive",
                      obstacles=args_dict.get("obstacles", False))
    smoothed = sio.SmoothedTeacher(teacher)
    memory = args_dict.get("memory", False)
    dataset = LeRobotDataset.create(repo_id, fps=5, features=sio.features(memory) if twist else features(),
                                    root=shard_root, robot_type="microduck", use_videos=True)
    kicks = {ds.P["kick_left"], ds.P["kick_right"]}
    walkish = [i for i in range(len(ds.PRIMITIVES)) if i not in kicks]
    stats = {"kept": 0, "failed": 0, "frames": 0, "noisy": 0}
    for ep in range(n_episodes):
        if args_dict.get("stages"):
            stage = int(rng.choice(args_dict["stages"]))
        else:
            stage = 3 if rng.random() < args_dict["stage3_share"] else int(rng.integers(1, 3))
        noisy = rng.random() < args_dict["noise_episode_share"]
        env.set_stage(stage)
        obs, info = env.reset(seed=100000 + wid * 10000 + ep)
        done, n = False, 0
        if twist:
            smoothed.reset()
            prev_cmd = np.zeros(3, np.float32)
            mem = sio.GoalMemory(seed=100000 + wid * 10000 + ep) if memory else None
            while not done:
                label = smoothed.act(obs)                       # smoothed teacher command
                dataset.add_frame({"observation.images.head": env.sim.render_head(),
                                   "observation.state": sio.student_state(env.sim, prev_cmd, mem),
                                   "action": label, "task": sio.TASK})
                n += 1
                act = label
                if noisy and rng.random() < args_dict["noise_p"]:
                    name = ds.PRIMITIVE_NAMES[int(rng.choice(walkish))]
                    act = np.array([*sio.TWIST[name], 0.0], np.float32)
                obs, r, term, trunc, info = env.step(act)
                obs = smoothed.fix_obs(obs)
                prev_cmd = act[:3].copy()
                done = term or trunc
        while not done:
            label = int(teacher.predict(obs, deterministic=True)[0])
            if env.sim.policy.behavior_mode is not None:       # kick running: env ignores actions
                label = int(np.argmax(obs[env_module.OBS_SLICES["prev_action"]]))
            onehot = np.zeros(len(ds.PRIMITIVES), np.float32)
            onehot[label] = 1.0
            dataset.add_frame({"observation.images.head": env.sim.render_head(),
                               "observation.state": env.sim.state_vector(),
                               "action": onehot, "task": TASK})
            n += 1
            act = label
            if noisy and rng.random() < args_dict["noise_p"]:
                act = int(rng.choice(walkish))
            obs, r, term, trunc, info = env.step(act)
            done = term or trunc
        if info["is_success"]:
            dataset.save_episode()
            stats["kept"] += 1
            stats["frames"] += n
            stats["noisy"] += int(noisy)
        else:
            dataset.clear_episode_buffer()
            stats["failed"] += 1
        out_q.put(("progress", wid, ep + 1))
    dataset.finalize()
    out_q.put(("done", wid, stats))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", type=Path, default=paths.DATA_DIR / "rl_football_v0")
    ap.add_argument("--repo-id", default="local/rl_football_v0")
    ap.add_argument("--teacher", type=Path, default=TEACHER)
    ap.add_argument("--stage3-share", type=float, default=0.7)
    ap.add_argument("--noise-episode-share", type=float, default=0.25)
    ap.add_argument("--noise-p", type=float, default=0.15)
    ap.add_argument("--action-mode", choices=["primitive", "twist"], default="twist",
                    help="twist: smoothed continuous command labels + 25-D state (student_io.py)")
    ap.add_argument("--memory", action="store_true", help="32-D state with student_io.GoalMemory")
    ap.add_argument("--obstacles", action="store_true",
                    help="opponent layouts (needs an obstacle teacher, e.g. ppo_opponents_v2_keeper_ladder)")
    ap.add_argument("--stages", type=int, nargs="+", default=None,
                    help="sample episode stages uniformly from these (overrides --stage3-share)")
    args = ap.parse_args()

    out = args.out.resolve()
    if out.exists():
        sys.exit(f"{out} exists - pick another --out or delete it")
    shard_dir = out.parent / f"{out.name}_shards"
    shutil.rmtree(shard_dir, ignore_errors=True)
    shard_dir.mkdir(parents=True)
    per = [args.episodes // args.workers + (1 if i < args.episodes % args.workers else 0)
           for i in range(args.workers)]
    args_dict = {"teacher": str(args.teacher), "stage3_share": args.stage3_share,
                 "noise_episode_share": args.noise_episode_share, "noise_p": args.noise_p,
                 "action_mode": args.action_mode, "memory": args.memory,
                 "obstacles": args.obstacles, "stages": args.stages}

    t0 = time.time()
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    procs, roots, repo_ids = [], [], []
    for i, n in enumerate(per):
        root = shard_dir / f"shard{i}"
        repo = f"{args.repo_id}_shard{i}"
        roots.append(root)
        repo_ids.append(repo)
        p = ctx.Process(target=worker, args=(i, n, str(root), repo, args_dict, q))
        p.start()
        procs.append(p)
    done, progress, totals = 0, [0] * args.workers, {"kept": 0, "failed": 0, "frames": 0, "noisy": 0}
    while done < len(procs):
        kind, wid, payload = q.get()
        if kind == "progress":
            progress[wid] = payload
            print(f"\r episodes {sum(progress)}/{args.episodes}  ({time.time() - t0:5.0f}s)", end="", flush=True)
        else:
            done += 1
            for k in totals:
                totals[k] += payload[k]
    for p in procs:
        p.join()
    t_gen = time.time() - t0
    print(f"\ngenerated in {t_gen:.0f}s: kept {totals['kept']} goal episodes "
          f"({totals['noisy']} with noise injection), dropped {totals['failed']} failures, "
          f"{totals['frames']} frames ({totals['frames'] / t_gen:.0f} frames/s)")

    from lerobot.datasets.aggregate import aggregate_datasets
    t1 = time.time()
    aggregate_datasets(repo_ids=repo_ids, aggr_repo_id=args.repo_id, roots=roots, aggr_root=out)
    shutil.rmtree(shard_dir)
    print(f"merged into {out} in {time.time() - t1:.0f}s (total {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
